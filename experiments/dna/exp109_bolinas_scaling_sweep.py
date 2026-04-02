# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Bolinas DNA transferred scaling sweep (marin#4251).

DNA parameter scaling sweep using Completed AdamH heuristics, adapted from the
text reference sweep (marin#2432). See bolinas-dna#109 for full context.

Subcommands:
    run_smoke_test                 ~20-step infrastructure validation
    run_reference_tuning_sweep     Vizier Bayesian optimization over AdamH hparams
    run_transfer_validation_sweep  Sweep key hypers in isolation at largest scale
    run_parameter_scaling_sweep    IsoFLOP parameter scaling via CompletedAdamH
"""

import logging
import os
from dataclasses import dataclass, replace
from datetime import timedelta
from functools import lru_cache

import jmp
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import DNALmDatasetFormat
from levanter.main.train_lm import TrainLmConfig
from levanter.optim import AdamHConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig

from experiments.defaults import default_tokenize, default_train
from experiments.dna.defaults import dna_effective_seq_len
from experiments.evals.task_configs import TRAITGYM_MENDELIAN_V2_255
from experiments.references.reference_hyperparameter_sweep import (
    SUGGESTIONS_FILENAME,
    VIZIER_DB_FILENAME,
    VizierOptimalConfig,
    VizierSuggestConfig,
    VizierUpdateConfig,
    _extract_adamh_hparams,
    _load_suggestions,
    run_vizier_optimal,
    run_vizier_suggest,
    run_vizier_update,
)
from experiments.scaling_law_sweeps.completed_adamh import CompletedAdamHHeuristic
from experiments.simple_train_config import SimpleTrainConfig
from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.execution.remote import remote
from marin.processing.tokenize import lm_mixture_data_config
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm

# =============================================================================
# Module-level constants
# =============================================================================

TOKENIZER = "bolinas-dna/tokenizer-char-bos"
DNA_BASE_SEQ_LEN = 255  # bp (256 - 1 for BOS)

TRAIN_DATASETS = {
    "cds": "bolinas-dna/genomes-v5-genome_set-animals-intervals-v5_255_128",
    "upstream": "bolinas-dna/genomes-v5-genome_set-animals-intervals-v1_255_128",
    "downstream": "bolinas-dna/genomes-v5-genome_set-animals-intervals-v15_255_128",
}
TRAIN_WEIGHTS = {"cds": 0.7319, "upstream": 0.2062, "downstream": 0.0619}

VALIDATION_DATASETS = {
    "val_cds": "bolinas-dna/genomes-v5-validation-intervals-v5_255_255",
    "val_upstream": "bolinas-dna/genomes-v5-validation-intervals-v1_255_255",
    "val_downstream": "bolinas-dna/genomes-v5-validation-intervals-v15_255_255",
}

DNA_TPU_TYPE: str | None = os.getenv("DNA_TPU_TYPE")
DNA_RESOURCES: ResourceConfig | None = ResourceConfig.with_tpu(DNA_TPU_TYPE) if DNA_TPU_TYPE else None

# Reference sweep sizing
REFERENCE_HIDDEN_SIZE = 512  # ~25M params with vocab_size=7
INITIALIZER_RANGES = (0.04, 0.02, 0.01, 0.005, 0.0025)
EPOCHS = (1,)

# Vizier search space (same as text reference sweep)
SEARCH_SPACE = {
    "lr": (0.00005, 0.03),
    "beta1": (0.5, 1.0),
    "adam_lr": (0.00005, 0.03),
    "beta2": (0.5, 1.0),
    "epsilon": (1e-15, 1e-3),
    "max_grad_norm": (0.1, 1.0),
    "z_loss_weight": (1e-7, 0.1),
}

# Vizier sweep parameters
NUM_LOOPS = 10
SUGGESTIONS_PER_LOOP = 4
TARGET_TOKENS = 2_500_000_000  # ~100:1 token-to-param for ~25M
FIXED_BATCH_SIZE = 16384  # tokens/batch: 16384*256 = 4M (16x reference_hyperparameter_sweep 64*4096=256K)
METRIC_KEY = "eval/loss"  # TODO: confirm from smoke test tracker_metrics.jsonl
METRIC_FILE = "tracker_metrics.jsonl"
METRIC_MODE = "min"
VIZIER_ALGORITHM = "DEFAULT"
STUDY_OWNER = "marin"
WANDB_PROJECT = "marin"


def _get_initializer_ranges() -> tuple[float, ...]:
    """Parse SWEEP_IR_VALUES env var (comma-separated) or return all IR values."""
    raw = os.getenv("SWEEP_IR_VALUES")
    if not raw:
        return INITIALIZER_RANGES
    values = tuple(float(v) for v in raw.split(","))
    invalid = set(values) - set(INITIALIZER_RANGES)
    if invalid:
        raise ValueError(f"Invalid IR values {invalid}. Must be in {INITIALIZER_RANGES}")
    return values


# Schedule (same as text reference)
LR_SCHEDULE = "linear"
WARMUP_FRACTION = 0.1
DECAY_FRACTION = 0.2


# DNA-specific heuristic instance (vocab_size=7 affects param counting)
DNA_HEURISTIC = CompletedAdamHHeuristic(tokenizer=TOKENIZER)

_EXPECTED_VOCAB_SIZE_WARNING = f"Tokenizer {TOKENIZER!r} not found in _KNOWN_VOCAB_SIZES"
logging.getLogger("marin.processing.tokenize.data_configs").addFilter(
    lambda record: _EXPECTED_VOCAB_SIZE_WARNING not in record.getMessage()
)


# =============================================================================
# Shared builders
# =============================================================================


@lru_cache(maxsize=1)
def _model_seq_len() -> int:
    """Model context size = DNA base seq len + special tokens (BOS)."""
    return dna_effective_seq_len(DNA_BASE_SEQ_LEN, TOKENIZER)


def _num_train_steps(target_tokens: int = TARGET_TOKENS, batch_size: int = FIXED_BATCH_SIZE) -> int:
    return target_tokens // (batch_size * _model_seq_len())


def _build_model_config(hidden_size: int, initializer_range: float = 0.02):
    """Build Qwen3Config via the heuristic's architecture formula."""
    config = DNA_HEURISTIC._build_model_config(hidden_size, _model_seq_len())
    return replace(config, initializer_range=initializer_range)


def _tokenize_dataset(name: str, dataset: str) -> ExecutorStep:
    return default_tokenize(
        name=name,
        dataset=dataset,
        tokenizer=TOKENIZER,
        format=DNALmDatasetFormat(lowercase_weight=0.01),
    )


def _build_data_mixture():
    """Tokenize train + validation datasets; validation has weight=0 (eval-only)."""
    tokenized = {
        region: _tokenize_dataset(f"bolinas-v5-{region}-char-bos", dataset) for region, dataset in TRAIN_DATASETS.items()
    }
    for region, dataset in VALIDATION_DATASETS.items():
        tokenized[region] = _tokenize_dataset(f"bolinas-v5-{region}-char-bos", dataset)
    # TRAIN_WEIGHTS only covers train keys; missing_weights_are_validation=True (default)
    # assigns weight=0 to val_* keys, making them validation-only.
    return lm_mixture_data_config(components=tokenized, weights=TRAIN_WEIGHTS)


# =============================================================================
# Reference sweep: AdamH config builder
# =============================================================================


def _build_adamh_config(
    *,
    learning_rate: float,
    beta1: float,
    adam_learning_rate: float,
    beta2: float,
    epsilon: float,
    max_grad_norm: float,
) -> AdamHConfig:
    return AdamHConfig(
        learning_rate=learning_rate,
        adam_lr=adam_learning_rate,
        min_lr_ratio=0.0,
        warmup=WARMUP_FRACTION,
        decay=DECAY_FRACTION,
        lr_schedule=LR_SCHEDULE,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        max_grad_norm=max_grad_norm,
        nesterov=False,
    )


# =============================================================================
# Reference sweep: base training config
# =============================================================================


def _final_checkpoint_only(num_steps: int) -> CheckpointerConfig:
    """Save a single permanent checkpoint at the final training step."""
    return CheckpointerConfig(
        save_interval=timedelta(days=365),  # no time-based saves
        keep=[dict(every=num_steps)],  # permanent checkpoint at final step only
    )


def _build_base_train_config(
    model_config,
    data_mixture,
    *,
    wandb_group: str,
    base_tags: tuple[str, ...],
    checkpointer: CheckpointerConfig,
    steps_per_eval: int,
) -> TrainLmOnPodConfig:
    """Build TrainLmOnPodConfig with placeholder optimizer values.

    Placeholders for lr/batch_size/steps/z_loss are replaced at runtime
    inside run_dna_vizier_train once Vizier suggestions are available.
    """
    placeholder_lr = SEARCH_SPACE["lr"][0]
    placeholder_beta1 = SEARCH_SPACE["beta1"][0]
    placeholder_adam_lr = SEARCH_SPACE["adam_lr"][0]
    placeholder_beta2 = SEARCH_SPACE["beta2"][0]
    placeholder_epsilon = SEARCH_SPACE["epsilon"][0]
    placeholder_max_grad_norm = SEARCH_SPACE["max_grad_norm"][0]
    placeholder_z_loss = SEARCH_SPACE["z_loss_weight"][0]
    placeholder_steps = TARGET_TOKENS // (FIXED_BATCH_SIZE * _model_seq_len())

    inner = TrainLmConfig(
        data=data_mixture,
        model=model_config,
        train_seq_len=_model_seq_len(),
        z_loss_weight=placeholder_z_loss,
        optimizer=_build_adamh_config(
            learning_rate=placeholder_lr,
            beta1=placeholder_beta1,
            adam_learning_rate=placeholder_adam_lr,
            beta2=placeholder_beta2,
            epsilon=placeholder_epsilon,
            max_grad_norm=placeholder_max_grad_norm,
        ),
        trainer=TrainerConfig(
            tracker=WandbConfig(
                project=WANDB_PROJECT,
                tags=list(base_tags),
                group=wandb_group,
                name=None,
                replicate_path=this_output_path(),
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=FIXED_BATCH_SIZE,
            num_train_steps=placeholder_steps,
            steps_per_eval=steps_per_eval,
            checkpointer=checkpointer,
            mesh=MeshConfig(axes={"replica": 1, "data": -1, "model": 1}),
            allow_nondivisible_batch_size=True,
        ),
    )

    return TrainLmOnPodConfig(
        train_config=inner,
        resources=DNA_RESOURCES,
        output_path=this_output_path(),
    )


# =============================================================================
# Reference sweep: Vizier train config and function
# =============================================================================


@dataclass(frozen=True)
class DnaVizierTrainConfig:
    """Config passed to run_dna_vizier_train at execution time."""

    suggestions_path: str
    suggestion_index: int
    base_train_config: TrainLmOnPodConfig
    target_tokens: int
    seq_len: int
    fixed_batch_size: int
    loop_index: int
    initializer_range: float
    epochs: int
    version: str
    wandb_group: str
    base_tags: tuple[str, ...]


def run_dna_vizier_train(config: DnaVizierTrainConfig) -> None:
    """Train a DNA model for a single Vizier suggestion."""
    suggestions = _load_suggestions(config.suggestions_path)["suggestions"]
    if config.suggestion_index >= len(suggestions):
        raise IndexError(f"Suggestion index {config.suggestion_index} out of range")

    suggestion = suggestions[config.suggestion_index]
    hparams = _extract_adamh_hparams(suggestion)
    batch_size = config.fixed_batch_size
    num_steps = config.target_tokens // (batch_size * config.seq_len)
    trial_id = int(suggestion["trial_id"])

    base = config.base_train_config
    num_params = base.train_config.model.total_trainable_params(DNA_HEURISTIC.vocab_size)

    run_name = (
        f"dna-bolinas-reference-{config.version}"
        f"-IR{config.initializer_range}-E{config.epochs}"
        f"-L{config.loop_index}-T{trial_id}"
    )
    new_tags = [
        *config.base_tags,
        f"lr={hparams['lr']}",
        f"beta1={hparams['beta1']}",
        f"adam_lr={hparams['adam_lr']}",
        f"beta2={hparams['beta2']}",
        f"eps={hparams['epsilon']}",
        f"mgn={hparams['max_grad_norm']}",
        f"zloss={hparams['z_loss_weight']}",
        f"bs={batch_size}",
        f"params={num_params}",
        f"tokens={config.target_tokens}",
        f"trial={trial_id}",
        f"loop={config.loop_index}",
    ]
    inner = replace(
        base.train_config,
        optimizer=_build_adamh_config(
            learning_rate=hparams["lr"],
            beta1=hparams["beta1"],
            adam_learning_rate=hparams["adam_lr"],
            beta2=hparams["beta2"],
            epsilon=hparams["epsilon"],
            max_grad_norm=hparams["max_grad_norm"],
        ),
        z_loss_weight=hparams["z_loss_weight"],
        trainer=replace(
            base.train_config.trainer,
            num_train_steps=num_steps,
            train_batch_size=batch_size,
            tracker=replace(
                base.train_config.trainer.tracker,
                tags=new_tags,
                name=run_name,
            ),
        ),
    )
    pod_config = replace(base, train_config=inner)
    run_levanter_train_lm(pod_config)


# =============================================================================
# Reference sweep: step builders
# =============================================================================


def _build_dna_suggest_step(
    study_id: str,
    loop_index: int,
    input_db_path,
) -> ExecutorStep:
    client_id = f"{study_id}-loop-{loop_index}"
    return ExecutorStep(
        name=f"{study_id}-suggest-loop{loop_index}",
        fn=remote(run_vizier_suggest, resources=ResourceConfig.with_cpu(), pip_dependency_groups=["vizier"]),
        config=VizierSuggestConfig(
            study_owner=STUDY_OWNER,
            study_id=study_id,
            input_db_path=input_db_path,
            output_path=this_output_path(),
            num_suggestions=SUGGESTIONS_PER_LOOP,
            client_id=client_id,
            metric_key=METRIC_KEY,
            mode=METRIC_MODE,
            algorithm=VIZIER_ALGORITHM,
            search_space=SEARCH_SPACE,
            loop_index=loop_index,
        ),
    )


def _build_dna_train_step(
    suggest_step: ExecutorStep,
    suggestion_index: int,
    base_train_config: TrainLmOnPodConfig,
    *,
    loop_index: int,
    study_id: str,
    initializer_range: float,
    epochs: int,
    base_tags: tuple[str, ...],
    wandb_group: str,
    version: str,
) -> ExecutorStep:
    return ExecutorStep(
        name=os.path.join(
            "checkpoints",
            f"{study_id}-loop{loop_index}-trial{suggestion_index}",
        ),
        fn=remote(run_dna_vizier_train, resources=ResourceConfig.with_cpu()),
        config=DnaVizierTrainConfig(
            suggestions_path=suggest_step / SUGGESTIONS_FILENAME,
            suggestion_index=suggestion_index,
            base_train_config=base_train_config,
            target_tokens=TARGET_TOKENS,
            seq_len=_model_seq_len(),
            fixed_batch_size=FIXED_BATCH_SIZE,
            loop_index=loop_index,
            initializer_range=initializer_range,
            epochs=epochs,
            version=version,
            wandb_group=wandb_group,
            base_tags=base_tags,
        ),
    )


def _build_dna_update_step(
    study_id: str,
    loop_index: int,
    suggest_step: ExecutorStep,
    training_steps: list[ExecutorStep],
) -> ExecutorStep:
    study_resource_name = f"owners/{STUDY_OWNER}/studies/{study_id}"
    return ExecutorStep(
        name=f"{study_id}-update-loop{loop_index}",
        fn=remote(run_vizier_update, resources=ResourceConfig.with_cpu(), pip_dependency_groups=["vizier"]),
        config=VizierUpdateConfig(
            study_id=study_id,
            study_resource_name=study_resource_name,
            input_db_path=suggest_step / VIZIER_DB_FILENAME,
            suggestions_path=suggest_step / SUGGESTIONS_FILENAME,
            run_paths=[step.as_input_name() for step in training_steps],
            metric_file=METRIC_FILE,
            metric_key=METRIC_KEY,
            mode=METRIC_MODE,
            output_path=this_output_path(),
            loop_index=loop_index,
        ),
    )


def _build_dna_optimal_step(
    study_id: str,
    last_update_step: ExecutorStep,
) -> ExecutorStep:
    study_resource_name = f"owners/{STUDY_OWNER}/studies/{study_id}"
    return ExecutorStep(
        name=f"{study_id}-optimal",
        fn=remote(run_vizier_optimal, resources=ResourceConfig.with_cpu(), pip_dependency_groups=["vizier"]),
        config=VizierOptimalConfig(
            study_id=study_id,
            study_resource_name=study_resource_name,
            input_db_path=last_update_step / VIZIER_DB_FILENAME,
            output_path=this_output_path(),
        ),
    )


# =============================================================================
# Subcommands (selected via SWEEP_COMMAND env var)
# =============================================================================

REFERENCE_VERSION = "v0.6"

COMMANDS = (
    "run_smoke_test",
    "run_reference_tuning_sweep",
    "run_transfer_validation_sweep",
    "run_parameter_scaling_sweep",
)


def run_smoke_test():
    """Run ~20-step infrastructure validation with the sweep's exact config."""
    mixture = _build_data_mixture()
    model_config = _build_model_config(REFERENCE_HIDDEN_SIZE)

    train_config = SimpleTrainConfig(
        resources=DNA_RESOURCES,
        train_batch_size=32,
        num_train_steps=20,
        learning_rate=1e-3,
        steps_per_eval=10,
        steps_per_task_eval=10,
        steps_per_export=20,
    )

    train_step = default_train(
        name=f"dna-bolinas-smoke-{REFERENCE_VERSION}",
        tokenized=mixture,
        model_config=model_config,
        train_config=train_config,
        tags=["dna", "bolinas", "smoke_test", REFERENCE_VERSION],
        eval_harness_tasks=[TRAITGYM_MENDELIAN_V2_255],
        eval_harness_max_packed_segments=1,
        use_default_validation=False,
    )

    executor_main(steps=[train_step], description=f"DNA Bolinas smoke test {REFERENCE_VERSION}")


def run_reference_tuning_sweep():
    """Vizier Bayesian optimization over AdamH hparams.

    Outer loop: EPOCHS x INITIALIZER_RANGES (independent Vizier studies).
    Inner: suggest -> train x N -> update, repeated for num_loops.
    Final: extract optimal trials per study.
    """
    version = REFERENCE_VERSION
    mixture = _build_data_mixture()
    all_optimal_steps = []

    # SWEEP_TEST_MODE=1: run a single initializer_range, 1 loop, 1 suggestion.
    # Use with v0.x versions to validate the full pipeline without burning compute.
    test_mode = os.getenv("SWEEP_TEST_MODE") == "1"

    num_loops = NUM_LOOPS
    suggestions_per_loop = SUGGESTIONS_PER_LOOP
    initializer_ranges = _get_initializer_ranges()
    if test_mode or os.getenv("CI") is not None:
        num_loops = 1
        suggestions_per_loop = 1
        initializer_ranges = INITIALIZER_RANGES[:1]

    for epochs in EPOCHS:
        for init_range in initializer_ranges:
            study_id = f"dna-bolinas-ref-{version}-IR{init_range}-E{epochs}"
            wandb_group = f"dna-bolinas-reference-sweep-{version}"
            base_tags = (
                "sweep",
                "dna",
                "bolinas",
                "reference",
                version,
                f"epochs={epochs}",
                f"initializer_range={init_range}",
            )

            model_config = _build_model_config(REFERENCE_HIDDEN_SIZE, init_range)
            num_steps = _num_train_steps()
            base_config = _build_base_train_config(
                model_config,
                mixture,
                wandb_group=wandb_group,
                base_tags=base_tags,
                checkpointer=_final_checkpoint_only(num_steps),
                steps_per_eval=num_steps // 2,  # 3 evals per run: step 0 (forced), midpoint, final
            )

            previous_update_step = None
            for loop_index in range(num_loops):
                input_db_path = previous_update_step / VIZIER_DB_FILENAME if previous_update_step else None
                suggest_step = _build_dna_suggest_step(study_id, loop_index, input_db_path)

                training_steps = [
                    _build_dna_train_step(
                        suggest_step,
                        i,
                        base_config,
                        loop_index=loop_index,
                        study_id=study_id,
                        initializer_range=init_range,
                        epochs=epochs,
                        base_tags=base_tags,
                        wandb_group=wandb_group,
                        version=version,
                    )
                    for i in range(suggestions_per_loop)
                ]

                update_step = _build_dna_update_step(study_id, loop_index, suggest_step, training_steps)
                previous_update_step = update_step

            optimal_step = _build_dna_optimal_step(study_id, previous_update_step)
            all_optimal_steps.append(optimal_step)

    executor_main(steps=all_optimal_steps, description=f"DNA Bolinas reference sweep {version}")


def run_transfer_validation_sweep():
    """Sweep key hypers (LR, beta1, beta2) in isolation at single-epoch scale."""
    raise NotImplementedError("Transfer validation sweep not yet implemented")


def run_parameter_scaling_sweep():
    """IsoFLOP parameter scaling sweep using CompletedAdamH heuristic."""
    raise NotImplementedError("Parameter scaling sweep not yet implemented")


if __name__ == "__main__":
    if not DNA_TPU_TYPE:
        raise ValueError("Set DNA_TPU_TYPE env var (e.g. v5p-8, v4-8)")
    command = os.environ.get("SWEEP_COMMAND")
    if command is None or command not in COMMANDS:
        raise ValueError(f"Set SWEEP_COMMAND to one of: {', '.join(COMMANDS)}")
    globals()[command]()
