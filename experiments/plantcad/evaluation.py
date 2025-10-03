#!/usr/bin/env python3
# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Custom DNA evaluation metrics for Levanter - unified module for training and standalone evaluation.
"""

import logging
import os
import json
import dataclasses
from dataclasses import dataclass
from collections.abc import Callable
from datasets import Dataset

import jax.numpy as jnp
import numpy as np
import ray
from datasets import load_dataset
from transformers import AutoTokenizer
import haliax as hax
import haliax.haxtyping as ht
import levanter
import fsspec
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM
from levanter.callbacks import StepInfo
from marin.utilities.json_encoder import CustomJsonEncoder

from experiments.plantcad.utils import get_available_gpus, get_nucleotide_token_ids, get_plantcad_tokenizer
from marin.execution.executor import InputName, this_output_path, versioned

logger = logging.getLogger("ray")


@dataclass
class DnaEvalBaseConfig:
    """Base configuration for DNA evaluation with fields needed for training callbacks"""

    model_config: str
    """Model configuration size (e.g., '300m', '100m', etc.)"""

    dataset_path: str = "plantcad/evolutionary-constraint-example"
    """Dataset repository path"""

    dataset_config: str | None = "10k"
    """Dataset configuration"""

    dataset_split: str = "validation"
    """Dataset split"""

    batch_size: int = 32
    """Batch size to use for inference"""

    max_samples: int | None = None
    """Maximum number of samples to evaluate (for quick testing)"""

    random_seed: int = 42
    """Random seed for data shuffling prior to downsampling"""


@dataclass
class DnaEvalConfig(DnaEvalBaseConfig):
    """Configuration for standalone DNA model evolutionary conservation evaluation"""

    checkpoint_path: str | InputName | None = None
    """Path to the model checkpoint directory (None for training callbacks)"""

    device: str = "cuda"
    """Device to use for model inference (e.g., 'cuda', 'cpu')"""

    dtype: str | None = None
    """Dtype to use for model inference (e.g., 'float32', 'float16', 'bfloat16' or any torch dtype)"""

    num_workers: int | None = None
    """Number of workers to use for parallel evaluation (defaults to number of GPUs if None)"""

    revision: str = versioned("0.1")
    """Revision number to force re-runs when needed"""

    output_dir: str = this_output_path()
    """Output directory for results"""


@dataclass
class ConservationResult:
    """Result of scoring an evaluation dataset for conservation."""

    scores: list[float]
    labels: list[int]


def resolve_checkpoint_path(checkpoint_path: str | InputName) -> str:
    """Resolve checkpoint path, downloading from HuggingFace if needed."""
    # Download HF checkpoint if it's a remote path
    protocol = fsspec.utils.get_protocol(str(checkpoint_path))
    if protocol != "hf":
        return str(checkpoint_path)

    # Remove protocol prefix to get path
    path = str(checkpoint_path).removeprefix("hf://")
    # Parse org/repo/path/to/folder format
    path_parts = path.split("/")
    if len(path_parts) >= 2:
        repo_id = "/".join(path_parts[:2])  # org/repo
        folder_path = "/".join(path_parts[2:]) if len(path_parts) > 2 else ""
    else:
        repo_id = path
        folder_path = ""

    # Download to HF cache
    api = HfApi()
    local_path = api.snapshot_download(
        repo_id=repo_id,
        allow_patterns=f"{folder_path}/*" if folder_path else "*",
    )

    # Update config to point to local path
    final_path = os.path.join(local_path, folder_path) if folder_path else local_path
    logger.info(f"Downloaded HF checkpoint to: {final_path}")

    return final_path


def load_eval_dataset(config: DnaEvalConfig) -> Dataset:
    """Load and validate evaluation dataset."""
    logger.info(
        f"Loading dataset from {config.dataset_path} (config={config.dataset_config}, split={config.dataset_split})"
    )

    dataset = load_dataset(
        config.dataset_path,
        config.dataset_config,
        split=config.dataset_split,
    )

    if config.max_samples is not None and len(dataset) > config.max_samples:
        logger.info(f"Downsampling dataset to {config.max_samples} samples")
        dataset = dataset.shuffle(seed=config.random_seed)
        dataset = dataset.select(range(config.max_samples))

    # Validate labels
    labels = np.array(dataset["label"])
    if not np.isin(labels, [0, 1]).all():
        raise ValueError(f"Label must be 0 or 1; got unique values: {np.unique(labels)}")

    logger.info(f"Loaded and validated dataset with {len(dataset)} samples")

    return dataset


TokenArray = ht.Int[ht.NamedArray, "batch position"]
LogitArray = ht.Float[ht.NamedArray, "batch position vocab"]
PositionArray = ht.Int[ht.NamedArray, "batch"]
ScoreArray = ht.Float[ht.NamedArray, "batch"]


def compute_masked_conservation(
    tokens: TokenArray,
    logit_function: Callable[[TokenArray], LogitArray],
    nucleotide_positions: PositionArray,
    nucleotide_token_ids: list[int],
    mask_token_id: int,
) -> ScoreArray:
    """Compute conservation scores using masked language modeling.

    Masks nucleotides at target positions and measures model's probability
    of predicting the original nucleotide.

    Args:
        tokens: Token sequences with axes (Batch, Position)
        logit_function: Function that takes tokens and returns model logits
        nucleotide_positions: Target positions for substitution with axes (Batch,)
        nucleotide_token_ids: List of nucleotide token IDs (typically A, C, T, G)
        mask_token_id: Mask token ID used to train model

    Returns:
        Conservation scores with axes (Batch,), higher values indicate
        better model prediction of the original nucleotide
    """
    # Create input sequences with a single masked token in each,
    # at the provided, target nucleotide position
    Batch, Position = tokens.axes

    # Use proper broadcasting for position comparison
    position_indices = hax.broadcast_axis(hax.arange(Position), Batch)  # (Position,) -> (Batch, Position)
    target_positions = hax.broadcast_axis(nucleotide_positions, Position)  # (Batch,) -> (Batch, Position)
    position_mask = position_indices == target_positions

    masked_tokens = hax.where(position_mask, mask_token_id, tokens)
    assert masked_tokens.axes == (Batch, Position)

    # Compute logits for full sequences
    logits = logit_function(masked_tokens)
    assert logits.ndim == 3, f"Expected 3 dimensions, got {logits.shape=}"
    Vocab = logits.axes[2]
    assert logits.axes == (Batch, Position, Vocab)

    # Select logits and create probabilities at target positions
    nt_logits = logits[Position, nucleotide_positions]
    assert nt_logits.axes == (Batch, Vocab)
    nt_probs = hax.nn.softmax(nt_logits, axis=Vocab)
    assert nt_probs.axes == (Batch, Vocab)

    # Get true nucleotide tokens at target positions
    nt_tokens = tokens[Position, nucleotide_positions]
    assert nt_tokens.axes == (Batch,)

    # Check that all target nucleotides are valid
    nt_mask = hax.named(jnp.isin(nt_tokens.array, jnp.array(nucleotide_token_ids)), (Batch,))
    if not nt_mask.array.all():
        invalid_positions = nucleotide_positions[Batch, ~nt_mask.array]
        invalid_tokens = nt_tokens[Batch, ~nt_mask.array]
        raise ValueError(
            "Found invalid sequences in batch with OOV nucleotides at target positions; "
            f"Target positions: {invalid_positions.array} "
            f"Valid nucleotide token IDs: {nucleotide_token_ids} "
            f"Invalid tokens: {invalid_tokens.array} "
        )

    # Extract probabilities for the true tokens
    result = hax.take(nt_probs, index=nt_tokens, axis=Vocab)
    assert result.axes == (Batch,)

    return result


def create_alternate_sequences(
    tokens: TokenArray,
    nucleotide_positions: PositionArray,
    nucleotide_token_ids: list[int],
) -> tuple[ht.Int[ht.NamedArray, "batch position variant"], ht.Int[ht.NamedArray, " batch"]]:
    """Create 4 alternative sequences with different nucleotides at target positions.

    For each input sequence, creates 4 variants where the nucleotide at the target position
    is replaced with each of the 4 possible nucleotides (A, C, T, G). Also determines which
    variant corresponds to the original sequence.

    Args:
        tokens: Token sequences with axes (Batch, Position)
        nucleotide_positions: Target positions for substitution with axes (Batch,)
        nucleotide_token_ids: List of 4 nucleotide token IDs (typically [A, C, T, G])

    Returns:
        tuple containing:
        - alt_sequences: Array with axes (Batch, Position, Variant) where Variant has size 4,
          containing the original sequences with nucleotides substituted at target positions
        - ref_indexes: Array with axes (Batch,) indicating which variant index corresponds
          to the original nucleotide for each sequence

    Raises:
        ValueError: If any target position contains a nucleotide not in nucleotide_token_ids
    """
    Batch, Position = tokens.axes

    # Create a new axis for the 4 nucleotide variants
    Variant = hax.Axis("variant", 4)

    # Create array of nucleotide token IDs to substitute
    nucleotide_ids = hax.named(jnp.array(nucleotide_token_ids), Variant)

    # Broadcast tokens to include the variant dimension: (Batch, Position) -> (Batch, Position, Variant)
    tokens_expanded = hax.broadcast_to(tokens, (Batch, Position, Variant))

    # Create position mask for target positions
    position_indices = hax.broadcast_axis(hax.arange(Position), Batch)  # (Position,) -> (Batch, Position)
    target_positions = hax.broadcast_axis(nucleotide_positions, Position)  # (Batch,) -> (Batch, Position)
    position_mask = position_indices == target_positions
    assert position_mask.axes == (Batch, Position)

    # Apply substitutions using where: replace tokens at target positions with nucleotide variants
    alt = hax.where(position_mask, nucleotide_ids, tokens_expanded)
    assert alt.axes == (Batch, Position, Variant)

    # Determine what index across the Variant axis corresponds to the "ref" sequence;
    # This should be which variant sequence has a token id equal to the actual token id in tokens_expanded
    ref_mask = alt[Position, nucleotide_positions] == tokens_expanded[Position, nucleotide_positions]
    assert ref_mask.axes == (Batch, Variant)
    ref_cts = hax.sum(ref_mask, axis=Variant)
    assert 0 <= ref_cts.max().item() <= 1
    if (invalid := ref_cts == 0).any().item():
        pos = nucleotide_positions[Batch, invalid]
        tok = tokens_expanded[Batch, invalid, Position, pos]
        raise ValueError(
            "Found invalid sequences in batch with OOV nucleotides at target positions; "
            f"Target positions: {pos} "
            f"Valid nucleotide token IDs: {nucleotide_token_ids} "
            f"Invalid tokens: {tok} "
        )
    ref = hax.argmax(ref_mask, axis=Variant)
    assert ref.axes == (Batch,)

    return alt, ref


def compute_sequence_logprob(
    logits: LogitArray,
    tokens: TokenArray,
) -> ScoreArray:
    """Compute log probabilities for token sequences.

    Sums log probabilities of predicting each next token in the sequence.

    Args:
        logits: Model logits with axes (Batch, Position, Vocab)
        tokens: Token sequences with axes (Batch, Position)

    Returns:
        Log probabilities with axes (Batch,), representing total sequence
        likelihood under a causal language model
    """
    Batch, Position, Vocab = logits.axes

    # Compute log probabilities of *all* tokens
    log_probs = hax.nn.log_softmax(logits, axis=Vocab)
    assert log_probs.axes == (Batch, Position, Vocab)

    # Align log probabilities to their corresponding true tokens, i.e. shift
    # next token logits from model one token to the right (first token is ignored)
    aligned_log_probs = log_probs[Position, :-1]
    aligned_tokens = tokens[Position, 1:]
    AlignedPosition = aligned_log_probs.resolve_axis(Position.name)
    assert aligned_log_probs.axes == (Batch, AlignedPosition, Vocab)
    assert aligned_tokens.axes == (Batch, AlignedPosition)

    # Select the log probabilities of only the true tokens for each sequence
    # and sum them to get a log probability by sequence
    token_log_probs = hax.take(aligned_log_probs, index=aligned_tokens, axis=Vocab)
    assert token_log_probs.axes == (Batch, AlignedPosition)
    sequence_log_probs = hax.sum(token_log_probs.astype(jnp.float32), axis=AlignedPosition)
    assert sequence_log_probs.axes == (Batch,)

    return sequence_log_probs


def compute_causal_conservation(
    tokens: TokenArray,
    logit_function: Callable[[TokenArray], LogitArray],
    nucleotide_positions: PositionArray,
    nucleotide_token_ids: list[int],
) -> ScoreArray:
    """Compute conservation scores using causal language modeling.

    Creates 4 nucleotide variants at target positions and compares their
    relative likelihoods to measure evolutionary conservation.

    Args:
        tokens: Token sequences with axes (Batch, Position)
        logit_function: Function that takes tokens and returns model logits
        nucleotide_positions: Target positions for substitution with axes (Batch,)
        nucleotide_token_ids: List of nucleotide token IDs (typically A, C, T, G)

    Returns:
        Conservation scores with axes (Batch,), representing the log probability
        of the original sequence relative to nucleotide variants
    """
    Batch, Position = tokens.axes

    # Create alternate/variant sequences with all possible nucleotides at target positions
    alt_sequences, ref_indexes = create_alternate_sequences(tokens, nucleotide_positions, nucleotide_token_ids)
    Variant = alt_sequences.resolve_axis("variant")
    assert alt_sequences.axes == (Batch, Position, Variant)
    assert ref_indexes.axes == (Batch,)

    # Stack variant and batch dimensions for model input
    # TODO: can this be done with Axis objects instead?
    batch_alt_sequences = hax.rearrange(
        alt_sequences, "{batch position variant} -> (batch_variant: batch variant) position"
    )
    VariantBatch = batch_alt_sequences.resolve_axis("batch_variant")

    # Run inference for all reference/alternate sequences
    logits = logit_function(batch_alt_sequences)
    # Always promote to full precision for zero-shot evaluation
    logits = logits.astype(jnp.float32)
    Vocab = logits.resolve_axis("vocab")
    assert logits.axes == (VariantBatch, Position, Vocab)

    # Compute marginal log probabilities for all variant sequences
    alternate_log_probs = compute_sequence_logprob(logits, batch_alt_sequences)

    # Unstack the variant dimension in order to renormalize the true (i.e. ref), input
    # sequences by the log probability of each variant sequence
    sequence_log_probs = hax.rearrange(
        alternate_log_probs, "(batch_variant: batch variant) -> batch variant", batch=Batch, variant=Variant
    )
    marginal_log_probs = hax.nn.log_softmax(sequence_log_probs, axis=Variant)
    assert marginal_log_probs.axes == (Batch, Variant)

    # Select the log probabilities of the true (i.e. ref) sequences
    ref_log_probs = marginal_log_probs[Variant, ref_indexes]
    assert ref_log_probs.axes == (Batch,)

    return ref_log_probs


def score_eval_dataset(
    tokenizer: AutoTokenizer,
    eval_dataset: Dataset,
    logit_function: Callable[[TokenArray], LogitArray],
    batch_size: int = 32,
    log_progress: bool = True,
) -> ConservationResult:
    """Score evaluation dataset based on zero-shot conservation prediction."""

    # Get nucleotide token mappings from tokenizer
    nucleotide_token_ids = list(get_nucleotide_token_ids(tokenizer).values())  # [id_A, id_C, id_T, id_G]

    all_scores = []
    all_labels = []
    total_processed = 0

    # Set batches for processing
    batches = eval_dataset.with_format(None).batch(batch_size=batch_size)
    total_batches = len(batches)
    progress_interval = max(1, total_batches // 20)  # Every 5%
    if log_progress:
        logger.info(f"Processing {len(eval_dataset)} samples in {total_batches} batches (batch_size={batch_size})")

    for batch_index, batch_data in enumerate(batches):
        # Tokenize sequences
        sequences = batch_data["seq"]
        labels = batch_data["label"]
        pos = batch_data["pos"]
        assert isinstance(sequences, list)
        assert isinstance(labels, list)
        assert isinstance(pos, list)

        # Tokenize and convert to JAX arrays
        tokenized = tokenizer(sequences, padding=True, truncation=True, max_length=512, return_tensors="np")
        tokens = hax.named(tokenized["input_ids"], ("batch", "position"))
        nucleotide_positions = hax.named(pos, ("batch",))

        # Compute conservation score for each example in batch
        scores = compute_causal_conservation(
            tokens=tokens,
            logit_function=logit_function,
            nucleotide_positions=nucleotide_positions,
            nucleotide_token_ids=nucleotide_token_ids,
        )
        assert len(scores.array) == len(labels)

        # Aggregate scores and labels
        all_scores.extend(scores.tolist())
        all_labels.extend(labels)
        total_processed += len(sequences)

        # Log progress every 5% of batches
        if log_progress and (batch_index % progress_interval == 0 or batch_index == total_batches - 1):
            progress_pct = ((batch_index + 1) / total_batches) * 100
            logger.info(
                f"Progress: {batch_index + 1}/{total_batches} batches ({progress_pct:.1f}%) - "
                f"{len(all_scores)} scores generated"
            )

    return ConservationResult(scores=all_scores, labels=all_labels)


# ------------------------------------------------------------------------------------------------
# In-training evaluation
# ------------------------------------------------------------------------------------------------


def create_dna_eval_callback(config: DnaEvalBaseConfig) -> Callable[[StepInfo], None]:
    """Create a training callback for DNA evaluation."""

    # Load tokenizer
    # TODO: fix this; how can the tokenizer be referenced during training without reloading?
    tokenizer = get_plantcad_tokenizer()

    # Load and validate dataset once when creating the callback
    dataset = load_eval_dataset(config)

    def dna_conservation_callback(step_info: StepInfo) -> None:
        step = step_info.step
        logger.debug(f"Running DNA conservation evaluation ({step=})")
        eval_model = step_info.eval_model

        # Create logit function for Levanter model
        def logit_function(
            tokens: ht.Int[ht.NamedArray, "batch position"],
        ) -> ht.Float[ht.NamedArray, "batch position vocab"]:
            logits = eval_model(tokens)
            return logits

        # Compute scores with binary labels
        scores = score_eval_dataset(
            tokenizer=tokenizer,
            logit_function=logit_function,
            eval_dataset=dataset,
            batch_size=config.batch_size,
            # TODO: make configurable or disable?
            log_progress=True,
        )

        # Evaluate scores and labels
        metrics = evaluate_conservation_scores(scores)

        # Log results
        levanter.tracker.log(
            {
                "eval/dna_conservation_roc": metrics["roc_auc"],
            },
            step=step_info.step,
        )

        logger.info(
            f"DNA conservation evaluation complete ({step=}): "
            f"ROC AUC = {metrics['roc_auc']:.4f}, n_samples = {metrics['n_total']}"
        )

    return dna_conservation_callback


# ------------------------------------------------------------------------------------------------
# Standalone evaluation
# ------------------------------------------------------------------------------------------------


def load_hf_model(checkpoint_path: str, dtype: str | None, device: str | None) -> AutoModelForCausalLM:
    # TODO: figure out if this can be a global import
    import torch

    # Resolve target dtype for model
    torch_dtype = None
    if dtype is not None:
        if not hasattr(torch, dtype):
            raise ValueError(
                f"Unsupported model dtype: '{dtype}'. Must be a valid torch dtype "
                f"(e.g., 'float32', 'float16', 'bfloat16')"
            )
        torch_dtype = getattr(torch, dtype)

    # Load model
    hf_model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path, trust_remote_code=True, **(dict(torch_dtype=torch_dtype) if torch_dtype is not None else {})
    )
    hf_model = hf_model.to(device=device, dtype=torch_dtype)
    return hf_model


def generate_conservation_scores(config: DnaEvalConfig, worker_id: int = 0, num_workers: int = 1) -> ConservationResult:
    # TODO: figure out if this can be a global import
    import torch

    if config.checkpoint_path is None:
        raise ValueError("checkpoint_path required for standalone evaluation")

    logger.info(f"Resolving checkpoint path: {config.checkpoint_path}")
    checkpoint_path = resolve_checkpoint_path(config.checkpoint_path)
    logger.info(f"Resolved checkpoint path to: {checkpoint_path}")

    logger.info(f"Loading tokenizer from {checkpoint_path}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    vocab_size = len(tokenizer.vocab)
    logger.info(f"Tokenizer vocab size: {vocab_size}")

    # Load HuggingFace torch model
    hf_model = load_hf_model(checkpoint_path, dtype=config.dtype, device=config.device)
    hf_model.eval()

    # Create logit function for HuggingFace model
    def logit_function(
        tokens: ht.Int[ht.NamedArray, "batch position"],
    ) -> ht.Float[ht.NamedArray, "batch position vocab"]:
        Batch, Position = tokens.axes
        # Convert haliax token array to torch tensor
        token_array = np.array(tokens.array, dtype=np.int64)
        # TODO: is there a better way to convert jax to torch device?
        input_ids = torch.from_numpy(token_array).to(str(tokens.array.device))

        with torch.inference_mode():
            outputs = hf_model(input_ids)
            # TODO: Try to get direct conversion to jax working here;
            #       jnp.asarray(outputs.logits) results in an inscrutable error:
            #       jaxlib._jax.XlaRuntimeError: INVALID_ARGUMENT: Unknown NumPy dtype dtype('V2') char V kind V itemsize 2 # noqa: E501
            logits = outputs.logits.float().cpu().numpy()
            assert logits.ndim == 3, f"Expected 3 dimensions, got {logits.shape=}"
            return hax.named(logits, (Batch, Position, "vocab"))

    # Load and validate evaluation dataset
    dataset = load_eval_dataset(config)

    # Shard dataset for this worker
    if num_workers > 1:
        dataset = dataset.shard(num_shards=num_workers, index=worker_id)
        logger.info(f"Worker {worker_id}: Processing {len(dataset)} samples")
    else:
        logger.info(f"Running inference on {len(dataset)} samples")

    # Generate raw conservation scores and labels
    result = score_eval_dataset(
        tokenizer=tokenizer,
        logit_function=logit_function,
        eval_dataset=dataset,
        batch_size=config.batch_size,
        log_progress=True,
    )

    logger.info(f"Generated {len(result.scores)} conservation scores")
    return result


def combine_conservation_results(results: list[ConservationResult]) -> ConservationResult:
    """Combine raw scores and labels from multiple workers."""
    all_scores = []
    all_labels = []

    for result in results:
        all_scores.extend(result.scores)
        all_labels.extend(result.labels)

    return ConservationResult(scores=all_scores, labels=all_labels)


def evaluate_conservation_scores(scores: ConservationResult) -> dict[str, float]:
    """Calculate ROC AUC and other metrics from combined scores."""
    from sklearn.metrics import roc_auc_score

    if len(scores.scores) == 0:
        raise ValueError("No valid conservation scores found")

    n_unmasked_total = len(scores.scores)
    valid_mask = ~np.isnan(scores.scores)
    filtered_scores = np.array(scores.scores)[valid_mask]
    filtered_labels = np.array(scores.labels)[valid_mask]

    if len(filtered_scores) == 0:
        raise ValueError("No valid (non-NaN) conservation scores found after filtering")

    if (n_filtered := n_unmasked_total - len(filtered_scores)) > 0:
        logger.info(f"Filtered out {n_filtered} samples with NaN scores")

    # Compute metrics
    roc_auc = roc_auc_score(filtered_labels, filtered_scores)
    n_positive = filtered_labels.sum()
    n_total = len(filtered_labels)

    results = {
        "roc_auc": roc_auc,
        "n_total": n_total,
        "n_positive": int(n_positive),
        "n_negative": n_total - int(n_positive),
        "balance": n_positive / n_total,
    }

    logger.info("=" * 50)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 50)
    logger.info(f"Total examples: {results['n_total']}")
    logger.info(f"ROC AUC: {results['roc_auc']:.6f}")
    logger.info(f"Balance: {results['balance']:.3f} ({results['n_positive']}/{results['n_total']})")
    logger.info("=" * 50)

    return results


def save_conservation_results(config: DnaEvalConfig, results: dict[str, float]) -> None:
    """Save conservation evaluation results along with config to JSON file."""
    os.makedirs(config.output_dir, exist_ok=True)
    results_file = os.path.join(config.output_dir, "results.json")
    logger.info(f"Saving final results to {results_file}")

    output_data = {"config": dataclasses.asdict(config), "results": results}

    with open(results_file, "w") as f:
        json.dump(output_data, f, indent=2, cls=CustomJsonEncoder)
    logger.info(f"Saved evaluation results to: {results_file}")


# TODO: fix this which forces only one checkpoint to run at a time
@ray.remote(max_calls=1, resources={"head_node": 1})
def run_conservation_eval(config: DnaEvalConfig) -> dict[str, float]:
    # Determine number of workers
    if config.num_workers is None:
        num_gpus = get_available_gpus(local_only=False)
        num_workers = num_gpus
        logger.info(f"Running conservation evaluation on {num_gpus} GPUs")
    else:
        num_workers = config.num_workers
        logger.info(f"Running conservation evaluation with {num_workers} workers")

    # Run parallel score generation
    executor = ray.remote(num_gpus=1)(generate_conservation_scores)
    futures = [executor.remote(config, worker_id, num_workers) for worker_id in range(num_workers)]
    results = ray.get(futures)

    # Combine results and evaluate
    combined_scores = combine_conservation_results(results)
    final_results = evaluate_conservation_scores(combined_scores)

    # Save results and config
    save_conservation_results(config, final_results)

    return final_results


# Usage examples:

# 1. Training callback (uses Levanter model from training state):
# config = DnaEvalConfig(
#     checkpoint_path="/path/to/checkpoint",  # Not used for callbacks
#     model_config="300m",
#     dataset_path="plantcad/evolutionary-constraint-example",
#     dataset_config="10k"
# )
# trainer.add_hook(create_dna_eval_callback(config), every=1000)

# 2. Standalone evaluation with HuggingFace checkpoint:
# config = DnaEvalConfig(
#     checkpoint_path="/path/to/hf/checkpoint",
#     model_config="300m",
#     device="cuda",  # or "cpu" for CPU inference
#     num_workers=None,  # defaults to number of GPUs
#     dataset_path="plantcad/evolutionary-constraint-example",
#     dataset_config="10k",
#     max_samples=1000
# )
# results = run_conservation_eval(config)
