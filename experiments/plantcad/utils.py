"""
Utility functions for PlantCAD experiments.
"""

from typing import Literal
from experiments.defaults import default_tokenize
from levanter.data.text import TextLmDatasetFormat
from levanter.models.lm_model import LmConfig
from levanter.models.llama import LlamaConfig
from marin.execution.executor import ExecutorStep, InputName

# Common tags for PlantCAD experiments
PLANTCAD_TAGS_BASE = ["plant", "genomics"]
PLANTCAD_TAGS_LR_TUNE = PLANTCAD_TAGS_BASE + ["lr-tune", "hyperparameter"]
PLANTCAD_TAGS_BATCH_TUNE = PLANTCAD_TAGS_BASE + ["batch-tune", "memory-test"]


def get_plantcad_config(model_size: Literal["nano", "10m", "30m", "100m"] = "30m") -> LlamaConfig:
    if model_size == "nano":
        # Testing configuration - keep small for fast iteration
        return LlamaConfig(
            seq_len=512,
            hidden_dim=32,
            intermediate_dim=128,
            num_heads=2,
            num_kv_heads=2,
            num_layers=2,
        )
    elif model_size == "10m":
        # Optimized 10M parameter configuration for genomic data
        return LlamaConfig(
            seq_len=512,
            hidden_dim=256,
            intermediate_dim=896,
            num_heads=8,
            num_kv_heads=8,
            num_layers=10,
        )
    elif model_size == "30m":
        # Optimized 30M parameter configuration for genomic data (default)
        return LlamaConfig(
            seq_len=512,
            hidden_dim=512,
            intermediate_dim=1792,
            num_heads=8,
            num_kv_heads=8,
            num_layers=8,
        )
    elif model_size == "100m":
        # Optimized 100M parameter configuration for genomic data
        return LlamaConfig(
            seq_len=512,
            hidden_dim=768,
            intermediate_dim=2688,
            num_heads=12,
            num_kv_heads=12,
            num_layers=12,
        )
    else:
        raise ValueError(f"Unknown model size: {model_size}. Choose from: 'nano', '10m', '30m', '100m'")


def create_dna_conservation_eval_step(
    checkpoint_step: ExecutorStep | InputName,
    max_steps: int = 99,
    max_samples: int = 1000,
    random_seed: int = 42,
) -> ExecutorStep:
    """
    Create an ExecutorStep for DNA model evaluation on evolutionary constraints.
    
    Args:
        checkpoint_step: Training step that produced the model checkpoint
        model_config: Model configuration (currently unused but kept for compatibility)
            - TODO: Is this necessary if the HF checkpoint isn't for an arch in `transformers`?
        max_samples: Maximum number of evaluation samples
        random_seed: Random seed for data shuffling
    
    Returns:
        ExecutorStep configured for DNA evolutionary constraint evaluation
    """
    from experiments.plantcad.evaluation import run_dna_evaluation, DnaEvalConfig
    
    return ExecutorStep(
        name=f"evaluation/dna-conservation/{checkpoint_step.name}",
        fn=run_dna_evaluation,
        config=DnaEvalConfig(
            checkpoint_path=checkpoint_step / "hf" / f"step-{max_steps}",
            max_samples=max_samples,
            random_seed=random_seed,
        ),
        # pip_dependency_groups=["eval"],
        # pip_dependency_groups=["dna"],
        description="Zero-shot evolutionary conservation prediction evaluation for DNA model",
    )


def get_plantcad_training_dataset(use_pretokenized: bool = True):
    # PlantCaduceus tokenizer for genomic sequences
    plantcad_tokenizer = "kuleshov-group/PlantCaduceus_l20"
    angiosperm_hf_id = "kuleshov-group/Angiosperm_16_genomes"
    
    # Create the base tokenization step
    tokenize_step = default_tokenize(
        name="angiosperm_16_genomes",  # path to store the tokenized data inside PREFIX
        dataset=angiosperm_hf_id,  # HuggingFace dataset ID
        tokenizer=plantcad_tokenizer,  # PlantCaduceus tokenizer for genomic sequences
        format=TextLmDatasetFormat(text_key="seq"),  # CRITICAL: Use 'seq' field instead of default 'text'
    )
    
    if use_pretokenized:
        # Download pre-tokenized dataset to HF cache and use local path
        from huggingface_hub import snapshot_download
        
        # Download the entire pre-tokenized dataset to HF cache
        # Based on HuggingFace Hub patterns for downloading datasets
        cached_path = snapshot_download(
            repo_id="plantcad/_dev_marin_plantcad1_v1_tokenized",
            repo_type="dataset",
            revision="main",
        )
        
        # Override the tokenize step output to point to the cached location
        return tokenize_step.with_output_path(cached_path)
    else:
        # Return the standard tokenization step
        return tokenize_step
