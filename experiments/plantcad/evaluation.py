"""
DNA Model Evaluation for Evolutionary Conservation Prediction

This module implements a zero-shot evaluation of DNA language models on 
evolutionary conservation prediction tasks using single nucleotide sequences.

The evaluation:
1. Loads evolutionary conservation dataset from HuggingFace
2. Tokenizes DNA sequences and runs model inference  
3. Extracts nucleotide probabilities at position 255 (middle of 512bp sequences)
4. Computes ROC AUC to measure evolutionary conservation discrimination ability
"""

import json
import os
from dataclasses import dataclass
from typing import List

import numpy as np
import ray
from datasets import Dataset, load_dataset

from levanter.models.lm_model import LmConfig
from marin.execution.executor import InputName, this_output_path, versioned

# Constants
MIDDLE_POSITION = 255  # Middle position in 512bp sequences
DEFAULT_BATCH_SIZE = 32


# TODO: Which fields should be versioned?
@dataclass
class DnaEvalConfig:
    """Configuration for DNA model evolutionary conservation evaluation"""
    
    checkpoint_path: str | InputName
    """Path to the model checkpoint"""
    
    model_config: LmConfig
    """Model configuration"""
    
    eval_dataset_repo: str = "kuleshov-group/cross-species-single-nucleotide-annotation"
    """Dataset repository ID"""
    
    eval_data_file: str = "Evolutionary_constraint/valid.tsv"
    """Specific data file to evaluate on"""
    
    max_samples: int = 1000
    """Maximum number of samples to evaluate (for quick testing)"""
    
    random_seed: int = versioned(42)
    """Random seed for data shuffling prior to downsampling"""
    
    revision: str = versioned("0.5")
    """Revision number to force re-runs when needed"""
    
    output_path: str = this_output_path()
    """Output path for results"""


def _load_and_prepare_dataset(config: DnaEvalConfig) -> Dataset:
    """Load and prepare the evaluation dataset."""
    print("Loading evaluation dataset...")
    dataset = load_dataset(
        config.eval_dataset_repo,
        data_files={"valid": config.eval_data_file}
    )
    
    eval_data = dataset["valid"]
    if len(eval_data) > config.max_samples:
        # Shuffle dataset before downsampling to ensure representative sample
        eval_data = eval_data.shuffle(seed=config.random_seed)
        eval_data = eval_data.select(range(config.max_samples))
    
    print(f"Evaluating on {len(eval_data)} samples")
    return eval_data


def _extract_nucleotide_probabilities(
    model, 
    tokenizer, 
    sequences: List[str], 
    device: str,
    batch_size: int = DEFAULT_BATCH_SIZE
) -> np.ndarray:
    """
    Extract nucleotide probabilities at the middle position for each sequence.
    
    Args:
        model: Loaded transformers model
        tokenizer: Loaded tokenizer
        sequences: List of DNA sequences
        device: Device to run inference on
        batch_size: Batch size for inference
        
    Returns:
        Array of probabilities for true nucleotides at middle position
    """
    import torch
    
    nucleotide_probs = []
    
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_sequences = sequences[i:i+batch_size]
            
            # Tokenize batch
            inputs = tokenizer(
                batch_sequences, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Forward pass
            outputs = model(**inputs)
            logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size)
            
            # Process each sequence in batch
            for j, seq in enumerate(batch_sequences):
                # Extract logits at middle position
                pos_logits = logits[j, MIDDLE_POSITION, :]
                assert pos_logits.shape == (model.config.vocab_size,), \
                    f"Expected shape ({model.config.vocab_size},), got {pos_logits.shape}"
                
                # Convert to probabilities
                probs = torch.softmax(pos_logits, dim=0)
                
                # TODO: Skip examples where middle position is not in ACTG
                # Get true nucleotide and its probability
                true_nucleotide = seq[MIDDLE_POSITION]
                token_id = tokenizer.convert_tokens_to_ids(true_nucleotide)
                nucleotide_prob = probs[token_id].cpu().item()
                
                nucleotide_probs.append(nucleotide_prob)
            
            print(f"Processed {min(i + batch_size, len(sequences))}/{len(sequences)} sequences")
    
    return np.array(nucleotide_probs)


# TODO: What is best practice for declaring deps that intersect with groups in pyproject.toml?
@ray.remote(runtime_env={"pip": ["scikit-learn"]}, max_calls=1)
def run_dna_evaluation(config: DnaEvalConfig) -> None:
    """
    Run DNA model evaluation on evolutionary conservation prediction task.
    
    This performs zero-shot evaluation by:
    1. Loading evolutionary conservation sequences (512bp each)
    2. Running model inference to get nucleotide probabilities at middle position  
    3. Computing ROC AUC to measure evolutionary conservation discrimination
    """
    print("🧬 Starting DNA Conservation Model Evaluation")
    print(f"Checkpoint: {config.checkpoint_path}")
    print(f"Max samples: {config.max_samples}")
    
    # Load and prepare dataset
    eval_data = _load_and_prepare_dataset(config)
    
    # Load model and tokenizer
    print("Loading tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint_path)
    
    print("Loading model...")
    from transformers import AutoModelForCausalLM
    import torch
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        config.checkpoint_path, 
        torch_dtype=torch.float16
    )
    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}")
    
    # Extract nucleotide probabilities
    print("Running inference...")
    sequences = eval_data["sequences"]
    labels = eval_data["label"]
    
    nucleotide_probs = _extract_nucleotide_probabilities(
        model=model,
        tokenizer=tokenizer, 
        sequences=sequences,
        device=device,
        batch_size=DEFAULT_BATCH_SIZE
    )
    
    # Validate results shape
    assert nucleotide_probs.shape == (len(eval_data),), \
        f"Expected shape ({len(eval_data)},), got {nucleotide_probs.shape}"
    
    # Compute ROC AUC
    from sklearn.metrics import roc_auc_score
    roc_auc = roc_auc_score(labels, nucleotide_probs)
    print(f"ROC AUC: {roc_auc:.4f}")
    
    # Save results
    _save_results(config, eval_data, roc_auc, device)
    print("🧬 DNA conservation evaluation completed!")


def _save_results(
    config: DnaEvalConfig, 
    eval_data: Dataset, 
    roc_auc: float, 
    device: str
) -> None:
    """Save evaluation results to JSON file."""
    results = {
        "dataset_repo": config.eval_dataset_repo,
        "dataset_file": config.eval_data_file,
        "num_samples": len(eval_data),
        "max_samples": config.max_samples,
        "checkpoint_path": str(config.checkpoint_path),
        "roc_auc": float(roc_auc),
        "metrics": {
            "evolutionary_conservation_roc_auc": float(roc_auc)
        },
        "model_device": str(device),
        "batch_size": DEFAULT_BATCH_SIZE,
        "middle_position": MIDDLE_POSITION,
    }
    
    os.makedirs(config.output_path, exist_ok=True)
    results_file = os.path.join(config.output_path, "results.json")
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {results_file}")