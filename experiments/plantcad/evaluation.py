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
import logging
import numpy as np
import ray
from datasets import load_dataset
from huggingface_hub import HfApi
import fsspec
from marin.execution.executor import InputName, this_output_path, versioned
from datasets import load_dataset
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

from marin.experiments.plantcad.utils import get_available_gpus

logger = logging.getLogger("ray")


@dataclass
class DnaEvalConfig:
    """Configuration for DNA model evolutionary conservation evaluation"""
    
    checkpoint_path: str | InputName
    """Path to the model checkpoint directory"""
    
    dataset_path: str = "plantcad/evolutionary-constraint-example"
    """Dataset repository path"""

    dataset_config: str | None = "10k"
    """Dataset configuration"""

    dataset_split: str = "validation"
    """Dataset split"""

    batch_size: int = 32
    """Batch size to use for inference"""

    num_workers: int = 4
    """Number of workers to use for data transformation and loading"""
    
    max_samples: int | None = None
    """Maximum number of samples to evaluate (for quick testing)"""
    
    random_seed: int = versioned(42)
    """Random seed for data shuffling prior to downsampling"""
    
    revision: str = versioned("0.1")
    """Revision number to force re-runs when needed"""
    
    output_path: str = this_output_path()
    """Output path for results"""


def _resolve_checkpoint(config: DnaEvalConfig) -> str:
    # Download HF checkpoint if it's a remote path
    protocol = fsspec.utils.get_protocol(str(config.checkpoint_path))
    if protocol != "hf":
        return str(config.checkpoint_path)

    # Remove protocol prefix to get path
    path = str(config.checkpoint_path).removeprefix("hf://")
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

@ray.remote(max_calls=1, num_gpus=get_available_gpus())
def run_conservation_eval(config: DnaEvalConfig) -> None:
    """Run DNA model evaluation on evolutionary conservation prediction task.
    
    See:
    - https://github.com/Open-Athena/biofoundation/blob/main/examples/marin_evolutionary_constraint.py#L12
    - https://github.com/Open-Athena/biofoundation/blob/e8ff2febc0f14a268b757ee35585ede5dbf8b4ae/biofoundation/model.py#L86
    - https://openathena.slack.com/archives/C0884476QSC/p1758039985337099?thread_ts=1758038598.680879&cid=C0884476QSC
    - https://github.com/Open-Athena/marin/blob/a22645a881b3ecf68def6fa219690b8e871a9f58/experiments/plantcad/evaluation.py
    """
    from biofoundation.model import HFCausalLM
    from biofoundation.inference import run_reflogprob_clm
    from sklearn.metrics import roc_auc_score

    logger.info(f"Loading tokenizer and model from {config.checkpoint_path}")
    checkpoint_path = _resolve_checkpoint(config)
    logger.info(f"Resolved checkpoint path to: {checkpoint_path}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path, trust_remote_code=True)
    model = HFCausalLM(model)

    logger.info(f"Loading dataset from {config.dataset_path} (config={config.dataset_config}, split={config.dataset_split})")
    dataset = load_dataset(
        config.dataset_path,
        config.dataset_config,
        split=config.dataset_split,
    )

    if config.max_samples is not None and len(dataset) > config.max_samples:
        logger.info(f"Downsampling dataset to {config.max_samples} samples")
        dataset = dataset.shuffle(seed=config.random_seed)
        dataset = dataset.select(range(config.max_samples))

    label = np.array(dataset["label"])
    if not np.in1d(label, [0, 1]).all():
        raise ValueError(f"Label must be 0 or 1; got {label.unique()=}")

    logger.info(f"Running inference on {len(dataset)} samples")
    pred = run_reflogprob_clm(
        model,
        tokenizer,
        dataset,
        data_transform_kwargs=dict(
            remove_columns=dataset.column_names,
            num_proc=config.num_workers,
        ),
        inference_kwargs=dict(
            per_device_eval_batch_size=config.batch_size,
            torch_compile=False,
            bf16_full_eval=True,
            dataloader_num_workers=config.num_workers,
            remove_unused_columns=False,
        ),
    )

    n_positive = np.sum(label)
    n_negative = len(label) - n_positive
    balance = np.mean(label)
    roc_auc = roc_auc_score(label, pred)
    results = {
        "n_positive": int(n_positive),
        "n_negative": int(n_negative),
        "balance": float(balance),
        "roc_auc": float(roc_auc),
    }
    logger.info(f"Evaluation results: {results}")
    os.makedirs(config.output_path, exist_ok=True)
    results_file = os.path.join(config.output_path, "results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to: {results_file}")