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
Utility functions for PlantCAD experiments.
"""

import ray
import jax
import re
import os
import fsspec
from typing import Literal
from transformers import AutoTokenizer
from experiments.defaults import default_tokenize
from levanter.data.text import TextLmDatasetFormat
from levanter.models.llama import LlamaConfig

# Constants for PlantCAD experiments
PLANTCAD_TOKENIZER = "kuleshov-group/PlantCaduceus_l20"
ANGIOSPERM_HF_ID = "kuleshov-group/Angiosperm_16_genomes"

# Dataset statistics
PLANTCAD_DATASET_EXAMPLES = 5_485_282
PLANTCAD_DATASET_TOKENS = 2_808_464_384

# Common tags for PlantCAD experiments
PLANTCAD_TAGS_BASE = ["plant", "genomics"]
PLANTCAD_TAGS_LR_TUNE = [*PLANTCAD_TAGS_BASE, "lr-tune", "hyperparameter"]
PLANTCAD_TAGS_BATCH_TUNE = [*PLANTCAD_TAGS_BASE, "batch-tune", "memory-test"]


def get_plantcad_tokenizer():
    """Get the PlantCAD tokenizer instance."""
    return AutoTokenizer.from_pretrained(PLANTCAD_TOKENIZER, trust_remote_code=True)


def get_nucleotide_token_ids(tokenizer: AutoTokenizer):
    """Get the token IDs for nucleotides A, C, T, G."""
    nucleotide_ids = {}
    # Use lowercase as this is what the plantcad tokenizer normalizes to first
    for nucleotide in ["a", "c", "t", "g"]:
        token_id = tokenizer.convert_tokens_to_ids(nucleotide)
        if token_id is None:
            raise ValueError(f"Nucleotide '{nucleotide}' not found in PlantCAD tokenizer vocabulary")
        nucleotide_ids[nucleotide] = token_id

    # Assert that all token IDs are unique
    token_id_values = list(nucleotide_ids.values())
    assert len(token_id_values) == len(set(token_id_values)), f"Token IDs are not unique: {nucleotide_ids}"

    return nucleotide_ids


def get_plantcad_config(model_size: Literal["nano", "10m", "30m", "100m", "300m", "600m", "1b"] = "30m") -> LlamaConfig:
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
        return LlamaConfig(
            seq_len=512,
            hidden_dim=256,
            intermediate_dim=896,
            num_heads=8,
            num_kv_heads=8,
            num_layers=10,
        )
    elif model_size == "30m":
        return LlamaConfig(
            seq_len=512,
            hidden_dim=512,
            intermediate_dim=1792,
            num_heads=8,
            num_kv_heads=8,
            num_layers=8,
        )
    elif model_size == "100m":
        return LlamaConfig(
            seq_len=512,
            hidden_dim=768,
            intermediate_dim=2688,
            num_heads=12,
            num_kv_heads=12,
            num_layers=12,
        )
    elif model_size == "300m":
        return LlamaConfig(
            seq_len=512,
            hidden_dim=1024,
            intermediate_dim=3072,
            num_heads=16,
            num_kv_heads=16,
            num_layers=22,
        )
    elif model_size == "600m":
        return LlamaConfig(
            seq_len=512,
            hidden_dim=1408,
            intermediate_dim=4224,
            num_heads=22,
            num_kv_heads=22,
            num_layers=24,
        )
    elif model_size == "1b":
        return LlamaConfig(
            seq_len=512,
            hidden_dim=1664,
            intermediate_dim=4992,
            num_heads=26,
            num_kv_heads=26,
            num_layers=28,
        )
    else:
        raise ValueError(
            f"Unknown model size: {model_size}. Choose from: 'nano', '10m', '30m', '100m', '300m', '600m', '1b'"
        )


def get_plantcad_training_dataset(use_pretokenized: bool = True):
    # Create the base tokenization step
    tokenize_step = default_tokenize(
        name="angiosperm_16_genomes",  # path to store the tokenized data inside PREFIX
        dataset=ANGIOSPERM_HF_ID,  # HuggingFace dataset ID
        tokenizer=PLANTCAD_TOKENIZER,  # PlantCaduceus tokenizer for genomic sequences
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


def get_checkpoints(checkpoint_dir: str) -> list[dict[str, str | int]]:
    """Retrieve checkpoint information from a given directory.

    Args:
        checkpoint_dir: Local or remote directory path containing model checkpoints, e.g.
            `<prefix>/checkpoints/model-train-432442/hf/step-<step_number>`.

    Returns:
        List of dictionaries containing checkpoint path and step number
    """
    fs, _ = fsspec.url_to_fs(checkpoint_dir)
    protocol = fsspec.utils.get_protocol(checkpoint_dir)
    paths = []
    for checkpoint_path in fs.glob(os.path.join(checkpoint_dir, "step-*")):
        if protocol != "file":
            checkpoint_path = fs.unstrip_protocol(checkpoint_path)
        match = re.search(r"step-(\d+)$", checkpoint_path.split("/")[-1])
        if not match:
            raise ValueError(f"Failed to extract step number from checkpoint path: {checkpoint_path}")
        step = int(match.group(1))
        paths.append(dict(path=checkpoint_path, step=step))
    # Sort by latest checkpoint first
    paths = sorted(paths, key=lambda x: x["step"], reverse=True)
    return paths


def get_available_gpus(local_only: bool = False) -> int:
    if local_only:
        gpu_devices = jax.devices("gpu")
        if not gpu_devices:
            raise ValueError("No GPU devices found on this system")
        return len(gpu_devices)
    else:
        cluster_resources = ray.cluster_resources()
        gpu_count = cluster_resources.get("GPU")
        if gpu_count is None:
            raise ValueError("No GPUs found in the Ray cluster")
        return int(gpu_count)
