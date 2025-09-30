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
PlantCAD tutorial experiment

This experiment demonstrates:
1. Using the PlantCaduceus tokenizer for genomic sequences
2. Training a small model on Angiosperm_16_genomes dataset
3. Run a zero-shot evolutionary conservation evaluation
"""

import logging

from experiments.defaults import default_train
from experiments.plantcad.utils import (
    get_plantcad_config,
    get_plantcad_training_dataset,
)
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main, ExecutorStep, InputName
from marin.resources import GpuConfig

logger = logging.getLogger("ray")

model_size = "nano"
use_pretokenized = True

# Get the PlantCAD training dataset
angiosperm_tokenized = get_plantcad_training_dataset(use_pretokenized=use_pretokenized)

# Create a model configuration suitable for genomic sequences
# Note: Genomic sequences are typically 512bp, matching the dataset
plant_model_config = get_plantcad_config(model_size)  # Using "nano" for quick tutorial

# Training configuration for a quick test
nano_plant_train_config = SimpleTrainConfig(
    resources=GpuConfig(gpu_count=1),
    train_batch_size=16,  # Smaller batch size for genomic data
    num_train_steps=100,  # Quick test - 100 steps
    learning_rate=3e-4,  # Conservative learning rate for genomic data
    weight_decay=0.1,
    steps_per_export=100,
)

# Define the training step
nano_angiosperm_model = default_train(
    name="plant-nano-angiosperm",
    tokenized=angiosperm_tokenized,
    model_config=plant_model_config,
    train_config=nano_plant_train_config,
    tags=["plant", "genomics", "angiosperm", "nano", "test"],
    # No eval harness for genomic data
    eval_harness_tasks=[],
    use_default_validation=False,  # No default validation for genomic data
)


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
        pip_dependency_groups=["dna"],
        description="Zero-shot evolutionary conservation prediction evaluation for DNA model",
    )


# Create DNA conservation evaluation step
dna_conservation_evaluation = create_dna_conservation_eval_step(
    checkpoint_step=nano_angiosperm_model,
    model_config=plant_model_config,
    max_samples=1000,
    random_seed=42,
)

if __name__ == "__main__":
    # Quick data exploration first
    logger.info("🧬 PlantCAD Experiment: Angiosperm Genomic Data")
    logger.info("=" * 60)
    logger.info("Dataset: kuleshov-group/Angiosperm_16_genomes")
    logger.info("Tokenizer: kuleshov-group/PlantCaduceus_l20")
    logger.info(f"Model: {plant_model_config}")
    logger.info(f"Training steps: {nano_plant_train_config.num_train_steps}")
    logger.info(f"Batch size: {nano_plant_train_config.train_batch_size}")
    logger.info("=" * 60)

    executor_main(
        steps=[
            angiosperm_tokenized,
            nano_angiosperm_model,
            dna_conservation_evaluation,
        ]
    )
