#!/usr/bin/env python3
"""
PlantCAD tutorial experiment

This experiment demonstrates:
1. Using the PlantCaduceus tokenizer for genomic sequences
2. Training a small model on Angiosperm_16_genomes dataset
3. Run a zero-shot evolutionary conservation evaluation
"""

import logging

from experiments.defaults import default_train
from experiments.plantcad.utils import get_plantcad_config, create_dna_conservation_eval_step, get_plantcad_training_dataset
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
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
    num_train_steps=100,   # Quick test - 100 steps
    learning_rate=3e-4,    # Conservative learning rate for genomic data
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
    logger.info(f"Dataset: kuleshov-group/Angiosperm_16_genomes")
    logger.info(f"Tokenizer: kuleshov-group/PlantCaduceus_l20")
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