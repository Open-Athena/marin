#!/usr/bin/env python3
"""
PlantCAD training experiment - single 300M model with optimized learning rate
"""

import logging

from experiments.defaults import default_train
from experiments.plantcad.utils import get_available_gpus, get_plantcad_config, get_plantcad_training_dataset, PLANTCAD_TAGS_BASE, PLANTCAD_DATASET_EXAMPLES
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.resources import GpuConfig

logger = logging.getLogger("ray")

# Run iteration 
run_number = 2

# Resources
num_gpus = get_available_gpus(local_only=True)
target_examples = PLANTCAD_DATASET_EXAMPLES * 10 # 10 epochs

# Best learning rate from tuning experiments
learning_rate = 1e-4

# Batch size
# TODO: How do you tune micro/macro batch size instead of gloabl batch?
#       This needs to not be a function of device count.
batch_size = 256 * num_gpus

# Shuffle training data
shuffle = True

# Training configuration
num_train_steps = target_examples // batch_size
steps_per_export = num_train_steps // 16
steps_per_eval = num_train_steps // 64

# Model configuration - use 300M by default
model_size = "300m"
plant_model_config = get_plantcad_config(model_size)

# PlantCAD1 training dataset
plant_data_tokenized = get_plantcad_training_dataset(use_pretokenized=True)

# Training configuration
train_config = SimpleTrainConfig(
    resources=GpuConfig(gpu_count=num_gpus),
    data_seed=42,
    train_batch_size=batch_size,
    steps_per_eval=steps_per_eval,
    num_train_steps=num_train_steps,
    learning_rate=learning_rate,
    steps_per_export=steps_per_export,
)

# Create training step
training_step = default_train(
    name=f"plantcad-train-{model_size}-r{run_number:02d}",
    tokenized=plant_data_tokenized,
    model_config=plant_model_config,
    train_config=train_config,
    tags=PLANTCAD_TAGS_BASE + ["training"],
    eval_harness_tasks=[],
    use_default_validation=False,
    shuffle=shuffle,
)


if __name__ == "__main__":
    logger.info("🧬 PlantCAD Training Experiment")
    logger.info("=" * 60)
    logger.info(f"Model: {model_size} ({plant_model_config})")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Target examples: {target_examples:,}")
    logger.info(f"Training steps: {num_train_steps:,}")
    logger.info(f"Steps per export: {steps_per_export:,}")
    logger.info(f"Steps per eval: {steps_per_eval:,}")
    logger.info("=" * 60)
    
    executor_main(
        steps=[
            plant_data_tokenized,
            training_step,
        ]
    )
