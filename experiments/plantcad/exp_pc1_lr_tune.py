#!/usr/bin/env python3
"""
PlantCAD learning rate tuning experiment
"""

import logging

from experiments.defaults import default_train
from experiments.plantcad.utils import get_available_gpus, get_plantcad_config, get_plantcad_training_dataset, PLANTCAD_TAGS_LR_TUNE, PLANTCAD_DATASET_EXAMPLES
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.resources import GpuConfig

logger = logging.getLogger("ray")

# Run iteration 
run_number = 9

# Resources
num_gpus = get_available_gpus(local_only=True)
target_examples = PLANTCAD_DATASET_EXAMPLES

# Learning rates to test
learning_rates = [1e-5, 3e-5, 1e-4, 3e-4]

# Fixed batch size
# TODO: How do you tune micro/macro batch size instead of gloabl batch?
#       This needs to not be a function of device count.
batch_size = 1024 * num_gpus

# Shuffle training data
shuffle = False

# Model configuration
model_size = "30m"  # Use optimized 30M config as default
plant_model_config = get_plantcad_config(model_size)

# PlantCAD1 training dataset
plant_data_tokenized = get_plantcad_training_dataset(use_pretokenized=True)

# Create training configurations for each learning rate
training_steps = []

for lr in learning_rates:
    train_config = SimpleTrainConfig(
        resources=GpuConfig(gpu_count=num_gpus),
        data_seed=42,
        train_batch_size=batch_size,
        steps_per_eval=50,
        num_train_steps=int(target_examples / batch_size),
        learning_rate=lr,
        steps_per_export=100,
    )
    
    training_step = default_train(
        name=f"plantcad-lr-tune-lr{lr:.0e}-r{run_number:02d}",
        tokenized=plant_data_tokenized,
        model_config=plant_model_config,
        train_config=train_config,
        tags=PLANTCAD_TAGS_LR_TUNE,
        eval_harness_tasks=[],
        use_default_validation=False,
        shuffle=shuffle,
    )
    
    training_steps.append(training_step)


if __name__ == "__main__":
    logger.info("🧬 PlantCAD Learning Rate Tuning Experiment")
    logger.info("=" * 60)
    logger.info(f"Model: {model_size} ({plant_model_config})")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Target examples: {target_examples:,}")
    logger.info(f"Learning rates to test: {learning_rates}")
    logger.info("=" * 60)
    
    executor_main(
        steps=[
            plant_data_tokenized,
            *training_steps,
        ]
    )
