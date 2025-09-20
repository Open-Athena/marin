#!/usr/bin/env python3
"""
PlantCAD learning rate tuning experiment
"""

import logging

from experiments.defaults import default_train
from experiments.plantcad.utils import get_plantcad_config, get_plantcad_training_dataset, PLANTCAD_TAGS_LR_TUNE
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.resources import GpuConfig

logger = logging.getLogger("ray")

# Run iteration 
run_number = 7

# Resources
num_gpus = 8
target_tokens = 5_485_282

# Learning rates to test
learning_rates = [1e-5, 3e-5, 1e-4, 3e-4]

# Fixed batch size
batch_size = 1024 * num_gpus

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
        train_batch_size=batch_size,
        steps_per_eval=50,
        num_train_steps=int(target_tokens / batch_size),
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
    )
    
    training_steps.append(training_step)


if __name__ == "__main__":
    logger.info("🧬 PlantCAD Learning Rate Tuning Experiment")
    logger.info("=" * 60)
    logger.info(f"Model: {model_size} ({plant_model_config})")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning rates to test: {learning_rates}")
    logger.info("=" * 60)
    
    executor_main(
        steps=[
            plant_data_tokenized,
            *training_steps,
        ]
    )
