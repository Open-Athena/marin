#!/usr/bin/env python3
"""
PlantCAD batch size tuning experiment

References:
- experiments/exp474_config_sweep.py (batch size sweeping).
"""

import dataclasses
import logging
import ray

from experiments.defaults import default_train
from experiments.plantcad.utils import get_plantcad_config, get_plantcad_training_dataset, PLANTCAD_TAGS_BATCH_TUNE
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.resources import GpuConfig

logger = logging.getLogger("ray")

# Run iteration 
run_number = 2

# Batch sizes to test
batch_sizes = [512, 1024, 2048, 4096, 8192]

# Model configuration
model_size = "30m"  # Use optimized 30M config as default
plant_model_config = get_plantcad_config(model_size)

# PlantCAD1 training dataset
plant_data_tokenized = get_plantcad_training_dataset(use_pretokenized=True)

# Create training configurations for each batch size
training_steps = []

def _failure_ok_train(*args, **kwargs):
    # Lift from https://github.com/marin-community/marin/blob/3e61da0ff82607e751499a4f8869781b0f053f0b/experiments/exp474_config_sweep.py#L155C1-L163C71
    from marin.training.training import run_levanter_train_lm
    try:
        return ray.get(run_levanter_train_lm.remote(*args, **kwargs))
    except Exception as e:
        logger.exception("Failed to run training", exc_info=e)
        return None

for i, batch_size in enumerate(batch_sizes):
    # Training configuration 
    train_config = SimpleTrainConfig(
        resources=GpuConfig(gpu_count=1),
        train_batch_size=batch_size,
        num_train_steps=3,  # Minimal steps to test memory fit
        learning_rate=1e-4, # Doesn't matter for this test
        steps_per_export=100, # No checkpoints (TODO: what's a better way? -1 seems to mean every step)
    )
    
    # Training executor step
    training_step = default_train(
        name=f"plantcad-batch-tune-bs{batch_size}-r{run_number:02d}",
        tokenized=plant_data_tokenized,
        model_config=plant_model_config,
        train_config=train_config,
        tags=PLANTCAD_TAGS_BATCH_TUNE,
        eval_harness_tasks=[],
        use_default_validation=False,
    )

    # Wrap with try/catch to handle OOMs
    training_step = dataclasses.replace(training_step, fn=_failure_ok_train)
    
    training_steps.append(training_step)


if __name__ == "__main__":
    logger.info("🧬 PlantCAD Batch Size Tuning Experiment")
    logger.info("=" * 60)
    logger.info(f"Model: {model_size} ({plant_model_config})")
    logger.info(f"Batch sizes to test: {batch_sizes}")
    logger.info("=" * 60)
    
    executor_main(
        steps=[
            plant_data_tokenized,
            *training_steps,
        ]
    )
