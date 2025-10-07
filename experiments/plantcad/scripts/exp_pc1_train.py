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
PlantCAD training experiment - single 300M model with optimized learning rate
"""

from dataclasses import replace
import logging

from experiments.defaults import default_train
from experiments.plantcad.utils import (
    get_available_gpus,
    get_plantcad_config,
    get_plantcad_training_dataset,
    PLANTCAD_TAGS_BASE,
    PLANTCAD_DATASET_EXAMPLES,
)
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.resources import GpuConfig
from experiments.plantcad.plugin import PlantCADEvaluationPlugin
from experiments.plantcad.evaluation import resolve_checkpoint_path

logger = logging.getLogger("ray")

# Run iteration
run_number = 16

# Resources
num_gpus = get_available_gpus(local_only=True)
target_examples = PLANTCAD_DATASET_EXAMPLES * 10  # 10 epochs

# Best learning rate from tuning experiments
learning_rate = 3e-4

# Batch size
# Ideally global batch would be fixed and device count wouldn't
# matter (w/ grad accum), however OOMs are unavoidable on GPUs
# unless the global batch varies as a function of device count.
# TODO: find out how to fix global batch on GPUs
micro_batch_size = 256
global_batch_size = micro_batch_size * num_gpus

# Shuffle training data
shuffle = True

# Training configuration
num_train_steps = target_examples // global_batch_size
steps_per_export = num_train_steps // 10
steps_per_cycle = num_train_steps // 10
steps_per_eval = num_train_steps // 100

# Model configuration - use 600M by default
model_size = "600m"
plant_model_config = get_plantcad_config(model_size)

# PlantCAD1 training dataset
plant_data_tokenized = get_plantcad_training_dataset(use_pretokenized=True)
plugin_class = PlantCADEvaluationPlugin

hf_checkpoint_path = (
    "hf://plantcad/_dev_marin_plantcad1_v2_train/local_store/checkpoints/plantcad-train-600m-r12-7ea0fc/hf/step-26782"
)
if hf_checkpoint_path is not None:
    hf_checkpoint_path = resolve_checkpoint_path(hf_checkpoint_path)

# Training configuration
train_config = SimpleTrainConfig(
    resources=GpuConfig(gpu_count=num_gpus),
    data_seed=42,
    # See marin/experiments/exp432_wsds.py
    lr_schedule="inv",
    warmup=0.05,
    decay=0.1,
    cycle_length=steps_per_cycle,
    train_batch_size=global_batch_size,
    per_device_eval_parallelism=micro_batch_size,
    steps_per_eval=steps_per_eval,
    num_train_steps=num_train_steps,
    learning_rate=learning_rate,
    steps_per_export=steps_per_export,
    initialize_from_hf=hf_checkpoint_path,
    # TODO: figure out why this is broken
    # eval_plugins=[
    #     EvalPluginConfig(
    #         plugin_class=f"{plugin_class.__module__}.{plugin_class.__qualname__}",
    #         steps=steps_per_cycle,
    #     )
    # ]
)

# Create training step
training_step = default_train(
    name=f"plantcad-train-{model_size}-r{run_number:02d}",
    tokenized=plant_data_tokenized,
    model_config=plant_model_config,
    train_config=train_config,
    tags=[*PLANTCAD_TAGS_BASE, "training"],
    eval_harness_tasks=[],
    use_default_validation=False,
    shuffle=shuffle,
)

# Add dependencies for DNA evals
# TODO: figure out why this seems to have no effect;
#       I still see "no module sklearn" errors
training_step = replace(training_step, pip_dependency_groups=["dna"])

if __name__ == "__main__":
    logger.info("🧬 PlantCAD Training Experiment")
    logger.info("=" * 60)
    logger.info(f"Model: {model_size} ({plant_model_config})")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Global batch size: {global_batch_size}")
    logger.info(f"Micro batch size: {micro_batch_size}")
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
