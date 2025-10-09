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
PlantCAD Evaluation Script

This script evaluates a trained PlantCAD model on evolutionary conservation prediction.
"""

import logging
from experiments.plantcad.evaluation import run_conservation_eval, DnaEvalConfig
from experiments.plantcad.utils import get_checkpoints
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned

logger = logging.getLogger("ray")

# Script parameters
# TODO: Create task/config for multi-checkpoint evaluation
checkpoint_dir = "hf://plantcad/_dev_marin_plantcad1_v3_train/local_store/checkpoints/plantcad-train-600m-r16-a1bc43/hf"
batch_size: int = 32  # 16  # biggest batch size that works for 600m + 40G A100
dtype: str | None = "bfloat16"
checkpoint_steps: list[int] | None = None

# Create an evaluation task for each checkpoint
eval_steps = []
for checkpoint in get_checkpoints(checkpoint_dir):
    path, step = checkpoint["path"], checkpoint["step"]
    if checkpoint_steps is not None and step not in checkpoint_steps:
        continue
    logger.info(f"Adding eval for checkpoint at {step=}: {path}")
    eval_step = ExecutorStep(
        name="evaluation/dna-conservation",
        fn=run_conservation_eval,
        config=DnaEvalConfig(
            checkpoint_path=versioned(path),
            batch_size=batch_size,
            dtype=dtype,
            output_dir=this_output_path(),
        ),
        description=f"Zero-shot evolutionary conservation evaluation for model {path}",
        pip_dependency_groups=["dna"],
    )
    eval_steps.append(eval_step)


def main():
    logger.info("🧬 PlantCAD Model Evaluation")
    logger.info("=" * 60)
    logger.info(f"Checkpoint dir: {checkpoint_dir}")
    logger.info(f"Batch size: {batch_size}")
    logger.info("=" * 60)
    executor_main(steps=eval_steps)


if __name__ == "__main__":
    main()
