#!/usr/bin/env python3
"""
PlantCAD Evaluation Script

This script evaluates a trained PlantCAD model on evolutionary conservation prediction.
"""

import logging
from experiments.plantcad.evaluation import run_conservation_eval, DnaEvalConfig
from marin.execution.executor import ExecutorStep, executor_main

logger = logging.getLogger("ray")

checkpoint_path = "hf://plantcad/_dev_marin_plantcad1_v1_lr_tune/local_store/checkpoints/plantcad-lr-tune-lr1e-04-r07-5e25ff/hf/step-668"
batch_size = 32
max_samples = None

def main():
    logger.info("🧬 PlantCAD Model Evaluation")
    logger.info("=" * 60)
    logger.info(f"Checkpoint path: {checkpoint_path}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Max samples: {max_samples}")
    logger.info("=" * 60)
    
    eval_step = ExecutorStep(
        name=f"evaluation/dna-conservation",
        fn=run_conservation_eval,
        config=DnaEvalConfig(
            checkpoint_path=checkpoint_path,
            batch_size=batch_size,
            max_samples=max_samples,
        ),
        description=f"Zero-shot evolutionary conservation evaluation for model {checkpoint_path}",
        pip_dependency_groups=["dna"],
    )
    
    executor_main(steps=[eval_step])


if __name__ == "__main__":
    main()
