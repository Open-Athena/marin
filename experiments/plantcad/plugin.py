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

"""PlantCAD evaluation plugin for Levanter training."""

import logging
from collections.abc import Callable

from levanter.eval import EvalPlugin
from levanter.callbacks import StepInfo
from levanter.utils.hf_utils import HfTokenizer
from jax.sharding import Mesh
from haliax.partitioning import ResourceMapping
from experiments.plantcad.evaluation import DnaEvalBaseConfig, create_dna_eval_callback

logger = logging.getLogger("ray")


class PlantCADEvaluationPlugin(EvalPlugin):
    """PlantCAD DNA conservation evaluation plugin for Levanter."""

    def __init__(self):
        logger.info("Initialized PlantCAD evaluation plugin")

    def create_callback(
        self,
        *,
        tokenizer: HfTokenizer,
        device_mesh: Mesh,
        compute_axis_mapping: ResourceMapping,
        parameter_axis_mapping: ResourceMapping,
        batch_size: int,
    ) -> Callable[[StepInfo], None]:
        """Create DNA conservation evaluation callback.

        Args:
            tokenizer: Tokenizer for the model
            device_mesh: JAX device mesh for distributed computation
            compute_axis_mapping: Axis mapping for computation
            parameter_axis_mapping: Axis mapping for parameter storage
            batch_size: Evaluation batch size

        Returns:
            Callback function for DNA conservation evaluation
        """
        # Cut batch size in half because current eval runs with
        # model in full precision rather than bf16
        # TODO: implement mixed-precision in eval
        config = DnaEvalBaseConfig(batch_size=batch_size // 2)
        logger.info(f"Creating conservation evaluation callback with config: {config}")
        return create_dna_eval_callback(
            config=config,
            tokenizer=tokenizer,
            device_mesh=device_mesh,
            compute_axis_mapping=compute_axis_mapping,
            parameter_axis_mapping=parameter_axis_mapping,
        )
