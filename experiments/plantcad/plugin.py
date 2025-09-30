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
from typing import Any
from collections.abc import Callable

from levanter.eval import EvalPlugin
from levanter.callbacks import StepInfo
from experiments.plantcad.evaluation import DnaEvalBaseConfig, create_dna_eval_callback

logger = logging.getLogger("ray")


class PlantCADEvaluationPlugin(EvalPlugin):
    """PlantCAD DNA conservation evaluation plugin for Levanter."""

    def __init__(self, config: dict[str, Any]):
        # Store the dict config directly
        self.config = DnaEvalBaseConfig(**config)
        logger.info(f"Initialized PlantCAD evaluation plugin with config: {self.config}")

    def create_callback(self, **kwargs) -> Callable[[StepInfo], None]:
        """Create DNA conservation evaluation callback."""
        return create_dna_eval_callback(self.config)
