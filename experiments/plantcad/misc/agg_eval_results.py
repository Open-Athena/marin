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

import json
import re
from pathlib import Path
import pandas as pd


def main():
    data = []
    base_dir = "~/sky_workdir/local_store/evaluation"

    for eval_dir in Path(base_dir).expanduser().glob("dna-conservation-*"):
        with open(eval_dir / "results.json") as f:
            result = json.load(f)
        config = result.get("config", {})
        results_data = result.get("results", {})
        checkpoint_path = config.get("checkpoint_path", "")
        roc_auc = results_data.get("roc_auc")

        if roc_auc and checkpoint_path:
            step_match = re.search(r"step-(\d+)$", checkpoint_path)
            step = int(step_match.group(1)) if step_match else None
            data.append({"roc_auc": roc_auc, "step": step, "checkpoint_path": checkpoint_path})

    if not data:
        print("No results found")
        return

    df = pd.DataFrame(data).sort_values("step", na_position="last")

    print(df.to_string(index=False, float_format="%.6f"))
    print(f"\n# Total: {len(df)}")
    print(df["roc_auc"].describe().to_string())


if __name__ == "__main__":
    main()
