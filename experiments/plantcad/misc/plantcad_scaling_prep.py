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

import pandas as pd
from io import StringIO
from pathlib import Path

DATA_DIR = Path("~/Downloads").expanduser()

RUN1_DATA = """composite_step,raw_step,eval_roc_auc,run_number
2678,2678,0.549341,r1
5356,5356,0.566597,r1
8034,8034,0.604521,r1
10712,10712,0.626729,r1
13390,13390,0.631095,r1
16068,16068,0.607316,r1
18746,18746,0.622988,r1
21424,21424,0.639093,r1
24102,24102,0.650973,r1
26780,26780,0.657882,r1
26782,26782,0.657452,r1"""

RUN2_DATA = """composite_step,raw_step,eval_roc_auc,run_number
29460,2678,0.665902,r2
32138,5356,0.672563,r2
34816,8034,0.673937,r2
37494,10712,0.675633,r2
40172,13390,0.678089,r2
42850,16068,0.684904,r2
45528,18746,0.680056,r2
48206,21424,0.677681,r2
50884,24102,0.679077,r2
53562,26780,0.681293,r2
53564,26782,0.680195,r2"""

MAX_DISTANCE = 100


def find_closest(
    target_step: int, source_df: pd.DataFrame, value_col: str, max_distance: int | None = None
) -> tuple[float | None, int | None, int]:
    """Find closest step in source dataframe and return value, matched step, and distance.
    Returns (None, None, distance) if distance exceeds max_distance."""
    idx = (source_df["step"] - target_step).abs().idxmin()
    closest_step = source_df.loc[idx, "step"]
    distance = abs(closest_step - target_step)

    if max_distance is not None and distance > max_distance:
        return None, None, distance

    value = source_df.loc[idx, value_col]
    return value, closest_step, distance


def load_csv_data(eval_loss_path: Path, gflops_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and prepare eval_loss and gflops dataframes."""
    # Load eval_loss data
    eval_loss_df = pd.read_csv(eval_loss_path)
    eval_loss_df.columns = ["step", "eval_loss", "eval_loss_min", "eval_loss_max"]
    eval_loss_df = eval_loss_df[["step", "eval_loss"]]

    # Load gflops data
    gflops_df = pd.read_csv(gflops_path)
    gflops_df.columns = ["step", "gflops", "gflops_min", "gflops_max"]
    gflops_df = gflops_df[["step", "gflops"]]

    return eval_loss_df, gflops_df


def merge_metrics(
    df: pd.DataFrame, eval_loss_df: pd.DataFrame, gflops_df: pd.DataFrame, max_distance: int | None = None
) -> pd.DataFrame:
    """Merge eval_loss and gflops data into the main dataframe."""
    # Initialize columns
    df["eval_loss"] = None
    df["eval_loss_step"] = None
    df["gflops"] = None
    df["gflops_step"] = None

    print("Finding closest steps (using raw_step)...")
    print("\nEval Loss matches:")
    for idx, row in df.iterrows():
        eval_loss, matched_step, dist = find_closest(row["raw_step"], eval_loss_df, "eval_loss", max_distance)
        df.at[idx, "eval_loss"] = eval_loss
        df.at[idx, "eval_loss_step"] = matched_step
        status = "IGNORED (too far)" if eval_loss is None else "OK"
        print(f"  raw_step={row['raw_step']}, matched_step={matched_step}, distance={dist} [{status}]")

    print("\nGFlops matches:")
    for idx, row in df.iterrows():
        gflops, matched_step, dist = find_closest(row["raw_step"], gflops_df, "gflops", max_distance)
        df.at[idx, "gflops"] = gflops
        df.at[idx, "gflops_step"] = matched_step
        status = "IGNORED (too far)" if gflops is None else "OK"
        print(f"  raw_step={row['raw_step']}, matched_step={matched_step}, distance={dist} [{status}]")

    return df


def prepare_run1_data() -> pd.DataFrame:
    """Prepare run 1 data with metrics."""
    print("=== Run 1 ===")
    df = pd.read_csv(StringIO(RUN1_DATA))

    eval_loss_df, gflops_df = load_csv_data(
        DATA_DIR / "wandb_export_2025-10-08T20_46_11.694-04_00.csv",
        DATA_DIR / "wandb_export_2025-10-08T20_47_02.420-04_00.csv",
    )

    df = merge_metrics(df, eval_loss_df, gflops_df, max_distance=MAX_DISTANCE)

    # Format and print result
    result = df[
        [
            "composite_step",
            "raw_step",
            "eval_roc_auc",
            "eval_loss",
            "eval_loss_step",
            "gflops",
            "gflops_step",
            "run_number",
        ]
    ]
    print("\n" + result.to_csv(index=False))
    return result


def prepare_run2_data() -> pd.DataFrame:
    """Prepare run 2 data with metrics."""
    print("=== Run 2 ===")
    df = pd.read_csv(StringIO(RUN2_DATA))

    eval_loss_df, gflops_df = load_csv_data(
        DATA_DIR / "wandb_export_2025-10-08T20_15_01.509-04_00.csv",
        DATA_DIR / "wandb_export_2025-10-08T20_15_46.851-04_00.csv",
    )

    df = merge_metrics(df, eval_loss_df, gflops_df, max_distance=MAX_DISTANCE)

    # Format and print result
    result = df[
        [
            "composite_step",
            "raw_step",
            "eval_roc_auc",
            "eval_loss",
            "eval_loss_step",
            "gflops",
            "gflops_step",
            "run_number",
        ]
    ]
    print("\n" + result.to_csv(index=False))
    return result


def prepare_run_data() -> pd.DataFrame:
    """Load and combine data from both runs."""
    # Process both runs
    run1_df = prepare_run1_data()
    print("\n" + "=" * 60 + "\n")
    run2_df = prepare_run2_data()

    # Combine the dataframes
    combined_df = pd.concat([run1_df, run2_df], ignore_index=True)

    # Create composite_gflops and insert it right after gflops
    composite_gflops = combined_df["gflops"].where(
        combined_df["run_number"] == "r1", combined_df["gflops"] + run1_df["gflops"].max()
    )
    combined_df.insert(loc=combined_df.columns.get_loc("gflops") + 1, column="composite_gflops", value=composite_gflops)

    print("\n" + "=" * 60)
    print("=== Combined Data ===")
    print("=" * 60 + "\n")
    print(combined_df.to_csv(index=False))

    return combined_df


# Run the main function
prepare_run_data()
