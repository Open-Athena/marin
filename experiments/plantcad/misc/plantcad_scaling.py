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
Scaling Law Analysis for PlantCAD ROC Data
Fits ROC to log of step count with FLOP information
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# A100 estimation:
# - 312 TFLOPS (https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf)
# - 3.12e+14 FLOPS * 32 A100s * 3 days * 86400 seconds/day = 2.59x10^21 FLOPS
# - 50% MFU ==> ~1x10^21 FLOPS
# H100 estimation:
# - 1,671 TFLOPS (https://www.nvidia.com/en-us/data-center/h100/)
# - 1,671/312 = 5.36x A100s
# - 32 * 72 = 2304 A100 hrs vs 8 * 8 = 64 H100 hours ==> 64 * 5.36 = 343 A100 hrs
# - PC1 training was ~36x more GPU hours

METRICS = """step,lr,roc,flops
1673,9.93779613054e-05,0.53528062,3404319038770249728
5019,9.30571259232e-05,0.5486141,10208889829526075392
6692,8.75795158208e-05,0.55787022,13613208868296327168
11711,6.45971667836e-05,0.5689541,23818031411037732864
16730,3.83249971491e-05,0.58587594,34024887597171470336
21749,1.77692745637e-05,0.5934248799999999,44231743783305216000
"""

# Set style for better plots
plt.style.use("classic")


def load_and_clean_data():
    """Load and clean the embedded CSV data"""
    from io import StringIO

    df = pd.read_csv(StringIO(METRICS))
    # Remove any empty rows
    df = df.dropna()
    # Convert flops to numpy float64 manually
    flop_values = []
    for flop_str in df["flops"]:
        flop_values.append(float(flop_str))
    df["flops"] = np.array(flop_values, dtype=np.float64)
    return df


def fit_scaling_law(steps, roc_values):
    """
    Fit ROC to log10(steps) using linear regression
    Model: ROC = a * log10(steps) + b
    """
    # Transform steps to log base 10 scale
    log_steps = np.log10(steps).reshape(-1, 1)

    # Fit linear regression
    model = LinearRegression()
    model.fit(log_steps, roc_values)

    # Get predictions
    roc_pred = model.predict(log_steps)

    # Calculate R²
    r2 = r2_score(roc_values, roc_pred)

    return model, r2


def calculate_steps_for_roc(target_roc, model):
    """
    Calculate steps needed to achieve target ROC value
    Inverse of: ROC = a * log₁₀(steps) + b
    Solved for steps: steps = 10^((ROC - b) / a)
    """
    coef = model.coef_[0]
    intercept = model.intercept_

    # Solve: target_roc = coef * log10(steps) + intercept
    # log10(steps) = (target_roc - intercept) / coef
    # steps = 10^((target_roc - intercept) / coef)

    log10_steps = (target_roc - intercept) / coef
    steps = 10**log10_steps
    return steps


def create_visualization(steps, roc_values, flops_values, model, r2):
    """Create single visualization of scaling law with FLOPs on x-axis and ROC projections"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    # Calculate ROC targets and their required steps/FLOPs
    roc_targets = [0.7, 0.8, 0.9]
    flops_per_step = np.mean(flops_values / steps)
    target_steps = []
    target_flops = []

    for roc in roc_targets:
        steps_needed = calculate_steps_for_roc(roc, model)
        flops_needed = steps_needed * flops_per_step
        target_steps.append(steps_needed)
        target_flops.append(flops_needed)

    # Create extended range for visualization using FLOPs
    max_target_flops = max(target_flops) if target_flops else max(flops_values) * 2
    max_flops = max(max_target_flops * 1.2, max(flops_values) * 2)
    flops_extended = np.logspace(np.log10(min(flops_values)), np.log10(max_flops), 1000)

    # Convert FLOPs back to steps for the model (since model was fitted on log(steps))
    steps_extended = flops_extended / flops_per_step
    log_steps_extended = np.log10(steps_extended)
    roc_extended = model.predict(log_steps_extended.reshape(-1, 1))

    # Plot original data using FLOPs
    ax.scatter(
        flops_values, roc_values, alpha=0.8, s=120, color="blue", label="Observed Data", zorder=5, edgecolors="darkblue"
    )

    # Plot fitted line using FLOPs
    ax.plot(flops_extended, roc_extended, "r-", linewidth=3, label="Best Fit", alpha=0.8)

    # Plot ROC target projections
    colors = ["green", "orange", "purple"]
    for roc, flops_needed, color in zip(roc_targets, target_flops, colors, strict=False):
        ax.scatter(
            [flops_needed],
            [roc],
            color=color,
            s=150,
            marker="*",
            label=f"ROC {roc:.1f}: {flops_needed:.2e} FLOPs",
            zorder=6,
            edgecolors="black",
            linewidth=1,
        )

        # Add vertical lines for each target
        ax.axvline(x=flops_needed, color=color, linestyle="--", alpha=0.6, linewidth=1.5)
        ax.axhline(y=roc, color=color, linestyle=":", alpha=0.4, linewidth=1)

    # Set log scale for x-axis with scientific notation
    ax.set_xscale("log")
    ax.set_xlabel("Total FLOPs", fontsize=12, fontweight="bold")
    ax.set_ylabel("ROC Score", fontsize=12, fontweight="bold")
    ax.set_title("PlantCAD + Llama 300M Scaling\n[Sorghum conservation ROC]", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, bbox_to_anchor=(1.05, 1), loc="upper left", numpoints=1, scatterpoints=1)

    # Format x-axis with scientific notation for log scale
    from matplotlib.ticker import LogFormatter

    ax.xaxis.set_major_formatter(LogFormatter(base=10, labelOnlyBase=False))

    # Add model equation and statistics
    coef = model.coef_[0]
    intercept = model.intercept_
    textstr = f"Model: ROC = {coef:.4f} * log₁₀(steps) + {intercept:.4f}\nR² = {r2:.4f}"
    ax.text(
        0.02,
        0.98,
        textstr,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # Create inset plot in lower left showing zoom of observed data region
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    axins = inset_axes(
        ax, width="30%", height="30%", loc="lower left", bbox_to_anchor=(0.05, 0.45, 1, 1), bbox_transform=ax.transAxes
    )

    # Plot the same data in the inset but zoomed to observed data range
    axins.scatter(flops_values, roc_values, alpha=0.8, s=80, color="blue", edgecolors="darkblue", zorder=5)

    # Plot fitted line in the observed range only
    flops_obs_extended = np.logspace(np.log10(min(flops_values) * 0.8), np.log10(max(flops_values) * 1.2), 100)
    steps_obs_extended = flops_obs_extended / flops_per_step
    log_steps_obs_extended = np.log10(steps_obs_extended)
    roc_obs_extended = model.predict(log_steps_obs_extended.reshape(-1, 1))
    axins.plot(flops_obs_extended, roc_obs_extended, "r-", linewidth=2, alpha=0.8)

    # Format inset
    axins.set_xscale("log")
    axins.set_xlim(min(flops_values) * 0.8, max(flops_values) * 1.2)
    axins.set_ylim(min(roc_values) - 0.005, max(roc_values) + 0.005)
    axins.grid(True, alpha=0.3)
    axins.tick_params(labelsize=8)
    axins.xaxis.set_major_formatter(LogFormatter(base=10, labelOnlyBase=False))

    # Add a title to the inset
    axins.set_title("Observed Data Detail", fontsize=10, pad=5)

    # Draw a box around the observed data region on the main plot
    from matplotlib.patches import Rectangle

    obs_flop_min = min(flops_values) * 0.8
    obs_flop_max = max(flops_values) * 1.2
    obs_roc_min = min(roc_values) - 0.005
    obs_roc_max = max(roc_values) + 0.005

    # Create rectangle patch
    rect = Rectangle(
        (obs_flop_min, obs_roc_min),
        obs_flop_max - obs_flop_min,
        obs_roc_max - obs_roc_min,
        linewidth=1.5,
        edgecolor="black",
        facecolor="none",
        linestyle="--",
        alpha=0.7,
    )
    ax.add_patch(rect)

    # Add connection lines from the box corners to the inset plot
    from matplotlib.patches import ConnectionPatch

    # Get the corners of the observed data box and inset plot
    # Top left corner of box to bottom left corner of inset
    con1 = ConnectionPatch(
        xyA=(obs_flop_min, obs_roc_max),
        coordsA=ax.transData,
        xyB=(0, 0),
        coordsB=axins.transAxes,
        shrinkA=0,
        shrinkB=0,
        color="black",
        alpha=0.7,
        linewidth=1.5,
        linestyle="-",
    )

    # Top right corner of box to bottom right corner of inset
    con2 = ConnectionPatch(
        xyA=(obs_flop_max, obs_roc_max),
        coordsA=ax.transData,
        xyB=(1, 0),
        coordsB=axins.transAxes,
        shrinkA=0,
        shrinkB=0,
        color="black",
        alpha=0.7,
        linewidth=1.5,
        linestyle="-",
    )

    fig.add_artist(con1)
    fig.add_artist(con2)

    return fig, dict(zip(roc_targets, zip(target_steps, target_flops, strict=False), strict=False))


def main():
    # Load data
    print("Loading PlantCAD metrics data...")
    df = load_and_clean_data()

    print(f"Loaded {len(df)} data points:")
    print(df.to_string())
    print()

    print("Training data with FLOP values:")
    for _i, row in df.iterrows():
        tflops = row["flops"] / 1e12
        print(f"  Step {row['step']:>5}: ROC {row['roc']:.6f}, {tflops:>8.0f} TFLOPs")
    print()

    # Extract steps, ROC values, and FLOP values
    steps = df["step"].values
    roc_values = df["roc"].values
    flops_values = df["flops"].values

    # Fit scaling law
    print("Fitting scaling law: ROC = a * log₁₀(steps) + b")
    model, r2 = fit_scaling_law(steps, roc_values)

    # Print model parameters
    coef = model.coef_[0]
    intercept = model.intercept_
    print("Fitted parameters:")
    print(f"  Coefficient (a): {coef:.6f}")
    print(f"  Intercept (b): {intercept:.6f}")
    print(f"  R-squared: {r2:.6f}")
    print()

    # Create visualization
    print("Creating visualization...")

    # Calculate and display ROC target projections
    roc_targets = [0.7, 0.8, 0.9]
    flops_per_step = np.mean(flops_values / steps)
    print("Steps and FLOPs required for target ROC values:")
    for roc in roc_targets:
        steps_needed = calculate_steps_for_roc(roc, model)
        flops_needed = steps_needed * flops_per_step
        print(f"  ROC {roc:.1f}: {steps_needed:,.0f} steps ({flops_needed:.2e} FLOPs)")
    print()

    # Single scaling law plot with projections
    fig, target_projections = create_visualization(steps, roc_values, flops_values, model, r2)
    fig.savefig("results/plantcad_scaling.png", dpi=300, bbox_inches="tight")
    print("Saved: results/plantcad_scaling.png")

    # Save model parameters
    with open("results/plantcad_scaling.txt", "w") as f:
        f.write("PlantCAD ROC Scaling Law Analysis\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Model: ROC = {coef:.6f} * log₁₀(steps) + {intercept:.6f}\n\n")
        f.write("Fitted parameters:\n")
        f.write(f"  Coefficient (a): {coef:.6f}\n")
        f.write(f"  Intercept (b): {intercept:.6f}\n")
        f.write(f"  R-squared: {r2:.6f}\n\n")
        f.write(f"Training data points: {len(df)}\n")
        f.write(f"Step range: {min(steps):,} - {max(steps):,}\n")
        f.write(f"ROC range: {min(roc_values):.6f} - {max(roc_values):.6f}\n")
        f.write(f"FLOP range: {min(flops_values):.2e} - {max(flops_values):.2e}\n\n")
        f.write("Steps and FLOPs required for target ROC values:\n")
        for roc, (steps_needed, flops_needed) in target_projections.items():
            f.write(f"  ROC {roc:.1f}: {steps_needed:,.0f} steps ({flops_needed:.2e} FLOPs)\n")

    print("Saved: results/plantcad_scaling.txt")
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
