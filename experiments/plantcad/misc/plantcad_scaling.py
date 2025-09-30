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
from scipy.optimize import curve_fit

# A100 estimation:
# - 312 TFLOPS (https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf)
# - 3.12e+14 FLOPS * 32 A100s * 3 days * 86400 seconds/day = 2.59x10^21 FLOPS
# - 50% MFU ==> ~1x10^21 FLOPS
# H100 estimation:
# - 1,671 TFLOPS (https://www.nvidia.com/en-us/data-center/h100/)
# - 1,671/312 = 5.36x A100s
# - 32 * 72 = 2304 A100 hrs vs 8 * 8 = 64 H100 hours ==> 64 * 5.36 = 343 A100 hrs
# - PC1 training was ~36x more GPU hours

# METRICS = """step,lr,roc_auc,flops
# 1673,9.93779613054e-05,0.53528062,3404319038770249728
# 5019,9.30571259232e-05,0.5486141,10208889829526075392
# 6692,8.75795158208e-05,0.55787022,13613208868296327168
# 11711,6.45971667836e-05,0.5689541,23818031411037732864
# 16730,3.83249971491e-05,0.58587594,34024887597171470336
# 21749,1.77692745637e-05,0.5934248799999999,44231743783305216000
# """

METRICS = """step,lr,roc_auc,flops
1673,9.93770809145e-05,0.535217,3.4043190387702497e+18
3346,9.7036921943e-05,0.546725,6.804570790755828e+18
5019,9.30542882997e-05,0.549917,1.0208889829526075e+19
6692,8.75831901794e-05,0.558042,1.3613208868296327e+19
8365,8.08332551969e-05,0.56029,1.701142697688957e+19
10038,7.30716710677e-05,0.565785,2.041574601565982e+19
11711,6.46127955405e-05,0.570048,2.3818031411037733e+19
13384,5.57357234356e-05,0.576358,2.722438409320032e+19
15057,4.68768230348e-05,0.583593,3.0624635845185896e+19
16730,3.83349033654e-05,0.585834,3.402488759717147e+19
18403,3.04164732369e-05,0.589215,3.742717299254939e+19
20076,2.34794097195e-05,0.588738,4.08294583879273e+19
21749,1.77572965185e-05,0.593178,4.423174378330522e+19
"""

# Set style for better plots
plt.style.use("classic")


def load_and_clean_data():
    """Load and clean the embedded CSV data"""
    from io import StringIO

    df = pd.read_csv(StringIO(METRICS))
    assert df.notna().all().all()
    return df


def fit_linear_scaling_law(steps, roc_values):
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


def sigmoid_function(log_steps, a, b, c, d):
    """
    Sigmoid function: ROC = a / (1 + exp(-b * (log_steps - c))) + d
    Parameters:
    - a: amplitude (max_roc - min_roc)
    - b: steepness
    - c: midpoint (inflection point on log scale)
    - d: baseline (min_roc)
    """
    return a / (1 + np.exp(-b * (log_steps - c))) + d


def fit_sigmoid_scaling_law(steps, roc_values, min_roc=0.5, max_roc=1.0):
    """
    Fit ROC to log10(steps) using sigmoid function
    Model: ROC = a / (1 + exp(-b * (log10(steps) - c))) + d
    Constrained to start at min_roc and asymptote to max_roc
    """
    # Transform steps to log base 10 scale
    log_steps = np.log10(steps)

    # For proper constraint: amplitude = max_roc - min_roc, baseline = min_roc
    # So max value will be amplitude + baseline = max_roc
    amplitude = max_roc - min_roc  # a (this is fixed)
    steepness = 1.0  # b
    midpoint = np.median(log_steps)  # c
    baseline = min_roc  # d (this is fixed)

    # Initial guess
    p0 = [amplitude, steepness, midpoint, baseline]

    # Strict parameter bounds to enforce constraints
    # a = exactly (max_roc - min_roc), d = exactly min_roc
    bounds = (
        [amplitude - 0.001, -np.inf, -np.inf, baseline - 0.001],
        [amplitude + 0.001, np.inf, np.inf, baseline + 0.001],
    )

    try:
        # Fit the sigmoid
        popt, pcov = curve_fit(sigmoid_function, log_steps, roc_values, p0=p0, bounds=bounds, maxfev=5000)

        # Ensure exact constraints by fixing the amplitude and baseline
        popt[0] = amplitude  # Force amplitude to be exactly max_roc - min_roc
        popt[3] = baseline  # Force baseline to be exactly min_roc

        # Get predictions with corrected parameters
        roc_pred = sigmoid_function(log_steps, *popt)

        # Calculate R²
        r2 = r2_score(roc_values, roc_pred)

        return popt, r2, sigmoid_function

    except Exception as e:
        print(f"Sigmoid fitting failed: {e}")
        # Return linear fit as fallback
        linear_model, linear_r2 = fit_linear_scaling_law(steps, roc_values)
        return None, linear_r2, None


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


def calculate_steps_for_sigmoid_roc(target_roc, sigmoid_params):
    """
    Calculate steps needed to achieve target ROC value for sigmoid model
    Inverse of: ROC = a / (1 + exp(-b * (log₁₀(steps) - c))) + d
    Solved for steps: steps = 10^(c - ln((a / (ROC - d)) - 1) / b)
    """
    a, b, c, d = sigmoid_params

    # Check if target is achievable
    if target_roc <= d or target_roc >= (a + d):
        return None  # Target ROC is outside sigmoid range

    # Solve: target_roc = a / (1 + exp(-b * (log10(steps) - c))) + d
    # target_roc - d = a / (1 + exp(-b * (log10(steps) - c)))
    # (target_roc - d) / a = 1 / (1 + exp(-b * (log10(steps) - c)))
    # (1 + exp(-b * (log10(steps) - c))) = a / (target_roc - d)
    # exp(-b * (log10(steps) - c)) = (a / (target_roc - d)) - 1
    # -b * (log10(steps) - c) = ln((a / (target_roc - d)) - 1)
    # log10(steps) - c = -ln((a / (target_roc - d)) - 1) / b
    # log10(steps) = c - ln((a / (target_roc - d)) - 1) / b
    # steps = 10^(c - ln((a / (target_roc - d)) - 1) / b)

    inner_term = (a / (target_roc - d)) - 1
    if inner_term <= 0:
        return None  # Mathematical error

    log10_steps = c - np.log(inner_term) / b
    steps = 10**log10_steps
    return steps


def create_visualization(
    steps, roc_values, flops_values, linear_model, linear_r2, sigmoid_params, sigmoid_r2, sigmoid_func
):
    """Create single visualization of scaling law with FLOPs on x-axis and ROC projections"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    # Create extended range for visualization using FLOPs
    flops_per_step = np.mean(flops_values / steps)
    max_flops = 1e23  # Limit to 1e+23
    flops_extended = np.logspace(np.log10(min(flops_values)), np.log10(max_flops), 1000)

    # Convert FLOPs back to steps for the models
    steps_extended = flops_extended / flops_per_step
    log_steps_extended = np.log10(steps_extended)

    # Linear model predictions
    linear_roc_extended = linear_model.predict(log_steps_extended.reshape(-1, 1))

    # Sigmoid model predictions (if available)
    sigmoid_roc_extended = None
    if sigmoid_params is not None and sigmoid_func is not None:
        sigmoid_roc_extended = sigmoid_func(log_steps_extended, *sigmoid_params)

    # Plotly default palette colors
    plotly_blue = "#1f77b4"
    plotly_red = "#d62728"
    plotly_green = "#2ca02c"

    # Plot original data using FLOPs
    ax.scatter(
        flops_values,
        roc_values,
        alpha=0.8,
        s=120,
        color=plotly_blue,
        label="Observed Data",
        zorder=5,
        edgecolors="#0f4c75",
    )

    # Plot linear fit using FLOPs
    ax.plot(
        flops_extended,
        linear_roc_extended,
        color=plotly_red,
        linestyle="-",
        linewidth=3,
        label=f"Linear Fit (R²={linear_r2:.3f})",
        alpha=0.8,
    )

    # Plot sigmoid fit using FLOPs (if available)
    if sigmoid_roc_extended is not None:
        ax.plot(
            flops_extended,
            sigmoid_roc_extended,
            color=plotly_green,
            linestyle="-",
            linewidth=3,
            label=f"Sigmoid Fit (R²={sigmoid_r2:.3f})",
            alpha=0.8,
        )

        # Add sigmoid target projections
        flops_per_step = np.mean(flops_values / steps)
        roc_targets = [0.7, 0.8, 0.9]
        target_colors = ["#ff7f0e", "#9467bd", "#8c564b"]  # Plotly orange, purple, brown

        for roc, color in zip(roc_targets, target_colors, strict=False):
            sigmoid_steps = calculate_steps_for_sigmoid_roc(roc, sigmoid_params)
            if sigmoid_steps is not None:
                sigmoid_flops = sigmoid_steps * flops_per_step

                # Add marker for sigmoid target
                ax.scatter(
                    [sigmoid_flops],
                    [roc],
                    color=color,
                    s=150,
                    marker="s",
                    label=f"Sigmoid ROC {roc:.1f}: {sigmoid_flops:.2e} FLOPs",
                    zorder=6,
                    edgecolors="black",
                    linewidth=1,
                )

                # Add vertical line
                ax.axvline(x=sigmoid_flops, color=color, linestyle=":", alpha=0.6, linewidth=2)

    # Set log scale for x-axis with scientific notation
    ax.set_xscale("log")
    ax.set_xlim(min(flops_values) * 0.1, 1e23)  # Limit x-axis to 1e+23 with more left padding
    ax.set_xlabel("Total FLOPs", fontsize=12, fontweight="bold")
    ax.set_ylabel("ROC Score", fontsize=12, fontweight="bold")
    ax.set_title("PlantCAD + Llama 300M Scaling\n[Sorghum conservation ROC]", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, bbox_to_anchor=(1.05, 1), loc="upper left", numpoints=1, scatterpoints=1)

    # Format x-axis with scientific notation for log scale
    from matplotlib.ticker import LogFormatter

    ax.xaxis.set_major_formatter(LogFormatter(base=10, labelOnlyBase=False))

    # Create inset plot in lower left showing zoom of observed data region
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    axins = inset_axes(
        ax, width="52%", height="52%", loc="lower left", bbox_to_anchor=(0.03, 0.42, 1, 1), bbox_transform=ax.transAxes
    )

    # Plot the same data in the inset but zoomed to observed data range
    axins.scatter(flops_values, roc_values, alpha=0.8, s=80, color=plotly_blue, edgecolors="#0f4c75", zorder=5)

    # Plot fitted lines in the observed range only
    flops_obs_extended = np.logspace(np.log10(min(flops_values) * 0.5), np.log10(max(flops_values) * 1.2), 100)
    steps_obs_extended = flops_obs_extended / flops_per_step
    log_steps_obs_extended = np.log10(steps_obs_extended)

    # Linear fit
    linear_roc_obs_extended = linear_model.predict(log_steps_obs_extended.reshape(-1, 1))
    axins.plot(flops_obs_extended, linear_roc_obs_extended, color=plotly_red, linestyle="-", linewidth=2, alpha=0.8)

    # Sigmoid fit (if available)
    if sigmoid_params is not None and sigmoid_func is not None:
        sigmoid_roc_obs_extended = sigmoid_func(log_steps_obs_extended, *sigmoid_params)
        axins.plot(
            flops_obs_extended, sigmoid_roc_obs_extended, color=plotly_green, linestyle="-", linewidth=2, alpha=0.8
        )

    # Format inset
    axins.set_xscale("log")
    axins.set_xlim(min(flops_values) * 0.5, max(flops_values) * 1.2)
    axins.set_ylim(min(roc_values) - 0.005, max(roc_values) + 0.005)
    axins.grid(True, alpha=0.3)
    axins.tick_params(labelsize=8)
    axins.xaxis.set_major_formatter(LogFormatter(base=10, labelOnlyBase=False))

    # Add a title to the inset
    axins.set_title("Observed Data Detail", fontsize=10, pad=5)

    # Draw a box around the observed data region on the main plot
    from matplotlib.patches import Rectangle

    obs_flop_min = min(flops_values) * 0.5
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

    return fig, {}


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
        print(f"  Step {row['step']:>5}: ROC {row['roc_auc']:.6f}, {tflops:>8.0f} TFLOPs")
    print()

    # Extract steps, ROC values, and FLOP values
    steps = df["step"].values
    roc_values = df["roc_auc"].values
    flops_values = df["flops"].values

    # Fit scaling laws
    print("Fitting linear scaling law: ROC = a * log₁₀(steps) + b")
    linear_model, linear_r2 = fit_linear_scaling_law(steps, roc_values)

    print("Fitting sigmoid scaling law: ROC = a / (1 + exp(-b * (log₁₀(steps) - c))) + d")
    sigmoid_params, sigmoid_r2, sigmoid_func = fit_sigmoid_scaling_law(steps, roc_values)

    # Print model parameters
    print("Linear model fitted parameters:")
    linear_coef = linear_model.coef_[0]
    linear_intercept = linear_model.intercept_
    print(f"  Coefficient (a): {linear_coef:.6f}")
    print(f"  Intercept (b): {linear_intercept:.6f}")
    print(f"  R-squared: {linear_r2:.6f}")
    print()

    if sigmoid_params is not None:
        print("Sigmoid model fitted parameters:")
        print(f"  Amplitude (a): {sigmoid_params[0]:.6f}")
        print(f"  Steepness (b): {sigmoid_params[1]:.6f}")
        print(f"  Midpoint (c): {sigmoid_params[2]:.6f}")
        print(f"  Baseline (d): {sigmoid_params[3]:.6f}")
        print(f"  R-squared: {sigmoid_r2:.6f}")
    else:
        print("Sigmoid fitting failed, using linear model only")
    print()

    # Calculate sigmoid target projections
    if sigmoid_params is not None:
        print("Steps and FLOPs required for target ROC values (sigmoid model):")
        roc_targets = [0.7, 0.8, 0.9]
        flops_per_step = np.mean(flops_values / steps)
        for roc in roc_targets:
            sigmoid_steps = calculate_steps_for_sigmoid_roc(roc, sigmoid_params)
            if sigmoid_steps is not None:
                sigmoid_flops = sigmoid_steps * flops_per_step
                print(f"  ROC {roc:.1f}: {sigmoid_steps:,.0f} steps ({sigmoid_flops:.2e} FLOPs)")
            else:
                print(f"  ROC {roc:.1f}: Not achievable with sigmoid model")
        print()

    # Create visualization
    print("Creating visualization...")

    # Single scaling law plot with both fits
    fig, _ = create_visualization(
        steps, roc_values, flops_values, linear_model, linear_r2, sigmoid_params, sigmoid_r2, sigmoid_func
    )
    fig.savefig("results/plantcad_scaling.png", dpi=300, bbox_inches="tight")
    print("Saved: results/plantcad_scaling.png")

    # Save model parameters
    with open("results/plantcad_scaling.txt", "w") as f:
        f.write("PlantCAD ROC Scaling Law Analysis\n")
        f.write("=" * 40 + "\n\n")

        # Linear model
        f.write(f"Linear Model: ROC = {linear_coef:.6f} * log₁₀(steps) + {linear_intercept:.6f}\n")
        f.write("Linear model fitted parameters:\n")
        f.write(f"  Coefficient (a): {linear_coef:.6f}\n")
        f.write(f"  Intercept (b): {linear_intercept:.6f}\n")
        f.write(f"  R-squared: {linear_r2:.6f}\n\n")

        # Sigmoid model
        if sigmoid_params is not None:
            f.write("Sigmoid Model: ROC = a / (1 + exp(-b * (log₁₀(steps) - c))) + d\n")
            f.write("Sigmoid model fitted parameters:\n")
            f.write(f"  Amplitude (a): {sigmoid_params[0]:.6f}\n")
            f.write(f"  Steepness (b): {sigmoid_params[1]:.6f}\n")
            f.write(f"  Midpoint (c): {sigmoid_params[2]:.6f}\n")
            f.write(f"  Baseline (d): {sigmoid_params[3]:.6f}\n")
            f.write(f"  R-squared: {sigmoid_r2:.6f}\n\n")
        else:
            f.write("Sigmoid fitting failed\n\n")

        f.write(f"Training data points: {len(df)}\n")
        f.write(f"Step range: {min(steps):,} - {max(steps):,}\n")
        f.write(f"ROC range: {min(roc_values):.6f} - {max(roc_values):.6f}\n")
        f.write(f"FLOP range: {min(flops_values):.2e} - {max(flops_values):.2e}\n")

    print("Saved: results/plantcad_scaling.txt")
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
