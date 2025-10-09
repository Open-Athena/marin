# ruff: noqa
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
from pathlib import Path
from io import StringIO
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr

# A100 estimation:
# - 312 TFLOPS (https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf)
# - 3.12e+14 FLOPS * 32 A100s * 3 days * 86400 seconds/day = 2.59x10^21 FLOPS
# - 50% MFU ==> ~1x10^21 FLOPS
# H100 SXM5 estimation:
# - 1,979 teraFLOPS (https://www.nvidia.com/en-us/data-center/h100/)
# -   1.979x10^15 FLOP/s
# - Cost per hour on lambda = $3.29
# -   9.14x10^-4 $/s
# - (9.14x10^-4 $/s) / (1.979x10^15 FLOP/s) = 4.618494189x10^-19 $/FLOP
# - Assuming 25% MFU: 4.618494189x10^-19 $/FLOP * 4 = 1.847397676x10^-18 $/FLOP
# - Corweave v Lambda multiplier: $6.16 / $3.29 = 1.8723404255

data_dir = Path("~/Downloads").expanduser()

# Cost per FLOP
COST_PER_FLOP = 3.458957351e-18  # $/FLOP on CoreWeave

METRICS = """composite_step,raw_step,eval_roc_auc,eval_loss,eval_loss_step,gflops,composite_gflops,gflops_step,run_number
2678,2678,0.549341,,,11007189685.037434,11007189685.037434,2676,r1
5356,5356,0.566597,1.2227721214294434,5356,22026714659.225075,22026714659.225075,5356,r1
8034,8034,0.604521,1.1921502351760864,8034,33038016107.31258,33038016107.31258,8034,r1
10712,10712,0.626729,,,44045205792.35001,44045205792.35001,10711,r1
13390,13390,0.631095,1.1775107383728027,13390,55060619003.48759,55060619003.48759,13390,r1
16068,16068,0.607316,1.1989717483520508,16068,66080143977.675224,66080143977.675224,16070,r1
18746,18746,0.622988,1.1834633350372314,18746,77074998373.56245,77074998373.56245,18744,r1
21424,21424,0.639093,1.1624643802642822,21424,88115082163.00044,88115082163.00044,21429,r1
24102,24102,0.650973,1.1481825113296509,24102,99105824795.8376,99105824795.8376,24102,r1
26780,26780,0.657882,1.1365783214569092,26780,110117126243.9251,110117126243.9251,26780,r1
26782,26782,0.657452,1.1366617679595947,26782,110125349770.02524,110125349770.02524,26782,r1
29460,2678,0.665902,1.1226577758789062,2670,11007189685.037434,121132539455.06267,2676,r2
32138,5356,0.672563,1.1122373342514038,5340,22026714659.225075,132152064429.2503,5356,r2
34816,8034,0.673937,1.095661997795105,8010,33033904344.26251,143159254114.28775,8033,r2
37494,10712,0.675633,1.0915319919586182,10680,44041094029.29994,154166443799.3252,10710,r2
40172,13390,0.678089,1.0902739763259888,13350,55052395477.38744,165177745247.4127,13388,r2
42850,16068,0.684904,1.081429123878479,16020,66071920451.57509,176197270221.60034,16068,r2
45528,18746,0.680056,1.0780466794967651,18690,77066774847.46231,187192124617.48755,18742,r2
48206,21424,0.677681,1.0775954723358154,21360,88106858636.9003,198232208406.92554,21427,r2
50884,24102,0.679077,1.0722960233688354,24030,99105824795.8376,209231174565.86285,24102,r2
53562,26780,0.681293,1.0778414011001587,26782,110117126243.9251,220242476013.95032,26780,r2
53564,26782,0.680195,1.0778414011001587,26782,110121238006.97516,220246587777.0004,26781,r2"""


def format_cost(flops):
    """Format cost in dollars with appropriate units"""
    cost = flops * COST_PER_FLOP
    if cost >= 1e6:
        return f"(${cost/1e6:.1f}M)"
    elif cost >= 1e3:
        return f"(${cost/1e3:.1f}K)"
    else:
        return f"(${cost:.0f})"


df = pd.read_csv(StringIO(METRICS))

# Convert gigaflops to flops
df["flops"] = df["composite_gflops"].astype(np.float64) * 1e9

# Create visualization
plt.figure(figsize=(12, 6))

# Separate data by run for subtle visual distinction
r1_data = df[df["run_number"] == "r1"]
r2_data = df[df["run_number"] == "r2"]


# Constrained sigmoid function: 0.5 + 0.5 * sigmoid
def sigmoid(x, k, x0):
    return 0.5 + 0.5 / (1 + np.exp(-k * (x - x0)))


# Fit sigmoid curve to all data combined
log_flops = np.log10(df["flops"])
y = df["eval_roc_auc"].values

# Fit sigmoid
popt, _ = curve_fit(sigmoid, log_flops, y, p0=[1, log_flops.mean()], maxfev=5000)

# Calculate R²
y_pred = sigmoid(log_flops, *popt)
r2 = r2_score(y, y_pred)


# Calculate crossing points for PlantCAD2-L and PlantCAD1-L
def solve_sigmoid_crossing(target_roc, k, x0):
    # Solve: target_roc = 0.5 + 0.5 / (1 + exp(-k * (x - x0)))
    # (target_roc - 0.5) * 2 = 1 / (1 + exp(-k * (x - x0)))
    # exp(-k * (x - x0)) = 1/((target_roc - 0.5) * 2) - 1
    return x0 - np.log(1 / ((target_roc - 0.5) * 2) - 1) / k


pc2l_log_flops = solve_sigmoid_crossing(0.67332978, *popt)
pc2l_flops = 10**pc2l_log_flops
pc1l_log_flops = solve_sigmoid_crossing(0.6898067999999999, *popt)
pc1l_flops = 10**pc1l_log_flops
target_flops = pc1l_flops

# Plot data points with distinction between r1 and r2
# r1 points: circles with gray border
plt.scatter(
    r1_data["flops"],
    r1_data["eval_roc_auc"],
    alpha=0.8,
    color="#1f77b4",
    s=50,
    marker="o",
    edgecolors="gray",
    linewidths=1.5,
    label="run 1",
)
# r2 points: circles with black border
plt.scatter(
    r2_data["flops"],
    r2_data["eval_roc_auc"],
    alpha=0.8,
    color="#1f77b4",
    s=50,
    marker="o",
    edgecolors="black",
    linewidths=1.5,
    label="run 2",
)

# Add baseline reference lines
baselines = {
    "PlantCAD1-S": 0.60242958,
    "PlantCAD2-S": 0.6251660400000001,
    "PlantCAD2-L": 0.67332978,
    "PlantCAD1-L": 0.6898067999999999,
}

for name, value in baselines.items():
    plt.axhline(y=value, color="black", linestyle="-", alpha=0.7, linewidth=1)

# Set x-axis limits with margins
min_flops = df["flops"].min() * 0.7  # margin below minimum
max_flops = target_flops * 1.3 if target_flops else df["flops"].max() * 1.3  # margin above target

# Plot sigmoid projection to full range
full_min_flops = df["flops"].min() * 0.7
full_max_flops = target_flops * 1.3 if target_flops else df["flops"].max() * 1.3
full_log_range = np.linspace(np.log10(full_min_flops), np.log10(full_max_flops), 300)
full_flops_range = 10**full_log_range

y_full = sigmoid(full_log_range, *popt)
plt.plot(full_flops_range, y_full, color="#1f77b4", alpha=0.8, linestyle="-", linewidth=2)

# Formatting
plt.xscale("log")
plt.xlim(min_flops, max_flops)

# Create custom x-axis labels with both FLOPs and cost
ax = plt.gca()
# Set specific tick locations at 1e19 and 1e20
tick_locs = [1e19, 1e20]
tick_labels = []
for flops in tick_locs:
    cost_str = format_cost(flops)
    tick_labels.append(f"{flops:.0e}\n{cost_str}")

ax.set_xticks(tick_locs)
ax.set_xticklabels(tick_labels, fontsize=11)

# Set y-tick label size
ax.tick_params(axis="y", labelsize=11)

plt.xlabel("FLOPs\n(Cost)\n[H100 hrs @ 25% MFU]", fontsize=13)
plt.ylabel("ROC AUC", fontsize=13)
plt.title("PlantCAD + Llama 600M scaling\n[Sorghum conservation 512bp]", fontsize=14)
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)

# Add text labels for baselines after formatting
xlim = plt.xlim()
# Position labels slightly to the right of y-axis
label_x = xlim[0] * 1.05
# Add labels with vertical offsets to avoid overlap
plt.text(label_x, 0.60242958 + 0.001, "PlantCAD1-S (512bp)", fontsize=11, color="black", va="bottom")
plt.text(label_x, 0.6251660400000001 + 0.001, "PlantCAD2-S (8192bp)", fontsize=11, color="black", va="bottom")
plt.text(label_x, 0.67332978 + 0.001, "PlantCAD2-L (8192bp)", fontsize=11, color="black", va="bottom")
plt.text(label_x, 0.6898067999999999 + 0.001, "PlantCAD1-L (512bp)", fontsize=11, color="black", va="bottom")

# Add R² annotation at very upper left, just inside axes
ylim = plt.ylim()
plt.text(
    0.02,
    0.98,
    f"Sigmoid fit: R² = {r2:.3f}",
    fontsize=9,
    color="black",
    ha="left",
    va="top",
    transform=ax.transAxes,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8),
)

# Add crossing point annotations for PlantCAD2-L and PlantCAD1-L
# PlantCAD2-L crossing point
plt.plot(
    pc2l_flops,
    0.67332978,
    "o",
    color="black",
    markersize=6,
    markerfacecolor="white",
    markeredgewidth=2,
    markeredgecolor="black",
    zorder=5,
)
pc2l_cost = format_cost(pc2l_flops)
plt.annotate(
    f"{pc2l_flops:.1e} FLOPs\n{pc2l_cost}",
    xy=(pc2l_flops, 0.67332978),
    xytext=(pc2l_flops * 1.3, 0.66),
    fontsize=10,
    color="black",
    ha="center",
    arrowprops=dict(arrowstyle="->", color="black", alpha=0.7, lw=1),
)

# PlantCAD1-L crossing point
plt.plot(
    pc1l_flops,
    0.6898067999999999,
    "o",
    color="black",
    markersize=6,
    markerfacecolor="white",
    markeredgewidth=2,
    markeredgecolor="black",
    zorder=5,
)
pc1l_cost = format_cost(pc1l_flops)
plt.annotate(
    f"{pc1l_flops:.1e} FLOPs\n{pc1l_cost}",
    xy=(pc1l_flops, 0.6898067999999999),
    xytext=(pc1l_flops * 0.65, 0.7),
    fontsize=10,
    color="black",
    ha="center",
    arrowprops=dict(arrowstyle="->", color="black", alpha=0.7, lw=1),
)

plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)

# Save plot
plt.savefig("plantcad_scaling.png", dpi=300, bbox_inches="tight")
plt.savefig("plantcad_scaling.pdf", dpi=300, bbox_inches="tight")
print("Plot saved to plantcad_scaling.png and plantcad_scaling.pdf")

# ============================================================================
# Loss vs ROC AUC plots
# ============================================================================


def plot_loss_vs_roc(df_loss, r1_loss, r2_loss, x_col, xlabel, title_suffix, output_prefix):
    """Create a loss vs ROC AUC plot with correlation and polynomial fits."""

    fig, ax = plt.subplots(figsize=(12, 6))

    # Calculate correlations
    x_vals = df_loss[x_col].values
    y_vals = df_loss["eval_roc_auc"].values
    pearson_corr, pearson_p = pearsonr(x_vals, y_vals)
    spearman_corr, spearman_p = spearmanr(x_vals, y_vals)

    # Fit and plot linear regression line first (so it appears behind points)
    coeffs_linear = np.polyfit(x_vals, y_vals, 1)
    poly_linear = np.poly1d(coeffs_linear)
    x_fit = np.linspace(x_vals.min(), x_vals.max(), 100)
    y_fit_linear = poly_linear(x_fit)
    plt.plot(x_fit, y_fit_linear, color="#1f77b4", alpha=0.6, linewidth=2, linestyle="--")

    # Fit and plot second-order polynomial
    coeffs_quad = np.polyfit(x_vals, y_vals, 2)
    poly_quad = np.poly1d(coeffs_quad)
    y_fit_quad = poly_quad(x_fit)
    plt.plot(x_fit, y_fit_quad, color="#1f77b4", alpha=0.8, linewidth=2, linestyle="-")

    # Plot data points with distinction between r1 and r2 (on top of line)
    plt.scatter(
        r1_loss[x_col],
        r1_loss["eval_roc_auc"],
        alpha=0.8,
        color="#1f77b4",
        s=50,
        marker="o",
        edgecolors="gray",
        linewidths=1.5,
        label="run 1",
    )
    plt.scatter(
        r2_loss[x_col],
        r2_loss["eval_roc_auc"],
        alpha=0.8,
        color="#1f77b4",
        s=50,
        marker="o",
        edgecolors="black",
        linewidths=1.5,
        label="run 2",
    )

    # Formatting
    plt.xlabel("Loss", fontsize=12)
    plt.ylabel("ROC AUC", fontsize=12)
    plt.title(f"PlantCAD + Llama 600M: Loss vs ROC AUC{title_suffix}\n[Sorghum conservation 512bp]", fontsize=14)
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)

    # Add correlation annotations in upper right
    corr_text = f"Pearson: r = {pearson_corr:.3f} (p = {pearson_p:.2e})\nSpearman: ρ = {spearman_corr:.3f} (p = {spearman_p:.2e})"
    plt.text(
        0.98,
        0.98,
        corr_text,
        fontsize=9,
        color="black",
        ha="right",
        va="top",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.9),
    )

    # Add quadratic formula annotation near the middle-right with short arrow
    a, b, c = coeffs_quad
    quad_formula = f"$y = {a:.3f}x^2 + {b:.3f}x + {c:.3f}$"

    # Find a point on the quadratic line in the middle-right area
    x_arrow = x_vals.min() + (x_vals.max() - x_vals.min()) * 0.55
    y_arrow = poly_quad(x_arrow)

    # Position text box above and to the right with short arrow
    plt.annotate(
        quad_formula,
        xy=(x_arrow, y_arrow),
        xytext=(20, 25),
        textcoords="offset points",
        fontsize=10,
        color="black",
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9),
        arrowprops=dict(arrowstyle="->", color="gray", lw=1.2, alpha=0.7, shrinkA=0, shrinkB=3),
    )

    # Add inset plot: composite_step vs eval_loss (both on log2 scale)
    # Position in bottom left, below the fit lines
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    ax_inset = inset_axes(
        ax, width="42%", height="30%", loc="lower left", bbox_to_anchor=(0.10, 0.18, 1, 1), bbox_transform=ax.transAxes
    )

    # Filter data for composite_step >= 15000
    r1_loss_filtered = r1_loss[r1_loss["composite_step"] >= 15000]
    r2_loss_filtered = r2_loss[r2_loss["composite_step"] >= 15000]

    # Fit second-order polynomial for inset (no label)
    x_inset_all = np.log2(pd.concat([r1_loss_filtered, r2_loss_filtered])["composite_step"])
    y_inset_all = np.log2(pd.concat([r1_loss_filtered, r2_loss_filtered])["eval_loss"])
    coeffs_inset_quad = np.polyfit(x_inset_all, y_inset_all, 2)
    poly_inset_quad = np.poly1d(coeffs_inset_quad)
    x_inset_fit = np.linspace(x_inset_all.min(), x_inset_all.max(), 100)
    y_inset_fit = poly_inset_quad(x_inset_fit)
    ax_inset.plot(x_inset_fit, y_inset_fit, color="#1f77b4", alpha=0.8, linewidth=1.5, linestyle="-")

    # Plot inset data with log2 scales
    ax_inset.scatter(
        np.log2(r1_loss_filtered["composite_step"]),
        np.log2(r1_loss_filtered["eval_loss"]),
        alpha=0.8,
        color="#1f77b4",
        s=25,
        marker="o",
        edgecolors="gray",
        linewidths=1.0,
    )
    ax_inset.scatter(
        np.log2(r2_loss_filtered["composite_step"]),
        np.log2(r2_loss_filtered["eval_loss"]),
        alpha=0.8,
        color="#1f77b4",
        s=25,
        marker="o",
        edgecolors="black",
        linewidths=1.0,
    )

    ax_inset.set_xlabel("log₂(Step)", fontsize=10)
    ax_inset.set_ylabel("log₂(Loss)", fontsize=10)
    ax_inset.tick_params(axis="both", which="major", labelsize=9)
    ax_inset.grid(True, alpha=0.3)
    ax_inset.patch.set_facecolor("white")
    ax_inset.patch.set_alpha(0.95)

    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)

    # Save plot
    plt.savefig(f"{output_prefix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_prefix}.pdf", dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_prefix}.png and {output_prefix}.pdf")


# Filter data where eval_loss is not null
df_loss = df[df["eval_loss"].notna()].copy()

# Add log2 transformed loss column
df_loss["log2_loss"] = np.log2(df_loss["eval_loss"])

# Separate by run
r1_loss = df_loss[df_loss["run_number"] == "r1"]
r2_loss = df_loss[df_loss["run_number"] == "r2"]

# Create plot 1: Linear scale
plot_loss_vs_roc(df_loss, r1_loss, r2_loss, "eval_loss", "Eval Loss", "", "plantcad_loss_vs_roc")

# Create plot 2: Log2 scale
plot_loss_vs_roc(df_loss, r1_loss, r2_loss, "log2_loss", "log₂(Eval Loss)", " (log₂ scale)", "plantcad_loss_vs_roc_log2")

# ============================================================================
# Loss vs LR plots
# ============================================================================


def plot_loss_vs_lr(loss_file, lr_file, base_lr=3e-4, output_prefix="plantcad_loss_vs_lr"):
    """Create a loss vs learning rate plot with cooldown period annotations.

    Args:
        loss_file: Path to CSV file containing loss data
        lr_file: Path to CSV file containing learning rate data
        base_lr: Base learning rate threshold for identifying cooldown periods
        output_prefix: Prefix for output files
    """
    from matplotlib.patches import Rectangle

    # Load loss data
    df_loss = pd.read_csv(loss_file)
    loss_steps = df_loss.iloc[:, 0].values  # Step column
    loss_values = df_loss.iloc[:, 1].values  # Loss column

    # Load LR data and identify cooldown intervals
    df_lr = pd.read_csv(lr_file)
    lr_steps = df_lr.iloc[:, 0].values  # Step column
    lr_values = df_lr.iloc[:, 1].values  # LR column

    # Identify continuous LR periods (cooldown vs stable)
    is_cooldown = lr_values < base_lr
    periods = []

    if len(lr_steps) > 0:
        current_state = is_cooldown[0]
        start_step = lr_steps[0]

        for i in range(1, len(lr_steps)):
            if is_cooldown[i] != current_state:
                periods.append({"start": start_step, "end": lr_steps[i - 1], "is_cooldown": current_state})
                current_state = is_cooldown[i]
                start_step = lr_steps[i]

        # Add the last period
        periods.append({"start": start_step, "end": lr_steps[-1], "is_cooldown": current_state})

    # Separate loss data by period type
    stable_steps = []
    stable_losses = []
    cooldown_steps = []
    cooldown_losses = []

    for i, step in enumerate(loss_steps):
        # Find which period this step belongs to
        for period in periods:
            if period["start"] <= step <= period["end"]:
                if period["is_cooldown"]:
                    cooldown_steps.append(step)
                    cooldown_losses.append(loss_values[i])
                else:
                    stable_steps.append(step)
                    stable_losses.append(loss_values[i])
                break

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 7))

    # Plotly default palette colors
    plotly_blue = "#636EFA"
    plotly_green = "#00CC96"

    # Plot rectangles for each period (these will be behind the line plot)
    for period in periods:
        color = plotly_blue if period["is_cooldown"] else plotly_green
        alpha = 1.0  # No transparency
        ax.axvspan(
            period["start"],
            period["end"],
            alpha=alpha,
            color=color,
            ymin=0,
            ymax=0.075,  # Cover bottom 7.5% of plot (50% taller)
            zorder=1,
        )

    # Fit and plot quadratic polynomials for stable and cooldown periods
    stable_poly = None
    cooldown_poly = None

    if len(stable_steps) >= 3:
        stable_coeffs = np.polyfit(stable_steps, stable_losses, 2)
        stable_poly = np.poly1d(stable_coeffs)
        stable_fit_x = np.linspace(min(stable_steps), max(stable_steps), 100)
        stable_fit_y = stable_poly(stable_fit_x)
        ax.plot(
            stable_fit_x,
            stable_fit_y,
            color=plotly_green,
            alpha=0.7,
            linewidth=2,
            linestyle="--",
            label="Stable fit",
            zorder=2,
        )

    if len(cooldown_steps) >= 3:
        cooldown_coeffs = np.polyfit(cooldown_steps, cooldown_losses, 2)
        cooldown_poly = np.poly1d(cooldown_coeffs)
        cooldown_fit_x = np.linspace(min(cooldown_steps), max(cooldown_steps), 100)
        cooldown_fit_y = cooldown_poly(cooldown_fit_x)
        ax.plot(
            cooldown_fit_x,
            cooldown_fit_y,
            color=plotly_blue,
            alpha=0.7,
            linewidth=2,
            linestyle="--",
            label="Cooldown fit",
            zorder=2,
        )

    # Plot loss line with dots on top (in black)
    ax.plot(loss_steps, loss_values, "o-", color="black", markersize=4, linewidth=1.5, label="Loss", zorder=3)

    # Add horizontal reference lines if both fits exist
    if stable_poly is not None and cooldown_poly is not None and len(cooldown_steps) > 0:
        # Get leftmost cooldown step and its y-value from the fit
        cooldown_start_step = min(cooldown_steps)
        cooldown_start_loss = cooldown_poly(cooldown_start_step)

        # Find where this y-value crosses the stable fit
        # Solve stable_poly(x) = cooldown_start_loss
        # a*x^2 + b*x + c - loss = 0
        stable_coeffs_arr = stable_poly.coefficients
        roots = np.roots([stable_coeffs_arr[0], stable_coeffs_arr[1], stable_coeffs_arr[2] - cooldown_start_loss])

        # Find the root that's within the stable range
        stable_step_range = (min(stable_steps), max(stable_steps))
        crossing_step = None
        for root in roots:
            if np.isreal(root) and stable_step_range[0] <= root.real <= stable_step_range[1]:
                crossing_step = root.real
                break

        if crossing_step is not None:
            # Draw horizontal reference line as a simple line segment
            ax.plot(
                [crossing_step, cooldown_start_step],
                [cooldown_start_loss, cooldown_start_loss],
                color="gray",
                linestyle=":",
                linewidth=1.5,
                alpha=0.7,
                zorder=2,
            )

            # Add markers at endpoints
            ax.plot(crossing_step, cooldown_start_loss, "o", color="gray", markersize=5, alpha=0.7, zorder=2)
            ax.plot(cooldown_start_step, cooldown_start_loss, "o", color="gray", markersize=5, alpha=0.7, zorder=2)

            # Calculate step difference, epoch count, and token count
            step_diff = abs(cooldown_start_step - crossing_step)
            epoch_diff = abs(step_diff / 2678)
            token_diff_b = abs(epoch_diff * 2.808)  # Tokens in billions

            # Add annotation - position it to the right and up with an arrow pointing at the line
            mid_x = (crossing_step + cooldown_start_step) / 2
            annotation_text = f"{step_diff:,.0f} steps  |  {epoch_diff:.1f} epochs  |  {token_diff_b:.1f}B tokens"
            ax.annotate(
                annotation_text,
                xy=(mid_x, cooldown_start_loss),
                xytext=(75, 50),
                textcoords="offset points",
                fontsize=11,
                color="black",
                ha="left",
                va="center",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.2, alpha=0.7),
            )

    # Add second horizontal reference line from rightmost stable point to cooldown fit crossing
    if stable_poly is not None and cooldown_poly is not None and len(stable_steps) > 0:
        # Get rightmost stable step and its y-value from the fit
        stable_end_step = max(stable_steps)
        stable_end_loss = stable_poly(stable_end_step)

        # Find where this y-value crosses the cooldown fit
        cooldown_coeffs_arr = cooldown_poly.coefficients
        roots = np.roots([cooldown_coeffs_arr[0], cooldown_coeffs_arr[1], cooldown_coeffs_arr[2] - stable_end_loss])

        # Find the root that's within the cooldown range
        cooldown_step_range = (min(cooldown_steps), max(cooldown_steps))
        crossing_step_2 = None
        for root in roots:
            if np.isreal(root) and cooldown_step_range[0] <= root.real <= cooldown_step_range[1]:
                crossing_step_2 = root.real
                break

        if crossing_step_2 is not None:
            # Draw horizontal reference line
            ax.plot(
                [crossing_step_2, stable_end_step],
                [stable_end_loss, stable_end_loss],
                color="gray",
                linestyle=":",
                linewidth=1.5,
                alpha=0.7,
                zorder=2,
            )

            # Add markers at endpoints
            ax.plot(crossing_step_2, stable_end_loss, "o", color="gray", markersize=5, alpha=0.7, zorder=2)
            ax.plot(stable_end_step, stable_end_loss, "o", color="gray", markersize=5, alpha=0.7, zorder=2)

            # Calculate step difference, epoch count, and token count
            step_diff_2 = abs(stable_end_step - crossing_step_2)
            epoch_diff_2 = abs(step_diff_2 / 2678)
            token_diff_b_2 = abs(epoch_diff_2 * 2.808)

            # Add annotation - position it far to the left with arrow pointing towards left side of line
            # Point arrow towards a position closer to the left endpoint (8% along the line)
            arrow_target_x = crossing_step_2 + (stable_end_step - crossing_step_2) * 0.08
            annotation_text_2 = (
                f"{step_diff_2:,.0f} steps  |  {epoch_diff_2:.1f} epochs  |  {token_diff_b_2:.1f}B tokens"
            )
            ax.annotate(
                annotation_text_2,
                xy=(arrow_target_x, stable_end_loss),
                xytext=(-350, -50),
                textcoords="offset points",
                fontsize=11,
                color="black",
                ha="left",
                va="center",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9),
                arrowprops=dict(arrowstyle="->", color="gray", lw=1.2, alpha=0.7),
            )

    # Formatting
    ax.set_xlabel("Step", fontsize=13)
    ax.set_ylabel("Loss", fontsize=13)
    ax.set_title("PlantCAD + Llama 600M: Loss vs Learning Rate\n[Sorghum conservation 512bp]", fontsize=14)
    ax.grid(True, alpha=0.3, zorder=0)
    ax.tick_params(axis="both", labelsize=11)

    # Create custom legend with both line and period markers
    legend_elements = [
        plt.Line2D([0], [0], color="black", marker="o", markersize=6, linewidth=1.5, label="Loss"),
        Rectangle((0, 0), 1, 1, fc=plotly_green, alpha=1.0, label=f"Stable (LR = {base_lr:.0e})"),
        Rectangle((0, 0), 1, 1, fc=plotly_blue, alpha=1.0, label=f"Cooldown (LR < {base_lr:.0e})"),
    ]

    # Add fit lines to legend if they exist
    if len(stable_steps) >= 3:
        legend_elements.append(
            plt.Line2D([0], [0], color=plotly_green, linewidth=2, linestyle="--", alpha=0.7, label="Stable fit")
        )
    if len(cooldown_steps) >= 3:
        legend_elements.append(
            plt.Line2D([0], [0], color=plotly_blue, linewidth=2, linestyle="--", alpha=0.7, label="Cooldown fit")
        )

    ax.legend(handles=legend_elements, loc="upper right", fontsize=11)

    # Adjust layout
    plt.subplots_adjust(left=0.08, right=0.97, top=0.93, bottom=0.1)

    # Save plot
    plt.savefig(f"{output_prefix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_prefix}.pdf", dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_prefix}.png and {output_prefix}.pdf")


# Create Loss vs LR plot
loss_file = data_dir / "wandb_export_2025-10-09T13_56_08.654-04_00.csv"
lr_file = data_dir / "wandb_export_2025-10-09T13_57_42.893-04_00.csv"
plot_loss_vs_lr(loss_file, lr_file)
