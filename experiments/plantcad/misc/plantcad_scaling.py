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
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit

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

# Cost per FLOP
COST_PER_FLOP = 3.458957351e-18  # $/FLOP on CoreWeave


def format_cost(flops):
    """Format cost in dollars with appropriate units"""
    cost = flops * COST_PER_FLOP
    if cost >= 1e6:
        return f"(${cost/1e6:.1f}M)"
    elif cost >= 1e3:
        return f"(${cost/1e3:.1f}K)"
    else:
        return f"(${cost:.0f})"


METRICS = """step,roc_auc,flops,iteration
1673,0.535217,3.4043190387702497e+18,v1
3346,0.546725,6.804570790755828e+18,v1
5019,0.549917,1.0208889829526075e+19,v1
6692,0.558042,1.3613208868296327e+19,v1
8365,0.56029,1.701142697688957e+19,v1
10038,0.565785,2.041574601565982e+19,v1
11711,0.570048,2.3818031411037733e+19,v1
13384,0.576358,2.722438409320032e+19,v1
15057,0.583593,3.0624635845185896e+19,v1
16730,0.585834,3.402488759717147e+19,v1
18403,0.589215,3.742717299254939e+19,v1
20076,0.588738,4.08294583879273e+19,v1
21749,0.593178,4.423174378330522e+19,v1
2679,0.549341,1.1019524974187643e+19,v2
5356,0.566597,2.2026714659225076e+19,v2
8038,0.604521,3.3054463159512859e+19,v2
10713,0.626729,4.4053429318450151e+19,v2
13392,0.631095,5.5068842529587724e+19,v2
16068,0.607316,6.6071920451575087e+19,v2
18745,0.622988,7.7079110136612520e+19,v2
21423,0.639093,8.8090411584700023e+19,v2
24102,0.650973,9.9105824795837596e+19,v2
26781,0.657882,1.1012123800697515e+20,v2
"""

# Parse data
from io import StringIO

df = pd.read_csv(StringIO(METRICS))

# Create visualization
plt.figure(figsize=(10, 6))

# Separate data by iteration
v1_data = df[df["iteration"] == "v1"]
v2_data = df[df["iteration"] == "v2"]


# Constrained sigmoid function: 0.5 + 0.5 * sigmoid
def sigmoid(x, k, x0):
    return 0.5 + 0.5 / (1 + np.exp(-k * (x - x0)))


# Fit sigmoid curves
v2_popt = None
target_flops = None
sigmoid_params = {}
r2_values = {}

for data, color, alpha, label in [(v1_data, "#A0A0A0", 0.5, "v1 fit"), (v2_data, "#1f77b4", 0.8, "v2 fit")]:
    if len(data) > 1:
        log_flops = np.log10(data["flops"])
        y = data["roc_auc"].values

        # Fit sigmoid
        popt, _ = curve_fit(sigmoid, log_flops, y, p0=[1, log_flops.mean()], maxfev=5000)
        sigmoid_params[color] = popt

        # Calculate R²
        y_pred = sigmoid(log_flops, *popt)
        r2 = r2_score(y, y_pred)
        r2_values[color] = r2

        # Store v2 parameters for projection
        if color == "#1f77b4":
            v2_popt = popt

            # Calculate crossing points for PlantCAD1 (0.695) and PlantCAD2 (0.72)
            def solve_sigmoid_crossing(target_roc, k, x0):
                # Solve: target_roc = 0.5 + 0.5 / (1 + exp(-k * (x - x0)))
                # (target_roc - 0.5) * 2 = 1 / (1 + exp(-k * (x - x0)))
                # exp(-k * (x - x0)) = 1/((target_roc - 0.5) * 2) - 1
                return x0 - np.log(1 / ((target_roc - 0.5) * 2) - 1) / k

            pc1_log_flops = solve_sigmoid_crossing(0.695, *popt)
            pc1_flops = 10**pc1_log_flops
            pc2_log_flops = solve_sigmoid_crossing(0.72, *popt)
            pc2_flops = 10**pc2_log_flops
            target_flops = pc2_flops

# Plot data points with detailed labels
plt.scatter(
    v1_data["flops"],
    v1_data["roc_auc"],
    alpha=0.6,
    color="#A0A0A0",
    s=30,
    marker="s",
    label=f'v1 (300M, R²={r2_values.get("#A0A0A0", 0):.3f})',
)
plt.scatter(
    v2_data["flops"],
    v2_data["roc_auc"],
    alpha=0.8,
    color="#1f77b4",
    s=50,
    label=f'v2 (600M, R²={r2_values.get("#1f77b4", 0):.3f})',
)

# Add threshold lines (without labels for legend) - black
plt.axhline(y=0.695, color="black", linestyle="-", alpha=0.7, linewidth=1)
plt.axhline(y=0.72, color="black", linestyle="-", alpha=0.7, linewidth=1)

# Set x-axis limits with margins
min_flops = df["flops"].min() * 0.7  # margin below minimum
max_flops = target_flops * 1.3 if target_flops else df["flops"].max() * 1.3  # margin above target

# Plot extended sigmoid projections to full range (no margins)
full_min_flops = df["flops"].min() * 0.7
full_max_flops = target_flops * 1.3 if target_flops else df["flops"].max() * 1.3
full_log_range = np.linspace(np.log10(full_min_flops), np.log10(full_max_flops), 300)
full_flops_range = 10**full_log_range

for color, popt in sigmoid_params.items():
    y_full = sigmoid(full_log_range, *popt)
    alpha_proj = 0.6 if color == "#A0A0A0" else 0.8
    plt.plot(full_flops_range, y_full, color=color, alpha=alpha_proj, linestyle="-", linewidth=2)

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
ax.set_xticklabels(tick_labels, fontsize=9)

plt.xlabel("FLOPs\n(Cost)\n[H100 hrs @ 25% MFU]")
plt.ylabel("ROC AUC")
plt.title("PlantCAD + Llama 600M scaling\n[Sorghum conservation ROC]")
plt.legend()
plt.grid(True, alpha=0.3)

# Add text labels for thresholds after formatting
xlim = plt.xlim()
plt.text(xlim[0] * 1.5, 0.695, "PlantCAD1 (0.695)", fontsize=10, color="black", va="bottom")
plt.text(xlim[0] * 1.5, 0.72, "PlantCAD2 (0.72)", fontsize=10, color="black", va="bottom")

# Add elegant crossing point annotations with black text and arrows
if "pc1_flops" in locals() and "pc2_flops" in locals():
    # PlantCAD1 crossing point
    plt.plot(
        pc1_flops,
        0.695,
        "o",
        color="black",
        markersize=6,
        markerfacecolor="white",
        markeredgewidth=2,
        markeredgecolor="black",
        zorder=5,
    )
    pc1_cost = format_cost(pc1_flops)
    plt.annotate(
        f"{pc1_flops:.1e} FLOPs\n{pc1_cost}",
        xy=(pc1_flops, 0.695),
        xytext=(pc1_flops * 0.4, 0.705),
        fontsize=9,
        color="black",
        ha="center",
        arrowprops=dict(arrowstyle="->", color="black", alpha=0.7, lw=1),
    )

    # PlantCAD2 crossing point
    plt.plot(
        pc2_flops,
        0.72,
        "o",
        color="black",
        markersize=6,
        markerfacecolor="white",
        markeredgewidth=2,
        markeredgecolor="black",
        zorder=5,
    )
    pc2_cost = format_cost(pc2_flops)
    plt.annotate(
        f"{pc2_flops:.1e} FLOPs\n{pc2_cost}",
        xy=(pc2_flops, 0.72),
        xytext=(pc2_flops * 0.4, 0.73),
        fontsize=9,
        color="black",
        ha="center",
        arrowprops=dict(arrowstyle="->", color="black", alpha=0.7, lw=1),
    )

plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)

# Save plot
plt.savefig("results/plantcad_scaling.png", dpi=300, bbox_inches="tight")
plt.savefig("results/plantcad_scaling.pdf", dpi=300, bbox_inches="tight")
print("Plot saved to marin/results/plantcad_scaling.png and plantcad_scaling.pdf")
