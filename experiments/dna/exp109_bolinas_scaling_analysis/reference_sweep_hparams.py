# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Visualize reference sweep hparams vs eval/loss with SHAP feature importance.

Usage:
    uv run --with lightgbm --with shap \
        experiments/dna/exp109_bolinas_scaling_analysis/reference_sweep_hparams.py [--refresh]

Caches wandb data locally; pass --refresh to re-fetch.
"""

import json
import sys
from pathlib import Path

import numpy as np

WANDB_PROJECT = "eric-czech/marin"
WANDB_GROUP = "dna-bolinas-reference-sweep-v0.6"
CACHE_PATH = Path("/tmp/sweep_v0.6_finished.json")
OUTPUT_PATH = Path("experiments/dna/exp109_bolinas_scaling_analysis/results/reference_sweep_v0.6_hparams.png")

HPARAM_KEYS = ["initializer_range", "lr", "adam_lr", "beta1", "beta2", "eps", "mgn", "zloss"]
HPARAM_LABELS = ["init_range", "lr", "adam_lr", "beta1", "beta2", "epsilon", "max_grad_norm", "z_loss"]
LOG_SCALE = {"initializer_range", "lr", "adam_lr", "eps", "zloss"}
CLIP_Y = 1.29


def fetch_data(project: str, group: str) -> list[dict]:
    import wandb

    api = wandb.Api()
    runs = api.runs(project, filters={"group": group})
    data = []
    for r in runs:
        if r.state != "finished":
            continue
        hparams = {}
        for tag in r.tags:
            if "=" in tag:
                k, v = tag.split("=", 1)
                try:
                    hparams[k] = float(v)
                except ValueError:
                    pass
        loop = next((int(t.split("=")[1]) for t in r.tags if t.startswith("loop=")), None)
        data.append(
            {
                "eval/loss": r.summary.get("eval/loss"),
                "eval/macro_loss": r.summary.get("eval/macro_loss"),
                "name": r.name,
                "completed_at": r.heartbeatAt,
                "loop": loop,
                **hparams,
            }
        )
    return data


def load_data(refresh: bool = False) -> list[dict]:
    if not refresh and CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            data = json.load(f)
        print(f"Loaded {len(data)} runs from cache ({CACHE_PATH})")
        return data
    data = fetch_data(WANDB_PROJECT, WANDB_GROUP)
    with open(CACHE_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Fetched {len(data)} finished runs, cached to {CACHE_PATH}")
    return data


def plot(data: list[dict], metric: str = "eval/loss") -> None:
    import math
    import matplotlib.pyplot as plt
    import lightgbm as lgb
    import shap
    from datetime import datetime
    from matplotlib.gridspec import GridSpec

    n_total = len(data)
    data = [d for d in data if isinstance(d.get(metric), float) and math.isfinite(d[metric])]
    n_feasible = len(data)

    X = np.array([[d[k] for k in HPARAM_KEYS] for d in data])
    y = np.array([d[metric] for d in data])
    ylim = (y.min() - 0.002, CLIP_Y + 0.004)

    model = lgb.LGBMRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.1, min_child_samples=2, random_state=42, verbose=-1
    ).fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    importance = np.abs(shap_values).mean(axis=0)
    importance = importance / importance.sum() * 100

    order = np.argsort(importance)
    sorted_labels = [HPARAM_LABELS[i] for i in order]
    sorted_importance = importance[order]
    sorted_colors = plt.cm.tab10(order)

    fig = plt.figure(figsize=(10, 10))
    gs_top = GridSpec(1, 4, figure=fig, top=0.95, bottom=0.75, wspace=0.08)
    gs_bot = GridSpec(3, 4, figure=fig, height_ratios=[1.5, 2, 2], top=0.67, bottom=0.05, hspace=0.45, wspace=0.08)

    # --- Timeline: eval/loss vs completion time, colored by loop ---
    ax_time = fig.add_subplot(gs_top[0, :])
    times = [datetime.fromisoformat(d["completed_at"].replace("Z", "+00:00")) for d in data]
    loops = np.array([d.get("loop", 0) or 0 for d in data])
    max_loop = int(loops.max()) if len(loops) else 0
    cmap = plt.cm.viridis

    for _d, t, yv, loop in zip(data, times, y, loops, strict=True):
        clipped = yv > CLIP_Y
        plot_y = CLIP_Y if clipped else yv
        c = cmap(loop / max(max_loop, 1))
        marker = "^" if clipped else "o"
        ax_time.scatter(t, plot_y, c=[c], s=25, edgecolors="k", linewidths=0.3, marker=marker, alpha=0.7, zorder=3)
        if clipped:
            ax_time.annotate(
                f"{yv:.2f}", (t, CLIP_Y), textcoords="offset points", xytext=(0, -8), ha="center", fontsize=5
            )

    # Lower envelope (cumulative min over time)
    sorted_idx = sorted(range(len(times)), key=lambda i: times[i])
    env_times, env_losses = [], []
    cur_min = float("inf")
    for i in sorted_idx:
        if y[i] < cur_min:
            cur_min = y[i]
            env_times.append(times[i])
            env_losses.append(cur_min)
    if env_times and times[sorted_idx[-1]] > env_times[-1]:
        env_times.append(times[sorted_idx[-1]])
        env_losses.append(cur_min)
    ax_time.step(env_times, env_losses, where="post", color="red", linewidth=1.5, alpha=0.8, zorder=4)

    ax_time.set_ylim(ylim)
    ax_time.set_ylabel(metric, fontsize=9)
    ax_time.set_xlabel("Completion time", fontsize=9)
    ax_time.tick_params(labelsize=7)
    for label in ax_time.get_xticklabels():
        label.set_rotation(20)
        label.set_ha("right")

    # Colorbar for loop
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max_loop))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_time, pad=0.02, aspect=20, fraction=0.03)
    cbar.set_label("Loop", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # --- SHAP importance bar chart ---
    ax_imp = fig.add_subplot(gs_bot[0, :])
    bars = ax_imp.barh(sorted_labels, sorted_importance, color=sorted_colors, edgecolor="k", linewidth=0.5)
    ax_imp.set_xlabel("SHAP importance (% mean |SHAP|)", fontsize=9)
    ax_imp.tick_params(labelsize=8)
    for bar, val in zip(bars, sorted_importance, strict=True):
        ax_imp.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", fontsize=7)

    # --- Hparam scatter plots ---
    colors = plt.cm.tab10(np.arange(len(HPARAM_LABELS)))
    for i, (hparam, label) in enumerate(zip(HPARAM_KEYS, HPARAM_LABELS, strict=True)):
        row = 1 + i // 4
        col = i % 4
        ax = fig.add_subplot(gs_bot[row, col])
        xs = [d[hparam] for d in data]
        normal = [(x, yv) for x, yv in zip(xs, y, strict=True) if yv <= CLIP_Y]
        clipped = [(x, yv) for x, yv in zip(xs, y, strict=True) if yv > CLIP_Y]
        if normal:
            ax.scatter(*zip(*normal, strict=True), alpha=0.7, s=25, edgecolors="k", linewidths=0.3, color=colors[i])
        if clipped:
            cx, _cy = zip(*clipped, strict=True)
            ax.scatter(
                cx, [CLIP_Y] * len(cx), alpha=0.7, s=25, edgecolors="k", linewidths=0.3, color=colors[i], marker="^"
            )
            for xv, yv in clipped:
                ax.annotate(
                    f"{yv:.2f}", (xv, CLIP_Y), textcoords="offset points", xytext=(0, -8), ha="center", fontsize=5
                )
        ax.set_xlabel(label, fontsize=8)
        ax.set_ylim(ylim)
        ax.tick_params(labelsize=7)
        if hparam in LOG_SCALE:
            ax.set_xscale("log")
        if col == 0:
            ax.set_ylabel(metric, fontsize=9)
        else:
            ax.set_yticklabels([])

    fig.suptitle(
        f"Bolinas DNA reference sweep v0.6\n" f"{n_feasible} feasible / {n_total} finished runs — metric: {metric}",
        fontsize=11,
        y=1.01,
    )
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    refresh = "--refresh" in sys.argv
    data = load_data(refresh=refresh)
    plot(data)
