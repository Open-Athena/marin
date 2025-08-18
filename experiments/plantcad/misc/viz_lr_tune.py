#!/usr/bin/env python3
"""
Visualize W&B runs for PlantCAD learning rate tuning experiments.
"""

import os
import pandas as pd
import wandb
from plotnine import ggplot, aes, geom_line, facet_wrap, theme_minimal, labs, scale_color_discrete, scale_x_log10, scale_y_log10


def fetch_wandb_runs(project_name: str = None, run_pattern: str = "plantcad-lr-tune-lr.*-r5"):
    """Fetch W&B runs matching the specified pattern."""
    api = wandb.Api()
    
    # If no project specified, try to infer from current environment
    if project_name is None:
        # Use the full project path with username
        project_name = "eric-czech/marin"
    
    # Fetch runs matching the pattern
    runs = api.runs(project_name, filters={"display_name": {"$regex": run_pattern}})
    
    return list(runs)


def list_available_runs(project_name: str = None, limit: int = 20):
    """List available W&B runs to help debug run patterns."""
    api = wandb.Api()
    
    if project_name is None:
        project_name = "eric-czech/marin"
    
    print(f"Listing recent runs in project '{project_name}':")
    runs = api.runs(project_name, per_page=limit)
    
    for i, run in enumerate(runs):
        print(f"  {i+1:2d}. {run.name}")
        if i >= limit - 1:
            break
    
    return list(runs)


def extract_metrics_data(runs, metrics=["eval/loss", "throughput/mfu", "train/loss"]):
    """Extract step-by-step metrics data from W&B runs."""
    all_data = []
    
    for run in runs:
        run_name = run.name
        
        for metric in metrics:
            # Get the history for this metric
            history = run.scan_history(keys=[metric, "_step"], page_size=10000)
            
            for row in history:
                if metric in row and "_step" in row:
                    all_data.append({
                        "run_name": run_name,
                        "metric": metric,
                        "step": row["_step"],
                        "value": row[metric]
                    })
    
    return pd.DataFrame(all_data)


def create_visualizations(df):
    """Create plotnine visualizations for each metric."""
    
    # Create a plot with facets for each metric (original scale)
    plot_original = (
        ggplot(df, aes(x="step", y="value", color="run_name"))
        + geom_line(size=1.2, alpha=0.8)
        + facet_wrap("metric", scales="free_y", ncol=1)
        + theme_minimal()
        + labs(
            title="PlantCAD Learning Rate Tuning - Metrics by Step",
            x="Training Step",
            y="Metric Value",
            color="Run"
        )
        + scale_color_discrete(name="Learning Rate Run")
    )
    
    # Create a log-scale plot for loss metrics only, starting from step 50
    loss_df = df[df["metric"].str.contains("loss")]
    loss_df_filtered = loss_df[loss_df["step"] >= 50]
    plot_log = (
        ggplot(loss_df_filtered, aes(x="step", y="value", color="run_name"))
        + geom_line(size=1.2, alpha=0.8)
        + facet_wrap("metric", scales="free_y", ncol=1)
        + theme_minimal()
        + labs(
            title="PlantCAD Learning Rate Tuning - Loss Metrics (Log Scale, Steps >= 50)",
            x="Training Step (Log Scale)",
            y="Loss Value (Log Scale)",
            color="Run"
        )
        + scale_color_discrete(name="Learning Rate Run")
        + scale_x_log10()
        + scale_y_log10()
    )
    
    return plot_original, plot_log


def main():
    """Main function to fetch data and create visualizations."""
    print("Fetching W&B runs...")
    runs = fetch_wandb_runs()
    
    if not runs:
        print("No runs found matching the pattern 'plantcad-lr-tune-lr.*-r5'")
        print("\nLet's see what runs are available:")
        list_available_runs()
        return None, None
    
    print(f"Found {len(runs)} runs:")
    for run in runs:
        print(f"  - {run.name}")
    
    print("\nExtracting metrics data...")
    df = extract_metrics_data(runs)
    
    if df.empty:
        print("No metrics data found.")
        return None, None
    
    print(f"Extracted {len(df)} data points across {df['metric'].nunique()} metrics")
    print(f"Metrics: {list(df['metric'].unique())}")
    
    print("\nCreating visualizations...")
    plot_original, plot_log = create_visualizations(df)
    
    # Show data filtering info
    loss_data_count = len(df[df["metric"].str.contains("loss")])
    loss_data_filtered_count = len(df[(df["metric"].str.contains("loss")) & (df["step"] >= 50)])
    print(f"Loss data points: {loss_data_count} total, {loss_data_filtered_count} after filtering steps >= 50")
    
    # Display the plots
    print("\nOriginal scale plot:")
    print(plot_original)
    print("\nLog scale plot (steps >= 50):")
    print(plot_log)
    
    # Create images directory if it doesn't exist
    images_dir = "experiments/plantcad/images"
    os.makedirs(images_dir, exist_ok=True)
    
    # Save both plots in the images directory
    plot_original_path = os.path.join(images_dir, "plantcad_lr_tune_metrics_original.png")
    plot_log_path = os.path.join(images_dir, "plantcad_lr_tune_metrics_log.png")
    
    plot_original.save(plot_original_path, width=12, height=10, dpi=300)
    print(f"Original scale plot saved as '{plot_original_path}'")
    
    plot_log.save(plot_log_path, width=12, height=8, dpi=300)
    print(f"Log scale plot saved as '{plot_log_path}'")
    
    return df, (plot_original, plot_log)


if __name__ == "__main__":
    result = main()
    if result[0] is not None:
        df, plots = result
        print(f"Successfully created 2 visualizations with {len(df)} data points!")
    else:
        print("No visualization created - check W&B project and run patterns.")
