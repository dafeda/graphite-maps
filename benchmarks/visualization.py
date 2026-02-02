"""Visualization utilities for benchmark results.

This module provides functions to create plots comparing ISIS vs standard
boosting across multiple dimensions.

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker


def plot_speedup_vs_dimension(
    df: pd.DataFrame,
    n_values: list[int] | None = None,
    ax: "plt.Axes | None" = None,
    figsize: tuple[float, float] = (10, 6),
) -> "plt.Figure":
    """Plot ISIS speedup factor as a function of problem dimension.

    Args:
        df: DataFrame from run_benchmark_suite
        n_values: Sample sizes to include (default: all)
        ax: Matplotlib axes to plot on (creates new figure if None)
        figsize: Figure size if creating new figure

    Returns:
        Matplotlib figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if n_values is None:
        n_values = sorted(df["n"].unique())

    # Compute speedup for each (n, p) configuration
    speedups = []
    for (n, p), group in df.groupby(["n", "p"]):
        isis_time = group[group["method"] == "isis"]["time_seconds"].mean()
        standard_time = group[group["method"] == "standard"]["time_seconds"].mean()
        speedup = standard_time / isis_time
        speedups.append({"n": n, "p": p, "speedup": speedup})

    speedup_df = pd.DataFrame(speedups)

    # Plot each n value as a separate line
    for n in n_values:
        data = speedup_df[speedup_df["n"] == n].sort_values("p")
        ax.plot(data["p"], data["speedup"], "o-", label=f"n={n}", markersize=8)

    ax.set_xlabel("Number of Features (p)", fontsize=12)
    ax.set_ylabel("Speedup (Standard Time / ISIS Time)", fontsize=12)
    ax.set_title("ISIS Speedup vs Problem Dimension", fontsize=14)
    ax.set_xscale("log")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="No speedup")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Format x-axis with readable numbers
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{x:,.0f}"))

    fig.tight_layout()
    return fig


def plot_metric_comparison(
    df: pd.DataFrame,
    metric: str = "test_mse",
    ax: "plt.Axes | None" = None,
    figsize: tuple[float, float] = (10, 6),
) -> "plt.Figure":
    """Plot metric comparison between ISIS and standard methods.

    Args:
        df: DataFrame from run_benchmark_suite
        metric: Which metric to compare
        ax: Matplotlib axes to plot on (creates new figure if None)
        figsize: Figure size if creating new figure

    Returns:
        Matplotlib figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Prepare data
    summary = (
        df.groupby(["n", "p", "method"])[metric].agg(["mean", "std"]).reset_index()
    )

    # Create grouped bar positions
    configs = summary.groupby(["n", "p"]).size().reset_index()[["n", "p"]]
    x = np.arange(len(configs))
    width = 0.35

    isis_means = []
    isis_stds = []
    standard_means = []
    standard_stds = []

    for _, row in configs.iterrows():
        n, p = row["n"], row["p"]
        isis_data = summary[
            (summary["n"] == n) & (summary["p"] == p) & (summary["method"] == "isis")
        ]
        std_data = summary[
            (summary["n"] == n)
            & (summary["p"] == p)
            & (summary["method"] == "standard")
        ]

        isis_means.append(isis_data["mean"].to_numpy()[0])
        isis_stds.append(isis_data["std"].to_numpy()[0])
        standard_means.append(std_data["mean"].to_numpy()[0])
        standard_stds.append(std_data["std"].to_numpy()[0])

    # Plot bars
    ax.bar(
        x - width / 2,
        standard_means,
        width,
        yerr=standard_stds,
        label="Standard",
        capsize=3,
        alpha=0.8,
    )
    ax.bar(
        x + width / 2,
        isis_means,
        width,
        yerr=isis_stds,
        label="ISIS",
        capsize=3,
        alpha=0.8,
    )

    # Labels
    labels = [f"n={row['n']}\np={row['p']:,}" for _, row in configs.iterrows()]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title(f"{metric.replace('_', ' ').title()} Comparison", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    return fig


def plot_benchmark_dashboard(
    df: pd.DataFrame,
    figsize: tuple[float, float] = (16, 12),
) -> "plt.Figure":
    """Create a comprehensive dashboard of benchmark results.

    Creates a 3x2 grid with:
    - Speedup vs dimension
    - Test MSE comparison
    - Recall comparison
    - Precision comparison
    - F1 score comparison
    - Time comparison

    Args:
        df: DataFrame from run_benchmark_suite
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(3, 2, figsize=figsize)

    # Speedup plot
    plot_speedup_vs_dimension(df, ax=axes[0, 0])

    # Test MSE
    plot_metric_comparison(df, metric="test_mse", ax=axes[0, 1])

    # Recall
    plot_metric_comparison(df, metric="recall", ax=axes[1, 0])

    # Precision
    plot_metric_comparison(df, metric="precision", ax=axes[1, 1])

    # F1 Score
    plot_metric_comparison(df, metric="f1_score", ax=axes[2, 0])

    # Time
    plot_metric_comparison(df, metric="time_seconds", ax=axes[2, 1])

    fig.suptitle("ISIS vs Standard Boosting - Benchmark Dashboard", fontsize=16, y=1.00)
    fig.tight_layout()
    return fig


def plot_scaling_analysis(
    df: pd.DataFrame,
    n_fixed: int | None = None,
    figsize: tuple[float, float] = (12, 5),
) -> "plt.Figure":
    """Plot how time and MSE scale with problem dimension.

    Args:
        df: DataFrame from run_benchmark_suite
        n_fixed: Fix sample size (default: use median n)
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    if n_fixed is None:
        # Use the closest n value to the median
        n_values = sorted(df["n"].unique())
        median_n = np.median(n_values)
        n_fixed = min(n_values, key=lambda x: abs(x - median_n))

    data = df[df["n"] == n_fixed]

    if len(data) == 0:
        raise ValueError(
            f"No data found for n={n_fixed}. Available n values: {sorted(df['n'].unique())}"
        )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Time scaling
    for method in ["standard", "isis"]:
        method_data = data[data["method"] == method]
        summary = method_data.groupby("p")["time_seconds"].agg(["mean", "std"])
        ax1.errorbar(
            summary.index,
            summary["mean"],
            yerr=summary["std"],
            label=method.capitalize(),
            marker="o",
            capsize=3,
        )

    ax1.set_xlabel("Number of Features (p)", fontsize=12)
    ax1.set_ylabel("Time (seconds)", fontsize=12)
    ax1.set_title(f"Time Scaling (n={n_fixed})", fontsize=14)
    ax1.set_xscale("log")
    # Only use log scale if all values are positive
    if data["time_seconds"].min() > 0:
        ax1.set_yscale("log")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # MSE scaling
    for method in ["standard", "isis"]:
        method_data = data[data["method"] == method]
        summary = method_data.groupby("p")["test_mse"].agg(["mean", "std"])
        ax2.errorbar(
            summary.index,
            summary["mean"],
            yerr=summary["std"],
            label=method.capitalize(),
            marker="o",
            capsize=3,
        )

    ax2.set_xlabel("Number of Features (p)", fontsize=12)
    ax2.set_ylabel("Test MSE", fontsize=12)
    ax2.set_title(f"Test MSE Scaling (n={n_fixed})", fontsize=14)
    ax2.set_xscale("log")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def save_all_plots(
    df: pd.DataFrame,
    output_dir: str = "benchmark_plots",
    dpi: int = 150,
) -> list[str]:
    """Generate and save all benchmark plots as PNG files.

    Args:
        df: DataFrame from run_benchmark_suite
        output_dir: Directory to save plots
        dpi: Resolution for PNG files

    Returns:
        List of saved file paths
    """
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_files = []

    # Generate each plot
    plots = [
        ("speedup", plot_speedup_vs_dimension(df)),
        ("test_mse", plot_metric_comparison(df, "test_mse")),
        ("recall", plot_metric_comparison(df, "recall")),
        ("precision", plot_metric_comparison(df, "precision")),
        ("dashboard", plot_benchmark_dashboard(df)),
        ("scaling", plot_scaling_analysis(df)),
    ]

    for name, fig in plots:
        filepath = output_path / f"{name}.png"
        fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved_files.append(str(filepath))

    return saved_files
