"""Benchmarking utilities for comparing regression methods."""

from benchmarks.benchmark_regression import (
    BenchmarkResult,
    compare_methods,
    format_comparison_table,
    run_benchmark_suite,
    run_single_benchmark,
    summarize_benchmarks,
)
from benchmarks.visualization import (
    plot_benchmark_dashboard,
    plot_metric_comparison,
    plot_scaling_analysis,
    plot_speedup_vs_dimension,
    save_all_plots,
)

__all__ = [
    "BenchmarkResult",
    "compare_methods",
    "format_comparison_table",
    "plot_benchmark_dashboard",
    "plot_metric_comparison",
    "plot_scaling_analysis",
    "plot_speedup_vs_dimension",
    "run_benchmark_suite",
    "run_single_benchmark",
    "save_all_plots",
    "summarize_benchmarks",
]
