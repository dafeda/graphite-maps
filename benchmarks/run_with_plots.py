"""Run benchmarks and generate all visualization plots."""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from benchmarks import run_benchmark_suite, save_all_plots

# Run benchmarks (or load existing results)
results = run_benchmark_suite(
    n_list=[100, 200], p_list=[10_000, 100_000], n_signal=5, n_repeats=10
)

# Generate all plots
saved_files = save_all_plots(results, output_dir="benchmark_plots")

print("\nPlots saved to:")
for filepath in saved_files:
    print(f"  {filepath}")
