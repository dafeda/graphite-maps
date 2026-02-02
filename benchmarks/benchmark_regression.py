"""Benchmarking module for comparing ISIS vs standard boosting regression.

This module provides tools for rigorous comparison of regression methods
across multiple dimensions:
- Computational efficiency (timing)
- Predictive accuracy (test MSE)
- Coefficient recovery (beta MSE)
- Feature selection quality (precision, recall, F1)

Example usage:
    from benchmarks.benchmark_regression import run_benchmark_suite, summarize_benchmarks

    # Run benchmarks
    results = run_benchmark_suite(
        n_list=[100, 200],
        p_list=[10_000, 100_000],
        n_signal=5,
        n_repeats=10,
    )

    # Get summary statistics
    summary = summarize_benchmarks(results)
    print(summary)
"""

import time
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd
from graphite_maps.linear_regression import linear_boost_ic_regression
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run.

    Attributes:
        method: Either "standard" or "isis"
        n: Number of samples
        p: Number of features
        n_signal: Number of true signal features
        seed: Random seed used for reproducibility
        time_seconds: Wall-clock time for the method
        train_mse: Mean squared error on training data
        test_mse: Mean squared error on held-out test data
        true_positives: Number of correct signal features found
        false_positives: Number of noise features incorrectly selected
        n_nonzero: Total number of non-zero coefficients
        beta_mse: MSE between estimated and true coefficients
        precision: TP / (TP + FP), or 0 if no features selected
        recall: TP / n_signal (sensitivity)
        f1_score: Harmonic mean of precision and recall
    """

    method: Literal["standard", "isis"]
    n: int
    p: int
    n_signal: int
    seed: int
    time_seconds: float
    train_mse: float
    test_mse: float
    true_positives: int
    false_positives: int
    n_nonzero: int
    beta_mse: float
    precision: float
    recall: float
    f1_score: float


def generate_benchmark_data(
    n: int,
    p: int,
    n_signal: int,
    seed: int,
    noise_std: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic data for benchmarking.

    Args:
        n: Number of samples
        p: Number of features
        n_signal: Number of true signal features
        seed: Random seed for reproducibility
        noise_std: Standard deviation of observation noise

    Returns:
        X: Feature matrix (n, p)
        y: Response vector (n,)
        true_beta: True coefficient vector (p,)
        signal_indices: Indices of true signal features
        X_test: Test feature matrix
        y_test: Test response vector
    """
    rng = np.random.default_rng(seed)

    # Generate all features as noise
    X = rng.standard_normal((n, p))

    # Select random positions for signal features
    signal_indices = rng.choice(p, size=n_signal, replace=False)

    # Generate true coefficients (sparse)
    true_beta = np.zeros(p)
    true_beta[signal_indices] = rng.choice([-1, 1], size=n_signal) * rng.uniform(
        0.5, 2.0, size=n_signal
    )

    # Generate response
    y = X @ true_beta + rng.normal(0, noise_std, size=n)

    # Generate test data with offset seed for independence
    rng_test = np.random.default_rng(seed + 1_000_000)
    X_test = rng_test.standard_normal((n, p))
    y_test = X_test @ true_beta + rng_test.normal(0, noise_std, size=n)

    return X, y, true_beta, signal_indices, X_test, y_test


def run_single_benchmark(
    n: int,
    p: int,
    n_signal: int,
    seed: int,
    use_isis: bool,
    learning_rate: float = 0.5,
    noise_std: float = 1.0,
) -> BenchmarkResult:
    """Run a single benchmark for one method.

    Args:
        n: Number of samples
        p: Number of features
        n_signal: Number of true signal features
        seed: Random seed for reproducibility
        use_isis: Whether to use ISIS pre-screening
        learning_rate: Learning rate for boosting
        noise_std: Standard deviation of observation noise

    Returns:
        BenchmarkResult with all metrics
    """
    # Generate data
    X, y, true_beta, signal_indices, X_test, y_test = generate_benchmark_data(
        n, p, n_signal, seed, noise_std
    )

    # Standardize
    scaler_X = StandardScaler().fit(X)
    scaler_y = StandardScaler().fit(y.reshape(-1, 1))

    X_scaled = scaler_X.transform(X)
    y_scaled = scaler_y.transform(y.reshape(-1, 1)).flatten()
    X_test_scaled = scaler_X.transform(X_test)
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    # Time the method
    start = time.perf_counter()
    H = linear_boost_ic_regression(
        X_scaled,
        y_scaled.reshape(-1, 1),
        learning_rate=learning_rate,
        use_isis=use_isis,
        verbose_level=0,
    )
    elapsed = time.perf_counter() - start

    beta_est = H.toarray().flatten()

    # Compute metrics
    signal_set = set(signal_indices)
    nonzero_set = set(np.where(np.abs(beta_est) > 1e-10)[0])

    tp = len(signal_set & nonzero_set)
    fp = len(nonzero_set - signal_set)
    n_nonzero = len(nonzero_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / n_signal if n_signal > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return BenchmarkResult(
        method="isis" if use_isis else "standard",
        n=n,
        p=p,
        n_signal=n_signal,
        seed=seed,
        time_seconds=elapsed,
        train_mse=float(mean_squared_error(y_scaled, X_scaled @ beta_est)),
        test_mse=float(mean_squared_error(y_test_scaled, X_test_scaled @ beta_est)),
        true_positives=tp,
        false_positives=fp,
        n_nonzero=n_nonzero,
        beta_mse=float(mean_squared_error(true_beta, beta_est)),
        precision=precision,
        recall=recall,
        f1_score=f1,
    )


def run_benchmark_suite(
    n_list: list[int],
    p_list: list[int],
    n_signal: int = 5,
    n_repeats: int = 10,
    learning_rate: float = 0.5,
    noise_std: float = 1.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run full benchmark suite across problem sizes and methods.

    Args:
        n_list: List of sample sizes to test
        p_list: List of feature dimensions to test
        n_signal: Number of true signal features
        n_repeats: Number of random seeds to run per configuration
        learning_rate: Learning rate for boosting
        noise_std: Standard deviation of observation noise
        verbose: Whether to print progress

    Returns:
        DataFrame with one row per benchmark run
    """
    results = []
    total = len(n_list) * len(p_list) * n_repeats * 2
    count = 0

    for n in n_list:
        for p in p_list:
            for seed in range(n_repeats):
                for use_isis in [False, True]:
                    if verbose:
                        count += 1
                        method_name = "ISIS" if use_isis else "Standard"
                        print(
                            f"[{count}/{total}] n={n}, p={p}, seed={seed}, "
                            f"method={method_name}",
                            end="",
                            flush=True,
                        )

                    result = run_single_benchmark(
                        n=n,
                        p=p,
                        n_signal=n_signal,
                        seed=seed,
                        use_isis=use_isis,
                        learning_rate=learning_rate,
                        noise_std=noise_std,
                    )
                    results.append(asdict(result))

                    if verbose:
                        print(f" -> {result.time_seconds:.2f}s")

    return pd.DataFrame(results)


def summarize_benchmarks(df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics grouped by (n, p, method).

    Args:
        df: DataFrame from run_benchmark_suite

    Returns:
        DataFrame with mean and std for each metric, grouped by configuration
    """
    metrics = [
        "time_seconds",
        "test_mse",
        "train_mse",
        "beta_mse",
        "true_positives",
        "false_positives",
        "precision",
        "recall",
        "f1_score",
    ]

    summary = df.groupby(["n", "p", "method"])[metrics].agg(["mean", "std"])
    summary.columns = ["_".join(col) for col in summary.columns]
    return summary.reset_index()


def compare_methods(
    df: pd.DataFrame,
    metric: str = "test_mse",
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
) -> pd.DataFrame:
    """Perform statistical comparison between ISIS and standard methods.

    Uses paired Wilcoxon signed-rank test for each (n, p) configuration.

    Args:
        df: DataFrame from run_benchmark_suite
        metric: Which metric to compare
        alternative: Alternative hypothesis for the test
            - "two-sided": ISIS != standard
            - "less": ISIS < standard (ISIS is better for MSE/time)
            - "greater": ISIS > standard (ISIS is better for recall/precision)

    Returns:
        DataFrame with test statistics and p-values for each configuration
    """
    comparisons = []

    for (n, p), group in df.groupby(["n", "p"]):
        isis_values = group[group["method"] == "isis"][metric].to_numpy()
        standard_values = group[group["method"] == "standard"][metric].to_numpy()

        if len(isis_values) != len(standard_values):
            raise ValueError(
                f"Unequal number of runs for n={n}, p={p}: "
                f"ISIS={len(isis_values)}, standard={len(standard_values)}"
            )

        # Paired Wilcoxon test
        try:
            stat, pvalue = stats.wilcoxon(
                isis_values, standard_values, alternative=alternative
            )
        except ValueError:
            # All differences are zero
            stat, pvalue = 0.0, 1.0

        # Also compute means for context
        isis_mean = np.mean(isis_values)
        standard_mean = np.mean(standard_values)
        diff_mean = isis_mean - standard_mean
        ratio = isis_mean / standard_mean if standard_mean != 0 else float("inf")

        comparisons.append(
            {
                "n": n,
                "p": p,
                "metric": metric,
                "isis_mean": isis_mean,
                "standard_mean": standard_mean,
                "difference": diff_mean,
                "ratio": ratio,
                "wilcoxon_stat": stat,
                "p_value": pvalue,
                "significant_0.05": pvalue < 0.05,
                "significant_0.01": pvalue < 0.01,
            }
        )

    return pd.DataFrame(comparisons)


def format_comparison_table(
    df: pd.DataFrame,
    metrics: list[str] | None = None,
) -> str:
    """Format benchmark summary as a readable comparison table.

    Args:
        df: DataFrame from run_benchmark_suite
        metrics: List of metrics to include (default: key metrics)

    Returns:
        Formatted string table
    """
    if metrics is None:
        metrics = ["time_seconds", "test_mse", "recall", "precision"]

    lines = []
    lines.append("=" * 80)
    lines.append("ISIS vs Standard Boosting - Benchmark Comparison")
    lines.append("=" * 80)

    for (n, p), group in df.groupby(["n", "p"]):
        lines.append(f"\nn={n}, p={p:,}")
        lines.append("-" * 40)

        isis = group[group["method"] == "isis"]
        standard = group[group["method"] == "standard"]

        for metric in metrics:
            isis_mean = isis[metric].mean()
            isis_std = isis[metric].std()
            std_mean = standard[metric].mean()
            std_std = standard[metric].std()

            # Format based on metric type
            if metric == "time_seconds":
                lines.append(
                    f"  Time:      Standard={std_mean:.3f}±{std_std:.3f}s, "
                    f"ISIS={isis_mean:.3f}±{isis_std:.3f}s "
                    f"({std_mean / isis_mean:.1f}x speedup)"
                )
            elif "mse" in metric.lower():
                lines.append(
                    f"  {metric}: Standard={std_mean:.4f}±{std_std:.4f}, "
                    f"ISIS={isis_mean:.4f}±{isis_std:.4f}"
                )
            else:
                lines.append(
                    f"  {metric}: Standard={std_mean:.3f}±{std_std:.3f}, "
                    f"ISIS={isis_mean:.3f}±{isis_std:.3f}"
                )

    lines.append("\n" + "=" * 80)
    return "\n".join(lines)
