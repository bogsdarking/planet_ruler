#!/usr/bin/env python
# Copyright 2025 Brandon Anderson
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Benchmark Analysis Script

Standalone Python script version of the Jupyter notebook for analyzing
benchmark results. Generates all plots and statistics.

Usage:
    python -m planet_ruler.benchmarks.analyze_benchmarks [--output-dir plots/]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from planet_ruler.benchmarks.analyzer import BenchmarkAnalyzer


def setup_plotting():
    """Configure matplotlib/seaborn styles."""
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (12, 6)


def load_data(analyzer):
    """Load and display basic statistics."""
    df = analyzer.get_results()

    print("=" * 70)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 70)
    print(f"Total benchmark runs: {len(df)}")
    print(f"Unique scenarios: {df['scenario_name'].nunique()}")
    print(f"Unique images: {df['image_name'].nunique()}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print()

    return df


def show_summary_stats(analyzer):
    """Display summary statistics table."""
    summary = analyzer.get_summary_stats()

    print("=" * 70)
    print("SUMMARY STATISTICS BY SCENARIO")
    print("=" * 70)
    print(summary.to_string(index=False))
    print()


def plot_performance_vs_accuracy(df, output_dir=None):
    """Plot runtime vs accuracy scatter plot."""
    success_df = df[df["convergence_status"] == "success"].copy()

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot all scenarios
    for scenario in success_df["scenario_name"].unique():
        scenario_df = success_df[success_df["scenario_name"] == scenario]
        ax.scatter(
            scenario_df["total_time"],
            scenario_df["relative_error"] * 100,
            label=scenario,
            alpha=0.6,
            s=100,
        )

    # Add reference lines
    ax.axhline(y=20, color="r", linestyle="--", alpha=0.3, label="20% error threshold")
    ax.axvline(x=15, color="g", linestyle="--", alpha=0.3, label="15s mobile target")

    ax.set_xlabel("Runtime (seconds)", fontsize=12)
    ax.set_ylabel("Relative Error (%)", fontsize=12)
    ax.set_title("Performance vs Accuracy Trade-off", fontsize=14, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_dir:
        plt.savefig(
            output_dir / "performance_vs_accuracy.png", dpi=150, bbox_inches="tight"
        )
        print(f"Saved: {output_dir / 'performance_vs_accuracy.png'}")

    plt.show()


def show_pareto_frontier(analyzer):
    """Display Pareto-optimal configurations."""
    pareto_df = analyzer.get_pareto_frontier()

    print("=" * 70)
    print(f"PARETO-OPTIMAL CONFIGURATIONS ({len(pareto_df)} found)")
    print("=" * 70)

    display_cols = [
        "scenario_name",
        "image_name",
        "total_time",
        "relative_error",
        "iterations",
    ]
    print(pareto_df[display_cols].sort_values("total_time").to_string(index=False))
    print()


def plot_runtime_distribution(df, output_dir=None):
    """Plot runtime distribution by scenario."""
    success_df = df[df["convergence_status"] == "success"].copy()

    fig, ax = plt.subplots(figsize=(14, 6))

    success_df.boxplot(column="total_time", by="scenario_name", ax=ax, rot=45)

    ax.axhline(y=15, color="r", linestyle="--", alpha=0.5, label="Mobile target (15s)")
    ax.set_xlabel("Scenario", fontsize=12)
    ax.set_ylabel("Runtime (seconds)", fontsize=12)
    ax.set_title("Runtime Distribution by Scenario", fontsize=14, fontweight="bold")
    plt.suptitle("")  # Remove auto-generated title
    ax.legend()

    plt.tight_layout()

    if output_dir:
        plt.savefig(
            output_dir / "runtime_distribution.png", dpi=150, bbox_inches="tight"
        )
        print(f"Saved: {output_dir / 'runtime_distribution.png'}")

    plt.show()


def plot_timing_breakdown(df, output_dir=None):
    """Plot stacked bar chart of timing phases."""
    success_df = df[df["convergence_status"] == "success"].copy()

    fig, ax = plt.subplots(figsize=(14, 6))

    phases = ["image_load_time", "detection_time", "optimization_time"]
    phase_labels = ["Image Load", "Detection", "Optimization"]

    mean_times = success_df.groupby("scenario_name")[phases].mean()

    mean_times.plot(
        kind="bar", stacked=True, ax=ax, color=["#1f77b4", "#ff7f0e", "#2ca02c"]
    )

    ax.set_xlabel("Scenario", fontsize=12)
    ax.set_ylabel("Time (seconds)", fontsize=12)
    ax.set_title("Timing Breakdown by Phase", fontsize=14, fontweight="bold")
    ax.legend(phase_labels, title="Phase")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()

    if output_dir:
        plt.savefig(output_dir / "timing_breakdown.png", dpi=150, bbox_inches="tight")
        print(f"Saved: {output_dir / 'timing_breakdown.png'}")

    plt.show()


def identify_bottlenecks(analyzer):
    """Analyze and display bottlenecks."""
    df = analyzer.get_results()

    print("=" * 70)
    print("BOTTLENECK ANALYSIS (phases >30% of total time)")
    print("=" * 70)

    for scenario in df["scenario_name"].unique():
        bottlenecks = analyzer.identify_bottlenecks(scenario, threshold=0.3)
        if bottlenecks:
            print(f"\n{scenario}:")
            for phase, fraction in bottlenecks.items():
                print(f"  {phase}: {fraction:.1%} of total time")
    print()


def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/results/plots"),
        help="Directory to save plots (default: benchmarks/results/plots)",
    )
    parser.add_argument(
        "--db",
        type=Path,
        help="Path to benchmark database (default: benchmarks/results/benchmark_results.db)",
    )
    parser.add_argument("--export-csv", type=Path, help="Export results to CSV file")

    args = parser.parse_args()

    # Setup
    setup_plotting()

    # Create output directory
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load analyzer
    analyzer = BenchmarkAnalyzer(db_path=args.db)

    # Run analysis
    df = load_data(analyzer)

    if len(df) == 0:
        print("No benchmark results found. Run benchmarks first:")
        print(
            "  python benchmarks/run_benchmarks.py benchmarks/configs/smoke_test.yaml"
        )
        return

    show_summary_stats(analyzer)
    show_pareto_frontier(analyzer)
    identify_bottlenecks(analyzer)

    # Generate plots
    print("=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)

    plot_performance_vs_accuracy(df, output_dir=args.output_dir)
    plot_runtime_distribution(df, output_dir=args.output_dir)
    plot_timing_breakdown(df, output_dir=args.output_dir)

    # Export if requested
    if args.export_csv:
        analyzer.export_csv(args.export_csv)
        print(f"\nExported results to: {args.export_csv}")

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
