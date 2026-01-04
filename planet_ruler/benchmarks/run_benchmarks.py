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
Command-line interface for running Planet Ruler benchmarks.

Usage:
    python -m planet_ruler.benchmarks.run_benchmarks <config_path> [options]

Examples:
    # Run smoke tests
    python -m planet_ruler.benchmarks.run_benchmarks benchmarks/configs/smoke_test.yaml

    # Run full suite with filtering
    python -m planet_ruler.benchmarks.run_benchmarks benchmarks/configs/full_suite.yaml --scenario gradient_multires

    # Run specific images
    python -m planet_ruler.benchmarks.run_benchmarks benchmarks/configs/full_suite.yaml --image pexels-claiton-17217951_exif

    # Parallel execution (future)
    python -m planet_ruler.benchmarks.run_benchmarks benchmarks/configs/full_suite.yaml --parallel
"""

import argparse
import sys
from pathlib import Path

from planet_ruler.benchmarks.runner import BenchmarkRunner


def main():
    parser = argparse.ArgumentParser(
        description="Run Planet Ruler performance benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m planet_ruler.benchmarks.run_benchmarks benchmarks/configs/smoke_test.yaml
  python -m planet_ruler.benchmarks.run_benchmarks benchmarks/configs/full_suite.yaml --scenario gradient_multires
  python -m planet_ruler.benchmarks.run_benchmarks benchmarks/configs/full_suite.yaml --image pexels-claiton-17217951_exif
        """,
    )

    parser.add_argument("config", type=Path, help="Path to YAML configuration file")

    parser.add_argument(
        "--scenario",
        action="append",
        help="Filter to specific scenario(s), can be specified multiple times",
    )

    parser.add_argument(
        "--image",
        action="append",
        help="Filter to specific image(s), can be specified multiple times",
    )

    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run scenarios in parallel (not yet implemented)",
    )

    parser.add_argument(
        "--db",
        type=Path,
        help="Path to output database (default: benchmarks/results/benchmark_results.db)",
    )

    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")

    args = parser.parse_args()

    # Validate config exists
    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    # Initialize runner
    try:
        runner = BenchmarkRunner(args.config, db_path=args.db)
    except Exception as e:
        print(f"Error initializing benchmark runner: {e}", file=sys.stderr)
        sys.exit(1)

    # Display info
    if not args.quiet:
        print("=" * 70)
        print("Planet Ruler Benchmark Suite")
        print("=" * 70)
        print(f"Config: {args.config}")
        print(f"Database: {runner.db_path}")
        if args.scenario:
            print(f"Scenarios: {', '.join(args.scenario)}")
        if args.image:
            print(f"Images: {', '.join(args.image)}")
        print(f"Git commit: {runner.git_commit or 'N/A'}")
        print("=" * 70)
        print()

    # Run benchmarks
    try:
        results = runner.run(
            parallel=args.parallel, scenarios=args.scenario, images=args.image
        )
    except Exception as e:
        print(f"Error running benchmarks: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Summary
    if not args.quiet:
        print()
        print("=" * 70)
        print("Benchmark Summary")
        print("=" * 70)
        print(f"Total runs: {len(results)}")

        successful = sum(1 for r in results if r.convergence_status == "success")
        print(f"Successful: {successful} ({successful/len(results)*100:.1f}%)")

        if successful > 0:
            successful_results = [
                r for r in results if r.convergence_status == "success"
            ]
            mean_time = sum(r.total_time for r in successful_results) / len(
                successful_results
            )
            mean_error = sum(
                r.relative_error
                for r in successful_results
                if r.relative_error is not None
            )
            mean_error = mean_error / len(
                [r for r in successful_results if r.relative_error is not None]
            )

            print(f"Mean runtime: {mean_time:.2f}s")
            print(f"Mean relative error: {mean_error*100:.2f}%")

        print("=" * 70)
        print(f"Results stored in: {runner.db_path}")
        print()
        print("Next steps:")
        print("  - Analyze results: jupyter notebook benchmarks/visualize.ipynb")
        print(
            "  - Export CSV: python -c 'from benchmarks.analyzer import BenchmarkAnalyzer; "
            'BenchmarkAnalyzer().export_csv("results.csv")\''
        )


if __name__ == "__main__":
    main()
