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
Analysis utilities for benchmark results.

Provides functions for querying, aggregating, and analyzing benchmark data
stored in SQLite database.
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


class BenchmarkAnalyzer:
    """
    Analyze benchmark results from SQLite database.

    Provides methods for querying, filtering, and analyzing benchmark results
    including summary statistics, Pareto frontiers, and parameter explosion
    for ablative studies.

    Args:
        db_path (str or Path, optional): Path to SQLite database.
            Defaults to {cwd}/results/benchmark_results.db.

    Attributes:
        db_path (Path): Path to benchmark results database.
    """

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            benchmark_dir = Path.cwd()
            db_path = benchmark_dir / "results" / "benchmark_results.db"

        self.db_path = Path(db_path)

        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Database not found: {self.db_path}\n"
                f"Run benchmarks first with:\n"
                f"  python -m planet_ruler.benchmarks.run_benchmarks benchmarks/configs/smoke_test.yaml"
            )

    def get_results(
        self,
        scenario: Optional[str] = None,
        image: Optional[str] = None,
        min_timestamp: Optional[str] = None,
        max_timestamp: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Query benchmark results as pandas DataFrame.

        Args:
            scenario (str, optional): Filter by scenario name.
            image (str, optional): Filter by image name.
            min_timestamp (str, optional): Filter results after this timestamp (ISO format).
            max_timestamp (str, optional): Filter results before this timestamp (ISO format).

        Returns:
            pd.DataFrame: Benchmark results with JSON columns parsed.
        """
        query = "SELECT * FROM benchmark_results WHERE 1=1"
        params = []

        if scenario is not None:
            query += " AND scenario_name = ?"
            params.append(scenario)

        if image is not None:
            query += " AND image_name = ?"
            params.append(image)

        if min_timestamp is not None:
            query += " AND timestamp >= ?"
            params.append(min_timestamp)

        if max_timestamp is not None:
            query += " AND timestamp <= ?"
            params.append(max_timestamp)

        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        # Parse JSON columns
        json_cols = [
            "fit_params",
            "free_parameters",
            "init_parameter_values",
            "parameter_limits",
            "best_parameters",
            "minimizer_config",
        ]
        for col in json_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: json.loads(x) if pd.notna(x) else {})

        return df

    def explode_parameters(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Explode JSON parameter columns into individual columns for each parameter.

        Creates columns like:
        - r, h, f, theta_x, theta_y, theta_z, w, x_c, y_c (fitted values)
        - r_init, h_init, ... (initial values)
        - r_lower, r_upper, h_lower, h_upper, ... (parameter limits)
        - r_free, h_free, ... (boolean - was parameter free?)
        - max_iter, resolution_stages, ... (fit parameters)

        Parameters
        ----------
        df : pd.DataFrame, optional
            DataFrame to explode. If None, loads all results.

        Returns
        -------
        pd.DataFrame
            DataFrame with exploded parameter columns for ablative analysis

        Examples
        --------
        >>> analyzer = BenchmarkAnalyzer()
        >>> df = analyzer.explode_parameters()
        >>> # Now can do: df[['r', 'r_init', 'r_lower', 'r_upper', 'r_free']]
        >>> # Or correlations: df[['r_init', 'relative_error']].corr()
        >>> # Or fit param studies: df[['max_iter', 'iterations', 'total_time']].corr()
        """
        if df is None:
            df = self.get_results()

        df = df.copy()

        # Get all unique parameters across all runs
        all_params = set()
        for params_dict in df["best_parameters"]:
            if isinstance(params_dict, dict):
                all_params.update(params_dict.keys())

        # Explode best_parameters (fitted values)
        for param in all_params:
            df[param] = df["best_parameters"].apply(
                lambda x: x.get(param) if isinstance(x, dict) else None
            )

        # Explode init_parameter_values
        for param in all_params:
            df[f"{param}_init"] = df["init_parameter_values"].apply(
                lambda x: x.get(param) if isinstance(x, dict) else None
            )

        # Explode parameter_limits (lower and upper bounds)
        for param in all_params:
            df[f"{param}_lower"] = df["parameter_limits"].apply(
                lambda x: (
                    x.get(param, [None, None])[0]
                    if isinstance(x, dict) and param in x
                    else None
                )
            )
            df[f"{param}_upper"] = df["parameter_limits"].apply(
                lambda x: (
                    x.get(param, [None, None])[1]
                    if isinstance(x, dict) and param in x
                    else None
                )
            )

        # Explode free_parameters (boolean indicators)
        for param in all_params:
            df[f"{param}_free"] = df["free_parameters"].apply(
                lambda x: param in x if isinstance(x, list) else False
            )

        # Expose minimizer method from minimizer_config JSON column
        df["minimizer"] = df["minimizer_config"].apply(
            lambda x: x if isinstance(x, str) else (
                x.get("method") or x.get("minimizer")
                if isinstance(x, dict) else None
            )
        )

        # Explode fit_params (max_iter, resolution_stages, etc.)
        all_fit_params = set()
        for fit_dict in df["fit_params"]:
            if isinstance(fit_dict, dict):
                all_fit_params.update(fit_dict.keys())

        for fit_param in all_fit_params:
            # Special handling for resolution_stages (list)
            if fit_param == "resolution_stages":
                df[fit_param] = df["fit_params"].apply(
                    lambda x: (
                        str(x.get(fit_param))
                        if isinstance(x, dict) and fit_param in x
                        else None
                    )
                )
            else:
                df[fit_param] = df["fit_params"].apply(
                    lambda x: x.get(fit_param) if isinstance(x, dict) else None
                )

        return df

    def get_summary_stats(self, scenario: Optional[str] = None) -> pd.DataFrame:
        """
        Get summary statistics across scenarios.

        Args:
            scenario (str, optional): Filter by scenario name.

        Returns:
            pd.DataFrame: Summary statistics with columns n_runs, mean_time,
                std_time, mean_rel_error, accuracy_rate, success_rate.
        """
        df = self.get_results(scenario=scenario)

        summary = (
            df.groupby("scenario_name")
            .agg(
                {
                    "total_time": ["count", "mean", "std", "min", "max"],
                    "relative_error": ["mean", "std", "min", "max"],
                    "within_uncertainty": "mean",
                    "convergence_status": lambda x: (x == "success").mean(),
                }
            )
            .round(4)
        )

        summary.columns = ["_".join(col).strip("_") for col in summary.columns]
        summary = summary.rename(
            columns={
                "total_time_count": "n_runs",
                "total_time_mean": "mean_time",
                "total_time_std": "std_time",
                "total_time_min": "min_time",
                "total_time_max": "max_time",
                "relative_error_mean": "mean_rel_error",
                "relative_error_std": "std_rel_error",
                "relative_error_min": "min_rel_error",
                "relative_error_max": "max_rel_error",
                "within_uncertainty_mean": "accuracy_rate",
                "convergence_status_<lambda>": "success_rate",
            }
        )

        return summary.reset_index()

    def get_pareto_frontier(
        self,
        metrics: Sequence[tuple],
        df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Identify the Pareto-optimal frontier across N metrics.

        A row is Pareto-optimal if no other row is at least as good on every
        metric and strictly better on at least one.

        Args:
            metrics: Sequence of ``(column, direction)`` pairs where
                ``direction`` is ``"min"`` or ``"max"``.  All listed columns
                must be present in the DataFrame.
            df: DataFrame to filter.  If None, loads all successful results.

        Returns:
            pd.DataFrame: Non-dominated rows sorted by the first metric.

        Example::

            pareto = analyzer.get_pareto_frontier(
                [("total_time", "min"), ("relative_error", "min")]
            )
        """
        if df is None:
            df = self.get_results()
        if "convergence_status" in df.columns:
            df = df[df["convergence_status"] == "success"].copy()
        else:
            df = df.copy()
        if df.empty:
            return df

        # Build a (n_rows, n_metrics) array where lower is always better
        cols = []
        for col, direction in metrics:
            vals = df[col].to_numpy(dtype=float)
            cols.append(vals if direction == "min" else -vals)
        mat = np.column_stack(cols)  # shape (n, k)

        n = len(mat)
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if dominated[i]:
                continue
            # Any row j that is ≤ i on all dims and < i on at least one
            leq = np.all(mat <= mat[i], axis=1)  # (n,)
            lt_any = np.any(mat < mat[i], axis=1)  # (n,)
            mask = leq & lt_any
            mask[i] = False
            if mask.any():
                dominated[i] = True

        first_col = metrics[0][0]
        return df[~dominated].sort_values(first_col)

    def score_pareto(
        self,
        df: pd.DataFrame,
        metrics: Sequence[tuple],
    ) -> pd.Series:
        """
        Compute a weighted score for each row (lower = better).

        Each metric is normalised to [0, 1] (min-max across the supplied
        DataFrame), then multiplied by its weight and summed.  "max" metrics
        are flipped before normalisation so that lower score always means
        better.

        Args:
            df: DataFrame to score (e.g. the output of ``get_pareto_frontier``).
            metrics: Sequence of ``(column, direction, weight)`` triples.

        Returns:
            pd.Series: Weighted scores indexed like ``df``.
        """
        score = pd.Series(0.0, index=df.index)
        for col, direction, weight in metrics:
            vals = df[col].astype(float)
            lo, hi = vals.min(), vals.max()
            if hi == lo:
                normalised = pd.Series(0.0, index=df.index)
            else:
                normalised = (vals - lo) / (hi - lo)
            if direction == "max":
                normalised = 1.0 - normalised
            score += weight * normalised
        return score

    def reliability(
        self,
        df: pd.DataFrame,
        threshold: float = 0.10,
        error_col: str = "relative_error",
    ) -> float:
        """
        Fraction of rows whose relative error is below *threshold*.

        Args:
            df: DataFrame slice to evaluate.
            threshold: Maximum acceptable relative error (default 0.10 = 10 %).
            error_col: Column containing relative error values.

        Returns:
            float: Fraction in [0, 1], or ``float("nan")`` if all values are
            NaN.
        """
        vals = df[error_col].dropna()
        if vals.empty:
            return float("nan")
        return float((vals < threshold).mean())

    def compare_scenarios(
        self, scenarios: List[str], metric: str = "total_time"
    ) -> pd.DataFrame:
        """
        Compare specific scenarios across images.

        Args:
            scenarios (list of str): Scenario names to compare.
            metric (str): Metric to compare.

        Returns:
            pd.DataFrame: Pivot table with images as rows, scenarios as columns.
        """
        df = self.get_results()
        df = df[df["scenario_name"].isin(scenarios)]

        pivot = df.pivot_table(
            values=metric, index="image_name", columns="scenario_name", aggfunc="mean"
        )

        return pivot

    def get_timing_breakdown(self, scenario: Optional[str] = None) -> pd.DataFrame:
        """
        Analyze timing breakdown across phases.

        Args:
            scenario (str, optional): Filter by scenario name.

        Returns:
            pd.DataFrame: Mean timing for each phase across scenarios.
        """
        df = self.get_results(scenario=scenario)

        timing_cols = ["image_load_time", "detection_time", "optimization_time"]

        breakdown = df.groupby("scenario_name")[timing_cols].agg(["mean", "std"])

        return breakdown

    def identify_bottlenecks(
        self, scenario: str, threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Identify timing bottlenecks for a scenario.

        Args:
            scenario (str): Scenario name.
            threshold (float): Fraction of total time to be considered bottleneck.

        Returns:
            dict: Phase name to fraction of total time mapping for bottleneck phases.
        """
        df = self.get_results(scenario=scenario)

        phases = {
            "image_load": "image_load_time",
            "detection": "detection_time",
            "optimization": "optimization_time",
        }

        bottlenecks = {}
        for phase_name, col in phases.items():
            mean_time = df[col].mean()
            mean_total = df["total_time"].mean()
            fraction = mean_time / mean_total if mean_total > 0 else 0

            if fraction >= threshold:
                bottlenecks[phase_name] = fraction

        return bottlenecks

    def export_csv(self, output_path: Path, scenario: Optional[str] = None):
        """
        Export results to CSV file.

        Args:
            output_path (Path): Output CSV file path.
            scenario (str, optional): Filter by scenario name.
        """
        df = self.get_results(scenario=scenario)

        # Convert JSON columns to strings for CSV
        json_cols = [
            "fit_params",
            "free_parameters",
            "init_parameter_values",
            "parameter_limits",
            "best_parameters",
            "minimizer_config",
        ]
        for col in json_cols:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: json.dumps(x) if x is not None else None
                )

        df.to_csv(output_path, index=False)
        print(f"Exported {len(df)} results to {output_path}")
