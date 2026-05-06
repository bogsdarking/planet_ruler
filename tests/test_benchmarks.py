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

"""Tests for benchmark infrastructure: analyzer, runner, and vet_images."""

import csv
import json
import math
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from planet_ruler.benchmarks.analyzer import BenchmarkAnalyzer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SYNTH_IMAGE = (
    Path(__file__).parent.parent
    / "planet_ruler/benchmarks/images/synth_iphone_13_h10km_clean.jpg"
)
SYNTH_ANNOTATION = (
    Path(__file__).parent.parent
    / "planet_ruler/benchmarks/annotations"
    / "synth_iphone_13_h10km_clean_limb_points.json"
)


def _make_analyzer_with_data(rows):
    """Create a BenchmarkAnalyzer backed by an in-memory SQLite DB."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE benchmark_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scenario_name TEXT NOT NULL,
            image_name TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            git_commit TEXT,
            detection_method TEXT NOT NULL,
            fit_params TEXT NOT NULL,
            free_parameters TEXT NOT NULL,
            init_parameter_values TEXT NOT NULL,
            parameter_limits TEXT NOT NULL,
            minimizer_config TEXT NOT NULL,
            minimizer_preset TEXT,
            image_width INTEGER,
            image_height INTEGER,
            altitude REAL,
            planet_name TEXT,
            expected_radius REAL,
            uncertainty_radius REAL,
            best_parameters TEXT NOT NULL,
            fitted_radius REAL,
            convergence_status TEXT,
            iterations INTEGER,
            absolute_error REAL,
            relative_error REAL,
            within_uncertainty INTEGER,
            total_time REAL NOT NULL,
            image_load_time REAL,
            detection_time REAL,
            optimization_time REAL,
            timing_details TEXT,
            error_message TEXT
        )
        """
    )
    for row in rows:
        conn.execute(
            """
            INSERT INTO benchmark_results
            (scenario_name, image_name, timestamp, git_commit,
             detection_method, fit_params, free_parameters,
             init_parameter_values, parameter_limits, minimizer_config,
             minimizer_preset, image_width, image_height, altitude,
             planet_name, expected_radius, uncertainty_radius,
             best_parameters, fitted_radius, convergence_status, iterations,
             absolute_error, relative_error, within_uncertainty,
             total_time, image_load_time, detection_time, optimization_time,
             timing_details, error_message)
            VALUES (
                ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
            )
            """,
            row,
        )
    conn.commit()
    conn.close()
    return BenchmarkAnalyzer(db_path=db_path), db_path


def _row(
    scenario="s1",
    image="img1",
    status="success",
    total_time=10.0,
    relative_error=0.05,
    minimizer_preset="balanced",
    minimizer_config="basinhopping",
):
    return (
        scenario, image, "2025-01-01T00:00:00", None,
        "manual", "{}", "[]", "{}", "{}", json.dumps(minimizer_config),
        minimizer_preset,
        800, 600, 10000.0, "Earth", 6371000.0, 10000.0,
        "{}", 6371000.0, status, 100,
        0.0, relative_error, 1,
        total_time, 0.1, 0.1, total_time - 0.2,
        None, None,
    )


# ---------------------------------------------------------------------------
# BenchmarkAnalyzer.get_pareto_frontier
# ---------------------------------------------------------------------------


class TestGetParetoFrontier:
    def test_2d_min_min_basic(self):
        # "a": fast but inaccurate; "c": slow but accurate
        # "b": dominated by "a" on both dims
        rows = [
            _row("a", total_time=1.0, relative_error=0.10),
            _row("b", total_time=5.0, relative_error=0.50),
            _row("c", total_time=10.0, relative_error=0.01),
        ]
        analyzer, db_path = _make_analyzer_with_data(rows)
        pareto = analyzer.get_pareto_frontier(
            [("total_time", "min"), ("relative_error", "min")]
        )
        db_path.unlink()
        # "a" and "c" are non-dominated; "b" is dominated by "a" on both dims
        assert set(pareto["scenario_name"]) == {"a", "c"}

    def test_returns_only_success_rows(self):
        rows = [
            _row("ok", status="success", total_time=1.0, relative_error=0.01),
            _row("bad", status="error", total_time=0.5, relative_error=0.001),
        ]
        analyzer, db_path = _make_analyzer_with_data(rows)
        pareto = analyzer.get_pareto_frontier(
            [("total_time", "min"), ("relative_error", "min")]
        )
        db_path.unlink()
        assert list(pareto["scenario_name"]) == ["ok"]

    def test_empty_returns_empty_dataframe(self):
        analyzer, db_path = _make_analyzer_with_data([])
        pareto = analyzer.get_pareto_frontier(
            [("total_time", "min"), ("relative_error", "min")]
        )
        db_path.unlink()
        assert isinstance(pareto, pd.DataFrame)
        assert len(pareto) == 0

    def test_3d_with_max_dimension(self):
        # reliability (max) is the third dim; row "b" has higher reliability
        rows = [
            _row("a", total_time=1.0, relative_error=0.01),
            _row("b", total_time=2.0, relative_error=0.02),
        ]
        analyzer, db_path = _make_analyzer_with_data(rows)
        # Both are non-dominated when we add a third max-dim where they differ
        df = analyzer.get_results()
        df.loc[df["scenario_name"] == "a", "rl"] = 0.5
        df.loc[df["scenario_name"] == "b", "rl"] = 0.9
        pareto = analyzer.get_pareto_frontier(
            [("total_time", "min"), ("relative_error", "min"), ("rl", "max")],
            df=df,
        )
        db_path.unlink()
        # "a" wins on time+error; "b" wins on reliability → neither dominates
        assert len(pareto) == 2

    def test_sorted_by_first_metric(self):
        rows = [
            _row("slow", total_time=9.0, relative_error=0.01),
            _row("fast", total_time=1.0, relative_error=0.05),
        ]
        analyzer, db_path = _make_analyzer_with_data(rows)
        pareto = analyzer.get_pareto_frontier(
            [("total_time", "min"), ("relative_error", "min")]
        )
        db_path.unlink()
        assert list(pareto["scenario_name"]) == ["fast", "slow"]


# ---------------------------------------------------------------------------
# BenchmarkAnalyzer.score_pareto
# ---------------------------------------------------------------------------


class TestScorePareto:
    def _df(self):
        return pd.DataFrame(
            {
                "total_time": [1.0, 5.0, 10.0],
                "relative_error": [0.10, 0.05, 0.01],
            }
        )

    def test_scores_in_0_1_range(self):
        analyzer, db_path = _make_analyzer_with_data([])
        df = self._df()
        scores = analyzer.score_pareto(
            df,
            [("total_time", "min", 0.5), ("relative_error", "min", 0.5)],
        )
        db_path.unlink()
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_lower_score_is_better(self):
        analyzer, db_path = _make_analyzer_with_data([])
        df = self._df()
        scores = analyzer.score_pareto(
            df,
            [("total_time", "min", 0.5), ("relative_error", "min", 0.5)],
        )
        db_path.unlink()
        # Row 0: best time (norm=0), worst error (norm=1)  → score 0.5
        # Row 2: worst time (norm=1), best error (norm=0)  → score 0.5
        # Equal weights on mirrored dims → row 0 score == row 2 score
        assert math.isclose(scores.iloc[0], scores.iloc[2], abs_tol=1e-9)
        # Row 1 scores strictly between 0 and 1
        assert 0.0 < scores.iloc[1] < 1.0

    def test_max_direction_flipped(self):
        analyzer, db_path = _make_analyzer_with_data([])
        df = pd.DataFrame({"reliability": [0.9, 0.1]})
        scores = analyzer.score_pareto(
            df, [("reliability", "max", 1.0)]
        )
        db_path.unlink()
        # Higher reliability = lower score (better)
        assert scores.iloc[0] < scores.iloc[1]

    def test_constant_column_no_division_by_zero(self):
        analyzer, db_path = _make_analyzer_with_data([])
        df = pd.DataFrame({"total_time": [5.0, 5.0, 5.0]})
        scores = analyzer.score_pareto(df, [("total_time", "min", 1.0)])
        db_path.unlink()
        assert all(s == 0.0 for s in scores)


# ---------------------------------------------------------------------------
# BenchmarkAnalyzer.reliability
# ---------------------------------------------------------------------------


class TestReliability:
    def _analyzer(self):
        analyzer, db_path = _make_analyzer_with_data([])
        return analyzer, db_path

    def test_correct_fraction(self):
        analyzer, db_path = self._analyzer()
        df = pd.DataFrame({"relative_error": [0.05, 0.08, 0.12, 0.20]})
        result = analyzer.reliability(df, threshold=0.10)
        db_path.unlink()
        assert math.isclose(result, 0.5)

    def test_all_pass(self):
        analyzer, db_path = self._analyzer()
        df = pd.DataFrame({"relative_error": [0.01, 0.02, 0.03]})
        result = analyzer.reliability(df, threshold=0.10)
        db_path.unlink()
        assert math.isclose(result, 1.0)

    def test_none_pass(self):
        analyzer, db_path = self._analyzer()
        df = pd.DataFrame({"relative_error": [0.15, 0.20, 0.25]})
        result = analyzer.reliability(df, threshold=0.10)
        db_path.unlink()
        assert math.isclose(result, 0.0)

    def test_nan_rows_excluded(self):
        analyzer, db_path = self._analyzer()
        df = pd.DataFrame({"relative_error": [0.05, float("nan"), 0.08]})
        result = analyzer.reliability(df, threshold=0.10)
        db_path.unlink()
        assert math.isclose(result, 1.0)

    def test_all_nan_returns_nan(self):
        analyzer, db_path = self._analyzer()
        df = pd.DataFrame({"relative_error": [float("nan"), float("nan")]})
        result = analyzer.reliability(df, threshold=0.10)
        db_path.unlink()
        assert math.isnan(result)


# ---------------------------------------------------------------------------
# BenchmarkAnalyzer.explode_parameters — minimizer extraction
# ---------------------------------------------------------------------------


class TestExplodeParametersMinimizer:
    def test_minimizer_string_column_extracted(self):
        rows = [_row(minimizer_config="basinhopping")]
        analyzer, db_path = _make_analyzer_with_data(rows)
        df = analyzer.explode_parameters()
        db_path.unlink()
        assert "minimizer" in df.columns
        assert df["minimizer"].iloc[0] == "basinhopping"

    def test_minimizer_dict_with_method_key(self):
        rows = [_row(minimizer_config={"method": "differential-evolution"})]
        analyzer, db_path = _make_analyzer_with_data(rows)
        df = analyzer.explode_parameters()
        db_path.unlink()
        assert df["minimizer"].iloc[0] == "differential-evolution"


# ---------------------------------------------------------------------------
# BenchmarkRunner — minimizer_preset stored in DB
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not SYNTH_IMAGE.exists(),
    reason="Synthetic benchmark images not present",
)
class TestRunnerMinimiserPreset:
    def test_stores_minimizer_preset(self, tmp_path):
        """After running a scenario the DB row contains minimizer_preset."""
        from planet_ruler.benchmarks.runner import BenchmarkRunner

        config_path = (
            Path(__file__).parent.parent
            / "planet_ruler/benchmarks/configs/smoke_test.yaml"
        )
        if not config_path.exists():
            pytest.skip("smoke_test.yaml not found")

        db_path = tmp_path / "test_runner.db"
        runner = BenchmarkRunner(config_path, db_path=db_path)

        # Build a minimal single scenario pointing at the synth image
        scenario = {
            "name": "test_preset_stored",
            "images": ["synth_iphone_13_h10km_clean"],
            "detection_method": "manual",
            "annotation_file": "synth_iphone_13_h10km_clean_limb_points.json",
            "planet_name": "Earth",
            "expected_radius": 6371000,
            "uncertainty_radius": 100000,
            "minimizer": "basinhopping",
            "minimizer_preset": "fast",
            "free_parameters": ["r"],
            "r_limits_km": [3000, 9000],
            "fit_params": {"max_iter": 5},
            "init_parameter_values_override": {"theta_z": 3.14159},
            "parameter_limits_override": {"theta_z": [-3.1416, 3.1416]},
        }

        result = runner._run_single(scenario, "synth_iphone_13_h10km_clean")
        runner._store_result(result)

        conn = sqlite3.connect(db_path)
        row = conn.execute(
            "SELECT minimizer_preset FROM benchmark_results"
            " WHERE scenario_name=?",
            ("test_preset_stored",),
        ).fetchone()
        conn.close()

        assert row is not None
        assert row[0] == "fast"


# ---------------------------------------------------------------------------
# vet_images — discover_pairs
# ---------------------------------------------------------------------------


class TestVetImagesDiscoverPairs:
    def test_discover_pairs_finds_synth(self):
        from planet_ruler.benchmarks.vet_images import discover_pairs

        bench_dir = Path(__file__).parent.parent / "planet_ruler/benchmarks"
        if not bench_dir.exists():
            pytest.skip("Benchmark directory not found")

        pairs = discover_pairs(bench_dir)
        stems = [img.stem for img, _ in pairs]
        assert any("synth_iphone_13" in s for s in stems)

    def test_discover_pairs_filter(self):
        from planet_ruler.benchmarks.vet_images import discover_pairs

        bench_dir = Path(__file__).parent.parent / "planet_ruler/benchmarks"
        if not bench_dir.exists():
            pytest.skip("Benchmark directory not found")

        pairs = discover_pairs(
            bench_dir,
            image_filter=["synth_iphone_13_h10km_clean"],
        )
        assert len(pairs) == 1
        assert pairs[0][0].stem == "synth_iphone_13_h10km_clean"


@pytest.mark.skipif(
    not SYNTH_IMAGE.exists() or not SYNTH_ANNOTATION.exists(),
    reason="Synthetic benchmark images not present",
)
class TestVetImageBasinhopping:
    def test_vet_image_returns_result(self):
        """Smoke test: vet_image() runs with BH and returns a VetResult."""
        from planet_ruler.benchmarks.vet_images import vet_image, VetResult

        result = vet_image(
            SYNTH_IMAGE,
            SYNTH_ANNOTATION,
            max_iter=10,
            minimizer_preset="fast",
        )
        assert isinstance(result, VetResult)
        assert result.fitted_altitude_km is not None
        assert result.convergence == "success"

    def test_vet_image_camera_always_fixed(self):
        """f and w are always fixed; vet_image accepts no fix_camera arg."""
        from planet_ruler.benchmarks.vet_images import vet_image
        import inspect
        assert "fix_camera" not in inspect.signature(vet_image).parameters
        result = vet_image(
            SYNTH_IMAGE,
            SYNTH_ANNOTATION,
            max_iter=10,
            minimizer_preset="fast",
        )
        assert result.fitted_altitude_km is not None


# ---------------------------------------------------------------------------
# BenchmarkRunner — constrain_radius integration
# ---------------------------------------------------------------------------


class TestRunnerConstrainRadius:
    def test_scenario_name_cr_encoding(self, tmp_path):
        """_make_grid_scenario_name includes cr{N} when a sagitta+arc fit_stages pipeline is set."""
        from planet_ruler.benchmarks.runner import BenchmarkRunner

        config_path = (
            Path(__file__).parent.parent
            / "planet_ruler/benchmarks/configs/smoke_test.yaml"
        )
        if not config_path.exists():
            pytest.skip("smoke_test.yaml not found")

        runner = BenchmarkRunner(config_path, db_path=tmp_path / "test.db")

        params_cr = {
            "minimizer": "basinhopping",
            "minimizer_preset": "fast",
            "free_parameters": ["r", "h", "theta_x", "theta_y", "theta_z"],
            "r_limits_km": [2000, 14000],
            "fit_stages": [{"method": "sagitta", "n_sigma": 3.0}, {"method": "arc"}],
            "h_limits_pct": 0.10,
            "fit_params": {},
        }
        name_cr = runner._make_grid_scenario_name(params_cr)
        assert "cr30" in name_cr

        params_rl = {**params_cr}
        del params_rl["fit_stages"]
        name_rl = runner._make_grid_scenario_name(params_rl)
        assert "cr" not in name_rl
        assert "r" in name_rl  # some rl code present

    def test_scenario_name_sagitta_only(self, tmp_path):
        """Sagitta-only fit_stages scenarios get the g_sag_only_ name prefix."""
        from planet_ruler.benchmarks.runner import BenchmarkRunner

        config_path = (
            Path(__file__).parent.parent
            / "planet_ruler/benchmarks/configs/smoke_test.yaml"
        )
        if not config_path.exists():
            pytest.skip("smoke_test.yaml not found")

        runner = BenchmarkRunner(config_path, db_path=tmp_path / "test.db")
        params = {
            "fit_stages": [{"method": "sagitta", "n_sigma": 2.0}],
            "free_parameters": ["r", "h", "theta_x", "theta_y", "theta_z"],
            "r_limits_km": [2000, 14000],
        }
        name = runner._make_grid_scenario_name(params)
        assert "sag_only" in name


@pytest.mark.skipif(
    not SYNTH_IMAGE.exists() or not SYNTH_ANNOTATION.exists(),
    reason="Synthetic benchmark images not present",
)
class TestRunnerConstrainRadiusIntegration:
    def _base_scenario(self):
        return {
            "name": "test_cr",
            "images": ["synth_iphone_13_h10km_clean"],
            "detection_method": "manual",
            "annotation_file": "synth_iphone_13_h10km_clean_limb_points.json",
            "planet_name": "Earth",
            "expected_radius": 6371000,
            "uncertainty_radius": 100000,
            "minimizer": "basinhopping",
            "minimizer_preset": "fast",
            "free_parameters": ["r"],
            "r_limits_km": [2000, 14000],
            "fit_params": {"max_iter": 5},
            "init_parameter_values_override": {"theta_z": 3.14159},
            "parameter_limits_override": {"theta_z": [-3.1416, 3.1416]},
        }

    def test_parameter_limits_reflect_sagitta_warm_start(self, tmp_path):
        """r bounds tighten after a sagitta stage relative to the wide initial [2000, 14000] km."""
        from planet_ruler.benchmarks.runner import BenchmarkRunner

        config_path = (
            Path(__file__).parent.parent
            / "planet_ruler/benchmarks/configs/smoke_test.yaml"
        )
        if not config_path.exists():
            pytest.skip("smoke_test.yaml not found")

        runner = BenchmarkRunner(config_path, db_path=tmp_path / "test.db")
        scenario = {
            **self._base_scenario(),
            "fit_stages": [
                {"method": "sagitta", "n_sigma": 2.0},
                {"method": "arc"},
            ],
        }
        result = runner._run_single(scenario, "synth_iphone_13_h10km_clean")

        r_limits = result.parameter_limits.get("r", [0, 1e12])
        r_range_km = (r_limits[1] - r_limits[0]) / 1000.0
        # Sagitta stage should tighten r limits well below [2000, 14000] km
        assert r_range_km < 12000.0

    def test_sagitta_only_skips_optimizer(self, tmp_path):
        """Sagitta-only fit_stages run: no optimizer, fitted_radius plausible."""
        from planet_ruler.benchmarks.runner import BenchmarkRunner

        config_path = (
            Path(__file__).parent.parent
            / "planet_ruler/benchmarks/configs/smoke_test.yaml"
        )
        if not config_path.exists():
            pytest.skip("smoke_test.yaml not found")

        runner = BenchmarkRunner(config_path, db_path=tmp_path / "test.db")
        scenario = {
            **self._base_scenario(),
            "name": "test_sag_only",
            "fit_stages": [{"method": "sagitta", "n_sigma": 2.0}],
        }
        result = runner._run_single(scenario, "synth_iphone_13_h10km_clean")

        assert result.convergence_status in ("success", "ok")
        assert result.iterations == 0
        assert result.fitted_radius is not None
        # Should be in the right ballpark for Earth (within 50%)
        assert 3e6 < result.fitted_radius < 1e7


# ---------------------------------------------------------------------------
# BenchmarkAnalyzer.get_results
# ---------------------------------------------------------------------------


def _row_ts(timestamp, scenario="s1", image="img1"):
    """Like _row but with a custom timestamp (index 2)."""
    r = list(_row(scenario, image))
    r[2] = timestamp
    return tuple(r)


@pytest.mark.unit
class TestAnalyzerGetResults:
    def test_returns_all_rows(self):
        rows = [_row("s1"), _row("s2"), _row("s3")]
        analyzer, db_path = _make_analyzer_with_data(rows)
        df = analyzer.get_results()
        db_path.unlink()
        assert len(df) == 3

    def test_filter_by_scenario(self):
        rows = [_row("s1"), _row("s2"), _row("s1")]
        analyzer, db_path = _make_analyzer_with_data(rows)
        df = analyzer.get_results(scenario="s1")
        db_path.unlink()
        assert len(df) == 2
        assert all(df["scenario_name"] == "s1")

    def test_filter_by_image(self):
        rows = [_row("s1", "img_a"), _row("s1", "img_b"), _row("s1", "img_a")]
        analyzer, db_path = _make_analyzer_with_data(rows)
        df = analyzer.get_results(image="img_b")
        db_path.unlink()
        assert len(df) == 1
        assert df["image_name"].iloc[0] == "img_b"

    def test_filter_by_min_timestamp(self):
        rows = [
            _row_ts("2025-01-15T00:00:00"),
            _row_ts("2025-06-15T00:00:00"),
        ]
        analyzer, db_path = _make_analyzer_with_data(rows)
        df = analyzer.get_results(min_timestamp="2025-04-01T00:00:00")
        db_path.unlink()
        assert len(df) == 1
        assert df["timestamp"].iloc[0] == "2025-06-15T00:00:00"

    def test_filter_by_max_timestamp(self):
        rows = [
            _row_ts("2025-01-15T00:00:00"),
            _row_ts("2025-06-15T00:00:00"),
        ]
        analyzer, db_path = _make_analyzer_with_data(rows)
        df = analyzer.get_results(max_timestamp="2025-04-01T00:00:00")
        db_path.unlink()
        assert len(df) == 1
        assert df["timestamp"].iloc[0] == "2025-01-15T00:00:00"

    def test_json_columns_parsed(self):
        rows = [_row()]
        analyzer, db_path = _make_analyzer_with_data(rows)
        df = analyzer.get_results()
        db_path.unlink()
        assert isinstance(df["fit_params"].iloc[0], dict)

    def test_empty_db_returns_empty_dataframe(self):
        analyzer, db_path = _make_analyzer_with_data([])
        df = analyzer.get_results()
        db_path.unlink()
        assert len(df) == 0


# ---------------------------------------------------------------------------
# BenchmarkAnalyzer.get_summary_stats
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAnalyzerGetSummaryStats:
    def test_returns_dataframe_with_n_runs(self):
        rows = [_row("s1"), _row("s1")]
        analyzer, db_path = _make_analyzer_with_data(rows)
        summary = analyzer.get_summary_stats()
        db_path.unlink()
        assert isinstance(summary, pd.DataFrame)
        assert "n_runs" in summary.columns

    def test_n_runs_correct(self):
        rows = [_row("s1"), _row("s1"), _row("s1")]
        analyzer, db_path = _make_analyzer_with_data(rows)
        summary = analyzer.get_summary_stats()
        db_path.unlink()
        row = summary[summary["scenario_name"] == "s1"]
        assert row["n_runs"].iloc[0] == 3

    def test_filter_by_scenario(self):
        rows = [_row("s1"), _row("s1"), _row("s2")]
        analyzer, db_path = _make_analyzer_with_data(rows)
        summary = analyzer.get_summary_stats(scenario="s1")
        db_path.unlink()
        assert len(summary) == 1
        assert summary["scenario_name"].iloc[0] == "s1"


# ---------------------------------------------------------------------------
# BenchmarkAnalyzer.compare_scenarios
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAnalyzerCompareScenarios:
    def test_returns_pivot_table(self):
        rows = [
            _row("scen_a", "img1", total_time=5.0),
            _row("scen_b", "img1", total_time=10.0),
        ]
        analyzer, db_path = _make_analyzer_with_data(rows)
        pivot = analyzer.compare_scenarios(["scen_a", "scen_b"], metric="total_time")
        db_path.unlink()
        assert isinstance(pivot, pd.DataFrame)
        assert "scen_a" in pivot.columns
        assert "scen_b" in pivot.columns

    def test_metric_values_correct(self):
        rows = [
            _row("scen_a", "img1", total_time=5.0),
            _row("scen_b", "img1", total_time=10.0),
        ]
        analyzer, db_path = _make_analyzer_with_data(rows)
        pivot = analyzer.compare_scenarios(["scen_a", "scen_b"], metric="total_time")
        db_path.unlink()
        assert pivot.loc["img1", "scen_a"] == pytest.approx(5.0)
        assert pivot.loc["img1", "scen_b"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# BenchmarkAnalyzer.get_timing_breakdown
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAnalyzerGetTimingBreakdown:
    def test_returns_dataframe(self):
        rows = [_row("s1"), _row("s1")]
        analyzer, db_path = _make_analyzer_with_data(rows)
        result = analyzer.get_timing_breakdown()
        db_path.unlink()
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_timing_columns_present(self):
        rows = [_row("s1")]
        analyzer, db_path = _make_analyzer_with_data(rows)
        result = analyzer.get_timing_breakdown()
        db_path.unlink()
        top_cols = result.columns.get_level_values(0)
        assert "image_load_time" in top_cols
        assert "optimization_time" in top_cols

    def test_filter_by_scenario(self):
        rows = [_row("s1"), _row("s2")]
        analyzer, db_path = _make_analyzer_with_data(rows)
        result = analyzer.get_timing_breakdown(scenario="s1")
        db_path.unlink()
        assert list(result.index) == ["s1"]


# ---------------------------------------------------------------------------
# BenchmarkAnalyzer.identify_bottlenecks
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAnalyzerIdentifyBottlenecks:
    def test_returns_dict(self):
        rows = [_row("s1")]
        analyzer, db_path = _make_analyzer_with_data(rows)
        result = analyzer.identify_bottlenecks("s1")
        db_path.unlink()
        assert isinstance(result, dict)

    def test_bottleneck_detected(self):
        # _row(total_time=10.0): optimization_time = 10.0 - 0.2 = 9.8
        # fraction = 9.8 / 10.0 = 0.98 > 0.5 → bottleneck
        rows = [_row("s1", total_time=10.0)]
        analyzer, db_path = _make_analyzer_with_data(rows)
        result = analyzer.identify_bottlenecks("s1", threshold=0.5)
        db_path.unlink()
        assert "optimization" in result

    def test_no_bottleneck_when_threshold_high(self):
        # With threshold=0.99, fraction 0.98 does not qualify
        rows = [_row("s1", total_time=10.0)]
        analyzer, db_path = _make_analyzer_with_data(rows)
        result = analyzer.identify_bottlenecks("s1", threshold=0.99)
        db_path.unlink()
        assert result == {}


# ---------------------------------------------------------------------------
# BenchmarkAnalyzer.export_csv
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAnalyzerExportCsv:
    def test_creates_csv_file(self, tmp_path):
        rows = [_row("s1"), _row("s1")]
        analyzer, db_path = _make_analyzer_with_data(rows)
        out = tmp_path / "results.csv"
        analyzer.export_csv(out)
        db_path.unlink()
        assert out.exists()

    def test_csv_row_count_matches_db(self, tmp_path):
        rows = [_row("s1"), _row("s2"), _row("s1")]
        analyzer, db_path = _make_analyzer_with_data(rows)
        out = tmp_path / "results.csv"
        analyzer.export_csv(out)
        db_path.unlink()
        with open(out) as f:
            reader = csv.reader(f)
            lines = list(reader)
        assert len(lines) - 1 == 3  # subtract header

    def test_csv_contains_expected_columns(self, tmp_path):
        rows = [_row("s1")]
        analyzer, db_path = _make_analyzer_with_data(rows)
        out = tmp_path / "results.csv"
        analyzer.export_csv(out)
        db_path.unlink()
        with open(out) as f:
            header = f.readline().strip().split(",")
        assert "scenario_name" in header
        assert "image_name" in header
        assert "total_time" in header

    def test_filter_by_scenario_exports_subset(self, tmp_path):
        rows = [_row("s1"), _row("s1"), _row("s2")]
        analyzer, db_path = _make_analyzer_with_data(rows)
        out = tmp_path / "results_s1.csv"
        analyzer.export_csv(out, scenario="s1")
        db_path.unlink()
        with open(out) as f:
            reader = csv.DictReader(f)
            exported = list(reader)
        assert all(r["scenario_name"] == "s1" for r in exported)
        assert len(exported) == 2


# ---------------------------------------------------------------------------
# vet_images — _load_annotation_as_target
# ---------------------------------------------------------------------------


def _write_annot_json(tmp_path, points, fmt="points", image_width=200, image_height=150):
    """Write a minimal annotation JSON file and return its path."""
    path = tmp_path / "annot_limb_points.json"
    if fmt == "points":
        data = {
            "image_path": str(tmp_path / "img.jpg"),
            "image_width": image_width,
            "image_height": image_height,
            "points": points,
        }
    else:  # "limb_points" nested format
        data = {
            "image_path": str(tmp_path / "img.jpg"),
            "image_width": image_width,
            "image_height": image_height,
            "limb_points": {"points": points},
        }
    with open(path, "w") as f:
        json.dump(data, f)
    return path


@pytest.mark.unit
class TestVetImagesLoadAnnotation:
    def test_returns_array_of_image_width(self, tmp_path):
        from planet_ruler.benchmarks.vet_images import _load_annotation_as_target

        path = _write_annot_json(tmp_path, [[50.0, 200.0]], image_width=200)
        result = _load_annotation_as_target(path, image_width=200)
        assert len(result) == 200

    def test_annotated_positions_filled(self, tmp_path):
        from planet_ruler.benchmarks.vet_images import _load_annotation_as_target

        path = _write_annot_json(tmp_path, [[50.0, 200.0]], image_width=200)
        result = _load_annotation_as_target(path, image_width=200)
        assert result[50] == pytest.approx(200.0)

    def test_unannotated_positions_are_nan(self, tmp_path):
        from planet_ruler.benchmarks.vet_images import _load_annotation_as_target

        path = _write_annot_json(tmp_path, [[50.0, 200.0]], image_width=200)
        result = _load_annotation_as_target(path, image_width=200)
        assert np.isnan(result[0])
        assert np.isnan(result[100])

    def test_out_of_bounds_x_skipped(self, tmp_path):
        from planet_ruler.benchmarks.vet_images import _load_annotation_as_target

        # Points with x=-1 and x=300 (> width=200) should not cause IndexError
        path = _write_annot_json(
            tmp_path, [[-1.0, 100.0], [50.0, 200.0], [300.0, 100.0]], image_width=200
        )
        result = _load_annotation_as_target(path, image_width=200)
        assert result[50] == pytest.approx(200.0)
        assert np.isnan(result[0])

    def test_points_flat_format_works(self, tmp_path):
        from planet_ruler.benchmarks.vet_images import _load_annotation_as_target

        path = _write_annot_json(tmp_path, [[30.0, 150.0]], fmt="points", image_width=200)
        result = _load_annotation_as_target(path, image_width=200)
        assert result[30] == pytest.approx(150.0)

    def test_limb_points_nested_format_works(self, tmp_path):
        from planet_ruler.benchmarks.vet_images import _load_annotation_as_target

        path = _write_annot_json(tmp_path, [[80.0, 180.0]], fmt="limb_points", image_width=200)
        result = _load_annotation_as_target(path, image_width=200)
        assert result[80] == pytest.approx(180.0)


# ---------------------------------------------------------------------------
# vet_images — _build_step2_config
# ---------------------------------------------------------------------------


EARTH_RADIUS_M = 6_371_000.0

_STEP1_CONFIG = {
    "free_parameters": ["h", "theta_x", "theta_y", "theta_z"],
    "init_parameter_values": {
        "r": EARTH_RADIUS_M,
        "h": 10_000.0,
        "theta_x": 0.05,
        "theta_y": 0.0,
        "theta_z": 3.14159,
        "f": 0.005,
        "w": 0.0076,
    },
    "parameter_limits": {
        "r": [EARTH_RADIUS_M * 0.999, EARTH_RADIUS_M * 1.001],
        "h": [1_500.0, 18_000.0],
        "theta_x": [-0.5, 0.5],
        "theta_y": [-0.5, 0.5],
        "theta_z": [2.0, 4.0],
        "f": [0.003, 0.01],
        "w": [0.005, 0.012],
    },
}


class _FakeObs:
    best_parameters = {
        "theta_x": 0.1,
        "theta_y": 0.0,
        "theta_z": 3.14,
        "f": 0.005,
        "w": 0.0076,
    }


@pytest.mark.unit
class TestVetImagesBuildStep2Config:
    def test_h_removed_from_free_parameters(self):
        from planet_ruler.benchmarks.vet_images import _build_step2_config

        import copy
        config2 = _build_step2_config(copy.deepcopy(_STEP1_CONFIG), 10_000.0, _FakeObs())
        assert "h" not in config2["free_parameters"]

    def test_r_added_to_free_parameters(self):
        from planet_ruler.benchmarks.vet_images import _build_step2_config

        import copy
        config2 = _build_step2_config(copy.deepcopy(_STEP1_CONFIG), 10_000.0, _FakeObs())
        assert "r" in config2["free_parameters"]

    def test_h_pinned_to_provided_value(self):
        from planet_ruler.benchmarks.vet_images import _build_step2_config

        import copy
        h_fit = 11_234.5
        config2 = _build_step2_config(copy.deepcopy(_STEP1_CONFIG), h_fit, _FakeObs())
        assert config2["init_parameter_values"]["h"] == pytest.approx(h_fit)

    def test_r_limits_wide(self):
        from planet_ruler.benchmarks.vet_images import _build_step2_config

        import copy
        config2 = _build_step2_config(copy.deepcopy(_STEP1_CONFIG), 10_000.0, _FakeObs())
        r_lo, r_hi = config2["parameter_limits"]["r"]
        # ±50% of Earth radius
        assert r_lo == pytest.approx(6_371_000 * 0.50, rel=0.01)
        assert r_hi == pytest.approx(6_371_000 * 1.50, rel=0.01)

    def test_step1_config_not_mutated(self):
        from planet_ruler.benchmarks.vet_images import _build_step2_config

        import copy
        original = copy.deepcopy(_STEP1_CONFIG)
        _build_step2_config(copy.deepcopy(_STEP1_CONFIG), 10_000.0, _FakeObs())
        assert "h" in original["free_parameters"]

    def test_orientation_warm_started_from_obs(self):
        from planet_ruler.benchmarks.vet_images import _build_step2_config

        import copy
        config2 = _build_step2_config(copy.deepcopy(_STEP1_CONFIG), 10_000.0, _FakeObs())
        # theta_x=0.1 is within [-0.5, 0.5], so it should be warm-started
        assert config2["init_parameter_values"]["theta_x"] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# vet_images — _save_csv
# ---------------------------------------------------------------------------


def _make_vet_result(name="img1", verdict="PASS", h_km=10.0, r_km=None):
    from planet_ruler.benchmarks.vet_images import VetResult

    return VetResult(
        image_name=name,
        annotation_file=f"{name}_limb_points.json",
        fitted_altitude_km=h_km,
        gps_altitude_km=10.5,
        fitted_radius_km=r_km,
        radius_error_pct=None if r_km is None else abs(r_km - 6371.0) / 6371.0 * 100,
        convergence="success",
        runtime_s=1.5,
        verdict=verdict,
        notes="test",
    )


@pytest.mark.unit
class TestVetImagesSaveCsv:
    def test_creates_csv_file(self, tmp_path):
        from planet_ruler.benchmarks.vet_images import _save_csv

        results = [_make_vet_result("img1"), _make_vet_result("img2")]
        out = tmp_path / "out.csv"
        _save_csv(results, out)
        assert out.exists()

    def test_header_row_correct(self, tmp_path):
        from planet_ruler.benchmarks.vet_images import _save_csv

        out = tmp_path / "out.csv"
        _save_csv([_make_vet_result()], out)
        with open(out) as f:
            header = f.readline().strip().split(",")
        assert "image_name" in header
        assert "verdict" in header
        assert "convergence" in header

    def test_row_count_matches_results(self, tmp_path):
        from planet_ruler.benchmarks.vet_images import _save_csv

        out = tmp_path / "out.csv"
        _save_csv([_make_vet_result("a"), _make_vet_result("b")], out)
        with open(out) as f:
            reader = csv.reader(f)
            lines = list(reader)
        assert len(lines) - 1 == 2  # subtract header

    def test_none_fields_written_as_empty_string(self, tmp_path):
        from planet_ruler.benchmarks.vet_images import _save_csv

        # r_km=None → fitted_radius_km field should be "" not "None"
        out = tmp_path / "out.csv"
        _save_csv([_make_vet_result(r_km=None)], out)
        with open(out) as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert row["fitted_radius_km"] == ""

    def test_creates_parent_directories(self, tmp_path):
        from planet_ruler.benchmarks.vet_images import _save_csv

        out = tmp_path / "subdir" / "nested" / "out.csv"
        _save_csv([_make_vet_result()], out)
        assert out.exists()


# ---------------------------------------------------------------------------
# vet_images — _print_results
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestVetImagesPrintResults:
    def test_does_not_raise(self, capsys):
        from planet_ruler.benchmarks.vet_images import _print_results

        results = [
            _make_vet_result("img1", verdict="PASS"),
            _make_vet_result("img2", verdict="WARN"),
            _make_vet_result("img3", verdict="FAIL"),
        ]
        _print_results(results)
        captured = capsys.readouterr()
        assert "PASS" in captured.out or "WARN" in captured.out

    def test_empty_list_does_not_raise(self, capsys):
        from planet_ruler.benchmarks.vet_images import _print_results

        _print_results([])


# ---------------------------------------------------------------------------
# BenchmarkRunner — _make_grid_scenario_name (additional cases)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (
        Path(__file__).parent.parent
        / "planet_ruler/benchmarks/configs/smoke_test.yaml"
    ).exists(),
    reason="smoke_test.yaml not found",
)
class TestRunnerMakeGridScenarioNameExtra:
    def _runner(self, tmp_path):
        from planet_ruler.benchmarks.runner import BenchmarkRunner

        config_path = (
            Path(__file__).parent.parent
            / "planet_ruler/benchmarks/configs/smoke_test.yaml"
        )
        return BenchmarkRunner(config_path, db_path=tmp_path / "test.db")

    def test_no_fit_stages_no_cr(self, tmp_path):
        runner = self._runner(tmp_path)
        params = {
            "minimizer": "basinhopping",
            "minimizer_preset": "fast",
            "free_parameters": ["r"],
            "r_limits_km": [2000, 14000],
            "fit_params": {"max_iter": 100},
        }
        name = runner._make_grid_scenario_name(params)
        assert "cr" not in name

    def test_minimizer_code_in_name(self, tmp_path):
        runner = self._runner(tmp_path)
        params = {
            "minimizer": "basinhopping",
            "minimizer_preset": "fast",
            "free_parameters": ["r"],
            "r_limits_km": [2000, 14000],
            "fit_params": {},
        }
        name = runner._make_grid_scenario_name(params)
        assert "bh" in name

    def test_free_param_code_r_only(self, tmp_path):
        runner = self._runner(tmp_path)
        params = {
            "minimizer": "basinhopping",
            "minimizer_preset": "fast",
            "free_parameters": ["r"],
            "r_limits_km": [2000, 14000],
            "fit_params": {},
        }
        name = runner._make_grid_scenario_name(params)
        # _fp_code(["r"]) → "r" (no h, no angles, no w/f)
        # The name segment should contain the fp code
        assert "_r_" in name or name.endswith("_r")

    def test_free_param_code_with_h(self, tmp_path):
        runner = self._runner(tmp_path)
        params_r = {
            "minimizer": "basinhopping",
            "minimizer_preset": "fast",
            "free_parameters": ["r"],
            "r_limits_km": [2000, 14000],
            "fit_params": {},
        }
        params_rh = {**params_r, "free_parameters": ["r", "h"]}
        name_r = runner._make_grid_scenario_name(params_r)
        name_rh = runner._make_grid_scenario_name(params_rh)
        assert name_r != name_rh
