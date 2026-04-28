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

import json
import math
import sqlite3
import tempfile
from pathlib import Path

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

    def test_vet_image_fix_camera(self):
        """--fix-camera removes f/w from search space without error."""
        from planet_ruler.benchmarks.vet_images import vet_image

        result = vet_image(
            SYNTH_IMAGE,
            SYNTH_ANNOTATION,
            max_iter=10,
            minimizer_preset="fast",
            fix_camera=True,
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
