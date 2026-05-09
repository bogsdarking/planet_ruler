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

"""Tests for benchmark CLI entry points: run_benchmarks and analyze_benchmarks."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from planet_ruler.benchmarks.run_benchmarks import main as run_main

SMOKE_CONFIG = (
    Path(__file__).parent.parent / "planet_ruler/benchmarks/configs/smoke_test.yaml"
)

_HAS_SMOKE = SMOKE_CONFIG.exists()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_result(status="success", total_time=5.0, rel_error=0.05):
    r = MagicMock()
    r.convergence_status = status
    r.total_time = total_time
    r.relative_error = rel_error
    return r


def _results_df(**overrides):
    """Minimal DataFrame matching the columns analyzed by analyze_benchmarks."""
    data = {
        "scenario_name": ["s1", "s1", "s2"],
        "image_name": ["img1", "img2", "img1"],
        "convergence_status": ["success", "success", "success"],
        "total_time": [5.0, 6.0, 4.0],
        "relative_error": [0.05, 0.08, 0.03],
        "image_load_time": [0.1, 0.1, 0.1],
        "detection_time": [0.1, 0.1, 0.1],
        "optimization_time": [4.8, 5.8, 3.8],
        "iterations": [100, 120, 80],
        "timestamp": ["2025-01-01", "2025-01-02", "2025-01-03"],
    }
    data.update(overrides)
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# run_benchmarks.main()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRunBenchmarksMain:
    def test_error_when_config_not_found(self, monkeypatch, tmp_path):
        missing = tmp_path / "does_not_exist.yaml"
        monkeypatch.setattr(sys, "argv", ["rb", str(missing)])
        with pytest.raises(SystemExit) as exc:
            run_main()
        assert exc.value.code == 1

    @pytest.mark.skipif(not _HAS_SMOKE, reason="smoke_test.yaml not found")
    def test_runner_init_error_exits(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["rb", str(SMOKE_CONFIG)])
        with patch(
            "planet_ruler.benchmarks.run_benchmarks.BenchmarkRunner",
            side_effect=Exception("init failed"),
        ):
            with pytest.raises(SystemExit) as exc:
                run_main()
        assert exc.value.code == 1

    @pytest.mark.skipif(not _HAS_SMOKE, reason="smoke_test.yaml not found")
    def test_quiet_suppresses_header(self, monkeypatch, tmp_path, capsys):
        db = tmp_path / "test.db"
        monkeypatch.setattr(
            sys, "argv", ["rb", str(SMOKE_CONFIG), "--db", str(db), "--quiet"]
        )
        mock_runner = MagicMock()
        mock_runner.db_path = db
        mock_runner.git_commit = "abc123"
        mock_runner.run.return_value = [_mock_result()]
        with patch(
            "planet_ruler.benchmarks.run_benchmarks.BenchmarkRunner",
            return_value=mock_runner,
        ):
            run_main()
        assert "Benchmark Suite" not in capsys.readouterr().out

    @pytest.mark.skipif(not _HAS_SMOKE, reason="smoke_test.yaml not found")
    def test_summary_printed_with_results(self, monkeypatch, tmp_path, capsys):
        db = tmp_path / "test.db"
        monkeypatch.setattr(sys, "argv", ["rb", str(SMOKE_CONFIG), "--db", str(db)])
        mock_runner = MagicMock()
        mock_runner.db_path = db
        mock_runner.git_commit = "abc123"
        mock_runner.run.return_value = [_mock_result(), _mock_result()]
        with patch(
            "planet_ruler.benchmarks.run_benchmarks.BenchmarkRunner",
            return_value=mock_runner,
        ):
            run_main()
        out = capsys.readouterr().out
        assert "Benchmark Summary" in out
        assert "Total runs: 2" in out

    @pytest.mark.skipif(not _HAS_SMOKE, reason="smoke_test.yaml not found")
    def test_overwrite_calls_init_database(self, monkeypatch, tmp_path):
        db = tmp_path / "test.db"
        db.write_text("placeholder")  # must exist for overwrite branch
        monkeypatch.setattr(
            sys,
            "argv",
            ["rb", str(SMOKE_CONFIG), "--db", str(db), "--overwrite", "--quiet"],
        )
        mock_runner = MagicMock()
        mock_runner.db_path = db
        mock_runner.git_commit = "abc123"
        mock_runner.run.return_value = []
        with patch(
            "planet_ruler.benchmarks.run_benchmarks.BenchmarkRunner",
            return_value=mock_runner,
        ):
            run_main()
        mock_runner._init_database.assert_called_once()

    @pytest.mark.skipif(not _HAS_SMOKE, reason="smoke_test.yaml not found")
    def test_run_error_exits(self, monkeypatch, tmp_path):
        db = tmp_path / "test.db"
        monkeypatch.setattr(sys, "argv", ["rb", str(SMOKE_CONFIG), "--db", str(db)])
        mock_runner = MagicMock()
        mock_runner.db_path = db
        mock_runner.git_commit = "abc123"
        mock_runner.run.side_effect = RuntimeError("run failed")
        with patch(
            "planet_ruler.benchmarks.run_benchmarks.BenchmarkRunner",
            return_value=mock_runner,
        ):
            with pytest.raises(SystemExit) as exc:
                run_main()
        assert exc.value.code == 1

    @pytest.mark.skipif(not _HAS_SMOKE, reason="smoke_test.yaml not found")
    def test_scenario_filter_passed_to_runner(self, monkeypatch, tmp_path):
        db = tmp_path / "test.db"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "rb",
                str(SMOKE_CONFIG),
                "--db",
                str(db),
                "--scenario",
                "s1",
                "--scenario",
                "s2",
                "--quiet",
            ],
        )
        mock_runner = MagicMock()
        mock_runner.db_path = db
        mock_runner.git_commit = "abc123"
        mock_runner.run.return_value = []
        with patch(
            "planet_ruler.benchmarks.run_benchmarks.BenchmarkRunner",
            return_value=mock_runner,
        ):
            run_main()
        kwargs = mock_runner.run.call_args[1]
        assert kwargs["scenarios"] == ["s1", "s2"]

    @pytest.mark.skipif(not _HAS_SMOKE, reason="smoke_test.yaml not found")
    def test_workers_implies_parallel(self, monkeypatch, tmp_path):
        db = tmp_path / "test.db"
        monkeypatch.setattr(
            sys,
            "argv",
            ["rb", str(SMOKE_CONFIG), "--db", str(db), "--workers", "2", "--quiet"],
        )
        mock_runner = MagicMock()
        mock_runner.db_path = db
        mock_runner.git_commit = "abc123"
        mock_runner.run.return_value = []
        with patch(
            "planet_ruler.benchmarks.run_benchmarks.BenchmarkRunner",
            return_value=mock_runner,
        ):
            run_main()
        kwargs = mock_runner.run.call_args[1]
        assert kwargs["parallel"] is True

    @pytest.mark.skipif(not _HAS_SMOKE, reason="smoke_test.yaml not found")
    def test_no_skip_passed_to_runner(self, monkeypatch, tmp_path):
        db = tmp_path / "test.db"
        monkeypatch.setattr(
            sys,
            "argv",
            ["rb", str(SMOKE_CONFIG), "--db", str(db), "--no-skip", "--quiet"],
        )
        mock_runner = MagicMock()
        mock_runner.db_path = db
        mock_runner.git_commit = "abc123"
        mock_runner.run.return_value = []
        with patch(
            "planet_ruler.benchmarks.run_benchmarks.BenchmarkRunner",
            return_value=mock_runner,
        ):
            run_main()
        kwargs = mock_runner.run.call_args[1]
        assert kwargs["skip_completed"] is False

    @pytest.mark.skipif(not _HAS_SMOKE, reason="smoke_test.yaml not found")
    def test_verbose_shows_config_path(self, monkeypatch, tmp_path, capsys):
        db = tmp_path / "test.db"
        monkeypatch.setattr(sys, "argv", ["rb", str(SMOKE_CONFIG), "--db", str(db)])
        mock_runner = MagicMock()
        mock_runner.db_path = db
        mock_runner.git_commit = None
        mock_runner.run.return_value = [_mock_result()]
        with patch(
            "planet_ruler.benchmarks.run_benchmarks.BenchmarkRunner",
            return_value=mock_runner,
        ):
            run_main()
        out = capsys.readouterr().out
        assert str(SMOKE_CONFIG) in out


# ---------------------------------------------------------------------------
# analyze_benchmarks — non-plotting functions
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAnalyzeBenchmarksLoadData:
    def test_returns_dataframe(self, capsys):
        from planet_ruler.benchmarks.analyze_benchmarks import load_data

        mock_analyzer = MagicMock()
        mock_analyzer.get_results.return_value = _results_df()
        result = load_data(mock_analyzer)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_prints_total_runs(self, capsys):
        from planet_ruler.benchmarks.analyze_benchmarks import load_data

        mock_analyzer = MagicMock()
        mock_analyzer.get_results.return_value = _results_df()
        load_data(mock_analyzer)
        assert "3" in capsys.readouterr().out


@pytest.mark.unit
class TestAnalyzeBenchmarksShowSummaryStats:
    def test_does_not_raise(self, capsys):
        from planet_ruler.benchmarks.analyze_benchmarks import show_summary_stats

        mock_analyzer = MagicMock()
        mock_analyzer.get_summary_stats.return_value = pd.DataFrame(
            {"scenario_name": ["s1"], "n_runs": [3], "mean_time": [5.0]}
        )
        show_summary_stats(mock_analyzer)
        assert "SUMMARY" in capsys.readouterr().out


@pytest.mark.unit
class TestAnalyzeBenchmarksShowParetoFrontier:
    def test_does_not_raise(self, capsys):
        from planet_ruler.benchmarks.analyze_benchmarks import show_pareto_frontier

        mock_analyzer = MagicMock()
        # get_pareto_frontier() returns a mock; indexing it with a list also returns a mock
        show_pareto_frontier(mock_analyzer)
        assert "PARETO" in capsys.readouterr().out


@pytest.mark.unit
class TestAnalyzeBenchmarksIdentifyBottlenecks:
    def test_bottleneck_printed(self, capsys):
        from planet_ruler.benchmarks.analyze_benchmarks import identify_bottlenecks

        df = _results_df()
        mock_analyzer = MagicMock()
        mock_analyzer.get_results.return_value = df
        mock_analyzer.identify_bottlenecks.return_value = {"optimization": 0.95}
        identify_bottlenecks(mock_analyzer)
        out = capsys.readouterr().out
        assert "optimization" in out

    def test_no_bottleneck_no_phase_printed(self, capsys):
        from planet_ruler.benchmarks.analyze_benchmarks import identify_bottlenecks

        df = _results_df()
        mock_analyzer = MagicMock()
        mock_analyzer.get_results.return_value = df
        mock_analyzer.identify_bottlenecks.return_value = {}
        identify_bottlenecks(mock_analyzer)
        out = capsys.readouterr().out
        assert "optimization" not in out


@pytest.mark.unit
class TestSetupPlotting:
    def test_does_not_raise(self):
        from planet_ruler.benchmarks.analyze_benchmarks import setup_plotting

        setup_plotting()


# ---------------------------------------------------------------------------
# analyze_benchmarks — plotting functions (plt.show patched)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestAnalyzeBenchmarksPlotting:
    """Smoke tests for plotting functions — just verify they run without error."""

    def test_plot_performance_vs_accuracy(self):
        from planet_ruler.benchmarks.analyze_benchmarks import (
            plot_performance_vs_accuracy,
        )

        df = _results_df()
        with patch("planet_ruler.benchmarks.analyze_benchmarks.plt.show"):
            plot_performance_vs_accuracy(df)

    def test_plot_performance_vs_accuracy_saves_file(self, tmp_path):
        from planet_ruler.benchmarks.analyze_benchmarks import (
            plot_performance_vs_accuracy,
        )

        df = _results_df()
        with patch("planet_ruler.benchmarks.analyze_benchmarks.plt.show"):
            plot_performance_vs_accuracy(df, output_dir=tmp_path)
        assert (tmp_path / "performance_vs_accuracy.png").exists()

    def test_plot_runtime_distribution(self):
        from planet_ruler.benchmarks.analyze_benchmarks import plot_runtime_distribution

        df = _results_df()
        with patch("planet_ruler.benchmarks.analyze_benchmarks.plt.show"):
            plot_runtime_distribution(df)

    def test_plot_runtime_distribution_saves_file(self, tmp_path):
        from planet_ruler.benchmarks.analyze_benchmarks import plot_runtime_distribution

        df = _results_df()
        with patch("planet_ruler.benchmarks.analyze_benchmarks.plt.show"):
            plot_runtime_distribution(df, output_dir=tmp_path)
        assert (tmp_path / "runtime_distribution.png").exists()

    def test_plot_timing_breakdown(self):
        from planet_ruler.benchmarks.analyze_benchmarks import plot_timing_breakdown

        df = _results_df()
        with patch("planet_ruler.benchmarks.analyze_benchmarks.plt.show"):
            plot_timing_breakdown(df)

    def test_plot_timing_breakdown_saves_file(self, tmp_path):
        from planet_ruler.benchmarks.analyze_benchmarks import plot_timing_breakdown

        df = _results_df()
        with patch("planet_ruler.benchmarks.analyze_benchmarks.plt.show"):
            plot_timing_breakdown(df, output_dir=tmp_path)
        assert (tmp_path / "timing_breakdown.png").exists()


# ---------------------------------------------------------------------------
# analyze_benchmarks.main()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAnalyzeBenchmarksMain:
    def _mock_analyzer(self, df=None):
        if df is None:
            df = _results_df()
        mock_az = MagicMock()
        mock_az.get_results.return_value = df
        mock_az.get_summary_stats.return_value = pd.DataFrame(
            {"scenario_name": ["s1"], "n_runs": [3], "mean_time": [5.0]}
        )
        return mock_az

    def test_empty_db_exits_early(self, monkeypatch, tmp_path, capsys):
        from planet_ruler.benchmarks.analyze_benchmarks import main as analyze_main

        monkeypatch.setattr(sys, "argv", ["analyze", "--db", str(tmp_path / "fake.db")])
        mock_az = MagicMock()
        mock_az.get_results.return_value = _results_df().iloc[
            0:0
        ]  # empty, keeps columns
        with patch(
            "planet_ruler.benchmarks.analyze_benchmarks.BenchmarkAnalyzer",
            return_value=mock_az,
        ):
            analyze_main()
        assert "No benchmark results" in capsys.readouterr().out

    def test_main_with_results_prints_summary(self, monkeypatch, tmp_path, capsys):
        from planet_ruler.benchmarks.analyze_benchmarks import main as analyze_main

        monkeypatch.setattr(sys, "argv", ["analyze", "--db", str(tmp_path / "fake.db")])
        mock_az = self._mock_analyzer()
        with (
            patch(
                "planet_ruler.benchmarks.analyze_benchmarks.BenchmarkAnalyzer",
                return_value=mock_az,
            ),
            patch("planet_ruler.benchmarks.analyze_benchmarks.plt.show"),
        ):
            analyze_main()
        out = capsys.readouterr().out
        assert "SUMMARY" in out

    def test_main_export_csv(self, monkeypatch, tmp_path, capsys):
        from planet_ruler.benchmarks.analyze_benchmarks import main as analyze_main

        csv_out = tmp_path / "out.csv"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "analyze",
                "--db",
                str(tmp_path / "fake.db"),
                "--export-csv",
                str(csv_out),
            ],
        )
        mock_az = self._mock_analyzer()
        with (
            patch(
                "planet_ruler.benchmarks.analyze_benchmarks.BenchmarkAnalyzer",
                return_value=mock_az,
            ),
            patch("planet_ruler.benchmarks.analyze_benchmarks.plt.show"),
        ):
            analyze_main()
        mock_az.export_csv.assert_called_once_with(csv_out)
