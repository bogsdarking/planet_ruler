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
Performance smoke tests for CI regression detection.

These tests run a small subset of benchmarks to catch performance regressions
before they reach production. They use the benchmark infrastructure but with
strict thresholds.
"""

import sqlite3
from pathlib import Path

import pytest
import yaml

from planet_ruler.benchmarks.runner import BenchmarkRunner


@pytest.fixture
def smoke_config_path():
    """Path to smoke test configuration."""
    return (
        Path(__file__).parent.parent
        / "planet_ruler"
        / "benchmarks"
        / "configs"
        / "smoke_test.yaml"
    )


@pytest.fixture
def smoke_thresholds(smoke_config_path):
    """Load acceptance thresholds from config."""
    with open(smoke_config_path) as f:
        config = yaml.safe_load(f)
    return config.get("thresholds", {})


@pytest.fixture
def temp_db(tmp_path):
    """Ephemeral DB; tests never write to the production benchmark DB."""
    return tmp_path / "smoke_test.db"


@pytest.mark.slow
@pytest.mark.benchmark
def test_performance_smoke_runtime(
    smoke_config_path, smoke_thresholds, temp_db
):
    """
    Test that all smoke test scenarios complete within time threshold.

    This is a critical performance regression test for CI.
    """
    max_time = smoke_thresholds.get("max_time", 15.0)

    runner = BenchmarkRunner(smoke_config_path, db_path=temp_db)
    results = runner.run(parallel=False)

    failures = []
    for result in results:
        if result.total_time > max_time:
            failures.append(
                f"{result.scenario_name} on {result.image_name}: "
                f"{result.total_time:.2f}s > {max_time}s"
            )

    assert not failures, (
        f"Performance regression detected - {len(failures)} scenarios "
        f"exceeded {max_time}s threshold:\n" + "\n".join(failures)
    )


@pytest.mark.slow
@pytest.mark.benchmark
def test_performance_smoke_accuracy(
    smoke_config_path, smoke_thresholds, temp_db
):
    """
    Test that all smoke test scenarios achieve acceptable accuracy.

    Ensures optimizations don't sacrifice measurement quality.
    """
    max_error = smoke_thresholds.get("max_relative_error", 0.20)

    runner = BenchmarkRunner(smoke_config_path, db_path=temp_db)
    results = runner.run(parallel=False)

    successful = [r for r in results if r.convergence_status == "success"]

    failures = []
    for result in successful:
        if result.relative_error is not None and result.relative_error > max_error:
            failures.append(
                f"{result.scenario_name} on {result.image_name}: "
                f"{result.relative_error*100:.1f}% error > {max_error*100:.0f}%"
            )

    assert not failures, (
        f"Accuracy regression detected - {len(failures)} scenarios exceeded "
        f"{max_error*100:.0f}% error threshold:\n" + "\n".join(failures)
    )


@pytest.mark.slow
@pytest.mark.benchmark
def test_performance_smoke_convergence(
    smoke_config_path, smoke_thresholds, temp_db
):
    """
    Test that smoke test scenarios converge reliably.

    Ensures optimization stability.
    """
    min_success_rate = smoke_thresholds.get("min_success_rate", 0.90)

    runner = BenchmarkRunner(smoke_config_path, db_path=temp_db)
    results = runner.run(parallel=False)

    total = len(results)
    successful = sum(1 for r in results if r.convergence_status == "success")
    success_rate = successful / total if total > 0 else 0

    assert success_rate >= min_success_rate, (
        f"Convergence regression detected: {success_rate:.1%} success rate "
        f"< {min_success_rate:.0%} threshold ({successful}/{total} succeeded)"
    )


@pytest.mark.slow
@pytest.mark.benchmark
def test_benchmark_runner_initialization(smoke_config_path, temp_db):
    """Test that benchmark runner initializes correctly."""
    runner = BenchmarkRunner(smoke_config_path, db_path=temp_db)

    assert runner.config_path.exists()
    assert "scenarios" in runner.config
    assert runner.db_path.parent.exists()


@pytest.mark.slow
@pytest.mark.benchmark
def test_benchmark_database_creation(smoke_config_path, tmp_path):
    """Test that benchmark database is created with correct schema."""
    db_path = tmp_path / "test_benchmark.db"
    BenchmarkRunner(smoke_config_path, db_path=db_path)

    assert db_path.exists()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
        " AND name='benchmark_results'"
    )
    assert cursor.fetchone() is not None

    cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
    indexes = [row[0] for row in cursor.fetchall()]
    assert "idx_scenario_image" in indexes
    assert "idx_timestamp" in indexes

    conn.close()
