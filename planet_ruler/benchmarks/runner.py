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
Benchmark runner for Planet Ruler performance testing.

This module provides infrastructure for running systematic performance benchmarks
across multiple scenarios, images, and parameter configurations.
"""

import itertools
import json
import os
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import warnings

import numpy as np
import yaml

# Import planet_ruler components
from planet_ruler.image import load_image


@dataclass
class BenchmarkResult:
    """
    Container for benchmark run results.

    Attributes:
        scenario_name (str): Name of benchmark scenario.
        image_name (str): Name of test image.
        timestamp (str): ISO format timestamp.
        git_commit (str): Git commit hash.
        detection_method (str): Limb detection method used.
        fit_params (dict): Fit parameters (max_iter, resolution_stages, etc.).
        free_parameters (list): List of free parameter names from fit.
        init_parameter_values (dict): Initial values from fit_config.
        parameter_limits (dict): Bounds from fit_config.
        minimizer_config (dict): Minimizer configuration.
        image_width (int): Image width in pixels.
        image_height (int): Image height in pixels.
        altitude (float): GPS altitude if available.
        planet_name (str): Target planet name.
        expected_radius (float): Expected planet radius in meters.
        uncertainty_radius (float): Acceptable error in meters.
        best_parameters (dict): Fitted parameter values {r, h, f, ...}.
        fitted_radius (float): Extracted radius from best_parameters['r'].
        convergence_status (str): Convergence status string.
        iterations (int): Number of iterations used.
        absolute_error (float): Absolute error in meters.
        relative_error (float): Relative error (fraction).
        within_uncertainty (bool): Whether error is within acceptable range.
        total_time (float): Total runtime in seconds.
        image_load_time (float): Image loading time in seconds.
        detection_time (float): Detection time in seconds.
        optimization_time (float): Optimization time in seconds.
        timing_details (str): JSON string with detailed timing breakdown.
        error_message (str): Error message if run failed.
    """

    # Identifiers
    scenario_name: str
    image_name: str
    timestamp: str
    git_commit: Optional[str]

    # Configuration (from fit_config)
    detection_method: str
    fit_params: Dict[str, Any]  # max_iter, resolution_stages, etc.
    free_parameters: List[str]  # Actual free parameter names from fit
    init_parameter_values: Dict[str, float]  # Initial values from fit_config
    parameter_limits: Dict[str, List[float]]  # Bounds from fit_config
    minimizer_config: Dict[str, Any]
    minimizer_preset: Optional[str]
    constrain_radius_n_sigma: Optional[float]
    perturbation_factors: Optional[str]

    # Image metadata
    image_width: int
    image_height: int
    altitude: Optional[float]
    planet_name: str
    expected_radius: float
    uncertainty_radius: float

    # Fit results (actual fitted parameters)
    best_parameters: Dict[
        str, float
    ]  # Actual fitted values {r, h, f, theta_x, ...}
    convergence_status: str
    iterations: Optional[int]

    # Accuracy (computed from best_parameters['r'])
    fitted_radius: Optional[float]  # Extracted from best_parameters['r']
    absolute_error: Optional[float]
    relative_error: Optional[float]
    within_uncertainty: Optional[bool]

    # Timing breakdown (seconds)
    total_time: float
    image_load_time: float
    detection_time: float
    optimization_time: float

    # Detailed timing (JSON string for flexibility)
    timing_details: Optional[str]

    # Error info
    error_message: Optional[str]


def _run_batch_worker(task: tuple) -> List["BenchmarkResult"]:
    """
    Module-level worker for ProcessPoolExecutor (must be picklable).

    Constructs a minimal BenchmarkRunner context from ``task`` without
    going through __init__ (which would re-expand the full config).
    Writes results to DB directly so the main process stays lean.
    """
    benchmark_dir, db_path, scenario, image_name, git_commit = task

    runner = object.__new__(BenchmarkRunner)
    runner.benchmark_dir = Path(benchmark_dir)
    runner.db_path = Path(db_path)
    runner.git_commit = git_commit

    try:
        batch = runner._run_with_augmentation(scenario, image_name)
        for result in batch:
            runner._store_result(result)
        return batch
    except Exception as exc:
        print(
            f"[worker error] {scenario.get('name')} / {image_name}: {exc}",
            flush=True,
        )
        return []


class BenchmarkRunner:
    """
    Execute and manage performance benchmarks for Planet Ruler.

    Runs benchmark scenarios from YAML configuration, measuring performance
    metrics like runtime, accuracy, and convergence. Stores results in SQLite
    for historical tracking and analysis.

    Args:
        config_path (str or Path): Path to YAML configuration file.
        db_path (str or Path, optional): Path to SQLite database for results.
            Defaults to benchmarks/results/benchmark_results.db.

    Attributes:
        config_path (Path): Path to configuration file.
        config (dict): Loaded configuration dictionary.
        benchmark_dir (Path): Base directory for benchmark files.
        db_path (Path): Path to SQLite database.
        git_commit (str): Current git commit hash.
    """

    def __init__(
        self,
        config_path: Union[str, Path],
        db_path: Optional[Union[str, Path]] = None,
    ):
        self.config_path = Path(config_path)
        self.config = self._load_config()

        # Determine base directory for benchmarks
        # Benchmarks are a dev/contributor workflow -- data lives in the repo,
        # not in the installed package. Search for repo working directory first.
        module_bench_dir = Path(__file__).parent

        # Walk up from CWD looking for a repo root (has planet_ruler/benchmarks/images)
        def _find_repo_bench_dir(start: Path) -> Optional[Path]:
            for parent in [start, *start.parents]:
                candidate = parent / "planet_ruler" / "benchmarks"
                if (candidate / "images").exists():
                    return candidate
            return None

        repo_bench_dir = _find_repo_bench_dir(Path.cwd())

        if repo_bench_dir is not None:
            self.benchmark_dir = repo_bench_dir
        else:
            # Fall back to module location -- will produce a clear error later
            # if data isn't present (e.g. running from a pip install without the repo)
            self.benchmark_dir = module_bench_dir
            if not (self.benchmark_dir / "images").exists():
                warnings.warn(
                    "Benchmark data directory not found. Benchmarks must be run from "
                    "within the planet_ruler git repository after `git lfs pull`. "
                    f"Expected images at: {self.benchmark_dir / 'images'}"
                )

        if db_path is None:
            db_path = self.benchmark_dir / "results" / "benchmark_results.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_database()
        self.git_commit = self._get_git_commit()

    def _load_config(self) -> Dict[str, Any]:
        """Load and validate YAML configuration."""
        with open(self.config_path) as f:
            config = yaml.safe_load(f)

        if "scenarios" not in config and "grid" not in config:
            raise ValueError("Config must contain 'scenarios' or 'grid' key")
        if "scenarios" not in config:
            config["scenarios"] = []

        return config

    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        import subprocess

        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                cwd=Path(__file__).parent.parent,
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    def _get_completed_keys(self) -> set:
        """Return set of (scenario_name, image_name) tuples already in DB."""
        conn = sqlite3.connect(self.db_path, timeout=30)
        cur = conn.cursor()
        cur.execute("SELECT scenario_name, image_name FROM benchmark_results")
        keys = {(row[0], row[1]) for row in cur.fetchall()}
        conn.close()
        return keys

    def _expected_names(self, scenario: Dict[str, Any]) -> List[str]:
        """
        Scenario_name values expected for one (scenario, image) batch.

        Includes the base name plus any augmentation variant names.
        """
        base = scenario["name"]
        aug = scenario.get("augmentation")
        if not aug or scenario.get("detection_method", "manual") != "manual":
            return [base]
        n_variants = int(aug.get("n_variants", aug.get("n_noisy_variants", 0)))
        return [base] + [f"{base}__aug{i + 1:02d}" for i in range(n_variants)]

    def _init_database(self):
        """Initialize SQLite database with results table."""
        conn = sqlite3.connect(self.db_path)
        # WAL mode allows concurrent readers + serialised writers without
        # blocking — required for parallel benchmark workers.
        conn.execute("PRAGMA journal_mode=WAL")
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS benchmark_results (
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
                constrain_radius_n_sigma REAL,
                perturbation_factors TEXT,

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

        # Create indexes for common queries
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_scenario_image
            ON benchmark_results(scenario_name, image_name)
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON benchmark_results(timestamp)
        """
        )

        conn.commit()
        conn.close()

    # ------------------------------------------------------------------
    # Top-level grid expansion (GridSearchCV-style ``grid:`` section)
    # ------------------------------------------------------------------

    # Abbreviations used in generated scenario names
    _PRESET_ABBR = {
        "fast": "fast",
        "balanced": "bal",
        "planet-ruler": "pr",
        "robust": "rob",
        "scipy-default": "sdf",
    }
    _MINIMIZER_ABBR = {
        "differential-evolution": "de",
        "dual-annealing": "da",
        "basinhopping": "bh",
    }

    @staticmethod
    def _fp_code(free_params: list) -> str:
        """Short code for a free_parameters list."""
        fp = set(free_params)
        has_angles = {"theta_x", "theta_y", "theta_z"}.issubset(fp)
        code = "r"
        if "h" in fp:
            code += "h"
        if has_angles:
            code += "a"
        if "w" in fp:
            code += "w"
        if "f" in fp:
            code += "f"
        return code

    @staticmethod
    def _rl_code(r_limits_km: list) -> str:
        """Short code for r_limits_km relative to Earth radius."""
        lo, hi = r_limits_km
        center = 6371.0
        pct = round(max(hi - center, center - lo) / center * 100)
        if pct <= 6:
            return "r05"
        if pct <= 11:
            return "r10"
        if pct <= 22:
            return "r20"
        if pct <= 52:
            return "r50"
        return "rck"

    def _make_grid_scenario_name(self, params: dict) -> str:
        """Compact, human-readable name encoding all grid axes."""
        if params.get("constrain_radius_only"):
            fp = self._fp_code(params.get("free_parameters", ["r"]))
            pf_val = params.get("perturbation_factor", 1.0)
            h_pf = (
                float(pf_val.get("h", 1.0))
                if isinstance(pf_val, dict)
                else float(pf_val)
            )
            phf = f"ph{int(h_pf * 100):03d}"
            return f"g_cr_only_{fp}_{phf}"

        m = self._MINIMIZER_ABBR.get(params.get("minimizer", ""), "xx")
        p = self._PRESET_ABBR.get(params.get("minimizer_preset", ""), "xx")
        fp = self._fp_code(params.get("free_parameters", ["r"]))
        cr_n = params.get("constrain_radius_n_sigma")
        if cr_n is not None:
            rl = f"cr{int(float(cr_n) * 10):02d}"  # cr10=1σ, cr30=3σ, cr50=5σ
        else:
            rl = self._rl_code(params.get("r_limits_km", [1000, 100000]))
        h_pct = params.get("h_limits_pct")
        hl = f"h{int(h_pct * 100):02d}" if h_pct else "hw"
        pf_val = params.get("perturbation_factor", 1.0)
        if isinstance(pf_val, dict):
            r_pf = float(pf_val.get("r", pf_val.get("default", 1.0)))
            h_pf = float(pf_val.get("h", 1.0))
        else:
            r_pf = float(pf_val)
            h_pf = float(pf_val)
        pf = f"p{int(r_pf * 100):03d}"
        phf = f"ph{int(h_pf * 100):03d}"
        it = params.get("fit_params", {}).get("max_iter", 1000)
        it_s = f"i{it}" if it < 1000 else f"i{it // 1000}k"
        return f"g_{m}_{p}_{fp}_{rl}_{hl}_{pf}_{phf}_{it_s}"

    def _expand_top_level_grid(self) -> List[Dict[str, Any]]:
        """
        Expand the top-level ``grid:`` config section into scenario dicts.

        Within ``param_grid``, every list-valued key is a sweep dimension
        (cross-product); scalar values are fixed for that sub-grid. Multiple
        dicts in the list are unioned (identical to sklearn param_grid).

        Special keys:
          ``images`` -- image stems to run each config on.
          ``fixed`` -- keys applied to every generated scenario.
          ``annotation_file_pattern`` -- template using ``{image}``.
          ``fit_params_max_iter`` -- shorthand for fit_params.max_iter.
          ``h_limits_pct`` -- ±fraction of GPS h; applied only when h free.
        """
        grid_cfg = self.config.get("grid")
        if not grid_cfg:
            return []

        images = grid_cfg.get("images", [])
        param_grid = grid_cfg.get("param_grid", [])
        fixed = grid_cfg.get("fixed", {})
        ann_pattern = grid_cfg.get(
            "annotation_file_pattern",
            "{image}_limb_points.json",
        )

        scenarios = []
        for spec in param_grid:
            sweep: Dict[str, list] = {}
            scalar: Dict[str, Any] = {}
            for k, v in spec.items():
                if isinstance(v, list):
                    sweep[k] = v
                else:
                    scalar[k] = v

            all_dims = {"_image": images, **sweep}
            keys = list(all_dims.keys())
            for combo in itertools.product(*all_dims.values()):
                params = dict(zip(keys, combo))
                image = params.pop("_image")

                # Resolve free_parameters from combo or scalar fallback
                fp = params.get(
                    "free_parameters",
                    scalar.get("free_parameters", ["r"]),
                )
                # Skip h_limits_pct combos when h is not free
                h_pct = params.get("h_limits_pct")
                if h_pct is not None and "h" not in fp:
                    continue

                scenario = {**fixed, **scalar, **params}

                # Promote fit_params_max_iter shorthand
                if "fit_params_max_iter" in scenario:
                    scenario["fit_params"] = {
                        "max_iter": scenario.pop("fit_params_max_iter")
                    }

                # Drop null h_limits_pct (no override wanted)
                if scenario.get("h_limits_pct") is None:
                    scenario.pop("h_limits_pct", None)

                scenario["images"] = [image]
                scenario["annotation_file"] = ann_pattern.format(image=image)
                scenario["name"] = self._make_grid_scenario_name(scenario)
                scenarios.append(scenario)

        return scenarios

    def _expand_scenarios(self) -> List[Dict[str, Any]]:
        """Expand explicit scenarios list plus top-level grid section."""
        expanded = list(self.config.get("scenarios", []))
        expanded.extend(self._expand_top_level_grid())
        return expanded

    def run(
        self,
        parallel: bool = False,
        workers: int = 0,
        scenarios: Optional[List[str]] = None,
        images: Optional[List[str]] = None,
        skip_completed: bool = True,
    ) -> List[BenchmarkResult]:
        """
        Run benchmark suite.

        Args:
            parallel (bool): Run batches in parallel using worker processes.
            workers (int): Number of worker processes (0 = half of CPU count).
            scenarios (list of str, optional): Filter to specific scenario names.
            images (list of str, optional): Filter to specific image names.
            skip_completed (bool): Skip (scenario, image) pairs already in DB.

        Returns:
            list of BenchmarkResult: All benchmark results (empty in parallel
            mode to avoid large IPC overhead; query the DB for summaries).
        """
        expanded_scenarios = self._expand_scenarios()

        if scenarios is not None:
            expanded_scenarios = [
                s for s in expanded_scenarios if s["name"] in scenarios
            ]

        # Build work list, optionally skipping already-complete pairs
        completed_keys: set = set()
        if skip_completed:
            completed_keys = self._get_completed_keys()

        pending: List[tuple] = []  # (scenario, image_name)
        n_skipped = 0

        for scenario in expanded_scenarios:
            scenario_images = scenario.get("images", [])
            if images is not None:
                scenario_images = [
                    img for img in scenario_images if img in images
                ]
            for image_name in scenario_images:
                expected = self._expected_names(scenario)
                if skip_completed and all(
                    (name, image_name) in completed_keys for name in expected
                ):
                    n_skipped += 1
                    continue
                pending.append((scenario, image_name))

        if n_skipped:
            print(
                f"Skipping {n_skipped} already-complete (scenario, image) pairs."
            )
        print(f"Running {len(pending)} pending batches.")

        results: List[BenchmarkResult] = []

        if not parallel or len(pending) == 0:
            for scenario, image_name in pending:
                batch = self._run_with_augmentation(scenario, image_name)
                for result in batch:
                    results.append(result)
                    self._store_result(result)
        else:
            n_workers = workers if workers > 0 else max(1, os.cpu_count() // 2)
            print(f"Parallel mode: {n_workers} workers.")
            tasks = [
                (
                    str(self.benchmark_dir),
                    str(self.db_path),
                    scenario,
                    image_name,
                    self.git_commit,
                )
                for scenario, image_name in pending
            ]
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_run_batch_worker, t): t for t in tasks}
                for fut in as_completed(futures):
                    try:
                        batch = fut.result()
                        results.extend(batch)
                    except Exception as exc:
                        task = futures[fut]
                        print(
                            f"[error] {task[2].get('name')} / {task[3]}: {exc}",
                            flush=True,
                        )

        return results

    def _run_with_augmentation(
        self, scenario: Dict[str, Any], image_name: str
    ) -> List[BenchmarkResult]:
        """
        Run a scenario on one image, including augmented annotation variants.

        If the scenario contains an ``augmentation`` key, the base annotation is
        run first, then each noisy variant is run and stored with a suffixed
        scenario name (e.g. ``my_scenario__aug01``).

        Augmentation config (all optional)::

            augmentation:
              n_noisy_variants: 5   # number of noisy copies (default 0)
              noise_sigma: 4.0      # Gaussian σ in pixels (default 4.0)
              seed: 0               # RNG seed (default 0)

        Returns:
            List containing the base result and any augmentation results.
        """
        augmentation = scenario.get("augmentation")

        if (
            not augmentation
            or scenario.get("detection_method", "manual") != "manual"
        ):
            name = scenario["name"]
            print(f"Running: {name} on {image_name}")
            return [self._run_single(scenario, image_name)]

        from planet_ruler.benchmarks.augment_annotations import (
            generate_noisy_variants,
        )

        n_variants = int(
            augmentation.get(
                "n_variants", augmentation.get("n_noisy_variants", 0)
            )
        )
        noise_sigma = float(augmentation.get("noise_sigma", 4.0))
        seed = int(augmentation.get("seed", 0))

        # Run base case first (no override — loads annotation from file as usual)
        base_name = scenario["name"]
        print(f"Running: {base_name} on {image_name} (base)")
        base_result = self._run_single(scenario, image_name)
        batch = [base_result]

        if n_variants <= 0 or base_result.convergence_status == "error":
            return batch

        # Re-load the annotation points so we can generate variants
        annotation_file = scenario.get("annotation_file")
        if not annotation_file:
            return batch

        annot_path = self.benchmark_dir / "annotations" / annotation_file
        try:
            with open(annot_path) as f:
                annotation_data = json.load(f)
            if "limb_points" in annotation_data:
                points = annotation_data["limb_points"]["points"]
            else:
                points = annotation_data["points"]
        except Exception:
            return batch

        variants = generate_noisy_variants(
            points, n_variants=n_variants, noise_sigma=noise_sigma, seed=seed
        )

        image_width = base_result.image_width

        for i, var_points in enumerate(variants):
            # Build sparse limb_target from variant points
            limb_target = np.full(image_width, np.nan)
            for x, y in var_points:
                x_idx = int(round(x))
                if 0 <= x_idx < image_width:
                    limb_target[x_idx] = y

            aug_name = f"{base_name}__aug{i + 1:02d}"
            print(f"Running: {aug_name} on {image_name}")
            result = self._run_single(
                scenario,
                image_name,
                limb_target_override=limb_target,
                scenario_name_override=aug_name,
            )
            batch.append(result)

        return batch

    def _run_single(
        self,
        scenario: Dict[str, Any],
        image_name: str,
        limb_target_override: Optional[np.ndarray] = None,
        scenario_name_override: Optional[str] = None,
    ) -> BenchmarkResult:
        """
        Run a single benchmark scenario on one image.

        Args:
            scenario: Scenario configuration dict.
            image_name: Image stem (no extension).
            limb_target_override: If provided, skip annotation file loading and
                use this sparse NaN array as the limb target directly. Useful
                for augmented annotation variants.
            scenario_name_override: If provided, use this as the stored scenario
                name instead of ``scenario['name']``.
        """
        start_time = time.time()
        effective_name = scenario_name_override or scenario["name"]

        # Initialize result with metadata
        result = BenchmarkResult(
            scenario_name=effective_name,
            image_name=image_name,
            timestamp=datetime.now().isoformat(),
            git_commit=self.git_commit,
            detection_method=scenario.get("detection_method", "manual"),
            fit_params=scenario.get("fit_params", {}),
            free_parameters=[],  # Will be filled from actual fit
            init_parameter_values={},  # Will be filled from actual fit
            parameter_limits={},  # Will be filled from actual fit
            minimizer_config=scenario.get("minimizer", {}),
            minimizer_preset=scenario.get("minimizer_preset"),
            constrain_radius_n_sigma=scenario.get("constrain_radius_n_sigma"),
            perturbation_factors=json.dumps(
                scenario.get("perturbation_factor", 1.0)
                if isinstance(scenario.get("perturbation_factor"), dict)
                else {"r": float(scenario.get("perturbation_factor", 1.0)),
                      "h": float(scenario.get("perturbation_factor", 1.0))}
            ),
            image_width=0,
            image_height=0,
            altitude=None,
            planet_name=scenario.get("planet_name", "Earth"),
            expected_radius=scenario.get("expected_radius", 6371e3),
            uncertainty_radius=scenario.get("uncertainty_radius", 10e3),
            best_parameters={},  # Will be filled from actual fit results
            fitted_radius=None,
            convergence_status="not_started",
            iterations=None,
            absolute_error=None,
            relative_error=None,
            within_uncertainty=None,
            total_time=0.0,
            image_load_time=0.0,
            detection_time=0.0,
            optimization_time=0.0,
            timing_details=None,
            error_message=None,
        )

        try:
            # Get image path and load image
            t0 = time.time()
            image_path = self._get_image_path(image_name)
            img_array = load_image(str(image_path))
            result.image_load_time = time.time() - t0
            result.image_width = img_array.shape[1]
            result.image_height = img_array.shape[0]

            # Get altitude from EXIF if available
            from planet_ruler.camera import get_gps_altitude

            result.altitude = get_gps_altitude(str(image_path))

            # Build fit config from scenario parameters (uses EXIF auto-config by default)
            fit_config = self._build_fit_config(scenario, image_path)

            # Use detection method directly (must match LimbObservation valid methods)
            limb_detection = scenario.get("detection_method", "manual")

            # Get minimizer (default to differential-evolution)
            minimizer_method = scenario.get(
                "minimizer", "differential-evolution"
            )
            minimizer_preset = scenario.get("minimizer_preset")
            minimizer_kwargs = scenario.get("minimizer_kwargs")

            # Create observation
            from planet_ruler.observation import LimbObservation

            obs = LimbObservation(
                image_filepath=str(image_path),
                fit_config=fit_config,
                limb_detection=limb_detection,
                minimizer=minimizer_method,
            )

            # Detection phase
            detect_start = time.time()

            # For manual detection, load and register annotations
            if limb_detection == "manual":
                if limb_target_override is not None:
                    # Augmented variant: use pre-built sparse target directly
                    obs.register_limb(limb_target_override)
                else:
                    # Normal path: load annotation from file
                    annotation_file = scenario.get("annotation_file")
                    if not annotation_file:
                        raise ValueError(
                            "Manual detection requires explicit 'annotation_file' "
                            "parameter.\nAdd to scenario: annotation_file: "
                            "'pexels-claiton-17217951_exif.json'"
                        )

                    annot_path = (
                        self.benchmark_dir / "annotations" / annotation_file
                    )
                    if not annot_path.exists():
                        raise FileNotFoundError(
                            f"Annotation file not found: {annot_path}\n"
                            f"Create annotation with:\n"
                            f"  python -m planet_ruler.annotate {image_path} --output-dir {annot_path.parent}"
                        )

                    with open(annot_path) as f:
                        annotation_data = json.load(f)

                    # If annotation stores ground-truth params (synthetic data),
                    # use them to initialise fixed parameters to their true
                    # values so the fit is not corrupted by EXIF rounding or
                    # perturbed-r_init artifacts.  Must apply to obs directly
                    # since LimbObservation copied fit_config on __init__.
                    # Also update _original_init_parameter_values since
                    # fit_limb restores from it on a cold start.
                    ann_params = annotation_data.get("params")
                    if ann_params:
                        free = set(obs.free_parameters or [])
                        for param, value in ann_params.items():
                            if (
                                param not in free
                                and param in obs.init_parameter_values
                            ):
                                obs.init_parameter_values[param] = float(value)
                                orig = getattr(
                                    obs,
                                    "_original_init_parameter_values",
                                    None,
                                )
                                if orig is not None and param in orig:
                                    orig[param] = float(value)

                    # Extract points from annotation format
                    if "limb_points" in annotation_data:
                        points = annotation_data["limb_points"]["points"]
                    elif "points" in annotation_data:
                        points = annotation_data["points"]
                    else:
                        raise ValueError(
                            f"Invalid annotation format in {annot_path}"
                        )

                    # Convert points to sparse target array
                    limb_target = np.full(img_array.shape[1], np.nan)
                    for x, y in points:
                        x_idx = int(round(x))
                        if 0 <= x_idx < img_array.shape[1]:
                            limb_target[x_idx] = y

                    # Register limb with observation
                    obs.register_limb(limb_target)

            # For other detection methods, detect_limb is called automatically during fit
            # (gradient-field, gradient-break, segmentation)

            # Apply data-driven r bounds from the registered limb arc before fitting.
            cr_n = scenario.get("constrain_radius_n_sigma")
            if cr_n is not None and limb_detection == "manual":
                obs.constrain_radius(sigma_px="auto", n_sigma=float(cr_n))

            result.detection_time = time.time() - detect_start

            # Optimization phase
            opt_start = time.time()

            if scenario.get("constrain_radius_only"):
                # Skip the optimizer — use the sagitta estimate as the final answer.
                result.best_parameters = dict(obs.init_parameter_values)
                result.fitted_radius = obs.init_parameter_values.get("r")
                result.free_parameters = list(scenario.get("free_parameters", []))
                result.init_parameter_values = dict(obs.init_parameter_values)
                result.parameter_limits = dict(obs.parameter_limits)
                result.iterations = 0
                result.convergence_status = "success"
            else:
                # Extract fit parameters from scenario
                fit_params = scenario.get("fit_params", {})
                loss_function = (
                    "gradient_field"
                    if limb_detection == "gradient-field"
                    else "l2"
                )

                # Call fit_limb with actual parameters
                fit_limb_kwargs = dict(
                    loss_function=loss_function,
                    max_iter=fit_params.get("max_iter", 15000),
                    resolution_stages=fit_params.get("resolution_stages"),
                    seed=0,
                    verbose=False,
                    minimizer=minimizer_method,
                    minimizer_kwargs=minimizer_kwargs,
                )
                if minimizer_preset is not None:
                    fit_limb_kwargs["minimizer_preset"] = minimizer_preset
                obs.fit_limb(**fit_limb_kwargs)

                result.optimization_time = time.time() - opt_start

                # Extract fit configuration that was actually used
                result.free_parameters = (
                    obs.free_parameters if hasattr(obs, "free_parameters") else []
                )
                result.init_parameter_values = (
                    obs.init_parameter_values
                    if hasattr(obs, "init_parameter_values")
                    else {}
                )
                result.parameter_limits = dict(obs.parameter_limits)

                # Extract results
                if obs.best_parameters is not None:
                    result.best_parameters = obs.best_parameters
                    result.fitted_radius = obs.best_parameters.get("r")

                    # Get iteration count from fit results
                    if obs.fit_results is not None:
                        result.iterations = getattr(obs.fit_results, "nit", None)
                        if result.iterations is None and hasattr(
                            obs.fit_results, "nfev"
                        ):
                            result.iterations = obs.fit_results.nfev

                result.convergence_status = "success"

            # Calculate accuracy metrics
            if result.fitted_radius is not None:
                result.absolute_error = abs(
                    result.fitted_radius - result.expected_radius
                )
                result.relative_error = (
                    result.absolute_error / result.expected_radius
                )
                result.within_uncertainty = (
                    result.absolute_error <= result.uncertainty_radius
                )

        except Exception as e:
            result.error_message = str(e)
            result.convergence_status = "error"
            import traceback

            traceback.print_exc()

        result.total_time = time.time() - start_time

        return result

    def _build_fit_config(
        self, scenario: Dict[str, Any], image_path: Path
    ) -> Dict[str, Any]:
        """
        Build fit configuration from EXIF data using create_config_from_image.
        """
        from planet_ruler.camera import create_config_from_image

        planet = scenario.get("planet_name", "earth").lower()
        limits_preset = scenario.get("limits_preset", "balanced")
        altitude_m = scenario.get("altitude_m")  # Optional override

        config = create_config_from_image(
            image_path=str(image_path),
            altitude_m=altitude_m,
            planet=planet,
            limits_preset=limits_preset,
        )

        # Allow scenarios to override r limits (in km).
        # Sampling of init values from these limits happens after all overrides
        # are applied (see perturb_params block below).
        r_limits_km = scenario.get("r_limits_km")
        if r_limits_km is not None:
            config["parameter_limits"]["r"] = [
                float(r_limits_km[0]) * 1000.0,
                float(r_limits_km[1]) * 1000.0,
            ]

        # Generalised parameter limits override (meters/radians, keyed by param name).
        # e.g. parameter_limits_override: {h: [9000, 11000], theta_x: [-0.2, 0.2]}
        param_limits_override = scenario.get("parameter_limits_override")
        if param_limits_override is not None:
            for param, bounds in param_limits_override.items():
                config["parameter_limits"][param] = [
                    float(bounds[0]),
                    float(bounds[1]),
                ]

        # Allow scenario to override init values for specific parameters.
        # Critical for synthetic data where the true theta_z differs from the
        # EXIF-derived default (e.g. init_parameter_values_override: {theta_z: 3.14159}).
        init_override = scenario.get("init_parameter_values_override")
        if init_override is not None:
            for param, value in init_override.items():
                config["init_parameter_values"][param] = float(value)

        # Allow scenario to fix which parameters are optimised.
        # e.g. free_parameters: ["r"]  → only optimise r, fix everything else.
        free_params = scenario.get("free_parameters")
        if free_params is not None:
            config["free_parameters"] = list(free_params)

        # h_limits_pct: set h bounds to ±pct of the GPS-derived init h.
        # Only applied when h is in free_parameters.
        h_pct = scenario.get("h_limits_pct")
        if h_pct is not None and "h" in config["free_parameters"]:
            h_init = config["init_parameter_values"].get("h", 10000)
            margin = h_init * float(h_pct)
            config["parameter_limits"]["h"] = [
                max(0.0, h_init - margin),
                h_init + margin,
            ]

        # Clip all init values to their parameter limits.  This is needed
        # when init values fall outside the (possibly tighter) r_limits_km
        # or h_limits_pct bounds we just applied.
        for param, bounds in config["parameter_limits"].items():
            if param in config["init_parameter_values"]:
                lo, hi = float(bounds[0]), float(bounds[1])
                v = config["init_parameter_values"][param]
                clipped = max(lo, min(hi, float(v)))
                if clipped != float(v):
                    config["init_parameter_values"][param] = clipped

        # Resample init values from their finalised bounds for any params listed
        # in perturb_params.  This runs after all limit overrides so sampling
        # always uses the correct scenario-specific bounds with no clipping risk.
        # perturbation_factor may be a scalar (applied to all params) or a dict
        # mapping param name → factor, with an optional "default" fallback key.
        perturb_params_list = scenario.get("perturb_params", [])
        if perturb_params_list:
            from collections import defaultdict
            from planet_ruler.camera import init_params_from_bounds

            pf = scenario.get("perturbation_factor", 1.0)
            if isinstance(pf, dict):
                default_factor = float(pf.get("default", 1.0))
                factor_groups: Dict[float, List[str]] = defaultdict(list)
                for p in perturb_params_list:
                    factor_groups[float(pf.get(p, default_factor))].append(p)
                for factor, group_params in factor_groups.items():
                    sampled = init_params_from_bounds(
                        param_limits=config["parameter_limits"],
                        perturbation_factor=factor,
                        seed=0,
                        params=group_params,
                    )
                    config["init_parameter_values"].update(sampled)
            else:
                sampled = init_params_from_bounds(
                    param_limits=config["parameter_limits"],
                    perturbation_factor=float(pf),
                    seed=0,
                    params=list(perturb_params_list),
                )
                config["init_parameter_values"].update(sampled)

        return config

    def _get_image_path(self, image_name: str) -> Path:
        """Get full path to image file."""
        # Check benchmarks/images first
        bench_path = self.benchmark_dir / "images" / f"{image_name}.jpg"
        if bench_path.exists():
            return bench_path

        # Fall back to tests/images (go up from benchmark_dir to repo root)
        repo_root = (
            self.benchmark_dir.parent.parent
        )  # planet_ruler/benchmarks -> planet_ruler -> repo_root
        test_path = (
            repo_root / "tests" / "images" / "airplane" / f"{image_name}.jpg"
        )
        if test_path.exists():
            return test_path

        raise FileNotFoundError(f"Image not found: {image_name}")

    def _store_result(self, result: BenchmarkResult):
        """Store result to SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Convert complex types to JSON strings
        data = asdict(result)
        data["fit_params"] = json.dumps(data["fit_params"])
        data["free_parameters"] = json.dumps(data["free_parameters"])
        data["init_parameter_values"] = json.dumps(
            data["init_parameter_values"]
        )
        data["parameter_limits"] = json.dumps(data["parameter_limits"])
        data["best_parameters"] = json.dumps(data["best_parameters"])
        data["minimizer_config"] = json.dumps(data["minimizer_config"])
        data["within_uncertainty"] = (
            int(data["within_uncertainty"])
            if data["within_uncertainty"] is not None
            else None
        )

        columns = ", ".join(data.keys())
        placeholders = ", ".join(["?" for _ in data])

        cursor.execute(
            f"INSERT INTO benchmark_results ({columns}) VALUES ({placeholders})",
            list(data.values()),
        )

        conn.commit()
        conn.close()
