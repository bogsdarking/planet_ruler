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

import json
import sqlite3
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import warnings

import numpy as np
import yaml
from PIL import Image

# Import planet_ruler components
from planet_ruler.image import load_image
from planet_ruler.observation import LimbObservation


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
    
    # Image metadata
    image_width: int
    image_height: int
    altitude: Optional[float]
    planet_name: str
    expected_radius: float
    uncertainty_radius: float
    
    # Fit results (actual fitted parameters)
    best_parameters: Dict[str, float]  # Actual fitted values {r, h, f, theta_x, ...}
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
    
    def __init__(self, config_path: Union[str, Path], db_path: Optional[Union[str, Path]] = None):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Determine base directory for benchmarks
        # Try module location first, fall back to CWD/planet_ruler/benchmarks
        module_bench_dir = Path(__file__).parent
        cwd_bench_dir = Path.cwd() / "planet_ruler" / "benchmarks"
        
        # Use module location if images/ exists there, otherwise use CWD-relative
        if (module_bench_dir / "images").exists():
            self.benchmark_dir = module_bench_dir
        elif cwd_bench_dir.exists():
            self.benchmark_dir = cwd_bench_dir
        else:
            # Default to module location (will be created if needed)
            self.benchmark_dir = module_bench_dir
        
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
        
        # Basic validation
        if "scenarios" not in config:
            raise ValueError("Config must contain 'scenarios' key")
        
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
                cwd=Path(__file__).parent.parent
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
    
    def _init_database(self):
        """Initialize SQLite database with results table."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
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
        """)
        
        # Create indexes for common queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_scenario_image 
            ON benchmark_results(scenario_name, image_name)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON benchmark_results(timestamp)
        """)
        
        conn.commit()
        conn.close()
    
    def _expand_scenarios(self) -> List[Dict[str, Any]]:
        """
        Expand scenarios with grid search or sampling.
        
        Returns list of individual scenario configurations.
        """
        expanded = []
        
        for scenario in self.config["scenarios"]:
            # Check for grid search
            has_grid = self._has_grid_params(scenario)
            has_sampling = "sampling" in scenario
            
            if has_grid and has_sampling:
                raise ValueError(f"Scenario {scenario['name']} has both grid and sampling")
            
            if has_grid:
                expanded.extend(self._expand_grid(scenario))
            elif has_sampling:
                expanded.extend(self._expand_sampling(scenario))
            else:
                # Single scenario
                expanded.append(scenario)
        
        return expanded
    
    def _has_grid_params(self, scenario: Dict[str, Any]) -> bool:
        """Check if scenario contains grid search parameters."""
        def check_dict(d):
            for v in d.values():
                if isinstance(v, dict):
                    if "grid" in v:
                        return True
                    if check_dict(v):
                        return True
                elif isinstance(v, list) and len(v) > 0:
                    if isinstance(v[0], dict) and check_dict(v[0]):
                        return True
            return False
        return check_dict(scenario)
    
    def _expand_grid(self, scenario: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Expand grid search parameters into individual scenarios."""
        # TODO: Implement proper grid expansion
        # For now, return as-is and log warning
        warnings.warn(f"Grid expansion not yet implemented for {scenario['name']}")
        return [scenario]
    
    def _expand_sampling(self, scenario: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate sampled parameter combinations."""
        # TODO: Implement Latin hypercube or random sampling
        warnings.warn(f"Sampling not yet implemented for {scenario['name']}")
        return [scenario]
    
    def run(self, 
            parallel: bool = False,
            scenarios: Optional[List[str]] = None,
            images: Optional[List[str]] = None) -> List[BenchmarkResult]:
        """
        Run benchmark suite.
        
        Args:
            parallel (bool): Whether to run scenarios in parallel.
            scenarios (list of str, optional): Filter to specific scenario names.
            images (list of str, optional): Filter to specific image names.
        
        Returns:
            list of BenchmarkResult: All benchmark results.
        """
        expanded_scenarios = self._expand_scenarios()
        
        # Filter scenarios if requested
        if scenarios is not None:
            expanded_scenarios = [s for s in expanded_scenarios if s["name"] in scenarios]
        
        results = []
        
        for scenario in expanded_scenarios:
            scenario_images = scenario.get("images", [])
            
            # Filter images if requested
            if images is not None:
                scenario_images = [img for img in scenario_images if img in images]
            
            for image_name in scenario_images:
                print(f"Running: {scenario['name']} on {image_name}")
                result = self._run_single(scenario, image_name)
                results.append(result)
                self._store_result(result)
        
        return results
    
    def _run_single(self, scenario: Dict[str, Any], image_name: str) -> BenchmarkResult:
        """Run a single benchmark scenario on one image."""
        start_time = time.time()
        
        # Initialize result with metadata
        result = BenchmarkResult(
            scenario_name=scenario["name"],
            image_name=image_name,
            timestamp=datetime.now().isoformat(),
            git_commit=self.git_commit,
            detection_method=scenario.get("detection_method", "manual"),
            fit_params=scenario.get("fit_params", {}),
            free_parameters=[],  # Will be filled from actual fit
            init_parameter_values={},  # Will be filled from actual fit
            parameter_limits={},  # Will be filled from actual fit
            minimizer_config=scenario.get("minimizer", {}),
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
            error_message=None
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
            minimizer_method = scenario.get("minimizer", "differential-evolution")
            minimizer_kwargs = scenario.get("minimizer_kwargs")
            
            # Create observation
            from planet_ruler.observation import LimbObservation
            obs = LimbObservation(
                image_filepath=str(image_path),
                fit_config=fit_config,
                limb_detection=limb_detection,
                minimizer=minimizer_method
            )
            
            # Detection phase
            detect_start = time.time()
            
            # For manual detection, load and register annotations
            if limb_detection == "manual":
                # Require explicit annotation file for manual detection
                annotation_file = scenario.get("annotation_file")
                if not annotation_file:
                    raise ValueError(
                        f"Manual detection requires explicit 'annotation_file' parameter.\n"
                        f"Add to scenario: annotation_file: 'pexels-claiton-17217951_exif.json'"
                    )
                
                annot_path = self.benchmark_dir / "annotations" / annotation_file
                if not annot_path.exists():
                    raise FileNotFoundError(
                        f"Annotation file not found: {annot_path}\n"
                        f"For manual detection, create annotation with:\n"
                        f"  python -m planet_ruler.annotate {image_path}\n"
                        f"Then convert with:\n"
                        f"  python -m planet_ruler.benchmarks.copy_test_annotations"
                    )
                
                with open(annot_path) as f:
                    annotation_data = json.load(f)
                
                # Extract points from annotation format
                if "limb_points" in annotation_data:
                    points = annotation_data["limb_points"]["points"]
                elif "points" in annotation_data:
                    points = annotation_data["points"]
                else:
                    raise ValueError(f"Invalid annotation format in {annot_path}")
                
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
            
            result.detection_time = time.time() - detect_start
            
            # Optimization phase
            opt_start = time.time()
            
            # Extract fit parameters from scenario
            fit_params = scenario.get("fit_params", {})
            loss_function = "gradient_field" if limb_detection == "gradient-field" else "l2"
            
            # Call fit_limb with actual parameters
            obs.fit_limb(
                loss_function=loss_function,
                max_iter=fit_params.get("max_iter", 15000),
                resolution_stages=fit_params.get("resolution_stages"),
                seed=0,
                verbose=False,
                minimizer=minimizer_method,
                minimizer_kwargs=minimizer_kwargs,
            )
            
            result.optimization_time = time.time() - opt_start
            
            # Extract fit configuration that was actually used
            result.free_parameters = obs.free_parameters if hasattr(obs, 'free_parameters') else []
            result.init_parameter_values = obs.init_parameter_values if hasattr(obs, 'init_parameter_values') else {}
            result.parameter_limits = fit_config.get("parameter_limits", {})
            
            # Extract results
            if obs.best_parameters is not None:
                result.best_parameters = obs.best_parameters
                result.fitted_radius = obs.best_parameters.get("r")  # Extract radius
                
                # Get iteration count from fit results
                if obs.fit_results is not None:
                    result.iterations = getattr(obs.fit_results, "nit", None)
                    if result.iterations is None and hasattr(obs.fit_results, "nfev"):
                        result.iterations = obs.fit_results.nfev
            
            # Calculate accuracy metrics
            if result.fitted_radius is not None:
                result.absolute_error = abs(result.fitted_radius - result.expected_radius)
                result.relative_error = result.absolute_error / result.expected_radius
                result.within_uncertainty = result.absolute_error <= result.uncertainty_radius
            
            result.convergence_status = "success"
            
        except Exception as e:
            result.error_message = str(e)
            result.convergence_status = "error"
            import traceback
            traceback.print_exc()
        
        result.total_time = time.time() - start_time
        
        return result
    
    def _build_fit_config(self, scenario: Dict[str, Any], image_path: Path) -> Dict[str, Any]:
        """
        Build fit configuration from EXIF data using create_config_from_image.
        """
        from planet_ruler.camera import create_config_from_image
        
        planet = scenario.get("planet_name", "earth").lower()
        perturbation_factor = scenario.get("perturbation_factor", 0.5)
        param_tolerance = scenario.get("param_tolerance", 0.1)
        altitude_m = scenario.get("altitude_m")  # Optional override
        
        config = create_config_from_image(
            image_path=str(image_path),
            altitude_m=altitude_m,
            planet=planet,
            param_tolerance=param_tolerance,
            perturbation_factor=perturbation_factor,
            seed=0  # Reproducible for benchmarks
        )
        return config
    
    def _get_image_path(self, image_name: str) -> Path:
        """Get full path to image file."""
        # Check benchmarks/images first
        bench_path = self.benchmark_dir / "images" / f"{image_name}.jpg"
        if bench_path.exists():
            return bench_path
        
        # Fall back to tests/images (go up from benchmark_dir to repo root)
        repo_root = self.benchmark_dir.parent.parent  # planet_ruler/benchmarks -> planet_ruler -> repo_root
        test_path = repo_root / "tests" / "images" / "airplane" / f"{image_name}.jpg"
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
        data["init_parameter_values"] = json.dumps(data["init_parameter_values"])
        data["parameter_limits"] = json.dumps(data["parameter_limits"])
        data["best_parameters"] = json.dumps(data["best_parameters"])
        data["minimizer_config"] = json.dumps(data["minimizer_config"])
        data["within_uncertainty"] = int(data["within_uncertainty"]) if data["within_uncertainty"] is not None else None
        
        columns = ", ".join(data.keys())
        placeholders = ", ".join(["?" for _ in data])
        
        cursor.execute(
            f"INSERT INTO benchmark_results ({columns}) VALUES ({placeholders})",
            list(data.values())
        )
        
        conn.commit()
        conn.close()