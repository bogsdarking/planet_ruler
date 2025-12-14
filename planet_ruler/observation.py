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

from __future__ import annotations
from typing import Optional, List, Dict, Literal
from scipy.optimize import differential_evolution
import yaml
import numpy as np
import pandas as pd
import logging
import warnings
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import cv2
from planet_ruler.plot import (
    plot_image,
    plot_limb,
    plot_diff_evol_posteriors,
    plot_full_limb,
    plot_segmentation_masks,
    plot_residuals,
)
from planet_ruler.image import (
    load_image,
    gradient_break,
    smooth_limb,
    fill_nans,
    MaskSegmenter,
)
from planet_ruler.annotate import TkLimbAnnotator
from planet_ruler.validation import validate_limb_config
from planet_ruler.fit import CostFunction, unpack_parameters, _validate_fit_results
from planet_ruler.geometry import limb_arc
from planet_ruler.dashboard import FitDashboard


# ============================================================================
# MINIMIZER PRESET CONFIGURATIONS
# ============================================================================

MINIMIZER_PRESETS = {
    "differential-evolution": {
        "fast": {
            "strategy": "best1bin",
            "popsize": 10,
            "mutation": [0.1, 1.5],
            "recombination": 0.7,
            "polish": True,
            "init": "sobol",
            "atol": 1.0,
            "tol": 0.01,
        },
        "balanced": {
            "strategy": "best2bin",
            "popsize": 15,
            "mutation": [0.1, 1.9],
            "recombination": 0.7,
            "polish": True,
            "init": "sobol",
            "atol": 1.0,
            "tol": 0.01,
        },
        "planet-ruler": {
            "strategy": "best2bin",
            "popsize": 15,
            "mutation": [0.1, 1.9],
            "recombination": 0.7,
            "polish": True,
            "init": "sobol",
            "atol": 0,
            "tol": 0.01,
        },
        "robust": {
            "strategy": "best2bin",
            "popsize": 20,
            "mutation": [0.5, 1.9],
            "recombination": 0.7,
            "polish": True,
            "init": "sobol",
            "atol": 0.1,
            "tol": 0.001,
        },
        "scipy-default": {
            # Exact scipy differential_evolution defaults
            # Use this to match original prototype behavior
            "strategy": "best1bin",
            "popsize": 15,
            "mutation": (0.5, 1),
            "recombination": 0.7,
            "polish": True,
            "init": "latinhypercube",
            "atol": 0,
            "tol": 0.01,
        },
    },
    "dual-annealing": {
        "fast": {
            "initial_temp": 10000,
            "restart_temp_ratio": 2e-5,
            "visit": 2.5,
            "accept": -5.0,
            "no_local_search": False,
        },
        "balanced": {
            "initial_temp": 20000,
            "restart_temp_ratio": 1e-4,
            "visit": 2.8,
            "accept": -10.0,
            "no_local_search": False,
        },
        "robust": {
            "initial_temp": 50000,
            "restart_temp_ratio": 5e-4,
            "visit": 3.0,
            "accept": -15.0,
            "no_local_search": False,
        },
        "scipy-default": {
            # Exact scipy dual_annealing defaults
            # Use this to match original prototype behavior
            "initial_temp": 5230.0,
            "restart_temp_ratio": 2e-05,
            "visit": 2.62,
            "accept": -5.0,
            "no_local_search": False,
        },
    },
    "basinhopping": {
        "fast": {"niter": 100, "T": 1.5, "stepsize": 0.5, "local_maxiter": 50},
        "balanced": {"niter": 200, "T": 2.0, "stepsize": 0.5, "local_maxiter": 100},
        "robust": {"niter": 500, "T": 3.0, "stepsize": 0.7, "local_maxiter": 200},
    },
}


# ============================================================================
# MAIN CLASSES
# ============================================================================


class PlanetObservation:
    """
    Base class for planet observations.

    Args:
        image_filepath (str): Path to image file.
    """

    def __init__(self, image_filepath: str):
        self.image = load_image(image_filepath)
        self._original_image = self.image.copy()
        self.image_filepath = image_filepath
        self.features = {}
        self._plot_functions = {}
        self._cwheel = ["y", "b", "r", "orange", "pink", "black"]

    def plot(self, gradient: bool = False, show: bool = True) -> None:
        """
        Display the observation and all current features.

        Args:
            gradient (bool): Show the image gradient instead of the raw image.
            show (bool): Show -- useful as False if intending to add more to the plot before showing.
        """
        plot_image(self.image, gradient=gradient, show=False)
        h_plus, l_plus = [], []
        for i, feature in enumerate(self.features):
            self._plot_functions[feature](
                self.features[feature], show=False, c=self._cwheel[i]
            )
            h_plus.append(Line2D([0], [0], color=self._cwheel[i], lw=2))
            l_plus.append(feature)
        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles + h_plus, labels + l_plus)

        if show:
            plt.show()


class LimbObservation(PlanetObservation):
    """
    Observation of a planet's limb (horizon).

    Args:
        image_filepath (str): Path to image file.
        fit_config (str): Path to fit config file.
        limb_detection (str): Method to locate the limb in the image.
        minimizer (str): Choice of minimizer. Supports 'differential-evolution',
            'dual-annealing', and 'basinhopping'.
    """

    def __init__(
        self,
        image_filepath: str,
        fit_config,
        limb_detection: Literal[
            "manual", "gradient-break", "gradient-field", "segmentation"
        ] = "manual",
        minimizer: Literal[
            "differential-evolution", "dual-annealing", "basinhopping"
        ] = "differential-evolution",
    ):
        super().__init__(image_filepath)

        # Runtime validation (Literal type hints alone don't enforce at runtime)
        valid_limb_methods = [
            "manual",
            "gradient-break",
            "gradient-field",
            "segmentation",
        ]
        assert limb_detection in valid_limb_methods, (
            f"Invalid limb_detection method '{limb_detection}'. "
            f"Must be one of: {valid_limb_methods}"
        )

        valid_minimizers = ["differential-evolution", "dual-annealing", "basinhopping"]
        assert minimizer in valid_minimizers, (
            f"Invalid minimizer '{minimizer}'. " f"Must be one of: {valid_minimizers}"
        )

        self.free_parameters = None
        self.init_parameter_values = None
        self._original_init_parameter_values = (
            None  # Store original values for warm start protection
        )
        self.parameter_limits = None
        self.load_fit_config(fit_config)
        self.limb_detection = limb_detection
        self._segmenter = None
        self.minimizer = minimizer

        self._raw_limb = None
        self.cost_function = None
        self.fit = None
        self.best_parameters = None
        self.fit_results = None

    def analyze(
        self,
        detect_limb_kwargs: dict = None,
        fit_limb_kwargs: dict = None,
    ) -> "LimbObservation":
        """
        Perform complete limb analysis: detection + fitting in one call.

        Args:
            detect_limb_kwargs (dict): Optional arguments for detect_limb()
            fit_limb_kwargs (dict): Optional arguments for fit_limb()

        Returns:
            self: For method chaining
        """
        if detect_limb_kwargs is None:
            detect_limb_kwargs = {}
        if fit_limb_kwargs is None:
            fit_limb_kwargs = {}

        self.detect_limb(**detect_limb_kwargs)
        self.fit_limb(**fit_limb_kwargs)

        return self

    @property
    def radius_km(self) -> float:
        """
        Get the fitted planetary radius in kilometers.

        Returns:
            float: Planetary radius in km, or 0 if not fitted
        """
        if self.best_parameters is None:
            return 0.0
        return self.best_parameters.get("r", 0.0) / 1000.0

    @property
    def altitude_km(self) -> float:
        """
        Get the observer altitude in kilometers.

        Returns:
            float: Observer altitude in km, or 0 if not fitted
        """
        if self.best_parameters is None:
            return 0.0
        return self.best_parameters.get("h", 0.0) / 1000.0

    @property
    def focal_length_mm(self) -> float:
        """
        Get the camera focal length in millimeters.

        Returns:
            float: Focal length in mm, or 0 if not fitted
        """
        if self.best_parameters is None:
            return 0.0
        return self.best_parameters.get("f", 0.0) * 1000.0

    @property
    def radius_uncertainty(self) -> float:
        """
        Get parameter uncertainty for radius.

        Automatically selects best method based on minimizer:
        - differential_evolution: Uses population spread (fast, exact)
        - dual_annealing/basinhopping: Uses Hessian approximation (fast, approximate)

        Returns:
            float: Radius uncertainty in km (1-sigma), or 0 if not available
        """
        if not hasattr(self, "fit_results") or self.fit_results is None:
            return 0.0

        try:
            from planet_ruler.uncertainty import calculate_parameter_uncertainty

            result = calculate_parameter_uncertainty(
                self,
                parameter="r",
                method="auto",  # Automatically selects best method
                scale_factor=1000.0,  # Convert m → km
                confidence_level=0.68,  # 1-sigma
            )
            return result["uncertainty"]
        except Exception as e:
            logging.warning(f"Could not calculate radius uncertainty: {e}")
            return 0.0

    def parameter_uncertainty(
        self,
        parameter: str,
        method: Literal["auto", "hessian", "profile", "bootstrap"] = "auto",
        scale_factor: float = 1.0,
        confidence_level: float = 0.68,
        **kwargs,
    ) -> Dict:
        """
        Get uncertainty for any fitted parameter.

        Args:
            parameter: Parameter name (e.g., 'r', 'h', 'f', 'theta_x')
            method: Uncertainty method
                - 'auto': Choose based on minimizer (recommended)
                - 'hessian': Fast Hessian approximation
                - 'profile': Slow but accurate profile likelihood
                - 'bootstrap': Multiple fits (very slow)
            scale_factor: Scale result (e.g., 1000.0 for m→km)
            confidence_level: Confidence level (0.68=1σ, 0.95=2σ)
            **kwargs: Additional arguments passed to uncertainty calculator

        Returns:
            dict with 'uncertainty', 'method', 'confidence_level', 'additional_info'

        Examples:
            # Radius uncertainty in km (1-sigma)
            obs.parameter_uncertainty('r', scale_factor=1000.0)

            # Altitude uncertainty in km (2-sigma / 95% CI)
            obs.parameter_uncertainty('h', scale_factor=1000.0, confidence_level=0.95)

            # Focal length uncertainty in mm (using profile likelihood)
            obs.parameter_uncertainty('f', scale_factor=1000.0, method='profile')
        """
        if not hasattr(self, "fit_results") or self.fit_results is None:
            return {
                "uncertainty": 0.0,
                "method": "none",
                "confidence_level": confidence_level,
                "additional_info": "No fit performed",
            }

        try:
            from planet_ruler.uncertainty import calculate_parameter_uncertainty

            return calculate_parameter_uncertainty(
                self,
                parameter=parameter,
                method=method,
                scale_factor=scale_factor,
                confidence_level=confidence_level,
                **kwargs,
            )
        except Exception as e:
            logging.warning(f"Could not calculate {parameter} uncertainty: {e}")
            return {
                "uncertainty": 0.0,
                "method": "error",
                "confidence_level": confidence_level,
                "additional_info": str(e),
            }

    def plot_3d(self, **kwargs) -> None:
        """
        Create 3D visualization of the planetary geometry.

        Args:
            **kwargs: Arguments passed to plot_3d_solution
        """
        from planet_ruler.plot import plot_3d_solution

        if self.best_parameters is None:
            raise ValueError("Must fit limb before plotting 3D solution")

        plot_3d_solution(**self.best_parameters, **kwargs)

    def load_fit_config(self, fit_config: str | dict) -> None:
        """
        Load the fit configuration from file, setting all parameters
        to their initial values. Missing values are filled with defaults.

        Args:
            fit_config (str or dict): Path to configuration file.
        """
        # Define default configuration values
        default_config = {
            "parameter_limits": {
                "theta_x": [-3.14, 3.14],
                "theta_y": [-3.14, 3.14],
                "theta_z": [-3.14, 3.14],
                "num_sample": [4000, 6000],
            },
            "init_parameter_values": {"theta_y": 0, "theta_z": 0},
        }

        # Load the provided configuration
        if isinstance(fit_config, dict):
            provided_config = fit_config
        else:
            with open(fit_config, "r") as f:
                provided_config = yaml.safe_load(f)

        # Merge configurations with provided values overriding defaults
        base_config = {}

        # Merge parameter_limits
        base_config["parameter_limits"] = default_config["parameter_limits"].copy()
        if "parameter_limits" in provided_config:
            base_config["parameter_limits"].update(provided_config["parameter_limits"])

        # Merge init_parameter_values
        base_config["init_parameter_values"] = default_config[
            "init_parameter_values"
        ].copy()
        if "init_parameter_values" in provided_config:
            base_config["init_parameter_values"].update(
                provided_config["init_parameter_values"]
            )

        # Copy other keys from provided config (like free_parameters)
        for key in provided_config:
            if key not in ["parameter_limits", "init_parameter_values"]:
                base_config[key] = provided_config[key]

        # Validate that initial values are within parameter limits and do not conflict
        validate_limb_config(base_config)

        self.free_parameters = base_config["free_parameters"]
        self.init_parameter_values = base_config["init_parameter_values"]
        # Store a deep copy of original initial values for warm start protection
        self._original_init_parameter_values = base_config[
            "init_parameter_values"
        ].copy()
        self.parameter_limits = base_config["parameter_limits"]

    def register_limb(self, limb: np.ndarray) -> "LimbObservation":
        """
        Register a detected limb.

        Args:
            limb (np.ndarray): Limb vector (y pixel coordinates).
        """
        self.features["limb"] = limb
        self._raw_limb = self.features["limb"].copy()
        self._plot_functions["limb"] = plot_limb
        return self

    def detect_limb(
        self,
        detection_method: Optional[
            Literal["manual", "gradient-break", "gradient-field", "segmentation"]
        ] = None,
        log: bool = False,
        y_min: int = 0,
        y_max: int = -1,
        window_length: int = 501,
        polyorder: int = 1,
        deriv: int = 0,
        delta: int = 1,
        segmentation_method: str = "sam",
        downsample_factor: int = 1,
        interactive: bool = True,
        **segmentation_kwargs,
    ) -> "LimbObservation":
        """
        Use the instance-defined method to find the limb in our observation.
        Kwargs are passed to the method.

        Args:
            detection_method (literal):
                Detection method. Must be one of
                    - manual
                    - gradient-break
                    - gradient-field
                    - segmentation
                Default (None) uses the class attribute self.limb_detection.
            log (bool): Use the log(gradient). Sometimes good for
                smoothing.
            y_min (int): Minimum y-position to consider.
            y_max (int): Maximum y-position to consider.
            window_length (int): Width of window to apply smoothing
                for each vertical. Larger means less noise but less
                sensitivity.
            polyorder (int): Polynomial order for smoothing.
            deriv (int): Derivative level for smoothing.
            delta (int): Delta for smoothing.
            segmentation_method (str): Model used for segmentation. Must be one
                of ['sam'].
            downsample_factor (int): Downsampling used for segmentation.
            interactive (bool): Prompts user to verify segmentation via annotation tool.
            segmentation_kwargs (dict): Kwargs passed to segmentation engine.

        """
        if detection_method is not None:
            self.limb_detection = detection_method

        if self.limb_detection == "gradient-break":
            limb = gradient_break(
                self.image,
                log=log,
                y_min=y_min,
                y_max=y_max,
                window_length=window_length,
                polyorder=polyorder,
                deriv=deriv,
                delta=delta,
            )
        elif self.limb_detection == "segmentation":
            self._segmenter = MaskSegmenter(
                image=self.image,
                method=segmentation_method,
                downsample_factor=downsample_factor,
                interactive=interactive,
                **segmentation_kwargs,
            )
            limb = self._segmenter.segment()

        elif self.limb_detection == "manual":
            annotator = TkLimbAnnotator(
                image_path=self.image_filepath, initial_stretch=1.0
            )
            annotator.run()  # Opens window

            # After closing window
            limb = annotator.get_target()  # Get sparse array

            if limb is not None:
                self.register_limb(limb)
            else:
                # No limb was annotated/insufficient points
                logging.warning("No limb detected from manual annotation")
            return self

        elif self.limb_detection == "gradient-field":
            print("Skipping detection step (not needed for gradient-field method)")
            return self

        # For non-manual methods, register the limb
        self.register_limb(limb)
        return self

    def smooth_limb(self, fill_nan=True, **kwargs) -> None:
        """
        Apply the smooth_limb function to current observation.

        Args:
            fill_nan (bool): Fill any NaNs in the limb.
        """
        self.features["limb"] = smooth_limb(self._raw_limb, **kwargs)
        if fill_nan:
            logging.info("Filling NaNs in fitted limb.")
            self.features["limb"] = fill_nans(self.features["limb"])

    def fit_limb(
        self,
        loss_function: Literal["l2", "l1", "log-l1", "gradient_field"] = "l2",
        max_iter: int = 15000,
        resolution_stages: Optional[List[int] | Literal["auto"]] = None,
        max_iter_per_stage: Optional[List[int]] = None,
        n_jobs: int = 1,
        seed: int = 0,
        image_smoothing: Optional[float] = None,
        kernel_smoothing: float = 5.0,
        directional_smoothing: int = 50,
        directional_decay_rate: float = 0.15,
        prefer_direction: Optional[Literal["up", "down"]] = None,
        minimizer: Optional[
            Literal["differential-evolution", "dual-annealing", "basinhopping"]
        ] = None,
        minimizer_preset: Literal[
            "fast", "balanced", "robust", "scipy-default"
        ] = "balanced",
        minimizer_kwargs: Optional[Dict] = None,
        warm_start: bool = False,
        dashboard: bool = False,
        dashboard_kwargs: Optional[Dict] = None,
        target_planet: str = "earth",
        verbose: bool = False,
    ) -> "LimbObservation":
        """
        Fit the limb to determine planetary parameters.

        Supports single-resolution or multi-resolution (coarse-to-fine) optimization.
        Multi-resolution is recommended for gradient_field loss to avoid local minima.

        Args:
            loss_function: Loss function type
                - 'l2', 'l1', 'log-l1': Traditional (requires detected limb)
                - 'gradient_field': Direct gradient alignment (no detection needed)
            max_iter: Maximum iterations (for single-resolution or total if multires)
            resolution_stages: Resolution strategy
                - None: Single resolution (original behavior)
                - 'auto': Auto-determine stages based on image size
                - List[int]: Custom stages, e.g., [4, 2, 1] = 1/4 → 1/2 → full
                NOTE: Multi-resolution only works with gradient_field loss functions.
                      Traditional loss functions (l2, l1, log-l1) require single resolution.
            max_iter_per_stage: Iterations per stage (auto if None)
            n_jobs: Number of parallel workers
            seed: Random seed for reproducibility
            image_smoothing: For gradient_field - Gaussian blur sigma applied to image
                before gradient computation. Removes high-frequency artifacts (crater rims,
                striations) that could mislead optimization. Different from kernel_smoothing.
            kernel_smoothing: For gradient_field - initial blur for gradient direction
                estimation. Makes the gradient field smoother for directional sampling.
            directional_smoothing: For gradient_field - sampling distance along gradients
            directional_decay_rate: For gradient_field - exponential decay for samples
            prefer_direction: For gradient_field - prefer 'up' or 'down' gradients
                where 'up' means dark-sky/bright-planet and v.v.
                (None = no preference, choose best gradient regardless of direction)
            minimizer (str): Choice of minimizer. Supports 'differential-evolution',
                'dual-annealing', and 'basinhopping'.
            minimizer_preset: Optimization strategy
                - 'fast': Quick convergence, may miss global minimum
                - 'balanced': Good trade-off (default)
                - 'robust': Thorough exploration, slower
            minimizer_kwargs: Override specific minimizer parameters (advanced)
            warm_start: If True, use previous fit's results as starting point
                (useful for iterative refinement). If False (default), use
                original init_parameter_values.
                Note: Multi-resolution stages always warm-start from previous
                stages automatically. This parameter is for warm-starting across
                separate fit_limb() calls.
            dashboard: Show live progress dashboard during optimization
            dashboard_kwargs: Additional kwargs for FitDashboard
                - output_capture: OutputCapture instance for print/log display
                - show_output: Show output section (default True if capture provided)
                - max_output_lines: Number of output lines to show (default 3)
                - min_refresh_delay: Fixed refresh delay (0.0 for adaptive, default)
                - refresh_frequency: Refresh every N iterations (default 1)
            target_planet: Reference planet for dashboard comparisons
                ('earth', 'mars', 'jupiter', 'saturn', 'moon', 'pluto')
            verbose: Print detailed progress

        Returns:
            self: For method chaining

        Examples:
            # Simple single-resolution fit
            obs.fit_limb()

            # Auto multi-resolution for gradient field
            obs.fit_limb(loss_function='gradient_field', resolution_stages='auto')

            # Remove image artifacts before optimization
            obs.fit_limb(
                loss_function='gradient_field',
                resolution_stages='auto',
                image_smoothing=2.0,  # Remove crater rims, striations
                kernel_smoothing=5.0  # Smooth gradient field
            )

            # Custom stages with robust optimization
            obs.fit_limb(
                loss_function='gradient_field',
                resolution_stages=[8, 4, 2, 1],
                minimizer_preset='robust'
            )

            # Override specific minimizer parameters
            obs.fit_limb(
                loss_function='gradient_field',
                minimizer_kwargs={'popsize': 25, 'atol': 0.5}
            )

            # Dashboard with output capture
            from planet_ruler.dashboard import OutputCapture
            capture = OutputCapture()
            with capture:
                obs.fit_limb(
                    loss_function='gradient_field',
                    dashboard=True,
                    dashboard_kwargs={'output_capture': capture}
                )

            # Iterative refinement with warm start
            obs.fit_limb(loss_function='gradient_field', resolution_stages='auto')
            obs.fit_limb(loss_function='gradient_field', warm_start=True,
                         minimizer_preset='robust')  # Refine with more thorough search
        """

        if minimizer is not None:
            self.minimizer = minimizer
        print(f"Using minimizer: {self.minimizer}")

        # ====================================================================
        # STEP 0: Warm start handling - protect original values
        # ====================================================================
        if (
            warm_start
            and hasattr(self, "best_parameters")
            and self.best_parameters is not None
        ):
            # Update init_parameter_values with previous fit's results
            # Only update free parameters (don't touch inferred ones like n_pix_x, n_pix_y, x0, y0)
            for param in self.free_parameters:
                if param in self.best_parameters:
                    self.init_parameter_values[param] = self.best_parameters[param]
            if verbose:
                print("Warm start: Using previous fit's results as starting point")
        elif (
            not warm_start
            and hasattr(self, "_original_init_parameter_values")
            and self._original_init_parameter_values is not None
        ):
            # Restore original initial parameter values when not using warm start
            self.init_parameter_values = self._original_init_parameter_values.copy()
            if verbose:
                print("Cold start: Using original initial parameter values")

        # ====================================================================
        # STEP 1: Save original image ONCE (if smoothing will be applied)
        # ====================================================================
        original_image = None
        if (
            image_smoothing is not None
            and image_smoothing > 0
            and "gradient_field" in loss_function
        ):
            original_image = self.image.copy()  # Save ONCE

            # Apply smoothing to self.image
            if verbose:
                print(f"Applying Gaussian blur to image (sigma={image_smoothing:.1f})")
            self.image = cv2.GaussianBlur(
                self.image.astype(np.float32),
                (0, 0),  # Kernel size auto-determined from sigma
                sigmaX=image_smoothing,
                sigmaY=image_smoothing,
            )

        # ====================================================================
        # STEP 2: Determine resolution strategy
        # ====================================================================
        use_multires = resolution_stages is not None

        # Multi-resolution only works with gradient_field loss functions
        if use_multires and "gradient_field" not in loss_function:
            logging.warning(
                f"Multi-resolution optimization is only supported for gradient_field loss functions. "
                f"Got loss_function='{loss_function}'. Falling back to single-resolution."
            )
            use_multires = False
            resolution_stages = None

        # ====================================================================
        # STEP 3: Run optimization (single or multi-resolution)
        # ====================================================================
        try:
            if not use_multires:
                result = self._fit_single_resolution(
                    loss_function=loss_function,
                    minimizer=minimizer,
                    max_iter=max_iter,
                    n_jobs=n_jobs,
                    seed=seed,
                    kernel_smoothing=kernel_smoothing,
                    directional_smoothing=directional_smoothing,
                    directional_decay_rate=directional_decay_rate,
                    prefer_direction=prefer_direction,
                    minimizer_preset=minimizer_preset,
                    minimizer_kwargs=minimizer_kwargs,
                    dashboard=dashboard,
                    dashboard_kwargs=dashboard_kwargs,
                    target_planet=target_planet,
                    verbose=verbose,
                )
            else:
                result = self._fit_multi_resolution(
                    loss_function=loss_function,
                    minimizer=minimizer,
                    resolution_stages=resolution_stages,
                    max_iter=max_iter,
                    max_iter_per_stage=max_iter_per_stage,
                    n_jobs=n_jobs,
                    seed=seed,
                    kernel_smoothing=kernel_smoothing,
                    directional_smoothing=directional_smoothing,
                    directional_decay_rate=directional_decay_rate,
                    prefer_direction=prefer_direction,
                    minimizer_preset=minimizer_preset,
                    minimizer_kwargs=minimizer_kwargs,
                    dashboard=dashboard,
                    dashboard_kwargs=dashboard_kwargs,
                    target_planet=target_planet,
                    verbose=verbose,
                )
        finally:
            # ================================================================
            # STEP 4: Restore original image ONCE (if it was smoothed)
            # ================================================================
            if original_image is not None:
                self.image = original_image
                if verbose:
                    print(f"Restored original unsmoothed image")

        return self

    def _fit_multi_resolution(
        self,
        loss_function: str,
        resolution_stages: List[int] | str,
        max_iter: int,
        max_iter_per_stage: Optional[List[int]],
        n_jobs: int,
        seed: int,
        kernel_smoothing: float,
        directional_smoothing: int,
        directional_decay_rate: float,
        prefer_direction: Optional[Literal["up", "down"]] = None,
        minimizer: Optional[
            Literal["differential-evolution", "dual-annealing", "basinhopping"]
        ] = None,
        minimizer_preset: Literal["fast", "balanced", "robust"] = "balanced",
        minimizer_kwargs: Optional[Dict] = None,
        dashboard: bool = False,
        dashboard_kwargs: Optional[Dict] = None,
        target_planet: str = "earth",
        verbose: bool = False,
    ) -> "LimbObservation":
        """
        Multi-resolution optimization (internal method).

        Note: self.image may be the smoothed version if image_smoothing was applied.
        """
        if minimizer is not None:
            self.minimizer = minimizer
            print(f"Using minimizer: {self.minimizer}")

        # Auto-generate stages
        if resolution_stages == "auto":
            min_dim = min(self.image.shape[:2])
            if min_dim >= 2000:
                resolution_stages = [4, 2, 1]
            elif min_dim >= 1000:
                resolution_stages = [2, 1]
            else:
                resolution_stages = [1]

        # Auto-determine iterations per stage
        if max_iter_per_stage is None:
            total_weight = sum(range(1, len(resolution_stages) + 1))
            max_iter_per_stage = [
                int(max_iter * (i + 1) / total_weight)
                for i in range(len(resolution_stages))
            ]

        if verbose:
            print(f"\n{'='*60}")
            print(f"Multi-Resolution Optimization")
            print(f"Stages: {resolution_stages}, Iterations: {max_iter_per_stage}")
            print(f"Loss: {loss_function}, Preset: {minimizer_preset}")
            print(f"{'='*60}\n")

        # Store current state (might be smoothed version)
        full_res_image = self.image.copy()
        original_params = self.init_parameter_values.copy()

        # Set up target resolution cost function (highest resolution if not native)

        # Scale parameters to target resolution
        target_downsample = int(resolution_stages[-1])
        target_res_params = self._scale_parameters_for_resolution(
            original_params, target_downsample
        )
        # Downsample image for this stage
        if target_downsample > 1:
            h, w = full_res_image.shape[:2]
            target_image = cv2.resize(
                full_res_image,
                (w // target_downsample, h // target_downsample),
                interpolation=cv2.INTER_AREA,
            )
        else:
            target_image = full_res_image.copy()

        if verbose:
            print("Setting up target resolution loss function...")
            print(f"Size: {full_res_image.shape[:2]} → {target_image.shape[:2]}")

        # Update to actual final resolution dimensions
        target_res_params["n_pix_x"] = target_image.shape[1]
        target_res_params["n_pix_y"] = target_image.shape[0]
        target_res_params["x0"] = int(target_image.shape[1] * 0.5)
        target_res_params["y0"] = int(target_image.shape[0] * 0.5)

        # Create temporary cost function at final resolution
        target_cost_fn = CostFunction(
            target=(
                target_image
                if "gradient_field" in loss_function
                else self.features.get("limb")
            ),
            function=limb_arc,
            free_parameters=self.free_parameters,
            init_parameter_values=target_res_params,
            loss_function=loss_function,
            kernel_smoothing=kernel_smoothing,
            directional_smoothing=directional_smoothing,
            directional_decay_rate=directional_decay_rate,
            prefer_direction=prefer_direction,
        )

        # Initialize dashboard for multi-stage if requested
        dash = None
        if dashboard:
            # Create dashboard with multi-stage awareness
            total_iters = sum(max_iter_per_stage)
            # Determine initial resolution label
            first_downsample = resolution_stages[0]
            if first_downsample == 1:
                initial_resolution_label = "full"
            else:
                initial_resolution_label = f"1/{first_downsample}x"

            dash = FitDashboard(
                initial_params=original_params,
                target_planet=target_planet,
                max_iter=max_iter_per_stage[0],  # First stage iterations
                free_params=self.free_parameters,
                total_stages=len(resolution_stages),
                cumulative_max_iter=total_iters,
                **(dashboard_kwargs or {}),
            )
            # Set initial resolution label
            dash.resolution_label = initial_resolution_label

        # Loop through resolution stages
        for stage_idx, (downsample, stage_iter) in enumerate(
            zip(resolution_stages, max_iter_per_stage)
        ):
            # Create resolution label for dashboard
            if downsample == 1:
                resolution_label = "full"
            else:
                resolution_label = f"1/{downsample}x"

            # Start new stage in dashboard (except first stage which starts automatically)
            if dash is not None and stage_idx > 0:
                dash.start_stage(stage_idx + 1, resolution_label, stage_iter)

            if verbose:
                print(f"\n{'─'*60}")
                print(
                    f"Stage {stage_idx + 1}/{len(resolution_stages)}: "
                    f"{resolution_label} ({stage_iter} iter)"
                )
                print(f"{'─'*60}")

            # Downsample image for this stage
            if downsample == resolution_stages[-1]:
                self.image = target_image
            else:
                # elif downsample > 1:
                h, w = full_res_image.shape[:2]
                self.image = cv2.resize(
                    full_res_image,
                    (w // downsample, h // downsample),
                    interpolation=cv2.INTER_AREA,
                )
            # else:
            #     self.image = full_res_image

            if verbose:
                print(f"Size: {full_res_image.shape[:2]} → {self.image.shape[:2]}")

            # Scale parameters for this resolution
            self.init_parameter_values = self._scale_parameters_for_resolution(
                original_params, 1.0 / downsample
            )

            # Warm start from previous stage
            if stage_idx > 0 and hasattr(self, "best_parameters"):
                if verbose:
                    print("Warm start from previous stage:")
                for param in self.free_parameters:
                    if param in self.best_parameters:
                        if verbose:
                            print(
                                f"    {param}: {self.init_parameter_values[param]:.4f} → {self.best_parameters.get(param, 'N/A')}"
                            )
                        self.init_parameter_values[param] = self.best_parameters[param]

            # Scale gradient parameters
            scaled_kernel_smoothing = max(0.5, kernel_smoothing / downsample)
            scaled_directional_smoothing = max(
                5, int(directional_smoothing / downsample)
            )

            if verbose:
                print(f"Gradient params:")
                if downsample > 1:
                    print(
                        f"  kernel_smoothing: {kernel_smoothing:.1f} → {scaled_kernel_smoothing:.1f}"
                    )
                    print(
                        f"  directional_smoothing: {directional_smoothing} → {scaled_directional_smoothing}"
                    )
                else:
                    print(f"  kernel_smoothing: {scaled_kernel_smoothing:.1f}")
                    print(f"  directional_smoothing: {scaled_directional_smoothing}")

            # Fit at this resolution
            if verbose:
                print(f"Starting optimization with:")
                for param in self.free_parameters:
                    print(f"  {param}: {self.init_parameter_values.get(param, 'N/A')}")

            self._fit_single_resolution(
                loss_function=loss_function,
                max_iter=stage_iter,
                n_jobs=n_jobs,
                seed=seed,
                kernel_smoothing=scaled_kernel_smoothing,
                directional_smoothing=scaled_directional_smoothing,
                directional_decay_rate=directional_decay_rate,
                prefer_direction=prefer_direction,
                minimizer=minimizer,
                minimizer_preset=minimizer_preset,
                minimizer_kwargs=minimizer_kwargs,
                dashboard=dashboard,  # Boolean flag
                dashboard_kwargs=dashboard_kwargs,
                target_planet=target_planet,
                verbose=verbose,
                dashboard_obj=dash,  # Pass dashboard object for multi-stage
            )

            if verbose and hasattr(self, "best_parameters"):
                print(f"Fitted parameters:")
                for param in self.free_parameters:
                    print(f"  {param}: {self.best_parameters.get(param, 'N/A')}")

                # Evaluate at target resolution
                target_cost = target_cost_fn.cost(self.fit_results.x)
                print(f"Cost: {target_cost:.6f}")
                if "gradient_field" in loss_function:
                    print(f"Flux: {1.0 - target_cost:.6f}")

        # Restore full resolution
        self.image = full_res_image

        # Note: We don't restore init_parameter_values here because:
        # 1. It's not needed for correctness (next fit will set its own)
        # 2. It would break warm_start (where we want to keep updated values)

        # Scale final solution back to full resolution
        if resolution_stages[-1] != 1:
            scale_factor = resolution_stages[-1]
            self.best_parameters = self._scale_parameters_for_resolution(
                self.best_parameters, scale_factor
            )
            # Update image dimensions to actual full resolution
            self.best_parameters["n_pix_x"] = full_res_image.shape[1]
            self.best_parameters["n_pix_y"] = full_res_image.shape[0]

            # Recompute fitted limb at full resolution
            inferred_parameters = {
                "n_pix_x": full_res_image.shape[1],
                "n_pix_y": full_res_image.shape[0],
                "x0": int(full_res_image.shape[1] * 0.5),
                "y0": int(full_res_image.shape[0] * 0.5),
            }
            full_res_params = self.best_parameters.copy()
            full_res_params.update(inferred_parameters)

            self.cost_function = CostFunction(
                target=(
                    full_res_image
                    if "gradient_field" in loss_function
                    else self.features.get("limb")
                ),
                function=limb_arc,
                free_parameters=self.free_parameters,
                init_parameter_values=full_res_params,
                loss_function=loss_function,
                kernel_smoothing=kernel_smoothing,
                directional_smoothing=directional_smoothing,
                directional_decay_rate=directional_decay_rate,
                prefer_direction=prefer_direction,
            )
            self.features["fitted_limb"] = self.cost_function.evaluate(
                self.best_parameters
            )

        if verbose:
            print(f"\n{'='*60}")
            print("Optimization Complete!")
            print(f"{'='*60}\n")

        # Validate fit results and issue warnings if needed
        _validate_fit_results(self)

        # Finalize dashboard after all stages complete
        if dash is not None:
            success = (
                self.fit_results.success
                if hasattr(self.fit_results, "success")
                else True
            )
            dash.finalize(success=success)

        return self

    def _fit_single_resolution(
        self,
        loss_function: str,
        max_iter: int,
        n_jobs: int,
        seed: int,
        kernel_smoothing: float,
        directional_smoothing: int,
        directional_decay_rate: float,
        prefer_direction: Optional[Literal["up", "down"]] = None,
        minimizer: Optional[
            Literal["differential-evolution", "dual-annealing", "basinhopping"]
        ] = None,
        minimizer_preset: Literal["fast", "balanced", "robust"] = "balanced",
        minimizer_kwargs: Optional[Dict] = None,
        dashboard: bool = False,
        dashboard_kwargs: Optional[Dict] = None,
        target_planet: str = "earth",
        verbose: bool = False,
        dashboard_obj: Optional["FitDashboard"] = None,
    ) -> "LimbObservation":
        """
        Internal method: single-resolution optimization.

        Note: self.image may be the smoothed version if image_smoothing was applied.
        """
        if minimizer is not None:
            self.minimizer = minimizer

        # Setup parameters
        # x0, y0 always inferred from image center (not free parameters)
        inferred_parameters = {
            "n_pix_x": self.image.shape[1],
            "n_pix_y": self.image.shape[0],
            "x0": int(self.image.shape[1] * 0.5),
            "y0": int(self.image.shape[0] * 0.5),
        }
        working_parameters = self.init_parameter_values.copy()
        working_parameters.update(inferred_parameters)

        # Choose target
        if "gradient_field" in loss_function:
            target = self.image
            if verbose:
                print(
                    f"Gradient field: kernel_smoothing={kernel_smoothing}, "
                    f"directional_smoothing={directional_smoothing}, decay={directional_decay_rate}"
                )
        else:
            if "limb" not in self.features:
                raise ValueError(
                    f"Loss '{loss_function}' requires detected limb. "
                    "Use detect_limb() or loss_function='gradient_field'"
                )
            target = self.features["limb"]

        # Create cost function for this resolution
        self.cost_function = CostFunction(
            target=target,
            function=limb_arc,
            free_parameters=self.free_parameters,
            init_parameter_values=working_parameters,
            loss_function=loss_function,
            kernel_smoothing=kernel_smoothing,
            directional_smoothing=directional_smoothing,
            directional_decay_rate=directional_decay_rate,
            prefer_direction=prefer_direction,
        )

        # Initialize dashboard if requested
        dash = dashboard_obj  # Use passed dashboard if provided (multi-stage)
        if dash is None and dashboard:
            # Create new dashboard for single-stage optimization
            dash = FitDashboard(
                initial_params=working_parameters,
                target_planet=target_planet,
                max_iter=max_iter,
                free_params=self.free_parameters,
                **(dashboard_kwargs or {}),
            )

        # Create callback for dashboard updates
        def dashboard_callback(xk, *args, **kwargs):
            """
            Universal callback for all minimizers.

            Signatures vary by minimizer:
            - differential_evolution: callback(xk, convergence=None)
            - dual_annealing: callback(x, f, context)
            - basinhopping: callback(x, f, accept)
            """
            if dash is not None:
                # Unpack parameters
                current_params = unpack_parameters(xk, self.free_parameters)
                full_params = working_parameters.copy()
                full_params.update(current_params)

                # Calculate current loss
                # For dual_annealing and basinhopping, loss is passed as second arg
                if len(args) > 0 and isinstance(args[0], (int, float)):
                    loss = args[0]
                else:
                    loss = self.cost_function.cost(xk)

                # Update dashboard
                dash.update(full_params, loss)
            return False  # Don't stop optimization

        # Get minimizer configuration
        if self.minimizer not in MINIMIZER_PRESETS:
            raise ValueError(f"Unknown minimizer: {self.minimizer}")

        if minimizer_preset not in MINIMIZER_PRESETS[self.minimizer]:
            raise ValueError(
                f"Unknown preset '{minimizer_preset}' for {self.minimizer}. "
                f"Choose from: {list(MINIMIZER_PRESETS[self.minimizer].keys())}"
            )

        # Start with preset, then apply overrides
        config = MINIMIZER_PRESETS[self.minimizer][minimizer_preset].copy()
        if minimizer_kwargs:
            config.update(minimizer_kwargs)

        # Prepare bounds and initial guess
        bounds = [self.parameter_limits[key] for key in self.free_parameters]
        x0 = [working_parameters[key] for key in self.free_parameters]

        # Clamp x0 to be strictly within bounds (avoid edge case violations)
        x0_clamped = []
        for val, (lower, upper) in zip(x0, bounds):
            # Ensure value is strictly within bounds with small epsilon
            epsilon = 1e-9 * (upper - lower)  # Relative epsilon
            clamped_val = max(lower + epsilon, min(upper - epsilon, val))
            if clamped_val != val:
                import warnings

                warnings.warn(
                    f"Initial parameter {val} clamped to [{lower}, {upper}]",
                    UserWarning,
                )
            x0_clamped.append(clamped_val)
        x0 = x0_clamped

        # Run minimizer
        if self.minimizer == "differential-evolution":
            updating = "deferred" if n_jobs > 1 else "immediate"

            self.fit_results = differential_evolution(
                self.cost_function.cost,
                bounds,
                x0=x0,
                workers=n_jobs,
                maxiter=max_iter,
                updating=updating,
                callback=dashboard_callback if dashboard else None,
                disp=verbose,
                seed=seed,
                **config,  # Apply preset + overrides
            )

        elif self.minimizer == "dual-annealing":
            from scipy.optimize import dual_annealing

            self.fit_results = dual_annealing(
                self.cost_function.cost,
                bounds=bounds,
                x0=x0,
                maxiter=max_iter,
                callback=dashboard_callback if dashboard else None,
                seed=seed,
                **config,
            )

        elif self.minimizer == "basinhopping":
            from scipy.optimize import basinhopping

            class BoundsChecker:
                def __init__(self, bounds):
                    self.bounds = bounds

                def __call__(self, **kwargs):
                    x = kwargs["x_new"]
                    return all(l <= xi <= u for xi, (l, u) in zip(x, self.bounds))

            local_maxiter = config.pop("local_maxiter", 100)
            minimizer_kwargs_local = {
                "method": "L-BFGS-B",
                "bounds": bounds,
                "options": {"maxiter": local_maxiter, "ftol": 1e-6},
            }

            self.fit_results = basinhopping(
                self.cost_function.cost,
                x0,
                minimizer_kwargs=minimizer_kwargs_local,
                accept_test=BoundsChecker(bounds),
                callback=dashboard_callback if dashboard else None,
                interval=20,
                disp=verbose,
                seed=seed,
                **config,
            )

        # Extract results
        best_parameters = unpack_parameters(self.fit_results.x, self.free_parameters)
        working_parameters.update(best_parameters)
        self.best_parameters = working_parameters
        self.features["fitted_limb"] = self.cost_function.evaluate(self.best_parameters)
        self._plot_functions["fitted_limb"] = plot_limb

        # Validate fit results and issue warnings if needed
        _validate_fit_results(self)

        # Finalize dashboard only if we created it (single-stage)
        # Multi-stage will finalize after all stages complete
        if dash is not None and dashboard_obj is None:
            success = (
                self.fit_results.success
                if hasattr(self.fit_results, "success")
                else True
            )
            dash.finalize(success=success)

        return self

    def _scale_parameters_for_resolution(
        self, params: Dict, scale_factor: float
    ) -> Dict:
        """
        Scale parameters for different image resolution.

        Args:
            params: Parameter dictionary
            scale_factor: Resolution scale (0.5 = half res, 2.0 = double res)

        Returns:
            Scaled parameters
        """
        scaled = params.copy()

        # Parameters that scale with image dimensions
        pixel_params = ["n_pix_x", "n_pix_y", "x0", "y0"]
        for key in pixel_params:
            if key in scaled:
                scaled[key] = int(scaled[key] * scale_factor)

        # Parameters that don't scale (physical units)
        # r, h, f, fov, theta_x, theta_y, theta_z remain unchanged

        return scaled

    def save_limb(self, filepath: str) -> None:
        """
        Save the detected limb position as a numpy array.

        Args:
            filepath (str): Path to save file.
        """
        np.save(filepath, self.features["limb"])

    def load_limb(self, filepath: str) -> None:
        """
        Load the detected limb position from a numpy array.

        Args:
            filepath (str): Path to save file.
        """
        self.features["limb"] = np.load(filepath)
        self.features["limb"] = fill_nans(self.features["limb"])
        self._plot_functions["limb"] = plot_limb
        self._raw_limb = self.features["limb"].copy()


def package_results(observation: LimbObservation) -> pd.DataFrame:
    """
    Consolidate the results of a fit to see final vs. initial values.

    Args:
        observation (object): Instance of LimbObservation (must have
            used differential evolution minimizer).

    Returns:
        results (pd.DataFrame): DataFrame of results including
            - fit value
            - initial value
            - parameter
    """
    full_fit_params = unpack_parameters(
        observation.fit_results.x, observation.free_parameters
    )

    results = []
    for key in observation.free_parameters:
        result = {
            "fit value": full_fit_params[key],
            "initial value": observation.init_parameter_values[key],
            "parameter": key,
        }
        results.append(result)
    results = pd.DataFrame.from_records(results)
    results = results.set_index(["parameter"])
    return results
