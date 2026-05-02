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
from planet_ruler.fit import LimbFitter, SagittaFitter, unpack_parameters, _validate_fit_results
from planet_ruler.geometry import limb_arc, focal_length, detector_size
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
            "initial_temp": 5230.0,
            "restart_temp_ratio": 2e-05,
            "visit": 2.62,
            "accept": -5.0,
            "no_local_search": False,
        },
    },
    "basinhopping": {
        "fast":        {"niter": 100,  "T": 1.5, "stepsize": 0.5, "local_maxiter": 50},
        "balanced":    {"niter": 200,  "T": 2.0, "stepsize": 0.5, "local_maxiter": 100},
        "robust":      {"niter": 500,  "T": 3.0, "stepsize": 0.7, "local_maxiter": 200},
        "super_robust":{"niter": 2000, "T": 5.0, "stepsize": 1.5, "local_maxiter": 500},
        "scipy-default": {
            # Exact scipy basinhopping defaults
            "niter": 100,
            "T": 1.0,
            "stepsize": 0.5,
        },
    },
    "scipy-minimize": {
        "fast":        {"method": "L-BFGS-B", "n_restarts": 5,   "ftol": 1e-6},
        "balanced":    {"method": "L-BFGS-B", "n_restarts": 20,  "ftol": 1e-8},
        "robust":      {"method": "Powell",   "n_restarts": 50,  "ftol": 1e-10},
        "super_robust":{"method": "Powell",   "n_restarts": 200, "ftol": 1e-12},
        "scipy-default": {
            # Exact scipy minimize defaults (L-BFGS-B, single run)
            "method": "L-BFGS-B",
            "n_restarts": 1,
        },
    },
    "shgo": {
        "fast":    {"n": 50,  "iters": 1},
        "balanced":{"n": 100, "iters": 2},
        "robust":  {"n": 200, "iters": 3},
        "scipy-default": {
            # Exact scipy shgo defaults
            "n": 100,
            "iters": 1,
        },
    },
}

# Per-detection-method preset overrides. Checked before MINIMIZER_PRESETS.
# Lets "balanced" mean different things for manual vs gradient-field fitting.
METHOD_MINIMIZER_PRESETS: dict = {
    "manual": {
        "basinhopping": {
            "fast":        {"niter": 100,  "T": 1.5, "stepsize": 0.5, "local_maxiter": 50},
            "balanced":    {"niter": 200,  "T": 2.0, "stepsize": 0.5, "local_maxiter": 100},
            "robust":      {"niter": 500,  "T": 3.0, "stepsize": 0.7, "local_maxiter": 200},
            "super_robust":{"niter": 2000, "T": 5.0, "stepsize": 1.5, "local_maxiter": 500},
        },
        "scipy-minimize": {
            "fast":        {"method": "L-BFGS-B", "n_restarts": 5,   "ftol": 1e-6},
            "balanced":    {"method": "L-BFGS-B", "n_restarts": 20,  "ftol": 1e-8},
            "robust":      {"method": "L-BFGS-B", "n_restarts": 50,  "ftol": 1e-10},
            "super_robust":{"method": "Powell",   "n_restarts": 200, "ftol": 1e-12},
        },
    },
}


def _resolve_preset_kwargs(minimizer: str, preset: str, detection_method: Optional[str] = None) -> dict:
    """Return preset kwargs, preferring method-specific overrides when available."""
    if detection_method:
        method_entry = METHOD_MINIMIZER_PRESETS.get(detection_method, {})
        minimizer_entry = method_entry.get(minimizer, {})
        if preset in minimizer_entry:
            return dict(minimizer_entry[preset])
    return dict(MINIMIZER_PRESETS.get(minimizer, {}).get(preset, {}))


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

    def crop_image(
        self,
        update_parameters: bool = True,
    ) -> "PlanetObservation":
        """
        Interactively crop the observation image with automatic parameter scaling.

        Opens a GUI tool allowing the user to select a rectangular region to crop.
        If the observation has camera parameters (e.g., LimbObservation), they are
        automatically scaled to match the cropped region.

        Note: The principal point may end up outside the cropped region bounds
        (negative coordinates). This is geometrically valid - it means the camera's
        optical axis was pointing at a location not visible in the cropped image.

        Args:
            update_parameters: If True and parameters exist, scale them for the crop

        Returns:
            self: For method chaining

        Example:
            >>> obs = LimbObservation("airplane.jpg", "config.yaml")
            >>> obs.crop_image()  # Opens interactive crop tool
            >>> obs.detect_limb()
            >>> obs.fit_arc()

        Notes:
            - Works for any PlanetObservation subclass
            - For LimbObservation: automatically scales detector_size, principal_point, etc.
            - Principal point can be outside cropped bounds (mathematically valid)
        """
        from planet_ruler.crop import TkImageCropper
        from pathlib import Path
        import logging
        import numpy as np
        from PIL import Image as PILImage

        logger = logging.getLogger(__name__)

        # Check if this observation has camera parameters (duck typing)
        has_parameters = (
            hasattr(self, "init_parameter_values")
            and self.init_parameter_values is not None
        )

        # Prepare current parameters for the cropper (if they exist)
        current_params = {}
        if has_parameters:
            current_params = {
                "w": self.init_parameter_values.get("w"),
                "n_pix_x": self.image.shape[1],
                "n_pix_y": self.image.shape[0],
                "x0": self.init_parameter_values.get("x0", self.image.shape[1] / 2),
                "y0": self.init_parameter_values.get("y0", self.image.shape[0] / 2),
            }

            # Include h_detector if present
            if "h_detector" in self.init_parameter_values:
                current_params["h_detector"] = self.init_parameter_values["h_detector"]

            logger.info("Opening interactive crop tool...")
            logger.info(f"Current detector width: {current_params['w']*1000:.3f}mm")
            logger.info(
                f"Current image size: {current_params['n_pix_x']}×{current_params['n_pix_y']}px"
            )
            logger.info(
                f"Current principal point: ({current_params['x0']:.0f}, {current_params['y0']:.0f})"
            )
        else:
            logger.info("Opening interactive crop tool (no parameter scaling)...")
            logger.info(
                f"Current image size: {self.image.shape[1]}×{self.image.shape[0]}px"
            )

        # Run the crop tool
        cropper = TkImageCropper(
            self.image_filepath, current_params if has_parameters else None
        )
        cropper.run()

        # Check if user completed the crop
        if cropper.cropped_image is None:
            logger.warning("Crop cancelled - no changes made")
            return self

        # Get results
        cropped_image = cropper.cropped_image
        crop_bounds = cropper.get_crop_bounds()
        logger.info(f"Crop complete: {crop_bounds}")

        # Update the image (always happens)
        if isinstance(cropped_image, PILImage.Image):
            self.image = np.array(cropped_image)
        else:
            self.image = cropped_image

        # Update original image reference
        self._original_image = self.image.copy()

        # Update parameters (only if they exist and user wants to)
        if has_parameters and update_parameters:
            scaled_params = cropper.scaled_parameters

            logger.info("Updating observation parameters...")
            logger.info(f"New detector width: {scaled_params['w']*1000:.3f}mm")
            logger.info(
                f"New image size: {scaled_params['n_pix_x']}×{scaled_params['n_pix_y']}px"
            )

            # ================================================================
            # CHECK: Inform user if principal point is out of bounds
            # (But don't change it - the math works fine)
            # ================================================================
            x0_in_bounds = 0 <= scaled_params["x0"] < scaled_params["n_pix_x"]
            y0_in_bounds = 0 <= scaled_params["y0"] < scaled_params["n_pix_y"]

            if not x0_in_bounds or not y0_in_bounds:
                logger.info(
                    f"Principal point ({scaled_params['x0']:.0f}, {scaled_params['y0']:.0f}) "
                    f"is outside cropped image bounds (0-{scaled_params['n_pix_x']}, "
                    f"0-{scaled_params['n_pix_y']}.  Should still be geometrically valid.)"
                )
            else:
                logger.info(
                    f"New principal point: ({scaled_params['x0']:.0f}, {scaled_params['y0']:.0f}) [✓ in bounds]"
                )

            # Update pixel dimensions
            self.init_parameter_values["n_pix_x"] = scaled_params["n_pix_x"]
            self.init_parameter_values["n_pix_y"] = scaled_params["n_pix_y"]

            # Update principal point
            self.init_parameter_values["x0"] = scaled_params["x0"]
            self.init_parameter_values["y0"] = scaled_params["y0"]

            # Update detector size (CRITICAL)
            if "w" in scaled_params:
                old_w = self.init_parameter_values.get("w", 0)
                self.init_parameter_values["w"] = scaled_params["w"]
                logger.info(
                    f"  Scaled detector width: {old_w*1000:.3f}mm → "
                    f"{scaled_params['w']*1000:.3f}mm"
                )

            # Update detector height if present
            if "h_detector" in scaled_params:
                self.init_parameter_values["h_detector"] = scaled_params["h_detector"]

            # Update parameter limits for detector width if they exist
            if hasattr(self, "parameter_limits") and self.parameter_limits is not None:
                if (
                    "w" in self.parameter_limits
                    and "w" in scaled_params
                    and "w" in current_params
                ):
                    # Scale the limits proportionally
                    crop_ratio = scaled_params["w"] / current_params["w"]
                    old_limits = self.parameter_limits["w"]
                    self.parameter_limits["w"] = [
                        old_limits[0] * crop_ratio,
                        old_limits[1] * crop_ratio,
                    ]
                    logger.info(
                        f"  Scaled detector width limits: "
                        f"[{old_limits[0]*1000:.3f}, {old_limits[1]*1000:.3f}]mm → "
                        f"[{self.parameter_limits['w'][0]*1000:.3f}, "
                        f"{self.parameter_limits['w'][1]*1000:.3f}]mm"
                    )

            # Note: Focal length (f) is NOT updated - it's a lens property
            # Field of view will automatically update via geometry.field_of_view(f, w)

            logger.info("Parameters updated successfully")

        elif has_parameters and not update_parameters:
            logger.info("Parameters NOT updated (update_parameters=False)")

        elif not has_parameters:
            logger.info(
                "No parameters to update (observation type has no camera parameters)"
            )

        return self


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

        valid_minimizers = ["differential-evolution", "dual-annealing", "basinhopping", "scipy-minimize", "shgo"]
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
        self.best_parameters = None
        self.fit_results = None
        self.stage_results: List[dict] = []

    def analyze(
        self,
        detect_limb_kwargs: Optional[dict] = None,
        fit_method: str = "arc",
        fit_method_kwargs: Optional[dict] = None,
        fit_stages: Optional[List[dict]] = None,
    ) -> "LimbObservation":
        """
        Perform complete limb analysis: detection + fitting in one call.

        Args:
            detect_limb_kwargs: Optional kwargs for detect_limb().
            fit_method: Single-stage method -- "arc" (default), "gradient", or "sagitta".
                Ignored when fit_stages is provided.
            fit_method_kwargs: Optional kwargs for the chosen fit method.
            fit_stages: If provided, run fit_limb(fit_stages) (multi-stage orchestrator).

        Returns:
            self: For method chaining.
        """
        self.detect_limb(**(detect_limb_kwargs or {}))
        if fit_stages is not None:
            self.fit_limb(fit_stages)
        elif fit_method == "sagitta":
            self.fit_sagitta(**(fit_method_kwargs or {}))
        elif fit_method == "arc":
            self.fit_arc(**(fit_method_kwargs or {}))
        elif fit_method == "gradient":
            self.fit_gradient(**(fit_method_kwargs or {}))
        else:
            raise ValueError(f"Unknown fit_method: {fit_method!r}")
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
        f = self.best_parameters.get("f")
        if f is None:
            w = self.best_parameters.get("w")
            fov = self.best_parameters.get("fov")
            if w is not None and fov is not None:
                f = focal_length(w, fov)
            else:
                return 0.0
        return f * 1000.0

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

    @property
    def methods(self) -> list:
        """Fit methods applied, in order."""
        return [s.get("method") for s in self.stage_results]

    @property
    def minimizers(self) -> list:
        """Minimizers used per stage (None for sagitta)."""
        return [s.get("minimizer") for s in self.stage_results]

    def reset_params(self) -> "LimbObservation":
        """Restore init_parameter_values to the original loaded values (cold start)."""
        self.init_parameter_values = self._original_init_parameter_values.copy()
        return self

    def _apply_updated_limits(self, updated_limits: dict) -> None:
        """Apply stage-returned bounds, always intersecting (never widening)."""
        for param, bounds in updated_limits.items():
            lo, hi = float(bounds[0]), float(bounds[1])
            if param in self.parameter_limits:
                cur_lo, cur_hi = self.parameter_limits[param]
                self.parameter_limits[param] = [max(cur_lo, lo), min(cur_hi, hi)]
            else:
                self.parameter_limits[param] = [lo, hi]

    def _apply_updated_init(self, updated_init: dict) -> None:
        """Apply stage-returned init values, clipping each to its current bounds."""
        for k, v in updated_init.items():
            if k in self.parameter_limits:
                lo, hi = self.parameter_limits[k]
                v = max(lo, min(hi, float(v)))
            self.init_parameter_values[k] = v

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
                image_path=self.image_filepath,
                image=self.image,
                initial_stretch=1.0,
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

    def fit_sagitta(
        self,
        n_sigma: float = 2.0,
        bias_correct: bool = False,
        uncertainty: str = "both",
    ) -> "LimbObservation":
        """
        Estimate planetary radius using the sagitta (arc-height) method.

        Runs SagittaFitter: a 2-D L-BFGS-B optimizer over (kappa, theta_x)
        using the exact projected-sagitta formula.  Updates self.init_parameter_values
        and self.parameter_limits so that a subsequent fit_arc/fit_gradient stage
        warm-starts from the result automatically.

        Args:
            n_sigma: Radius bound width in sigma units.
            bias_correct: If True and y0 is provided, correct for camera tilt bias using
                the apex y-offset to estimate theta_x, then refine kappa via 1-D minimization.
            uncertainty: Bound-width method — "ols" (cheap, single pass),
                "jackknife" (leave-one-out), or "both" (quadrature sum, default).

        Returns:
            self: For method chaining.
        """
        if "limb" not in self.features:
            raise ValueError(
                "Must detect limb before fitting. Call detect_limb() first."
            )

        f = self.init_parameter_values.get("f")
        w_sensor = self.init_parameter_values.get("w")
        fov = self.init_parameter_values.get("fov")

        if f is None and w_sensor is not None and fov is not None:
            f = focal_length(w_sensor, fov)
        elif w_sensor is None and f is not None and fov is not None:
            w_sensor = detector_size(f, fov)

        if f is None or w_sensor is None:
            raise ValueError(
                "fit_sagitta requires at least two of (f, fov, w) in "
                "init_parameter_values to compute the pixel focal length."
            )

        n_pix_x = self.image.shape[1]
        f_px = f / w_sensor * n_pix_x
        h = self.init_parameter_values["h"]
        y0 = self.image.shape[0] / 2.0

        fitter = SagittaFitter(
            limb=self.features["limb"],
            h=h,
            f_px=f_px,
            y0=y0,
            free_parameters=self.free_parameters,
            init_parameter_values=self.init_parameter_values,
            parameter_limits=self.parameter_limits,
            sigma_px="auto",
            n_sigma=n_sigma,
            bias_correct=bias_correct,
            uncertainty=uncertainty,
        )

        result = fitter.fit()

        self._apply_updated_limits(result.get("updated_limits", {}))
        self._apply_updated_init(result.get("updated_init", {}))

        working_parameters = self.init_parameter_values.copy()
        working_parameters.update(
            {
                "n_pix_x": self.image.shape[1],
                "n_pix_y": self.image.shape[0],
                "x0": int(self.image.shape[1] * 0.5),
                "y0": int(self.image.shape[0] * 0.5),
            }
        )
        self.best_parameters = working_parameters

        stage_entry = {
            "method": "sagitta",
            "n_sigma": n_sigma,
            "uncertainty": uncertainty,
            "bias_correct": bias_correct,
        }
        stage_entry.update(
            {
                k: v
                for k, v in result.items()
                if k not in ("updated_init", "updated_limits")
            }
        )
        self.stage_results.append(stage_entry)

        for w_msg in result.get("warnings", []):
            logging.warning(w_msg)

        return self

    def fit_arc(
        self,
        loss_function: Literal["l2", "l1", "log-l1"] = "l2",
        minimizer: Optional[
            Literal[
                "differential-evolution",
                "dual-annealing",
                "basinhopping",
                "scipy-minimize",
                "shgo",
            ]
        ] = None,
        minimizer_preset: Literal[
            "fast", "balanced", "robust", "super_robust", "scipy-default"
        ] = "balanced",
        minimizer_kwargs: Optional[Dict] = None,
        max_iter: int = 15000,
        seed: int = 0,
        verbose: bool = False,
        n_jobs: int = 1,
        _dashboard=None,
    ) -> "LimbObservation":
        """
        Fit the planet limb arc using a pixel-space cost function.

        Requires a detected limb (call detect_limb() first, or register_limb()).
        Reads self.init_parameter_values as the initial guess and writes fitted
        free-parameter values back so a chained call warm-starts automatically.

        Args:
            loss_function: "l2" (default), "l1", or "log-l1".
            minimizer: Minimizer name. Defaults to self.minimizer.
            minimizer_preset: Preset config.
            minimizer_kwargs: Override specific minimizer kwargs.
            max_iter: Maximum iterations.
            seed: Random seed.
            verbose: Print progress.
            n_jobs: Parallel workers. Effective only for differential-evolution
                and shgo; emits a UserWarning for other minimizers.
            _dashboard: Internal — FitDashboard instance passed by fit_limb.

        Returns:
            self: For method chaining.
        """
        if "limb" not in self.features:
            raise ValueError(
                "Must detect limb before fitting. Call detect_limb() first."
            )

        effective_minimizer = minimizer or self.minimizer
        if verbose:
            print(f"Using minimizer: {effective_minimizer}")

        valid_presets = set(MINIMIZER_PRESETS.get(effective_minimizer, {}).keys())
        if valid_presets and minimizer_preset not in valid_presets:
            raise ValueError(
                f"Unknown preset {minimizer_preset!r} for minimizer {effective_minimizer!r}. "
                f"Valid presets: {sorted(valid_presets)}"
            )

        preset_kw = _resolve_preset_kwargs(
            effective_minimizer, minimizer_preset, self.limb_detection
        )
        if minimizer_kwargs:
            preset_kw.update(minimizer_kwargs)

        working_parameters = self.init_parameter_values.copy()
        working_parameters.update(
            {
                "n_pix_x": self.image.shape[1],
                "n_pix_y": self.image.shape[0],
                "x0": int(self.image.shape[1] * 0.5),
                "y0": int(self.image.shape[0] * 0.5),
            }
        )

        if _dashboard is not None:
            _free = self.free_parameters

            def _arc_db_callback(xk, convergence=None, *args, **kwargs):
                _dashboard.update(unpack_parameters(xk, _free), convergence or 0.0)
                return False

            preset_kw["callback"] = _arc_db_callback

        fitter = LimbFitter(
            target=self.features["limb"],
            free_parameters=self.free_parameters,
            init_parameter_values=working_parameters,
            parameter_limits=self.parameter_limits,
            loss_function=loss_function,
            minimizer=effective_minimizer,
            minimizer_kwargs=preset_kw,
            max_iter=max_iter,
            seed=seed,
            verbose=verbose,
            n_jobs=n_jobs,
        )

        result = fitter.fit()

        self._apply_updated_limits(result.get("updated_limits", {}))
        self._apply_updated_init(result.get("updated_init", {}))

        self.best_parameters = result["best_parameters"]
        self.fit_results = result["fit_results"]

        kwargs = self.init_parameter_values.copy()
        kwargs.update(self.best_parameters)
        self.features["fitted_limb"] = limb_arc(**kwargs)
        self._plot_functions["fitted_limb"] = plot_limb

        _validate_fit_results(self)

        self.stage_results.append(
            {
                "method": "arc",
                "loss_function": loss_function,
                "minimizer": effective_minimizer,
                "minimizer_preset": minimizer_preset,
                "best_parameters": self.best_parameters,
                "fit_results": self.fit_results,
                "status": result.get("status", "ok"),
                "warnings": result.get("warnings", []),
            }
        )

        return self

    def fit_gradient(
        self,
        resolution_stages: Optional[List[int] | Literal["auto"]] = None,
        minimizer: Optional[
            Literal[
                "differential-evolution",
                "dual-annealing",
                "basinhopping",
                "scipy-minimize",
                "shgo",
            ]
        ] = None,
        minimizer_preset: Literal[
            "fast", "balanced", "robust", "super_robust", "scipy-default"
        ] = "balanced",
        minimizer_kwargs: Optional[Dict] = None,
        image_smoothing: Optional[float] = None,
        kernel_smoothing: float = 5.0,
        directional_smoothing: int = 50,
        directional_decay_rate: float = 0.15,
        prefer_direction: Optional[Literal["up", "down"]] = None,
        max_iter: int = 15000,
        max_iter_per_stage: Optional[List[int]] = None,
        seed: int = 0,
        verbose: bool = False,
        n_jobs: int = 1,
        _dashboard=None,
    ) -> "LimbObservation":
        """
        Fit using gradient field alignment, optionally with coarse-to-fine stages.

        Does not require a pre-detected limb; operates directly on the image.
        Multi-resolution stages warm-start from the previous stage automatically.

        Args:
            resolution_stages: Downsampling factors, e.g. [4, 2, 1] (coarse→fine).
                None → single full-resolution stage. "auto" → inferred from image size.
            minimizer: Minimizer name. Defaults to self.minimizer.
            minimizer_preset: Preset config.
            minimizer_kwargs: Override specific minimizer kwargs.
            image_smoothing: Gaussian blur sigma applied before optimisation
                (removes high-frequency artifacts; original image restored after).
            kernel_smoothing: Gradient field kernel size.
            directional_smoothing: Directional sampling distance along gradients.
            directional_decay_rate: Exponential decay for directional samples.
            prefer_direction: "up" (dark sky / bright planet), "down", or None.
            max_iter: Total maximum iterations (split across stages if multi-res).
            max_iter_per_stage: Explicit per-stage iteration budget.
            seed: Random seed.
            verbose: Print progress.
            n_jobs: Parallel workers. Effective only for differential-evolution
                and shgo; emits a UserWarning for other minimizers.

        Returns:
            self: For method chaining.
        """
        effective_minimizer = minimizer or self.minimizer
        if verbose:
            print(f"Using minimizer: {effective_minimizer}")

        valid_presets = set(MINIMIZER_PRESETS.get(effective_minimizer, {}).keys())
        if valid_presets and minimizer_preset not in valid_presets:
            raise ValueError(
                f"Unknown preset {minimizer_preset!r} for minimizer {effective_minimizer!r}. "
                f"Valid presets: {sorted(valid_presets)}"
            )

        original_image = None
        if image_smoothing is not None and image_smoothing > 0:
            original_image = self.image.copy()
            if verbose:
                print(f"Applying Gaussian blur to image (sigma={image_smoothing:.1f})")
            self.image = cv2.GaussianBlur(
                self.image.astype(np.float32),
                (0, 0),
                sigmaX=image_smoothing,
                sigmaY=image_smoothing,
            )

        if resolution_stages is None:
            resolution_stages = [1]
        elif resolution_stages == "auto":
            min_dim = min(self.image.shape[:2])
            if min_dim >= 2000:
                resolution_stages = [4, 2, 1]
            elif min_dim >= 1000:
                resolution_stages = [2, 1]
            else:
                resolution_stages = [1]

        if max_iter_per_stage is None:
            total_weight = sum(range(1, len(resolution_stages) + 1))
            max_iter_per_stage = [
                int(max_iter * (i + 1) / total_weight)
                for i in range(len(resolution_stages))
            ]

        result: dict = {}
        try:
            full_res_image = self.image.copy()

            for stage_idx, (downsample, stage_iter) in enumerate(
                zip(resolution_stages, max_iter_per_stage)
            ):
                if downsample > 1:
                    img_h, img_w = full_res_image.shape[:2]
                    stage_image = cv2.resize(
                        full_res_image,
                        (img_w // downsample, img_h // downsample),
                        interpolation=cv2.INTER_AREA,
                    )
                else:
                    stage_image = full_res_image

                scaled_kernel = max(0.5, kernel_smoothing / downsample)
                scaled_directional = max(5, int(directional_smoothing / downsample))

                working_params = self.init_parameter_values.copy()
                if stage_idx > 0 and self.best_parameters is not None:
                    for param in self.free_parameters:
                        if param in self.best_parameters:
                            working_params[param] = self.best_parameters[param]

                if verbose:
                    print(
                        f"\nStage {stage_idx + 1}/{len(resolution_stages)}: "
                        f"1/{downsample}x ({stage_iter} iter)"
                    )

                preset_kw = _resolve_preset_kwargs(
                    effective_minimizer, minimizer_preset, self.limb_detection
                )
                if minimizer_kwargs:
                    preset_kw.update(minimizer_kwargs)

                if _dashboard is not None:
                    if stage_idx > 0:
                        _lbl = "full" if downsample == 1 else f"1/{downsample}x"
                        _dashboard.start_stage(stage_idx + 1, _lbl, stage_iter)
                    _free = self.free_parameters

                    def _grad_db_callback(xk, convergence=None, *args, **kwargs):
                        _dashboard.update(
                            unpack_parameters(xk, _free), convergence or 0.0
                        )
                        return False

                    preset_kw["callback"] = _grad_db_callback

                fitter = LimbFitter(
                    target=stage_image,
                    free_parameters=self.free_parameters,
                    init_parameter_values=working_params,
                    parameter_limits=self.parameter_limits,
                    loss_function="gradient_field",
                    minimizer=effective_minimizer,
                    minimizer_kwargs=preset_kw,
                    max_iter=stage_iter,
                    seed=seed,
                    verbose=verbose,
                    kernel_smoothing=scaled_kernel,
                    directional_smoothing=scaled_directional,
                    directional_decay_rate=directional_decay_rate,
                    prefer_direction=prefer_direction,
                    n_jobs=n_jobs,
                )

                result = fitter.fit()

                self._apply_updated_limits(result.get("updated_limits", {}))
                self._apply_updated_init(result.get("updated_init", {}))

                self.best_parameters = result["best_parameters"]
                self.fit_results = result["fit_results"]

            if resolution_stages and resolution_stages[-1] != 1:
                self.best_parameters["n_pix_x"] = full_res_image.shape[1]
                self.best_parameters["n_pix_y"] = full_res_image.shape[0]
                self.best_parameters["x0"] = int(full_res_image.shape[1] * 0.5)
                self.best_parameters["y0"] = int(full_res_image.shape[0] * 0.5)

        finally:
            if original_image is not None:
                self.image = original_image

        if self.best_parameters is not None:
            kwargs = self.init_parameter_values.copy()
            kwargs.update(self.best_parameters)
            self.features["fitted_limb"] = limb_arc(**kwargs)
            self._plot_functions["fitted_limb"] = plot_limb

        _validate_fit_results(self)

        self.stage_results.append(
            {
                "method": "gradient",
                "resolution_stages": resolution_stages,
                "minimizer": effective_minimizer,
                "minimizer_preset": minimizer_preset,
                "best_parameters": self.best_parameters,
                "fit_results": self.fit_results,
                "status": result.get("status", "ok"),
                "warnings": result.get("warnings", []),
            }
        )

        return self

    def fit_limb(
        self,
        stages: List[dict],
        dashboard: bool = False,
        dashboard_kwargs: Optional[Dict] = None,
        target_planet: str = "earth",
        n_jobs: int = 1,
    ) -> "LimbObservation":
        """
        Run a multi-stage fit by sequentially calling fit_sagitta / fit_arc / fit_gradient.

        Resets self.stage_results before running so the list reflects only this call.
        Individual fit_* calls accumulate into stage_results without resetting.

        Args:
            stages: Ordered list of stage dicts, each with a "method" key
                ("sagitta", "arc", or "gradient") plus kwargs for that method.
            dashboard: Show a live FitDashboard during optimization.
            dashboard_kwargs: Extra kwargs forwarded to FitDashboard.__init__.
            target_planet: Planet name for dashboard reference radius.
            n_jobs: Parallel workers forwarded to each arc/gradient stage.
                Effective only for differential-evolution and shgo.

        Returns:
            self: For method chaining.

        Example::

            obs.fit_limb([
                {"method": "sagitta", "n_sigma": 2.0},
                {"method": "arc", "minimizer": "basinhopping",
                 "minimizer_preset": "robust"},
            ])
        """
        self.stage_results = []

        db = None
        if dashboard:
            # Count optimizer sub-stages (sagitta uses cheap regression, not counted)
            total_opt_stages = 0
            first_max_iter = None
            cumulative_max_iter = 0
            for s in stages:
                m = s.get("method")
                mi = s.get("max_iter", 15000)
                if m == "arc":
                    total_opt_stages += 1
                    if first_max_iter is None:
                        first_max_iter = mi
                    cumulative_max_iter += mi
                elif m == "gradient":
                    res = s.get("resolution_stages")
                    mips = s.get("max_iter_per_stage")
                    if mips is not None:
                        n = len(mips)
                        stage0_mi = mips[0]
                    elif res is not None and res != "auto":
                        n = len(res)
                        tw = sum(range(1, n + 1))
                        stage0_mi = int(mi * 1 / tw)
                    else:
                        n = 1
                        stage0_mi = mi
                    total_opt_stages += n
                    if first_max_iter is None:
                        first_max_iter = stage0_mi
                    cumulative_max_iter += mi

            working_params = self.init_parameter_values.copy()
            working_params.update(
                {
                    "n_pix_x": self.image.shape[1],
                    "n_pix_y": self.image.shape[0],
                    "x0": int(self.image.shape[1] * 0.5),
                    "y0": int(self.image.shape[0] * 0.5),
                }
            )
            db_kw = dict(dashboard_kwargs or {})
            db = FitDashboard(
                initial_params=working_params,
                target_planet=target_planet,
                total_stages=max(1, total_opt_stages),
                max_iter=first_max_iter or 15000,
                cumulative_max_iter=cumulative_max_iter or None,
                free_params=self.free_parameters,
                **db_kw,
            )

        try:
            for stage in stages:
                stage = dict(stage)
                method = stage.pop("method")
                if method == "sagitta":
                    self.fit_sagitta(**stage)
                elif method == "arc":
                    stage.setdefault("n_jobs", n_jobs)
                    self.fit_arc(_dashboard=db, **stage)
                elif method == "gradient":
                    stage.setdefault("n_jobs", n_jobs)
                    self.fit_gradient(_dashboard=db, **stage)
                else:
                    raise ValueError(f"Unknown fit method: {method!r}")
        finally:
            if db is not None:
                db.finalize()

        return self

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
