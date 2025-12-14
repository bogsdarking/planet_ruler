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
from typing import Optional
import numpy as np
import pandas as pd
import math
import warnings
from planet_ruler.image import (
    bidirectional_gradient_blur,
    bilinear_interpolate,
    gradient_field,
)
from typing import Callable


def unpack_parameters(params: list, template: list) -> dict:
    """
    Turn a list of parameters back into a dict.

    Args:
        params (list): Values of dictionary elements in a list.
        template (list): Ordered list of target keys.

    Returns:
        param_dict (dict): Parameter dictionary.
    """
    return {key: params[i] for i, key in enumerate(template)}


def pack_parameters(params: dict, template: dict) -> list:
    """
    Turn a dict of parameters (or defaults) into a list.

    Args:
        params (dict): Parameter dictionary (subset or full keys of template).
        template (dict): Template (full) parameter dictionary.

    Returns:
        param_list (list): List of parameter values.
    """
    return [params[key] if key in params else template[key] for key in template]


class CostFunction:
    """
    Wrapper to simplify interface with the minimization at hand.

    Args:
        target (np.ndarray): True value(s), e.g., the actual limb position.
                            For gradient_field loss, this should be the image.
        function (Callable): Function mapping parameters to target of interest.
        free_parameters (list): List of free parameter names.
        init_parameter_values (dict): Initial values for named parameters.
        loss_function (str): Type of loss function, must be one of
                            ['l2', 'l1', 'log-l1', 'gradient_field'].
        kernel_smoothing: For gradient_field - initial blur for gradient direction
            estimation. Makes the gradient field smoother for directional sampling.
        directional_smoothing: For gradient_field - sampling distance along gradients
        directional_decay_rate: For gradient_field - exponential decay for samples
        prefer_direction: For gradient_field - prefer 'up' or 'down' gradients
            where 'up' means dark-sky/bright-planet and v.v.
            (None = no preference, choose best gradient regardless of direction)
    """

    def __init__(
        self,
        target: np.ndarray,
        function: Callable,
        free_parameters: list,
        init_parameter_values,
        loss_function="l2",
        kernel_smoothing=5.0,
        directional_smoothing=30,
        directional_decay_rate=0.15,
        prefer_direction: Optional[str] = None,
    ):
        self.function = function
        self.free_parameters = free_parameters
        self.init_parameter_values = init_parameter_values
        self.loss_function = loss_function
        self.prefer_direction = prefer_direction

        # For traditional loss functions, target is the detected limb
        if loss_function in ["l2", "l1", "log-l1"]:
            self.x = np.arange(len(target))
            self.target = target

        # For gradient field loss, target is the image
        elif (
            loss_function == "gradient_field"
            or loss_function == "gradient_field_simple"
        ):
            self.target = None
            self.x = np.arange(target.shape[1])  # Image width

            # Use the new standalone gradient_field function
            grad_data = gradient_field(
                target,
                kernel_smoothing=kernel_smoothing,
                directional_smoothing=directional_smoothing,
                directional_decay_rate=directional_decay_rate,
            )

            # Store the gradient field components as instance variables
            self.grad_mag = grad_data["grad_mag"]
            self.grad_angle = grad_data["grad_angle"]
            self.grad_x = grad_data["grad_x"]
            self.grad_y = grad_data["grad_y"]
            self.grad_sin = grad_data["grad_sin"]
            self.grad_cos = grad_data["grad_cos"]
            self.grad_mag_dx = grad_data["grad_mag_dx"]
            self.grad_mag_dy = grad_data["grad_mag_dy"]
            self.grad_sin_dx = grad_data["grad_sin_dx"]
            self.grad_sin_dy = grad_data["grad_sin_dy"]
            self.grad_cos_dx = grad_data["grad_cos_dx"]
            self.grad_cos_dy = grad_data["grad_cos_dy"]
            self.image_height = grad_data["image_height"]
            self.image_width = grad_data["image_width"]

        else:
            raise ValueError(f"Unrecognized loss function: {loss_function}")

    def cost(self, params: np.ndarray | dict) -> float:
        """
        Compute prediction and use desired metric to reduce difference
        from truth to a cost. AKA loss function.

        Args:
            params (np.ndarray | dict): Parameter values, either packed
                into array or as dict.

        Returns:
            cost (float): Cost given parameters.
        """

        if self.loss_function == "gradient_field":
            return self._gradient_field_cost(params)
        elif self.loss_function == "gradient_field_simple":
            return self._gradient_field_cost_simple(params)

        # Traditional loss functions
        y = self.evaluate(params)

        if self.loss_function == "l2":
            cost = np.nanmean(pow(y - self.target, 2))
        elif self.loss_function == "l1":
            abs_diff = abs(y - self.target)
            cost = np.nanmean(abs_diff)
        elif self.loss_function == "log-l1":
            abs_diff = abs(y - self.target)
            cost = np.nanmean([math.log(float(x) + 1) for x in abs_diff.flatten()])
        else:
            raise ValueError("Unrecognized loss function.")

        return cost

    def _gradient_field_cost_simple(
        self, params: np.ndarray | dict, y_coords: np.ndarray = None
    ) -> float:
        """
        Simplified gradient field cost using signed flux method.

        Key differences from full version:
        - Hard boundary penalty (no blending)
        - Simpler out-of-bounds handling
        - Same core signed flux computation

        The signed flux naturally discriminates:
        - Coherent edges (limb): gradients aligned → large |flux|
        - Incoherent features (striations): mixed gradients → cancellation → small |flux|
        """
        if y_coords is None:
            # Get predicted horizon curve
            y_coords = self.evaluate(params)

        # Handle invalid coordinates
        if np.any(np.isnan(y_coords)) or np.any(np.isinf(y_coords)):
            return 1e10

        # Check what fraction is in bounds
        in_bounds = (y_coords >= 0) & (y_coords < self.image_height)
        fraction_in_bounds = np.mean(in_bounds)

        # HARD BOUNDARY: Need at least 30% in frame
        if fraction_in_bounds < 0.3:
            # Simple distance penalty
            center_y = self.image_height / 2
            mean_dist = np.mean(np.abs(y_coords - center_y))
            return 5.0 + mean_dist / self.image_height

        # ONLY process in-bounds pixels
        x_in = self.x[in_bounds]
        y_in = y_coords[in_bounds]

        # Compute curve normal direction at in-bounds points
        dy_dx_full = np.gradient(y_coords)
        dy_dx = dy_dx_full[in_bounds]

        # Tangent vector: (1, dy/dx)
        tangent_x = np.ones_like(dy_dx)
        tangent_y = dy_dx
        tangent_mag = np.sqrt(tangent_x**2 + tangent_y**2)

        # Normalize tangent
        tangent_x_unit = tangent_x / tangent_mag
        tangent_y_unit = tangent_y / tangent_mag

        # Normal: rotate tangent 90° CCW: (x,y) → (-y, x)
        normal_x_unit = -tangent_y_unit
        normal_y_unit = tangent_x_unit

        normal_angle = np.arctan2(normal_y_unit, normal_x_unit)

        # Integer indices and fractional offsets
        ix = x_in.astype(int)
        iy_float = y_in.copy()
        iy = iy_float.astype(int)

        fx = x_in - ix
        fy = iy_float - iy

        # Clip to valid range
        iy = np.clip(iy, 0, self.image_height - 1)
        ix = np.clip(ix, 0, self.image_width - 1)

        # Taylor expansion interpolation
        mag = (
            self.grad_mag[iy, ix]
            + fx * self.grad_mag_dx[iy, ix]
            + fy * self.grad_mag_dy[iy, ix]
        )

        sin_val = (
            self.grad_sin[iy, ix]
            + fx * self.grad_sin_dx[iy, ix]
            + fy * self.grad_sin_dy[iy, ix]
        )

        cos_val = (
            self.grad_cos[iy, ix]
            + fx * self.grad_cos_dx[iy, ix]
            + fy * self.grad_cos_dy[iy, ix]
        )

        angle = np.arctan2(sin_val, cos_val)

        # Compute SIGNED flux perpendicular to curve
        # gradient · normal = mag × cos(angle_difference)
        angle_diff = np.arctan2(
            np.sin(angle - normal_angle), np.cos(angle - normal_angle)
        )

        # Signed contributions - coherent gradients sum, incoherent cancel
        flux_contribution = mag * np.cos(angle_diff)

        # Net flux (can be positive or negative)
        net_flux = np.sum(flux_contribution)

        # Normalize by arc length
        ds = np.sqrt(1 + dy_dx**2)
        arc_length = np.sum(ds)
        flux_density = net_flux / (arc_length + 1e-10)

        # Normalize by typical gradient magnitude (contrast-invariant)
        typical_mag = np.mean(self.grad_mag)
        normalized_flux = flux_density / (typical_mag + 1e-10)

        # Cost = negative absolute flux
        # Maximize |flux| = prefer coherent gradients
        # Penalize weak/canceling flux = incoherent features
        cost = 1.0 - np.abs(normalized_flux)

        return cost

    def _gradient_field_cost(
        self, params: np.ndarray | dict, y_coords: np.ndarray = None
    ) -> float:
        """
        Gradient field cost based on flux through the limb curve.

        Computes the total "flow" of gradient field through the proposed limb,
        where each pixel contributes based on:
        - Gradient strength (stronger = more contribution)
        - Alignment with curve normal (perpendicular = full contribution)

        This naturally prefers strong, well-aligned gradients without arbitrary thresholds.
        """
        if y_coords is None:
            # Get predicted horizon curve (sub-pixel precision!)
            y_coords = self.evaluate(params)

        # Handle invalid coordinates
        if np.any(np.isnan(y_coords)) or np.any(np.isinf(y_coords)):
            return 1e10

        # Determine which pixels are in bounds
        in_bounds = (y_coords >= 0) & (y_coords < self.image_height)
        fraction_in_bounds = np.mean(in_bounds)

        # Require minimum fraction in frame
        if fraction_in_bounds < 0.3:
            # Distance penalty for out-of-frame
            center_y = self.image_height / 2
            mean_dist = np.mean(np.abs(y_coords - center_y))
            return 5.0 + mean_dist / self.image_height

        # ONLY process in-bounds pixels from here on
        x_in = self.x[in_bounds]
        y_in = y_coords[in_bounds]

        # Compute curve normal direction at in-bounds points
        dy_dx_full = np.gradient(y_coords)
        dy_dx = dy_dx_full[in_bounds]

        # Tangent vector: (1, dy/dx)
        tangent_x = np.ones_like(dy_dx)
        tangent_y = dy_dx
        tangent_mag = np.sqrt(tangent_x**2 + tangent_y**2)

        # Normalize tangent
        tangent_x_unit = tangent_x / tangent_mag
        tangent_y_unit = tangent_y / tangent_mag

        # Normal: rotate tangent 90° CCW: (x,y) → (-y, x)
        normal_x_unit = -tangent_y_unit
        normal_y_unit = tangent_x_unit

        normal_angle = np.arctan2(normal_y_unit, normal_x_unit)

        # Integer indices and fractional offsets (in-bounds only)
        ix = x_in.astype(int)
        iy_float = y_in.copy()
        iy = iy_float.astype(int)

        fx = x_in - ix
        fy = iy_float - iy

        # Clip to valid range (should already be valid, but safety check)
        iy = np.clip(iy, 0, self.image_height - 1)
        ix = np.clip(ix, 0, self.image_width - 1)

        # Taylor expansion interpolation
        mag = (
            self.grad_mag[iy, ix]
            + fx * self.grad_mag_dx[iy, ix]
            + fy * self.grad_mag_dy[iy, ix]
        )

        sin_val = (
            self.grad_sin[iy, ix]
            + fx * self.grad_sin_dx[iy, ix]
            + fy * self.grad_sin_dy[iy, ix]
        )

        cos_val = (
            self.grad_cos[iy, ix]
            + fx * self.grad_cos_dx[iy, ix]
            + fy * self.grad_cos_dy[iy, ix]
        )

        angle = np.arctan2(sin_val, cos_val)

        # Compute flux perpendicular to curve
        # gradient · normal = mag × cos(angle_difference)
        angle_diff = np.arctan2(
            np.sin(angle - normal_angle), np.cos(angle - normal_angle)
        )

        # SIGNED flux: positive and negative contributions can cancel
        # - Coherent gradients (all same direction) → large |flux|
        # - Incoherent gradients (random directions) → cancellation → small |flux|
        flux_contribution = mag * np.cos(angle_diff)

        # Net flux (can be positive or negative)
        net_flux = np.sum(flux_contribution)

        # Normalize by arc length
        ds = np.sqrt(1 + dy_dx**2)
        arc_length = np.sum(ds)
        flux_density = net_flux / (arc_length + 1e-10)

        # Normalize by typical gradient magnitude (contrast-invariant)
        typical_mag = np.mean(self.grad_mag)
        normalized_flux = flux_density / (typical_mag + 1e-10)

        # Cost = negative absolute flux (unless direction preferred)
        # Maximize |flux| = prefer coherent gradients in either direction
        # Penalize weak/canceling flux = incoherent features
        if self.prefer_direction is not None:
            if self.prefer_direction == "up":
                normalized_flux *= -1
            elif self.prefer_direction == "down":
                normalized_flux *= 1
            else:
                raise ValueError(
                    f"Unrecognized prefer_direction ({self.prefer_direction}) -- use 'up', 'down', or None"
                )
        else:
            normalized_flux = np.abs(normalized_flux)

        cost = 1.0 - normalized_flux

        return cost

    def evaluate(self, params: np.ndarray | dict) -> np.ndarray:
        """
        Compute prediction given parameters.

        Args:
            params (np.ndarray | dict): Parameter values, either packed
                into array or as dict.

        Returns:
            prediction (np.ndarray): Prediction value(s).
        """
        kwargs = self.init_parameter_values.copy()
        if type(params) == np.ndarray:
            kwargs.update(unpack_parameters(params.tolist(), self.free_parameters))
        else:
            kwargs.update(params)

        y = self.function(**kwargs)

        return y


def calculate_parameter_uncertainty(
    observation,
    parameter: str = "r",
    method: str = "auto",
    uncertainty_type: str = "std",
    scale_factor: float = 1.0,
) -> dict:
    """
    Calculate parameter uncertainty from fitting results.

    Provides a flexible interface for uncertainty estimation that works
    with different optimization methods and uncertainty metrics.

    Args:
        observation: LimbObservation object with completed fit
        parameter (str): Parameter name to calculate uncertainty for (default: "r")
        method (str): Uncertainty calculation method:
            - "auto": Automatically detect method from fit results
            - "differential_evolution": Use DE population posteriors
            - "bootstrap": Use bootstrap resampling (future implementation)
            - "hessian": Use Hessian-based uncertainty (future implementation)
        uncertainty_type (str): Type of uncertainty measure:
            - "std": Standard deviation of parameter distribution
            - "ptp": Peak-to-peak range (max - min)
            - "iqr": Interquartile range (75th - 25th percentile)
            - "ci": Confidence interval (returns dict with bounds)
        scale_factor (float): Scale factor to apply to results (e.g., 1000 for km units)

    Returns:
        dict: Dictionary containing uncertainty information:
            - "value": Fitted parameter value (scaled)
            - "uncertainty": Uncertainty estimate (scaled)
            - "method": Method used for uncertainty calculation
            - "type": Type of uncertainty measure used
            - "raw_data": Raw parameter samples if available

    Raises:
        ValueError: If uncertainty method is not supported or data is insufficient
        AttributeError: If observation doesn't have required fit results
    """

    # Get final parameter value
    if not hasattr(observation, "best_parameters"):
        raise AttributeError("Observation must have completed fit with best_parameters")

    final_params = observation.init_parameter_values.copy()
    final_params.update(observation.best_parameters)

    if parameter not in final_params:
        raise ValueError(f"Parameter '{parameter}' not found in fitted parameters")

    fitted_value = final_params[parameter] / scale_factor

    # Auto-detect method based on available data
    if method == "auto":
        if hasattr(observation, "fit_results") and hasattr(
            observation.fit_results, "population"
        ):
            method = "differential_evolution"
        else:
            raise ValueError(
                "No supported uncertainty method detected. Available fit results insufficient."
            )

    # Calculate uncertainty based on method
    if method == "differential_evolution":
        if not (
            hasattr(observation, "fit_results")
            and hasattr(observation.fit_results, "population")
        ):
            raise AttributeError(
                "Differential evolution posteriors not available in fit_results"
            )

        # Use unpack_diff_evol_posteriors from same module
        population_df = unpack_diff_evol_posteriors(observation)

        if parameter not in population_df.columns:
            raise ValueError(
                f"Parameter '{parameter}' not found in population posteriors"
            )

        param_samples = population_df[parameter] / scale_factor

        # Calculate uncertainty based on type
        if uncertainty_type == "std":
            uncertainty = param_samples.std()
        elif uncertainty_type == "ptp":
            uncertainty = param_samples.max() - param_samples.min()
        elif uncertainty_type == "iqr":
            uncertainty = param_samples.quantile(0.75) - param_samples.quantile(0.25)
        elif uncertainty_type == "ci":
            # Return 95% confidence interval
            lower = param_samples.quantile(0.025)
            upper = param_samples.quantile(0.975)
            uncertainty = {"lower": lower, "upper": upper, "width": upper - lower}
        else:
            raise ValueError(f"Unsupported uncertainty type: {uncertainty_type}")

        raw_data = param_samples.values

    elif method == "bootstrap":
        # Future implementation for bootstrap uncertainty
        raise NotImplementedError(
            "Bootstrap uncertainty calculation not yet implemented"
        )

    elif method == "hessian":
        # Future implementation for Hessian-based uncertainty
        raise NotImplementedError(
            "Hessian-based uncertainty calculation not yet implemented"
        )

    else:
        raise ValueError(f"Unsupported uncertainty calculation method: {method}")

    return {
        "value": fitted_value,
        "uncertainty": uncertainty,
        "method": method,
        "type": uncertainty_type,
        "raw_data": raw_data if "raw_data" in locals() else None,
        "scale_factor": scale_factor,
        "parameter": parameter,
    }


def format_parameter_result(uncertainty_result: dict, units: str = "") -> str:
    """
    Format parameter uncertainty results for display.

    Args:
        uncertainty_result (dict): Result from calculate_parameter_uncertainty
        units (str): Units to display (e.g., "km", "m", "degrees")

    Returns:
        str: Formatted string representation of result
    """
    value = uncertainty_result["value"]
    uncertainty = uncertainty_result["uncertainty"]
    param = uncertainty_result["parameter"]

    if uncertainty_result.get("type") == "ci":
        return (
            f"{param} = {value:.1f} {units} "
            f"(95% CI: {uncertainty['lower']:.1f}-{uncertainty['upper']:.1f} {units})"
        )
    else:
        uncertainty_type_name = {"std": "±", "ptp": "range ±", "iqr": "IQR ±"}.get(
            uncertainty_result.get("type", "std"), "±"
        )

        return f"{param} = {value:.1f} {uncertainty_type_name}{uncertainty:.1f} {units}"


def unpack_diff_evol_posteriors(observation) -> "pd.DataFrame":
    """
    Extract the final state population of a differential evolution
    minimization and organize as a DataFrame.

    Args:
        observation (object): Instance of LimbObservation (must have
            used differential evolution minimizer).

    Returns:
        population (pd.DataFrame): Population (rows) and properties (columns).
    """
    import pandas as pd

    pop = []
    en = observation.fit_results["population_energies"]
    for i, sol in enumerate(observation.fit_results["population"]):
        mse = en[i]
        updated = observation.init_parameter_values.copy()
        updated.update(unpack_parameters(sol, observation.free_parameters))
        updated["mse"] = mse
        pop.append(updated)
    pop = pd.DataFrame.from_records(pop)

    return pop


def _validate_fit_results(observation):
    """
    Validate fit results and issue warnings for problematic parameter combinations.

    This function checks for common issues that suggest the data may be insufficient
    to properly detect planetary curvature, helping users understand when their
    results may be unreliable.

    Args:
        observation: LimbObservation object with best_parameters from fitting
    """
    # Skip validation if no fit results are available
    if (
        not hasattr(observation, "best_parameters")
        or observation.best_parameters is None
    ):
        return

    # Extract altitude and radius if available
    h = observation.best_parameters.get("h")
    r = observation.best_parameters.get("r")

    # Skip if essential parameters are missing
    if h is None or r is None:
        return

    # Individual parameter warnings
    if h < 1000:  # Less than 1 km altitude
        warnings.warn(
            f"Fitted altitude ({h:.0f} m) is very low. At such low altitudes, "
            "planetary curvature is difficult to detect and the fit may be unreliable. "
            "Your data may be insufficient to accurately measure planetary radius.",
            UserWarning,
            stacklevel=3,
        )

    if r < 1000000:  # Less than 1000 km radius
        warnings.warn(
            f"Fitted radius ({r/1000:.0f} km) is very small. This suggests the "
            "optimization may have difficulty detecting planetary curvature in your data. "
            "Consider checking your image scale or using a different approach.",
            UserWarning,
            stacklevel=3,
        )

    if r > 100000000:  # Greater than 100,000 km radius
        warnings.warn(
            f"Fitted radius ({r/1000:.0f} km) is very large. This may indicate "
            "calibration issues or that the observed curvature is too subtle to measure "
            "reliably with the available data.",
            UserWarning,
            stacklevel=3,
        )

    # Combined problematic conditions (less strict thresholds)
    if h < 5000 and r > 50000000:  # 5 km altitude + 50,000 km radius
        warnings.warn(
            f"Combination of relatively low altitude ({h/1000:.1f} km) and large radius "
            f"({r/1000:.0f} km) suggests the data may be insufficient to reliably detect "
            "planetary curvature. The apparent 'flat' horizon may reflect measurement "
            "limitations rather than actual planetary geometry.",
            UserWarning,
            stacklevel=3,
        )
