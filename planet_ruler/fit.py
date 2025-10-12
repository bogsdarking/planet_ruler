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
import numpy as np
import math
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
        function (Callable): Function mapping parameters to target of interest.
        free_parameters (list): List of free parameter names.
        init_parameter_values (dict): Initial values for named parameters.
        loss_function (str): Type of loss function, must be one of ['l2', 'l1', 'log-l1'].

    Returns:
        param_list (list): List of parameter values.
    """

    def __init__(
        self,
        target: np.ndarray,
        function: Callable,
        free_parameters: list,
        init_parameter_values,
        loss_function="l2",
    ):

        self.function = function
        self.free_parameters = free_parameters
        self.init_parameter_values = init_parameter_values
        self.x = np.arange(len(target))
        self.target = target
        self.loss_function = loss_function

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
        y = self.evaluate(params)

        if self.loss_function == "l2":
            cost = np.nanmean(pow(y - self.target, 2))
        elif self.loss_function == "l1":
            abs_diff = abs(y - self.target)
            cost = np.nanmean(abs_diff)
        elif self.loss_function == "log-l1":
            abs_diff = abs(y - self.target)
            # cost = np.sum(abs_diff) + np.sum(pow(abs_diff+1, -0.5))
            cost = np.nanmean([math.log(float(x) + 1) for x in abs_diff.flatten()])
        else:
            raise ValueError("Unrecognized loss function.")

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

        # Import here to avoid circular imports
        try:
            from planet_ruler.observation import unpack_diff_evol_posteriors

            population_df = unpack_diff_evol_posteriors(observation)
        except ImportError:
            raise ImportError("Could not import unpack_diff_evol_posteriors function")

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

    if uncertainty_result["type"] == "ci":
        return (
            f"{param} = {value:.1f} {units} "
            f"(95% CI: {uncertainty['lower']:.1f}-{uncertainty['upper']:.1f} {units})"
        )
    else:
        uncertainty_type_name = {"std": "±", "ptp": "range ±", "iqr": "IQR ±"}.get(
            uncertainty_result["type"], "±"
        )

        return f"{param} = {value:.1f} {uncertainty_type_name}{uncertainty:.1f} {units}"
