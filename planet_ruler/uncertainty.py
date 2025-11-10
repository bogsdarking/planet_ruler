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

import numpy as np
from typing import Dict, Optional, Literal
from scipy.optimize import approx_fprime
import logging


def calculate_parameter_uncertainty(
    observation,
    parameter: str,
    method: Literal["auto", "hessian", "profile", "bootstrap"] = "auto",
    scale_factor: float = 1.0,
    confidence_level: float = 0.68,  # 1-sigma
    n_bootstrap: int = 20,
    **kwargs,
) -> Dict:
    """
    Calculate uncertainty for a fitted parameter using appropriate method.

    Automatically selects best method based on minimizer used:
    - differential_evolution: Population spread (fast, exact)
    - dual_annealing/basinhopping: Hessian approximation (fast, approximate)
    - any: Profile likelihood or bootstrap (slow, accurate)

    Args:
        observation: LimbObservation instance with completed fit
        parameter: Parameter name (e.g., 'r', 'h', 'f')
        method: Uncertainty estimation method
            - 'auto': Choose based on minimizer and available data
            - 'hessian': Inverse Hessian at optimum (fast)
            - 'profile': Profile likelihood (slow but accurate)
            - 'bootstrap': Multiple fits with different seeds (slow)
        scale_factor: Scale result (e.g., 1000.0 to convert m→km)
        confidence_level: Confidence level (0.68=1σ, 0.95=2σ)
        n_bootstrap: Number of bootstrap iterations

    Returns:
        dict with keys:
            - 'uncertainty': The uncertainty value
            - 'method': Method used
            - 'confidence_level': Confidence level
            - 'additional_info': Method-specific details
    """

    if not hasattr(observation, "fit_results") or observation.fit_results is None:
        logging.warning("No fit results available")
        return {
            "value": 0.0,
            "uncertainty": 0.0,
            "parameter": parameter,
            "method": "none",
            "confidence_level": confidence_level,
            "additional_info": "No fit performed",
        }

    if parameter not in observation.free_parameters:
        logging.warning(f"Parameter '{parameter}' was not a free parameter in fit")
        fitted_value = observation.best_parameters.get(parameter, 0.0) / scale_factor
        return {
            "value": fitted_value,
            "uncertainty": 0.0,
            "parameter": parameter,
            "method": "none",
            "confidence_level": confidence_level,
            "additional_info": f"{parameter} was fixed",
        }

    # Auto-select method
    if method == "auto":
        if observation.minimizer == "differential-evolution" and hasattr(
            observation.fit_results, "population"
        ):
            method = "population"
        else:
            method = "hessian"

    # Get fitted value
    fitted_value = observation.best_parameters[parameter] / scale_factor

    # Call method
    if method == "population":
        result = _uncertainty_from_population(
            observation, parameter, scale_factor, confidence_level
        )
    elif method == "hessian":
        result = _uncertainty_from_hessian(
            observation, parameter, scale_factor, confidence_level
        )
    elif method == "profile":
        result = _uncertainty_from_profile(
            observation, parameter, scale_factor, confidence_level, **kwargs
        )
    elif method == "bootstrap":
        result = _uncertainty_from_bootstrap(
            observation, parameter, scale_factor, confidence_level, n_bootstrap
        )
    else:
        raise ValueError(f"Unknown uncertainty method: {method}")

    # Add missing keys
    result["value"] = fitted_value
    result["parameter"] = parameter

    return result


def _uncertainty_from_population(
    observation, parameter: str, scale_factor: float, confidence_level: float
) -> Dict:
    """Uncertainty from differential evolution population."""

    if not hasattr(observation.fit_results, "population"):
        return {
            "uncertainty": 0.0,
            "method": "population",
            "confidence_level": confidence_level,
            "additional_info": "No population data available",
        }

    # Get parameter index
    param_idx = observation.free_parameters.index(parameter)

    # Extract parameter values from population
    population = observation.fit_results.population
    param_values = population[:, param_idx] / scale_factor

    # Calculate spread
    std = np.std(param_values, ddof=1)

    # Scale to desired confidence level (assumes normal distribution)
    from scipy import stats

    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    uncertainty = z_score * std

    return {
        "uncertainty": uncertainty,
        "method": "population",
        "confidence_level": confidence_level,
        "additional_info": {
            "population_size": len(population),
            "std": std,
            "mean": np.mean(param_values),
            "median": np.median(param_values),
            "min": np.min(param_values),
            "max": np.max(param_values),
        },
    }


def _uncertainty_from_hessian(
    observation, parameter: str, scale_factor: float, confidence_level: float
) -> Dict:
    """
    Uncertainty from inverse Hessian (covariance) at optimum.

    This approximates the cost function as quadratic near the minimum.
    Fast but may be inaccurate for highly nonlinear problems.
    """

    param_idx = observation.free_parameters.index(parameter)
    x_opt = observation.fit_results.x

    # Approximate Hessian using finite differences
    eps = np.sqrt(np.finfo(float).eps)
    n_params = len(x_opt)
    hessian = np.zeros((n_params, n_params))

    # Central finite difference approximation
    for i in range(n_params):
        for j in range(i, n_params):
            # Compute second derivative
            x_pp = x_opt.copy()
            x_pm = x_opt.copy()
            x_mp = x_opt.copy()
            x_mm = x_opt.copy()

            x_pp[i] += eps
            x_pp[j] += eps
            x_pm[i] += eps
            x_pm[j] -= eps
            x_mp[i] -= eps
            x_mp[j] += eps
            x_mm[i] -= eps
            x_mm[j] -= eps

            f_pp = observation.cost_function.cost(x_pp)
            f_pm = observation.cost_function.cost(x_pm)
            f_mp = observation.cost_function.cost(x_mp)
            f_mm = observation.cost_function.cost(x_mm)

            h_ij = (f_pp - f_pm - f_mp + f_mm) / (4 * eps * eps)
            hessian[i, j] = h_ij
            hessian[j, i] = h_ij

    try:
        # Invert Hessian to get covariance
        covariance = np.linalg.inv(hessian)

        # Extract variance for this parameter
        variance = covariance[param_idx, param_idx]

        if variance < 0:
            logging.warning(
                f"Negative variance for {parameter} - Hessian may be poorly conditioned"
            )
            return {
                "uncertainty": 0.0,
                "method": "hessian",
                "confidence_level": confidence_level,
                "additional_info": "Negative variance - singular Hessian",
            }

        std = np.sqrt(variance) / scale_factor

        # Scale to desired confidence level
        from scipy import stats

        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        uncertainty = z_score * std

        return {
            "uncertainty": uncertainty,
            "method": "hessian",
            "confidence_level": confidence_level,
            "additional_info": {
                "std": std,
                "variance": variance / scale_factor**2,
                "condition_number": np.linalg.cond(hessian),
            },
        }

    except np.linalg.LinAlgError:
        logging.warning("Hessian inversion failed - poorly conditioned")
        return {
            "uncertainty": 0.0,
            "method": "hessian",
            "confidence_level": confidence_level,
            "additional_info": "Hessian inversion failed",
        }


def _uncertainty_from_profile(
    observation,
    parameter: str,
    scale_factor: float,
    confidence_level: float,
    n_points: int = 20,
    search_range: float = 0.2,
) -> Dict:
    """
    Uncertainty from profile likelihood.

    For each value of the target parameter, re-optimize all other parameters
    and find where the cost exceeds a threshold based on chi-squared statistics.

    This is the most accurate but also the slowest method.
    """

    from scipy.optimize import minimize
    from scipy import stats

    param_idx = observation.free_parameters.index(parameter)
    x_opt = observation.fit_results.x
    cost_min = observation.cost_function.cost(x_opt)

    # Chi-squared threshold for confidence level
    # For 1 parameter, delta_cost = chi2.ppf(confidence_level, df=1) / 2
    delta_cost_threshold = stats.chi2.ppf(confidence_level, df=1) / 2

    # Search range around optimum
    param_value_opt = x_opt[param_idx]
    param_range = abs(search_range * param_value_opt)

    # Ensure we have a reasonable search range even for small parameter values
    if param_range < 1e-6:
        param_range = search_range * (
            observation.parameter_limits[parameter][1]
            - observation.parameter_limits[parameter][0]
        )

    param_values = np.linspace(
        param_value_opt - param_range, param_value_opt + param_range, n_points
    )

    costs = []

    # Profile: fix parameter, optimize others
    for param_value in param_values:
        # Create cost function with this parameter fixed
        def profile_cost(x_free):
            x_full = x_opt.copy()
            free_idx = [i for i in range(len(x_opt)) if i != param_idx]
            x_full[free_idx] = x_free
            x_full[param_idx] = param_value
            return observation.cost_function.cost(x_full)

        # Optimize other parameters
        x0_free = np.delete(x_opt, param_idx)
        bounds_free = [
            observation.parameter_limits[p]
            for i, p in enumerate(observation.free_parameters)
            if i != param_idx
        ]

        result = minimize(profile_cost, x0_free, method="L-BFGS-B", bounds=bounds_free)

        costs.append(result.fun)

    costs = np.array(costs)
    delta_costs = costs - cost_min

    # Find parameter values where delta_cost crosses threshold
    # We need to find crossings on both sides of the optimum
    try:
        # Split into left and right sides of optimum
        opt_idx = np.argmin(delta_costs)

        # Left side (lower parameter values)
        left_params = param_values[: opt_idx + 1]
        left_costs = delta_costs[: opt_idx + 1]

        # Right side (higher parameter values)
        right_params = param_values[opt_idx:]
        right_costs = delta_costs[opt_idx:]

        # Find where profile crosses threshold on each side
        lower_bound = None
        upper_bound = None

        # Left side: find where delta_cost crosses threshold
        if len(left_params) > 1:
            # Find the crossing point by linear interpolation
            for i in range(len(left_costs) - 1):
                if left_costs[i] <= delta_cost_threshold < left_costs[i + 1]:
                    # Linear interpolation
                    frac = (delta_cost_threshold - left_costs[i]) / (
                        left_costs[i + 1] - left_costs[i]
                    )
                    lower_bound = left_params[i] + frac * (
                        left_params[i + 1] - left_params[i]
                    )
                    break

            # If no crossing found, use furthest point that's below threshold
            if lower_bound is None:
                below_threshold = left_costs <= delta_cost_threshold
                if np.any(below_threshold):
                    lower_bound = left_params[np.where(below_threshold)[0][0]]

        # Right side: find where delta_cost crosses threshold
        if len(right_params) > 1:
            # Find the crossing point by linear interpolation
            for i in range(len(right_costs) - 1):
                if right_costs[i] <= delta_cost_threshold < right_costs[i + 1]:
                    # Linear interpolation
                    frac = (delta_cost_threshold - right_costs[i]) / (
                        right_costs[i + 1] - right_costs[i]
                    )
                    upper_bound = right_params[i] + frac * (
                        right_params[i + 1] - right_params[i]
                    )
                    break

            # If no crossing found, use furthest point that's below threshold
            if upper_bound is None:
                below_threshold = right_costs <= delta_cost_threshold
                if np.any(below_threshold):
                    upper_bound = right_params[np.where(below_threshold)[0][-1]]

        if lower_bound is None or upper_bound is None:
            logging.warning(
                f"Could not find confidence bounds - profile may not cross threshold. "
                f"Try increasing search_range (currently {search_range})"
            )
            return {
                "uncertainty": 0.0,
                "method": "profile",
                "confidence_level": confidence_level,
                "additional_info": {
                    "error": "Could not find confidence bounds",
                    "min_delta_cost": np.min(delta_costs),
                    "max_delta_cost": np.max(delta_costs),
                    "threshold": delta_cost_threshold,
                    "suggestion": "Increase search_range parameter",
                },
            }

        # Symmetric uncertainty (average of distances from optimum)
        lower_uncertainty = abs(param_value_opt - lower_bound) / scale_factor
        upper_uncertainty = abs(upper_bound - param_value_opt) / scale_factor
        uncertainty = (lower_uncertainty + upper_uncertainty) / 2

        return {
            "uncertainty": uncertainty,
            "method": "profile",
            "confidence_level": confidence_level,
            "additional_info": {
                "lower_bound": lower_bound / scale_factor,
                "upper_bound": upper_bound / scale_factor,
                "optimal_value": param_value_opt / scale_factor,
                "lower_uncertainty": lower_uncertainty,
                "upper_uncertainty": upper_uncertainty,
                "n_evaluations": n_points,
                "cost_min": cost_min,
                "threshold": delta_cost_threshold,
            },
        }

    except Exception as e:
        logging.warning(f"Profile likelihood failed: {e}")
        return {
            "uncertainty": 0.0,
            "method": "profile",
            "confidence_level": confidence_level,
            "additional_info": f"Failed: {str(e)}",
        }


def _uncertainty_from_bootstrap(
    observation,
    parameter: str,
    scale_factor: float,
    confidence_level: float,
    n_bootstrap: int,
) -> Dict:
    """
    Uncertainty from bootstrap: run optimizer multiple times with different seeds.

    This is robust but slow, especially for gradient_field loss functions.
    """

    param_idx = observation.free_parameters.index(parameter)
    param_values = []

    logging.info(f"Running {n_bootstrap} bootstrap iterations...")

    for i in range(n_bootstrap):
        # Run fit with different seed
        observation_copy = observation  # Could deep copy if needed

        # Re-run optimization with new seed
        # This is a simplified version - in practice you'd want to:
        # 1. Save original state
        # 2. Re-run fit with seed=i
        # 3. Extract parameter
        # 4. Restore original state

        # For now, just show the structure
        logging.warning(
            "Bootstrap not fully implemented - needs careful state management"
        )
        break

    if len(param_values) == 0:
        return {
            "uncertainty": 0.0,
            "method": "bootstrap",
            "confidence_level": confidence_level,
            "additional_info": "Bootstrap not fully implemented",
        }

    param_values = np.array(param_values) / scale_factor
    std = np.std(param_values, ddof=1)

    from scipy import stats

    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    uncertainty = z_score * std

    return {
        "uncertainty": uncertainty,
        "method": "bootstrap",
        "confidence_level": confidence_level,
        "additional_info": {
            "n_bootstrap": n_bootstrap,
            "std": std,
            "mean": np.mean(param_values),
            "median": np.median(param_values),
        },
    }
