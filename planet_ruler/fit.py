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
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import pandas as pd
import math
import warnings
from scipy.optimize import differential_evolution, minimize, minimize_scalar
from planet_ruler.image import (
    bidirectional_gradient_blur,
    bilinear_interpolate,
    gradient_field,
)
from planet_ruler.geometry import _r_from_K, limb_arc_sagitta, limb_arc
from typing import Callable
import logging

logger = logging.getLogger(__name__)

_PARALLEL_MINIMIZERS = frozenset({"differential-evolution", "shgo"})


# ============================================================================
# PARAMETER PACKING
# ============================================================================


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


# ============================================================================
# FITTER INTERFACE
# ============================================================================


class BaseFitter(ABC):
    """Common interface for planet-ruler fitting strategies.

    Each subclass encapsulates one strategy and returns a stage-result dict
    from .fit().  The dict always contains:

      updated_init   – {param: value} updates to warm-start the next stage
      updated_limits – {param: [lo, hi]} tighter bounds for the next stage
      status         – "ok" | "flat_arc" | "too_few_points" | "error"
      warnings       – list[str]
    """

    @abstractmethod
    def fit(self) -> dict: ...


# ============================================================================
# COST FUNCTIONS
# ============================================================================


class BaseCostFunction:
    """Shared evaluation logic for all cost function variants."""

    def __init__(
        self,
        target: np.ndarray,
        function: Callable,
        free_parameters: list,
        init_parameter_values: dict,
    ):
        self.function = function
        self.free_parameters = free_parameters
        self.init_parameter_values = init_parameter_values

    def evaluate(self, params: np.ndarray | dict) -> np.ndarray:
        """Compute prediction given parameters.

        For sparse manual annotation (is_sparse=True), only computes at
        annotated x-coordinates stored in self.x.
        """
        kwargs = self.init_parameter_values.copy()
        if isinstance(params, np.ndarray):
            kwargs.update(unpack_parameters(params.tolist(), self.free_parameters))
        else:
            kwargs.update(params)
        if getattr(self, "is_sparse", False):
            kwargs["x_coords"] = self.x
        return self.function(**kwargs)


class L2CostFunction(BaseCostFunction):
    """Cost function for l2, l1, and log-l1 loss against a detected limb."""

    def __init__(
        self,
        target: np.ndarray,
        function: Callable,
        free_parameters: list,
        init_parameter_values: dict,
        loss_function: str = "l2",
    ):
        _VALID = {"l2", "l1", "log-l1"}
        if loss_function not in _VALID:
            raise ValueError(f"Unrecognized loss function: {loss_function!r}")
        super().__init__(target, function, free_parameters, init_parameter_values)
        self.loss_function = loss_function
        valid_mask = ~np.isnan(target)
        self.x = np.where(valid_mask)[0]
        self.target = target[valid_mask]
        self.is_sparse = np.sum(valid_mask) < len(target)
        if self.is_sparse:
            logger.info(
                f"Sparse annotation detected: {np.sum(valid_mask)}/{len(target)} pixels "
                f"annotated ({100*np.sum(valid_mask)/len(target):.1f}%) - using optimized computation"
            )

    def cost(self, params: np.ndarray | dict) -> float:
        y = self.evaluate(params)
        if self.is_sparse and y is not None and len(y) != len(self.target):
            y = y[self.x]
        if self.loss_function == "l2":
            return np.mean((y - self.target) ** 2)
        elif self.loss_function == "l1":
            return np.mean(np.abs(y - self.target))
        elif self.loss_function == "log-l1":
            abs_diff = np.abs(y - self.target)
            return np.mean([math.log(float(v) + 1) for v in abs_diff.flatten()])
        raise ValueError(f"Unrecognized loss function: {self.loss_function}")


class GradientFieldCostFunction(BaseCostFunction):
    """Cost function using gradient field flux alignment."""

    def __init__(
        self,
        target: np.ndarray,
        function: Callable,
        free_parameters: list,
        init_parameter_values: dict,
        loss_function: str = "gradient_field",
        kernel_smoothing: float = 5.0,
        directional_smoothing: int = 30,
        directional_decay_rate: float = 0.15,
        prefer_direction: Optional[str] = None,
    ):
        super().__init__(target, function, free_parameters, init_parameter_values)
        self.loss_function = loss_function
        self.prefer_direction = prefer_direction
        self.x = np.arange(target.shape[1])
        self.target = None

        grad_data = gradient_field(
            target,
            kernel_smoothing=kernel_smoothing,
            directional_smoothing=directional_smoothing,
            directional_decay_rate=directional_decay_rate,
        )
        self.grad_mag = grad_data["grad_mag"]
        self.mean_grad_mag = np.mean(self.grad_mag)
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

    def cost(self, params: np.ndarray | dict) -> float:
        if self.loss_function == "gradient_field_simple":
            return self._gradient_field_cost_simple(params)
        return self._gradient_field_cost(params)

    def _gradient_field_cost_simple(
        self, params: np.ndarray | dict, y_coords: np.ndarray = None
    ) -> float:
        if y_coords is None:
            y_coords = self.evaluate(params)
        if np.any(np.isnan(y_coords)) or np.any(np.isinf(y_coords)):
            return 1e10

        in_bounds = (y_coords >= 0) & (y_coords < self.image_height)
        fraction_in_bounds = np.mean(in_bounds)
        if fraction_in_bounds < 0.3:
            center_y = self.image_height / 2
            mean_dist = np.mean(np.abs(y_coords - center_y))
            return 5.0 + mean_dist / self.image_height

        x_in = self.x[in_bounds]
        y_in = y_coords[in_bounds]
        dy_dx_full = np.gradient(y_coords)
        dy_dx = dy_dx_full[in_bounds]

        tangent_x = np.ones_like(dy_dx)
        tangent_y = dy_dx
        tangent_mag = np.sqrt(tangent_x**2 + tangent_y**2)
        tangent_x_unit = tangent_x / tangent_mag
        tangent_y_unit = tangent_y / tangent_mag
        normal_x_unit = -tangent_y_unit
        normal_y_unit = tangent_x_unit
        normal_angle = np.arctan2(normal_y_unit, normal_x_unit)

        ix = x_in.astype(int)
        iy_float = y_in.copy()
        iy = np.clip(iy_float.astype(int), 0, self.image_height - 1)
        ix = np.clip(ix, 0, self.image_width - 1)
        fx = x_in - ix
        fy = iy_float - iy

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
        angle_diff = np.arctan2(
            np.sin(angle - normal_angle), np.cos(angle - normal_angle)
        )
        flux_contribution = mag * np.cos(angle_diff)
        net_flux = np.sum(flux_contribution)
        ds = np.sqrt(1 + dy_dx**2)
        arc_length = np.sum(ds)
        flux_density = net_flux / (arc_length + 1e-10)
        normalized_flux = flux_density / (self.mean_grad_mag + 1e-10)
        return 1.0 - np.abs(normalized_flux)

    def _gradient_field_cost(
        self, params: np.ndarray | dict, y_coords: np.ndarray = None
    ) -> float:
        if y_coords is None:
            y_coords = self.evaluate(params)
        if np.any(np.isnan(y_coords)) or np.any(np.isinf(y_coords)):
            return 1e10

        in_bounds = (y_coords >= 0) & (y_coords < self.image_height)
        fraction_in_bounds = np.mean(in_bounds)
        if fraction_in_bounds < 0.3:
            center_y = self.image_height / 2
            mean_dist = np.mean(np.abs(y_coords - center_y))
            return 5.0 + mean_dist / self.image_height

        x_in = self.x[in_bounds]
        y_in = y_coords[in_bounds]
        dy_dx_full = np.gradient(y_coords)
        dy_dx = dy_dx_full[in_bounds]

        tangent_x = np.ones_like(dy_dx)
        tangent_y = dy_dx
        tangent_mag = np.sqrt(tangent_x**2 + tangent_y**2)
        tangent_x_unit = tangent_x / tangent_mag
        tangent_y_unit = tangent_y / tangent_mag
        normal_x_unit = -tangent_y_unit
        normal_y_unit = tangent_x_unit
        normal_angle = np.arctan2(normal_y_unit, normal_x_unit)

        ix = x_in.astype(int)
        iy_float = y_in.copy()
        iy = np.clip(iy_float.astype(int), 0, self.image_height - 1)
        ix = np.clip(ix, 0, self.image_width - 1)
        fx = x_in - ix
        fy = iy_float - iy

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
        angle_diff = np.arctan2(
            np.sin(angle - normal_angle), np.cos(angle - normal_angle)
        )
        flux_contribution = mag * np.cos(angle_diff)
        net_flux = np.sum(flux_contribution)
        ds = np.sqrt(1 + dy_dx**2)
        arc_length = np.sum(ds)
        flux_density = net_flux / (arc_length + 1e-10)
        normalized_flux = flux_density / (self.mean_grad_mag + 1e-10)

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

        return 1.0 - normalized_flux


# ============================================================================
# LIMB FITTER
# ============================================================================


class LimbFitter(BaseFitter):
    """Image-space fitter using a pixel-level cost function and a scipy minimizer.

    Accepts one target (image for gradient_field, limb array for l2/l1/log-l1)
    at a fixed resolution. The caller is responsible for downsampling and
    parameter scaling before instantiation.

    minimizer_kwargs must already have preset values resolved by the caller.
    """

    def __init__(
        self,
        target: np.ndarray,
        free_parameters: list,
        init_parameter_values: dict,
        parameter_limits: dict,
        loss_function: str,
        minimizer: str,
        minimizer_kwargs: dict,
        max_iter: int = 15000,
        seed: int = 0,
        verbose: bool = False,
        kernel_smoothing: float = 5.0,
        directional_smoothing: int = 50,
        directional_decay_rate: float = 0.15,
        prefer_direction=None,
        n_jobs: int = 1,
    ):
        self.target = target
        self.free_parameters = free_parameters
        self.init_parameter_values = init_parameter_values
        self.parameter_limits = parameter_limits
        self.loss_function = loss_function
        self.minimizer = minimizer
        self.minimizer_kwargs = dict(minimizer_kwargs) if minimizer_kwargs else {}
        self.max_iter = max_iter
        self.seed = seed
        self.verbose = verbose
        self.kernel_smoothing = kernel_smoothing
        self.directional_smoothing = directional_smoothing
        self.directional_decay_rate = directional_decay_rate
        self.prefer_direction = prefer_direction
        self.n_jobs = n_jobs

    def fit(self) -> dict:
        working_parameters = dict(self.init_parameter_values)

        # For gradient_field, infer pixel params from image dimensions
        if "gradient_field" in self.loss_function and self.target.ndim >= 2:
            working_parameters.update(
                {
                    "n_pix_x": self.target.shape[1],
                    "n_pix_y": self.target.shape[0],
                    "x0": int(self.target.shape[1] * 0.5),
                    "y0": int(self.target.shape[0] * 0.5),
                }
            )

        if self.loss_function in ("l2", "l1", "log-l1"):
            cost_fn = L2CostFunction(
                target=self.target,
                function=limb_arc,
                free_parameters=self.free_parameters,
                init_parameter_values=working_parameters,
                loss_function=self.loss_function,
            )
        else:
            cost_fn = GradientFieldCostFunction(
                target=self.target,
                function=limb_arc,
                free_parameters=self.free_parameters,
                init_parameter_values=working_parameters,
                loss_function=self.loss_function,
                kernel_smoothing=self.kernel_smoothing,
                directional_smoothing=self.directional_smoothing,
                directional_decay_rate=self.directional_decay_rate,
                prefer_direction=self.prefer_direction,
            )

        bounds = [self.parameter_limits[key] for key in self.free_parameters]
        x0 = [working_parameters[key] for key in self.free_parameters]

        x0_clamped = []
        for val, (lo, hi) in zip(x0, bounds):
            epsilon = 1e-9 * (hi - lo)
            clamped = max(lo + epsilon, min(hi - epsilon, val))
            if clamped != val:
                warnings.warn(
                    f"Initial parameter {val} clamped to [{lo}, {hi}]", UserWarning
                )
            x0_clamped.append(clamped)
        x0 = x0_clamped

        config = dict(self.minimizer_kwargs)

        if self.n_jobs != 1 and self.minimizer not in _PARALLEL_MINIMIZERS:
            warnings.warn(
                f"n_jobs={self.n_jobs} has no effect for minimizer "
                f"{self.minimizer!r}. Only "
                f"{sorted(_PARALLEL_MINIMIZERS)} support parallel workers.",
                UserWarning,
                stacklevel=2,
            )

        if self.minimizer == "differential-evolution":
            workers = config.pop("workers", self.n_jobs)
            fit_results = differential_evolution(
                cost_fn.cost,
                bounds,
                x0=x0,
                maxiter=self.max_iter,
                seed=self.seed,
                disp=self.verbose,
                workers=workers,
                **config,
            )

        elif self.minimizer == "basinhopping":
            from scipy.optimize import basinhopping

            local_maxiter = config.pop("local_maxiter", 100)
            minimizer_kwargs_local = {
                "method": "L-BFGS-B",
                "bounds": bounds,
                "options": {"maxiter": local_maxiter, "ftol": 1e-6},
            }

            class BoundsChecker:
                def __init__(self, bounds):
                    self.bounds = bounds

                def __call__(self, **kwargs):
                    x = kwargs["x_new"]
                    return all(l <= xi <= u for xi, (l, u) in zip(x, self.bounds))

            fit_results = basinhopping(
                cost_fn.cost,
                x0,
                minimizer_kwargs=minimizer_kwargs_local,
                accept_test=BoundsChecker(bounds),
                seed=self.seed,
                disp=self.verbose,
                interval=20,
                **config,
            )

        elif self.minimizer == "scipy-minimize":
            method = config.pop("method", "L-BFGS-B")
            n_restarts = config.pop("n_restarts", 1)
            rng = np.random.default_rng(self.seed)
            lo = np.array([b[0] for b in bounds])
            hi = np.array([b[1] for b in bounds])
            best_result = None
            for i in range(n_restarts):
                x_start = x0 if i == 0 else rng.uniform(lo, hi).tolist()
                res = minimize(
                    cost_fn.cost,
                    x_start,
                    method=method,
                    bounds=bounds,
                    options={"maxiter": self.max_iter, **config},
                )
                if best_result is None or res.fun < best_result.fun:
                    best_result = res
            fit_results = best_result

        elif self.minimizer == "dual-annealing":
            from scipy.optimize import dual_annealing

            fit_results = dual_annealing(
                cost_fn.cost,
                bounds=bounds,
                x0=x0,
                maxiter=self.max_iter,
                seed=self.seed,
                **config,
            )

        elif self.minimizer == "shgo":
            from scipy.optimize import shgo

            n = config.pop("n", 64)
            iters = config.pop("iters", 1)
            workers = config.pop("workers", self.n_jobs)
            fit_results = shgo(
                cost_fn.cost,
                bounds=bounds,
                n=n,
                iters=iters,
                workers=workers,
                options={"maxiter": self.max_iter, **config},
            )

        else:
            raise ValueError(f"Unknown minimizer: {self.minimizer}")

        best_params = unpack_parameters(fit_results.x, self.free_parameters)
        working_parameters.update(best_params)
        updated_init = {k: v for k, v in best_params.items()}

        return {
            "best_parameters": working_parameters,
            "fit_results": fit_results,
            "updated_init": updated_init,
            "updated_limits": {},
            "status": "ok",
            "warnings": [],
        }


# ============================================================================
# SAGITTA ESTIMATOR
# ============================================================================


def estimate_radius_via_sagitta(
    limb: np.ndarray,
    h: float,
    f_px: float,
    x0: Optional[float] = None,
    y0: Optional[float] = None,
    sigma_px: "float | str" = "auto",
    n_sigma: float = 1.0,
    bias_correct: bool = False,
    uncertainty: str = "ols",
) -> dict:
    """Estimate planetary radius from a sparse limb arc using OLS on the sagitta.

    Fits K = r / sqrt(h*(2r+h)) by regressing observed sagittae against the
    exact hyperbola model A(u) = sqrt(f_px^2 + u^2) - f_px.

    Args:
        limb: 1-D array of length n_pix_x; NaN where not annotated.
        h: Observer altitude in metres.
        f_px: Focal length in pixels.
        x0: Optional override for apex x-coordinate (auto-detected if None).
        y0: Image vertical centre in pixels, required for bias_correct=True.
        sigma_px: Annotation noise in pixels, or "auto" to derive from RMS residuals.
        n_sigma: Bound width in sigma units.
        bias_correct: If True and y0 is provided, correct for camera tilt bias using
            the apex y-offset to estimate theta_x, then refine kappa via 1-D minimization.
        uncertainty: "ols" | "jackknife" | "both".  Controls which K_sigma drives bounds.

    Returns:
        dict with keys: r, r_low, r_high, r_sigma, K, K_sigma, K_sigma_jack,
            n_points, residual_rms, arc_angle_deg, x_apex, y_apex,
            theta_x_est, status, warnings.
    """
    nan_r = float("nan")
    base = dict(
        r=nan_r, r_low=nan_r, r_high=nan_r, r_sigma=nan_r,
        K=nan_r, K_sigma=nan_r, K_sigma_jack=nan_r,
        n_points=0, residual_rms=nan_r, arc_angle_deg=nan_r,
        x_apex=nan_r, y_apex=nan_r, theta_x_est=nan_r,
        status="", warnings=[],
    )

    # --- Extract valid points ---
    xs = np.where(~np.isnan(limb))[0].astype(float)
    ys = limb[~np.isnan(limb)].astype(float)
    n = len(xs)
    base["n_points"] = n
    if n < 4:
        base["status"] = "too_few_points"
        return base

    # --- Apex ---
    ix_apex = int(np.argmin(ys))
    y_apex = ys[ix_apex]
    if x0 is not None:
        x_apex = float(x0)
    elif 0 < ix_apex < n - 1:
        # Sub-pixel apex via 3-point quadratic interpolation so that a
        # continuous minimum falling between two samples is located correctly.
        h_lo = xs[ix_apex - 1] - xs[ix_apex]
        h_hi = xs[ix_apex + 1] - xs[ix_apex]
        d_lo = ys[ix_apex - 1] - ys[ix_apex]
        d_hi = ys[ix_apex + 1] - ys[ix_apex]
        denom = h_lo * h_hi * (h_lo - h_hi)
        if abs(denom) > 0:
            a_q = (d_lo * h_hi - d_hi * h_lo) / denom
            b_q = (d_hi * h_lo**2 - d_lo * h_hi**2) / denom
            if abs(a_q) > 0:
                x_interp = xs[ix_apex] - b_q / (2.0 * a_q)
                x_apex = float(x_interp) if xs[ix_apex - 1] <= x_interp <= xs[ix_apex + 1] else xs[ix_apex]
            else:
                x_apex = xs[ix_apex]
        else:
            x_apex = xs[ix_apex]
    else:
        x_apex = xs[ix_apex]
    base["x_apex"] = float(x_apex)
    base["y_apex"] = float(y_apex)

    # Centred coordinates
    u = xs - x_apex          # horizontal offset from apex
    s = ys - y_apex           # sagitta from apex (≥ 0)
    A = np.sqrt(f_px**2 + u**2) - f_px   # exact hyperbola basis at theta_x=0

    if np.max(A) < 1e-6:
        base["status"] = "flat_arc"
        return base

    # --- OLS: s = s0 + (1/K)*A ---
    X = np.column_stack([np.ones(n), A])
    coeffs, _, _, _ = np.linalg.lstsq(X, s, rcond=None)
    s0_est, c_est = coeffs

    if abs(c_est) < 1e-8:
        base["status"] = "flat_arc"
        return base

    K_est = 1.0 / abs(c_est)

    residuals = s - X @ coeffs
    residual_rms = float(np.sqrt(np.mean(residuals**2)))
    base["residual_rms"] = residual_rms

    sigma_s = float(sigma_px) if sigma_px != "auto" else residual_rms

    # OLS uncertainty of c_est
    A_centered = A - A.mean()
    A_var = float(np.sum(A_centered**2))
    if A_var < 1e-12:
        base["status"] = "flat_arc"
        return base
    c_sigma = sigma_s / np.sqrt(A_var)
    K_sigma_ols = K_est**2 * c_sigma   # propagation: d(1/c)/dc = -K^2

    # --- Arc angle ---
    phi_lo = np.arctan(u.min() / f_px)
    phi_hi = np.arctan(u.max() / f_px)
    base["arc_angle_deg"] = float((phi_hi - phi_lo) * 180.0 / np.pi)

    # --- Jackknife ---
    K_sigma_jack = nan_r
    if uncertainty in ("jackknife", "both"):
        K_loo = np.empty(n)
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            X_loo = np.column_stack([np.ones(n - 1), A[mask]])
            c_loo = np.linalg.lstsq(X_loo, s[mask], rcond=None)[0][1]
            K_loo[i] = 1.0 / abs(c_loo) if abs(c_loo) > 1e-12 else nan_r
        valid = K_loo[np.isfinite(K_loo)]
        if len(valid) >= 2:
            K_sigma_jack = float(
                np.sqrt((n - 1) / n * np.sum((valid - valid.mean()) ** 2))
            )

    K_sigma = K_sigma_ols
    if uncertainty == "jackknife" and np.isfinite(K_sigma_jack):
        K_sigma = K_sigma_jack
    elif uncertainty == "both" and np.isfinite(K_sigma_jack):
        K_sigma = max(K_sigma_ols, K_sigma_jack)

    # --- Bias correction ---
    theta_x_est = nan_r
    if bias_correct:
        if y0 is None:
            base["warnings"].append(
                "bias_correct=True requires y0; skipping bias correction."
            )
        else:
            delta_y = float(y_apex - y0)
            theta_x_est = float(np.arctan(delta_y / f_px))

            def _cost_kappa(log_kappa: float) -> float:
                kap = np.exp(log_kappa)
                r_t = _r_from_K(1.0 / kap, h)
                s_pred = np.array(
                    [limb_arc_sagitta(float(ui), theta_x_est, f_px, r_t, h) for ui in u]
                )
                s0_opt = float(np.mean(s - s_pred))
                return float(np.sum((s - s0_opt - s_pred) ** 2))

            try:
                log_kappa_est = -np.log(max(K_est, 1e-12))
                res_bc = minimize_scalar(
                    _cost_kappa,
                    bounds=(log_kappa_est - 3.0, log_kappa_est + 3.0),
                    method="bounded",
                )
                K_est = 1.0 / float(np.exp(res_bc.x))
                # Recompute K_sigma with updated K_est (same c_sigma)
                K_sigma = K_est**2 * c_sigma
                if uncertainty == "jackknife" and np.isfinite(K_sigma_jack):
                    K_sigma = K_sigma_jack
                elif uncertainty == "both" and np.isfinite(K_sigma_jack):
                    K_sigma = max(K_sigma, K_sigma_jack)
            except Exception as exc:
                base["warnings"].append(f"bias_correct minimisation failed: {exc}")

    # --- Radius and bounds ---
    r = float(_r_from_K(K_est, h))
    K_low = max(K_est - n_sigma * K_sigma, 1e-12)
    K_high = K_est + n_sigma * K_sigma
    r_low = float(_r_from_K(K_low, h))
    r_high = float(_r_from_K(K_high, h))
    r_sigma = (r_high - r_low) / (2.0 * n_sigma)

    base.update(
        dict(
            r=r, r_low=r_low, r_high=r_high, r_sigma=r_sigma,
            K=float(K_est), K_sigma=float(K_sigma),
            K_sigma_jack=float(K_sigma_jack) if np.isfinite(K_sigma_jack) else nan_r,
            theta_x_est=float(theta_x_est) if np.isfinite(theta_x_est) else nan_r,
            status="ok",
        )
    )
    return base


class SagittaFitter(BaseFitter):
    """2-D optimizer over (kappa, theta_x) using the exact projected-sagitta formula.

    Accurate to < 0.05 km for all tested camera tilts.  Fits only r and theta_x;
    other free_parameters are handled by a subsequent LimbFitter stage.
    """

    def __init__(
        self,
        limb: np.ndarray,
        h: float,
        f_px: float,
        y0: float,
        free_parameters: list,
        init_parameter_values: dict,
        parameter_limits: dict,
        sigma_px: "float | str" = "auto",
        n_sigma: float = 1.0,
        uncertainty: str = "jackknife",
        bias_correct: bool = False,
    ):
        self.limb = limb
        self.h = h
        self.f_px = f_px
        self.y0 = y0
        self.free_parameters = free_parameters
        self.init_parameter_values = init_parameter_values
        self.parameter_limits = parameter_limits
        self.sigma_px = sigma_px
        self.n_sigma = n_sigma
        self.uncertainty = uncertainty
        self.bias_correct = bias_correct

    def fit(self) -> dict:
        warn: list[str] = []

        # Step 1: OLS initial guess (uncertainty="ols" — cheapest; we only need K and theta_x_est)
        est = estimate_radius_via_sagitta(
            self.limb,
            h=self.h,
            f_px=self.f_px,
            y0=self.y0,
            sigma_px=self.sigma_px,
            n_sigma=self.n_sigma,
            bias_correct=self.bias_correct,
            uncertainty="ols",
        )
        warn.extend(est["warnings"])

        if est["status"] == "too_few_points":
            return {
                "updated_init": {},
                "updated_limits": {},
                "status": "too_few_points",
                "warnings": warn,
            }

        xs = np.where(~np.isnan(self.limb))[0].astype(float)
        ys = self.limb[~np.isnan(self.limb)].astype(float)
        u = xs - float(est["x_apex"])
        s = ys - float(est["y_apex"])

        K0 = float(est["K"]) if np.isfinite(est["K"]) else 100.0
        tx0 = float(est["theta_x_est"]) if np.isfinite(est["theta_x_est"]) else 0.0
        log_kappa0 = np.log(1.0 / max(K0, 1e-12))

        tx_lo, tx_hi = self.parameter_limits.get("theta_x", [-np.pi, np.pi])
        log_kappa_bounds = (-3.0, 3.0)
        tx_bounds = (float(tx_lo), float(tx_hi))

        # Step 2: 2-D L-BFGS-B over [log_kappa, theta_x]
        def cost_2d(params: np.ndarray) -> float:
            log_kap, tx = params
            r_t = _r_from_K(1.0 / np.exp(log_kap), self.h)
            s_pred = np.array(
                [limb_arc_sagitta(float(ui), tx, self.f_px, r_t, self.h) for ui in u]
            )
            s0 = float(np.mean(s - s_pred))
            return float(np.sum((s - s0 - s_pred) ** 2))

        try:
            res2d = minimize(
                cost_2d,
                x0=[log_kappa0, tx0],
                method="L-BFGS-B",
                bounds=[log_kappa_bounds, tx_bounds],
                options={"maxiter": 500, "ftol": 1e-12},
            )
            K_opt = 1.0 / float(np.exp(res2d.x[0]))
            theta_x_opt = float(res2d.x[1])
        except Exception as exc:
            warn.append(f"SagittaFitter 2-D minimisation failed: {exc}")
            K_opt = K0
            theta_x_opt = tx0

        r_opt = float(_r_from_K(K_opt, self.h))

        # Step 3: uncertainty bounds — jackknife (theta_x fixed) or OLS fallback
        K_sigma_ols = float(est.get("K_sigma", 0.0))
        K_sigma_jack = float("nan")

        if self.uncertainty in ("jackknife", "both"):
            n = len(u)
            K_loo = np.empty(n)
            for i in range(n):
                mask = np.ones(n, dtype=bool)
                mask[i] = False
                u_loo, s_loo = u[mask], s[mask]

                def cost_1d(log_kap: float) -> float:
                    r_t = _r_from_K(1.0 / np.exp(log_kap), self.h)
                    s_pred = np.array(
                        [
                            limb_arc_sagitta(
                                float(ui), theta_x_opt, self.f_px, r_t, self.h
                            )
                            for ui in u_loo
                        ]
                    )
                    s0 = float(np.mean(s_loo - s_pred))
                    return float(np.sum((s_loo - s0 - s_pred) ** 2))

                try:
                    res1d = minimize_scalar(
                        cost_1d, bounds=(-3.0, 3.0), method="bounded"
                    )
                    K_loo[i] = 1.0 / float(np.exp(res1d.x))
                except Exception:
                    K_loo[i] = K_opt

            valid = K_loo[np.isfinite(K_loo)]
            if len(valid) >= 2:
                K_sigma_jack = float(
                    np.sqrt((n - 1) / n * np.sum((valid - valid.mean()) ** 2))
                )

        if self.uncertainty == "ols":
            K_sigma = K_sigma_ols
        elif self.uncertainty == "jackknife":
            K_sigma = K_sigma_jack if np.isfinite(K_sigma_jack) else K_sigma_ols
        else:  # "both" — independent sources, combine in quadrature
            K_sigma = (
                float(np.sqrt(K_sigma_ols**2 + K_sigma_jack**2))
                if np.isfinite(K_sigma_jack)
                else K_sigma_ols
            )

        K_low = max(K_opt - self.n_sigma * K_sigma, 1e-12)
        K_high = K_opt + self.n_sigma * K_sigma
        r_low = float(_r_from_K(K_low, self.h))
        r_high = float(_r_from_K(K_high, self.h))

        extra_fp = [
            p
            for p in self.free_parameters
            if p not in ("r", "theta_x")
        ]
        if extra_fp:
            warn.append(
                f"SagittaFitter does not fit: {extra_fp}. "
                "Pass a subsequent LimbFitter stage to fit these parameters."
            )

        return {
            "r": r_opt,
            "r_low": r_low,
            "r_high": r_high,
            "theta_x_est": theta_x_opt,
            "theta_x_sigma": float("nan"),
            "K_sigma_jack": K_sigma_jack,
            "updated_init": {"r": r_opt, "theta_x": theta_x_opt},
            "updated_limits": {"r": [r_low, r_high]},
            "status": "ok",
            "warnings": warn,
        }


# ============================================================================
# UNCERTAINTY AND DIAGNOSTICS
# ============================================================================


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

        population_df = unpack_diff_evol_posteriors(observation)

        if parameter not in population_df.columns:
            raise ValueError(
                f"Parameter '{parameter}' not found in population posteriors"
            )

        param_samples = population_df[parameter] / scale_factor

        if uncertainty_type == "std":
            uncertainty = param_samples.std()
        elif uncertainty_type == "ptp":
            uncertainty = param_samples.max() - param_samples.min()
        elif uncertainty_type == "iqr":
            uncertainty = param_samples.quantile(0.75) - param_samples.quantile(0.25)
        elif uncertainty_type == "ci":
            lower = param_samples.quantile(0.025)
            upper = param_samples.quantile(0.975)
            uncertainty = {"lower": lower, "upper": upper, "width": upper - lower}
        else:
            raise ValueError(f"Unsupported uncertainty type: {uncertainty_type}")

        raw_data = param_samples.values

    elif method == "bootstrap":
        raise NotImplementedError(
            "Bootstrap uncertainty calculation not yet implemented"
        )

    elif method == "hessian":
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
    if (
        not hasattr(observation, "best_parameters")
        or observation.best_parameters is None
    ):
        return

    h = observation.best_parameters.get("h")
    r = observation.best_parameters.get("r")

    if h is None or r is None:
        return

    if h < 1000:
        warnings.warn(
            f"Fitted altitude ({h:.0f} m) is very low. At such low altitudes, "
            "planetary curvature is difficult to detect and the fit may be unreliable. "
            "Your data may be insufficient to accurately measure planetary radius.",
            UserWarning,
            stacklevel=3,
        )

    if r < 1000000:
        warnings.warn(
            f"Fitted radius ({r/1000:.0f} km) is very small. This suggests the "
            "optimization may have difficulty detecting planetary curvature in your data. "
            "Consider checking your image scale or using a different approach.",
            UserWarning,
            stacklevel=3,
        )

    if r > 100000000:
        warnings.warn(
            f"Fitted radius ({r/1000:.0f} km) is very large. This may indicate "
            "calibration issues or that the observed curvature is too subtle to measure "
            "reliably with the available data.",
            UserWarning,
            stacklevel=3,
        )

    if h < 5000 and r > 50000000:
        warnings.warn(
            f"Combination of relatively low altitude ({h/1000:.1f} km) and large radius "
            f"({r/1000:.0f} km) suggests the data may be insufficient to reliably detect "
            "planetary curvature. The apparent 'flat' horizon may reflect measurement "
            "limitations rather than actual planetary geometry.",
            UserWarning,
            stacklevel=3,
        )
