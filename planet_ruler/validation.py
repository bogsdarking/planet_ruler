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

# planet_ruler/validation.py

"""
Validation functions for planet_ruler.
"""

import math
import logging

logger = logging.getLogger(__name__)


def validate_limb_config(config: dict, strict: bool = True) -> None:
    """
    Validate that a planet_ruler configuration is internally consistent.

    Checks:
    1. Initial parameter values are within their specified limits
    2. Theta parameter limits are wide enough (avoid r-h coupling issues)
    3. Radius limits span reasonable range for robust optimization

    Args:
        config: Configuration dictionary with keys:
            - 'init_parameter_values': dict of parameter initial values
            - 'parameter_limits': dict of [lower, upper] limits
        strict: If True, raise exceptions. If False, only log warnings.

    Raises:
        AssertionError: If strict=True and validation fails

    Example:
        >>> config = {
        ...     'init_parameter_values': {'r': 6371000, 'theta_x': 0.1},
        ...     'parameter_limits': {'r': [1e6, 1e8], 'theta_x': [-3.14, 3.14]}
        ... }
        >>> validate_config(config)  # Passes
    """
    init_values = config.get("init_parameter_values", {})
    param_limits = config.get("parameter_limits", {})

    # Check 1: Initial values within limits
    # This replicates the original strict behavior from observation.py
    for param, value in init_values.items():
        # Check that parameter has limits defined
        if param not in param_limits:
            msg = (
                f"Parameter '{param}' has initial value ({value}) "
                f"but no limits defined in parameter_limits."
            )
            if strict:
                raise AssertionError(msg)
            else:
                logger.warning(f"⚠️  {msg}")
                continue

        lower, upper = param_limits[param]

        # Check lower bound
        if value < lower:
            msg = (
                f"Initial value for parameter '{param}' ({value}) "
                f"violates stated lower limit ({lower})."
            )
            if strict:
                raise AssertionError(msg)
            else:
                logger.warning(f"⚠️  {msg}")

        # Check upper bound
        if value > upper:
            msg = (
                f"Initial value for parameter '{param}' ({value}) "
                f"violates stated upper limit ({upper})."
            )
            if strict:
                raise AssertionError(msg)
            else:
                logger.warning(f"⚠️  {msg}")

    # Check 2: Theta parameter limits (orientation angles)
    # These must be wide to avoid coupling issues with r and h
    theta_params = ["theta_x", "theta_y", "theta_z"]
    for param in theta_params:
        if param in param_limits:
            lower, upper = param_limits[param]
            range_size = upper - lower

            # Warn if range is less than π (180°)
            if range_size < math.pi:
                msg = (
                    f"Parameter '{param}' has tight limits [{lower:.3f}, {upper:.3f}] rad "
                    f"(range = {range_size:.3f} < π). "
                    f"Tight theta limits may prevent reaching the true solution due to "
                    f"geometric coupling with radius (r) and altitude (h). "
                    f"Recommended: Use wide limits ≈[-π, π] or omit to use defaults."
                )
                if strict:
                    logger.warning(f"⚠️  {msg}")
                else:
                    logger.warning(f"⚠️  {msg}")

    # Check 3: Radius limits span reasonable range
    if "r" in param_limits:
        lower, upper = param_limits["r"]
        ratio = upper / lower

        # Warn if range is less than 2 orders of magnitude (100x)
        if ratio < 100:
            msg = (
                f"Radius limits [{lower/1000:.0f}, {upper/1000:.0f}] km "
                f"span only {ratio:.1f}x (less than 2 orders of magnitude). "
                f"Consider wider range for robust optimization, e.g., [1000, 100000] km. "
                f"Tight limits may cause optimization issues or falsely appear data-driven."
            )
            logger.warning(f"⚠️  {msg}")

    logger.debug(f"Configuration validation {'passed' if strict else 'completed'}")
