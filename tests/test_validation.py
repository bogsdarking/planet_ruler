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
Tests for configuration validation.
"""

import pytest
import math
from planet_ruler.validation import validate_limb_config  # Adjust import path as needed


class TestValidateConfig:
    """Test suite for validate_limb_config function."""

    def test_valid_configuration(self):
        """Test that a valid configuration passes validation."""
        config = {
            "init_parameter_values": {
                "r": 6371000,
                "h": 10000,
                "theta_x": 0.1,
                "f": 0.006,
            },
            "parameter_limits": {
                "r": [1000000, 100000000],
                "h": [9000, 11000],
                "theta_x": [-3.14, 3.14],  # Wide - good!
                "f": [0.005, 0.007],
            },
        }
        # Should not raise
        validate_limb_config(config, strict=True)

    def test_init_value_below_lower_limit(self):
        """Test that init value below lower limit raises error."""
        config = {
            "init_parameter_values": {"r": 500000},  # Too low
            "parameter_limits": {"r": [1000000, 100000000]},
        }
        with pytest.raises(AssertionError, match="violates stated lower limit"):
            validate_limb_config(config, strict=True)

    def test_init_value_above_upper_limit(self):
        """Test that init value above upper limit raises error."""
        config = {
            "init_parameter_values": {"r": 150000000},  # Too high
            "parameter_limits": {"r": [1000000, 100000000]},
        }
        with pytest.raises(AssertionError, match="violates stated upper limit"):
            validate_limb_config(config, strict=True)

    def test_missing_limits_for_init_parameter_strict(self):
        """Test that missing limits for init parameter raises error in strict mode."""
        config = {
            "init_parameter_values": {"r": 6371000, "theta_x": 0.0},
            "parameter_limits": {"r": [1000000, 100000000]},
            # theta_x has init but no limits
        }
        with pytest.raises(AssertionError, match="no limits defined"):
            validate_limb_config(config, strict=True)

    def test_missing_limits_for_init_parameter_non_strict(self, caplog):
        """Test that missing limits for init parameter only warns in non-strict mode."""
        config = {
            "init_parameter_values": {"r": 6371000, "theta_x": 0.0},
            "parameter_limits": {"r": [1000000, 100000000]},
        }
        # Should not raise, but should warn
        validate_limb_config(config, strict=False)
        assert "no limits defined" in caplog.text.lower()

    def test_tight_theta_limits_warning(self, caplog):
        """Test that tight theta limits generate a warning."""
        config = {
            "init_parameter_values": {"theta_x": 0.1},
            "parameter_limits": {"theta_x": [-0.5, 0.5]},  # Too tight!
        }
        validate_limb_config(config, strict=True)
        assert "tight limits" in caplog.text.lower()
        assert "theta_x" in caplog.text

    def test_wide_theta_limits_no_warning(self, caplog):
        """Test that wide theta limits don't generate a warning."""
        config = {
            "init_parameter_values": {"theta_x": 0.1},
            "parameter_limits": {"theta_x": [-math.pi, math.pi]},  # Wide - good!
        }
        validate_limb_config(config, strict=True)
        # Should not warn about theta limits
        assert "tight limits" not in caplog.text.lower() or "theta_x" not in caplog.text

    def test_tight_radius_limits_warning(self, caplog):
        """Test that tight radius limits generate a warning."""
        config = {
            "init_parameter_values": {"r": 6371000},
            "parameter_limits": {"r": [6000000, 7000000]},  # Only ~1.2x range
        }
        validate_limb_config(config, strict=True)
        assert "radius limits" in caplog.text.lower()

    def test_wide_radius_limits_no_warning(self, caplog):
        """Test that wide radius limits don't generate a warning."""
        config = {
            "init_parameter_values": {"r": 6371000},
            "parameter_limits": {"r": [1000000, 100000000]},  # 100x range - good!
        }
        validate_limb_config(config, strict=True)
        # Should not warn about radius limits
        assert (
            "radius limits" not in caplog.text.lower()
            or "are tight" not in caplog.text.lower()
        )

    def test_empty_config(self):
        """Test that empty config doesn't crash."""
        config = {"init_parameter_values": {}, "parameter_limits": {}}
        # Should not raise
        validate_limb_config(config, strict=True)

    def test_multiple_parameters(self):
        """Test validation with multiple parameters."""
        config = {
            "init_parameter_values": {
                "r": 6371000,
                "h": 10000,
                "f": 0.006,
                "w": 0.0076,
                "theta_x": 0.1,
                "theta_y": 0.0,
                "theta_z": 0.0,
            },
            "parameter_limits": {
                "r": [1000000, 100000000],
                "h": [9000, 11000],
                "f": [0.0054, 0.0066],
                "w": [0.00684, 0.00836],
                "theta_x": [-3.14, 3.14],
                "theta_y": [-3.14, 3.14],
                "theta_z": [-3.14, 3.14],
            },
        }
        # Should not raise
        validate_limb_config(config, strict=True)

    def test_boundary_values(self):
        """Test that boundary values (exactly at limits) are accepted."""
        config = {
            "init_parameter_values": {
                "r": 1000000,  # Exactly at lower limit
                "h": 11000,  # Exactly at upper limit
            },
            "parameter_limits": {"r": [1000000, 100000000], "h": [9000, 11000]},
        }
        # Should not raise - boundary values are valid
        validate_limb_config(config, strict=True)


class TestValidateConfigIntegration:
    """Integration tests with realistic planet_ruler configs."""

    def test_earth_observation_config(self):
        """Test a realistic Earth observation configuration."""
        config = {
            "description": "Auto-generated from airplane.jpg (planet: earth)",
            "free_parameters": ["r", "h", "theta_x", "theta_y", "theta_z", "f", "w"],
            "init_parameter_values": {
                "r": 8892000,  # Perturbed from 6371000
                "h": 10668,
                "f": 0.0061,
                "w": 0.0076,
                "theta_x": 0.0234,
                "theta_y": 0.0,
                "theta_z": 0.0,
            },
            "parameter_limits": {
                "r": [1000000, 100000000],
                "h": [9601, 11735],
                "f": [0.00549, 0.00671],
                "w": [0.00684, 0.00836],
                "theta_x": [-3.14, 3.14],
                "theta_y": [-3.14, 3.14],
                "theta_z": [-3.14, 3.14],
            },
        }
        validate_limb_config(config, strict=True)

    def test_mars_observation_config(self):
        """Test a realistic Mars observation configuration."""
        config = {
            "description": "Mars rover observation",
            "free_parameters": ["r", "h", "theta_x", "theta_y", "theta_z", "f"],
            "init_parameter_values": {
                "r": 5084250,  # Perturbed Mars radius
                "h": 4500,  # Gale Crater elevation
                "f": 0.024,  # Mastcam focal length
                "theta_x": 0.015,
                "theta_y": 0.0,
                "theta_z": 0.0,
            },
            "parameter_limits": {
                "r": [1000000, 100000000],
                "h": [4050, 4950],
                "f": [0.0216, 0.0264],
                "theta_x": [-3.14, 3.14],
                "theta_y": [-3.14, 3.14],
                "theta_z": [-3.14, 3.14],
            },
        }
        validate_limb_config(config, strict=True)

    def test_jupiter_observation_config(self):
        """Test configuration for a gas giant (Jupiter)."""
        config = {
            "description": "Jupiter observation from space probe",
            "free_parameters": ["r", "h", "theta_x", "theta_y", "theta_z"],
            "init_parameter_values": {
                "r": 52433250,  # Perturbed Jupiter radius
                "h": 1000000,  # 1000 km altitude
                "theta_x": 0.08,
                "theta_y": 0.0,
                "theta_z": 0.0,
            },
            "parameter_limits": {
                "r": [10000000, 150000000],  # Wider for gas giants
                "h": [900000, 1100000],
                "theta_x": [-3.14, 3.14],
                "theta_y": [-3.14, 3.14],
                "theta_z": [-3.14, 3.14],
            },
        }
        validate_limb_config(config, strict=True)


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
