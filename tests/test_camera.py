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
Tests for automatic camera parameter extraction.
"""

import pytest
import math
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add the parent directory to sys.path to import camera module directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import camera module directly to avoid the torch import issue
import importlib.util

camera_spec = importlib.util.spec_from_file_location(
    "planet_ruler.camera",
    os.path.join(os.path.dirname(__file__), "..", "planet_ruler", "camera.py"),
)

if camera_spec is None:
    raise ImportError("Could not load camera module spec")

camera_module = importlib.util.module_from_spec(camera_spec)

if camera_spec.loader is None:
    raise ImportError("Could not load camera module loader")

camera_spec.loader.exec_module(camera_module)

# Import what we need from the camera module
CAMERA_DB = camera_module.CAMERA_DB
PLANET_RADII = camera_module.PLANET_RADII
get_camera_model = camera_module.get_camera_model
get_sensor_statistics_by_type = camera_module.get_sensor_statistics_by_type
infer_camera_type = camera_module.infer_camera_type
extract_camera_parameters = camera_module.extract_camera_parameters
get_gps_altitude = camera_module.get_gps_altitude
get_initial_radius = camera_module.get_initial_radius
create_config_from_image = camera_module.create_config_from_image
calculate_sensor_dimensions = camera_module.calculate_sensor_dimensions
get_focal_length_mm = camera_module.get_focal_length_mm
get_focal_length_35mm_equiv = camera_module.get_focal_length_35mm_equiv


class TestCameraDatabase:
    """Test the camera database structure and contents."""

    def test_database_has_required_fields(self):
        """Test that all cameras have required fields."""
        for model, specs in CAMERA_DB.items():
            if model == "default":
                continue

            # All cameras must have a type
            assert "type" in specs
            assert specs["type"] in [
                "phone",
                "compact",
                "dslr",
                "mirrorless",
                "unknown",
            ]

            # Check sensor dimensions (either directly or in cameras array)
            if "cameras" in specs:
                # Multi-camera format: check each camera module
                assert (
                    len(specs["cameras"]) > 0
                ), f"Camera {model} has empty cameras array"
                for cam in specs["cameras"]:
                    assert (
                        "sensor_width" in cam or "sensor_height" in cam
                    ), f"Camera {model} missing sensor dimensions in camera module"
            else:
                # Single-camera format: check top level
                assert (
                    "sensor_width" in specs or "sensor_height" in specs
                ), f"Camera {model} missing sensor dimensions"

    def test_database_has_known_cameras(self):
        """Test that database includes some known cameras."""
        assert "iPhone 13" in CAMERA_DB
        assert "Canon PowerShot G12" in CAMERA_DB
        assert "default" in CAMERA_DB

    def test_sensor_dimensions_positive(self):
        """Test that all sensor dimensions are positive."""
        for model, specs in CAMERA_DB.items():
            if "cameras" in specs:
                # Multi-camera format: check each camera module
                for cam in specs["cameras"]:
                    assert (
                        cam.get("sensor_width", 1) > 0
                    ), f"Camera {model} has non-positive sensor_width"
                    assert (
                        cam.get("sensor_height", 1) > 0
                    ), f"Camera {model} has non-positive sensor_height"
            else:
                # Single-camera format: check top level
                assert (
                    specs.get("sensor_width", 1) > 0
                ), f"Camera {model} has non-positive sensor_width"
                assert (
                    specs.get("sensor_height", 1) > 0
                ), f"Camera {model} has non-positive sensor_height"


class TestPlanetDatabase:
    """Test the planet radius database."""

    def test_planet_database_has_common_planets(self):
        """Test that database includes common planets."""
        assert "earth" in PLANET_RADII
        assert "mars" in PLANET_RADII
        assert "jupiter" in PLANET_RADII
        assert "moon" in PLANET_RADII

    def test_planet_radii_positive(self):
        """Test that all planet radii are positive."""
        for planet, radius in PLANET_RADII.items():
            assert radius > 0

    def test_planet_radii_order_of_magnitude(self):
        """Test that planet radii are in reasonable ranges."""
        assert PLANET_RADII["moon"] < PLANET_RADII["earth"] < PLANET_RADII["jupiter"]
        assert 1_000_000 < PLANET_RADII["earth"] < 10_000_000  # Earth: ~6000 km


class TestGetCameraModel:
    """Test camera model detection from EXIF."""

    def test_exact_match_with_make_and_model(self):
        """Test exact match when Make and Model are separate."""
        exif = {"Make": "Canon", "Model": "PowerShot G12"}
        model = get_camera_model(exif)
        assert model == "Canon PowerShot G12"

    def test_model_already_contains_make(self):
        """Test when Model already includes Make."""
        exif = {"Make": "Canon", "Model": "Canon PowerShot G12"}
        model = get_camera_model(exif)
        assert model == "Canon PowerShot G12"

    def test_partial_match(self):
        """Test partial matching for unknown models."""
        # iPhone 13 Pro Max is not in the DB; expect the longest partial match
        exif = {"Make": "Apple", "Model": "iPhone 13 Pro Max"}
        model = get_camera_model(exif)
        assert model == "iPhone 13 Pro"

    def test_no_exif_data(self):
        """Test with empty EXIF data."""
        exif = {}
        model = get_camera_model(exif)
        assert model is None

    def test_model_only(self):
        """Test with only Model field."""
        exif = {"Model": "iPhone 13"}
        model = get_camera_model(exif)
        assert model == "iPhone 13"


class TestSensorStatistics:
    """Test sensor dimension statistics by camera type."""

    def test_phone_statistics(self):
        """Test statistics for phone cameras."""
        stats = get_sensor_statistics_by_type("phone")
        assert stats is not None
        assert "sensor_width_median" in stats
        assert "sensor_width_min" in stats
        assert "sensor_width_max" in stats
        assert (
            stats["sensor_width_min"]
            <= stats["sensor_width_median"]
            <= stats["sensor_width_max"]
        )
        assert stats["count"] > 0

    def test_compact_statistics(self):
        """Test statistics for compact cameras."""
        stats = get_sensor_statistics_by_type("compact")
        assert stats is not None
        assert stats["count"] > 0

    def test_unknown_type_returns_none(self):
        """Test that unknown camera type returns None."""
        stats = get_sensor_statistics_by_type("nonexistent_type")
        assert stats is None


class TestInferCameraType:
    """Test camera type inference from EXIF."""

    def test_infer_phone_from_iphone(self):
        """Test iPhone is inferred as phone."""
        exif = {"Make": "Apple", "Model": "iPhone 99"}
        camera_type = infer_camera_type(exif)
        assert camera_type == "phone"

    def test_infer_phone_from_samsung(self):
        """Test Samsung is inferred as phone."""
        exif = {"Make": "Samsung", "Model": "SM-G999999"}
        camera_type = infer_camera_type(exif)
        assert camera_type == "phone"

    def test_infer_dslr_from_canon_eos(self):
        """Test Canon EOS is inferred as DSLR."""
        exif = {"Make": "Canon", "Model": "Canon EOS 999D"}
        camera_type = infer_camera_type(exif)
        assert camera_type == "dslr"

    def test_infer_compact_from_powershot(self):
        """Test PowerShot is inferred as compact."""
        exif = {"Make": "Canon", "Model": "PowerShot SX999"}
        camera_type = infer_camera_type(exif)
        assert camera_type == "compact"

    def test_unknown_camera_returns_none(self):
        """Test unknown camera returns None."""
        exif = {"Make": "UnknownBrand", "Model": "UnknownModel"}
        camera_type = infer_camera_type(exif)
        assert camera_type is None


class TestCalculateSensorDimensions:
    """Test sensor dimension calculation from focal length ratio."""

    def test_calculation_with_known_values(self):
        """Test calculation with known focal length values."""
        # Example: 6.1mm lens that's 28mm equivalent
        focal_mm = 6.1
        focal_35mm = 28.0
        width, height = calculate_sensor_dimensions(focal_mm, focal_35mm)

        # Crop factor = 28/6.1 ≈ 4.59
        # Sensor width = 36/4.59 ≈ 7.8mm
        assert 7.0 < width < 8.5
        assert 4.5 < height < 6.0

    def test_full_frame_returns_36mm(self):
        """Test that full frame (1:1 ratio) returns 36mm."""
        width, height = calculate_sensor_dimensions(50.0, 50.0)
        assert abs(width - 36.0) < 0.1
        assert abs(height - 24.0) < 0.1


class TestGetInitialRadius:
    """Test initial radius generation."""

    def test_earth_returns_exact_radius(self):
        """Test Earth returns the exact known radius."""
        r = get_initial_radius("earth")
        assert r == float(PLANET_RADII["earth"])

    def test_mars_returns_exact_radius(self):
        """Test Mars returns the exact known radius."""
        r = get_initial_radius("mars")
        assert r == float(PLANET_RADII["mars"])

    def test_unknown_planet_returns_10000_km(self):
        """Test unknown planet returns 10,000 km."""
        r = get_initial_radius("unknown_planet")
        assert r == 10_000_000.0

    def test_deterministic_multiple_calls(self):
        """Test that multiple calls give the same result."""
        results = [get_initial_radius("earth") for _ in range(5)]
        assert len(set(results)) == 1


class TestSampleParamFromBounds:
    """Test the sample_param_from_bounds helper."""

    def test_returns_value_within_bounds(self):
        """Test that result is within [lo, hi]."""
        sample_param_from_bounds = camera_module.sample_param_from_bounds
        lo, hi = 1.0, 10.0
        result = sample_param_from_bounds(lo, hi, perturbation_factor=1.0)
        assert lo <= result <= hi

    def test_reproducible_with_seed(self):
        """Test that same seed gives same result."""
        sample_param_from_bounds = camera_module.sample_param_from_bounds
        lo, hi = 100.0, 200.0
        r1 = sample_param_from_bounds(lo, hi, seed=42)
        r2 = sample_param_from_bounds(lo, hi, seed=42)
        assert r1 == r2

    def test_different_seeds_give_different_results(self):
        """Test that different seeds give different results."""
        sample_param_from_bounds = camera_module.sample_param_from_bounds
        lo, hi = 0.0, 1000.0
        r1 = sample_param_from_bounds(lo, hi, seed=1)
        r2 = sample_param_from_bounds(lo, hi, seed=2)
        assert r1 != r2

    def test_perturbation_factor_restricts_range(self):
        """Test that perturbation_factor < 1.0 restricts sampling to a sub-interval."""
        sample_param_from_bounds = camera_module.sample_param_from_bounds
        lo, hi = 0.0, 100.0
        # perturbation_factor=0.5 → sample from [25, 75]
        for seed in range(20):
            result = sample_param_from_bounds(lo, hi, perturbation_factor=0.5, seed=seed)
            assert 25.0 <= result <= 75.0

    def test_perturbation_factor_zero_gives_midpoint(self):
        """Test that perturbation_factor=0.0 returns the midpoint."""
        sample_param_from_bounds = camera_module.sample_param_from_bounds
        lo, hi = 2.0, 8.0
        result = sample_param_from_bounds(lo, hi, perturbation_factor=0.0, seed=99)
        assert result == 5.0


class TestExtractCameraParameters:
    """Test camera parameter extraction from images."""

    @patch.object(camera_module, "extract_exif")
    @patch.object(camera_module, "get_image_dimensions")
    def test_known_camera_high_confidence(self, mock_dims, mock_exif):
        """Test extraction with known camera model."""
        mock_dims.return_value = (4000, 3000)
        mock_exif.return_value = {
            "Make": "Canon",
            "Model": "PowerShot G12",
            "FocalLength": (61, 10),  # 6.1mm as tuple
        }

        params = extract_camera_parameters("dummy.jpg")

        assert params["camera_model"] == "Canon PowerShot G12"
        assert params["camera_type"] == "compact"
        assert params["confidence"] == "high"
        assert params["sensor_width_mm"] == 7.6
        assert params["focal_length_mm"] == 6.1

    @patch.object(camera_module, "extract_exif")
    @patch.object(camera_module, "get_image_dimensions")
    def test_unknown_phone_medium_low_confidence(self, mock_dims, mock_exif):
        """Test extraction with unknown phone (inferred type)."""
        mock_dims.return_value = (4000, 3000)
        mock_exif.return_value = {
            "Make": "Samsung",
            "Model": "SM-G999999",  # Not in database
            "FocalLength": (4.5, 1),
        }

        params = extract_camera_parameters("dummy.jpg")

        assert params["camera_type"] == "phone"
        assert params["confidence"] == "medium-low"
        assert params["sensor_width_min"] is not None  # Has data-driven limits
        assert params["sensor_width_max"] is not None

    @patch.object(camera_module, "extract_exif")
    @patch.object(camera_module, "get_image_dimensions")
    def test_calculated_from_focal_ratio(self, mock_dims, mock_exif):
        """Test sensor calculation from focal length ratio."""
        mock_dims.return_value = (4000, 3000)
        mock_exif.return_value = {
            "Make": "Unknown",
            "Model": "Unknown Camera",
            "FocalLength": (6.1, 1),
            "FocalLengthIn35mmFilm": 28,
        }

        params = extract_camera_parameters("dummy.jpg")

        assert params["confidence"] == "medium"
        assert params["camera_type"] == "calculated"
        assert 7.0 < params["sensor_width_mm"] < 8.5  # Calculated value


class TestGPSAltitude:
    """Test GPS altitude extraction."""

    @patch.object(camera_module, "extract_exif")
    def test_gps_altitude_present(self, mock_exif):
        """Test extraction when GPS altitude is present."""
        mock_exif.return_value = {
            "GPSInfo": {
                6: (10668, 1),  # Altitude as tuple (numerator, denominator)
                5: 0,  # Above sea level
            }
        }

        altitude = get_gps_altitude("dummy.jpg")
        assert altitude == 10668.0

    @patch.object(camera_module, "extract_exif")
    def test_gps_altitude_below_sea_level(self, mock_exif):
        """Test altitude below sea level is negative."""
        mock_exif.return_value = {"GPSInfo": {6: (100, 1), 5: 1}}  # Below sea level

        altitude = get_gps_altitude("dummy.jpg")
        assert altitude == -100.0

    @patch.object(camera_module, "extract_exif")
    def test_no_gps_data(self, mock_exif):
        """Test returns None when no GPS data."""
        mock_exif.return_value = {}

        altitude = get_gps_altitude("dummy.jpg")
        assert altitude is None


class TestCreateConfigFromImage:
    """Test complete config generation from images."""

    @patch.object(camera_module, "extract_camera_parameters")
    @patch.object(camera_module, "get_gps_altitude")
    @patch.object(camera_module, "get_initial_radius")
    def test_config_with_gps_altitude(self, mock_radius, mock_gps, mock_params):
        """Test config generation with GPS altitude."""
        mock_params.return_value = {
            "focal_length_mm": 6.1,
            "sensor_width_mm": 7.6,
            "sensor_height_mm": 5.7,
            "sensor_width_min": None,
            "sensor_width_max": None,
            "camera_model": "Canon PowerShot G12",
            "camera_type": "compact",
            "confidence": "high",
            "image_width_px": 4000,
            "image_height_px": 3000,
        }
        mock_gps.return_value = 10668.0
        mock_radius.return_value = 6371000

        config = create_config_from_image("dummy.jpg", planet="earth")

        assert config["init_parameter_values"]["r"] == 6371000
        assert config["init_parameter_values"]["h"] == 10668.0
        assert "theta_x" in config["init_parameter_values"]
        assert "r" in config["free_parameters"]
        assert "h" in config["free_parameters"]
        assert "theta_x" in config["free_parameters"]
        assert config["camera_info"]["planet"] == "earth"

    @patch.object(camera_module, "extract_camera_parameters")
    @patch.object(camera_module, "get_gps_altitude")
    def test_config_without_gps_requires_manual_altitude(self, mock_gps, mock_params):
        """Test that missing GPS altitude raises error."""
        mock_params.return_value = {
            "focal_length_mm": 6.1,
            "sensor_width_mm": 7.6,
            "sensor_height_mm": 5.7,
            "sensor_width_min": None,
            "sensor_width_max": None,
            "camera_model": "Canon PowerShot G12",
            "camera_type": "compact",
            "confidence": "high",
            "image_width_px": 4000,
            "image_height_px": 3000,
        }
        mock_gps.return_value = None  # No GPS

        with pytest.raises(ValueError, match="Altitude is required"):
            create_config_from_image("dummy.jpg")

    @patch.object(camera_module, "extract_camera_parameters")
    @patch.object(camera_module, "get_gps_altitude")
    @patch.object(camera_module, "get_initial_radius")
    def test_config_with_manual_altitude(self, mock_radius, mock_gps, mock_params):
        """Test config generation with manually provided altitude."""
        mock_params.return_value = {
            "focal_length_mm": 6.1,
            "sensor_width_mm": 7.6,
            "sensor_height_mm": 5.7,
            "sensor_width_min": None,
            "sensor_width_max": None,
            "camera_model": "Canon PowerShot G12",
            "camera_type": "compact",
            "confidence": "high",
            "image_width_px": 4000,
            "image_height_px": 3000,
        }
        mock_gps.return_value = None
        mock_radius.return_value = PLANET_RADII["mars"]

        config = create_config_from_image("dummy.jpg", altitude_m=4500, planet="mars")

        assert config["init_parameter_values"]["h"] == 4500
        assert config["camera_info"]["planet"] == "mars"
        assert config["camera_info"]["altitude_source"] == "manual"

    @patch.object(camera_module, "extract_camera_parameters")
    @patch.object(camera_module, "get_gps_altitude")
    @patch.object(camera_module, "get_initial_radius")
    def test_config_with_data_driven_sensor_limits(
        self, mock_radius, mock_gps, mock_params
    ):
        """Test config uses data-driven limits for unknown camera."""
        mock_params.return_value = {
            "focal_length_mm": 4.5,
            "sensor_width_mm": 7.6,  # Median
            "sensor_height_mm": 5.7,
            "sensor_width_min": 4.8,  # Min from phone DB
            "sensor_width_max": 9.8,  # Max from phone DB
            "sensor_height_min": 3.6,
            "sensor_height_max": 7.3,
            "camera_model": "Unknown Phone",
            "camera_type": "phone",
            "confidence": "medium-low",
            "image_width_px": 4000,
            "image_height_px": 3000,
        }
        mock_gps.return_value = 10000.0
        mock_radius.return_value = 6371000

        config = create_config_from_image("dummy.jpg", altitude_m=10000)

        # Should use data-driven limits, not tight ±10%
        assert config["parameter_limits"]["w"][0] == 4.8 / 1000  # Min from DB
        assert config["parameter_limits"]["w"][1] == 9.8 / 1000  # Max from DB

    @patch.object(camera_module, "extract_camera_parameters")
    @patch.object(camera_module, "get_gps_altitude")
    @patch.object(camera_module, "get_initial_radius")
    def test_config_parameter_limits_include_constraints(
        self, mock_radius, mock_gps, mock_params
    ):
        """Test that config includes appropriate parameter limits."""
        mock_params.return_value = {
            "focal_length_mm": 6.1,
            "sensor_width_mm": 7.6,
            "sensor_height_mm": 5.7,
            "sensor_width_min": None,
            "sensor_width_max": None,
            "camera_model": "Canon PowerShot G12",
            "camera_type": "compact",
            "confidence": "high",
            "image_width_px": 4000,
            "image_height_px": 3000,
        }
        mock_gps.return_value = 10000.0
        mock_radius.return_value = 6371000

        # Let GPS mock supply altitude so the GPS path is exercised
        config = create_config_from_image(
            "dummy.jpg", limits_preset="balanced"
        )

        # r bounds scale with preset around planet init radius (±20% balanced)
        r_init = config["init_parameter_values"]["r"]
        assert abs(config["parameter_limits"]["r"][0] - r_init * 0.80) < 1
        assert abs(config["parameter_limits"]["r"][1] - r_init * 1.20) < 1

        # focal length has ±5% limits (balanced preset)
        f_init = config["init_parameter_values"]["f"]
        assert abs(config["parameter_limits"]["f"][0] - f_init * 0.95) < 1e-6
        assert abs(config["parameter_limits"]["f"][1] - f_init * 1.05) < 1e-6

        # GPS altitude gets h_gps tolerance (±10% for balanced)
        h_init = config["init_parameter_values"]["h"]
        assert abs(config["parameter_limits"]["h"][0] - h_init * 0.90) < 1
        assert abs(config["parameter_limits"]["h"][1] - h_init * 1.10) < 1

    @patch.object(camera_module, "extract_camera_parameters")
    @patch.object(camera_module, "get_gps_altitude")
    @patch.object(camera_module, "get_initial_radius")
    def test_param_tolerance_deprecated(
        self, mock_radius, mock_gps, mock_params
    ):
        """param_tolerance raises DeprecationWarning; balanced is used."""
        mock_params.return_value = {
            "focal_length_mm": 6.1,
            "sensor_width_mm": 7.6,
            "sensor_height_mm": 5.7,
            "sensor_width_min": None,
            "sensor_width_max": None,
            "camera_model": "Canon PowerShot G12",
            "camera_type": "compact",
            "confidence": "high",
            "image_width_px": 4000,
            "image_height_px": 3000,
        }
        mock_gps.return_value = 10000.0
        mock_radius.return_value = 6371000

        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = create_config_from_image(
                "dummy.jpg", altitude_m=10000, param_tolerance=0.99
            )
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "limits_preset" in str(w[0].message)
        # Falls back to balanced — r bounds are ±20%, not ±99%
        r_init = config["init_parameter_values"]["r"]
        assert abs(config["parameter_limits"]["r"][0] - r_init * 0.80) < 1

    @patch.object(camera_module, "extract_camera_parameters")
    @patch.object(camera_module, "get_gps_altitude")
    @patch.object(camera_module, "get_initial_radius")
    def test_r_init_is_true_planet_radius(self, mock_radius, mock_gps, mock_params):
        """Test that r_init is always the true planet radius from get_initial_radius."""
        mock_params.return_value = {
            "focal_length_mm": 6.1,
            "sensor_width_mm": 7.6,
            "sensor_height_mm": 5.7,
            "sensor_width_min": None,
            "sensor_width_max": None,
            "camera_model": "Canon PowerShot G12",
            "camera_type": "compact",
            "confidence": "high",
            "image_width_px": 4000,
            "image_height_px": 3000,
        }
        mock_gps.return_value = 10000.0
        mock_radius.return_value = PLANET_RADII["earth"]

        config = create_config_from_image("dummy.jpg", altitude_m=10000)

        assert config["init_parameter_values"]["r"] == PLANET_RADII["earth"]


class TestConfigIntegration:
    """Integration tests for realistic workflows."""

    @patch.object(camera_module, "Image")
    @patch.object(camera_module, "extract_exif")
    def test_real_workflow_airplane_photo(self, mock_exif, mock_image):
        """Test realistic workflow: airplane photo with GPS."""
        # Mock image
        mock_img = MagicMock()
        mock_img.size = (4000, 3000)
        mock_img._getexif = Mock(
            return_value={
                271: "Apple",  # Make
                272: "iPhone 13",  # Model
                37386: (4.2, 1),  # FocalLength
                41989: 26,  # FocalLengthIn35mmFilm
                34853: {6: (10668, 1), 5: 0},  # GPSInfo  # Altitude  # Above sea level
            }
        )
        mock_image.open.return_value.__enter__ = Mock(return_value=mock_img)
        mock_image.open.return_value.__exit__ = Mock()

        # Mock EXIF extraction
        mock_exif.return_value = {
            "Make": "Apple",
            "Model": "iPhone 13",
            "FocalLength": (4.2, 1),
            "FocalLengthIn35mmFilm": 26,
            "GPSInfo": {6: (10668, 1), 5: 0},
        }

        config = create_config_from_image("airplane.jpg", planet="earth")

        # Verify config is reasonable
        assert config["camera_info"]["camera_model"] == "iPhone 13"
        assert config["camera_info"]["altitude_source"] == "gps"
        assert "r" in config["free_parameters"]
        assert (
            len(config["free_parameters"]) >= 5
        )  # r, h, theta_x, theta_y, theta_z, ...


class TestInitParamsFromBounds:
    """Test the init_params_from_bounds helper."""

    def setup_method(self):
        self.init_params_from_bounds = camera_module.init_params_from_bounds

    def test_returns_dict_with_requested_keys(self):
        """Test that returned dict has keys matching requested params."""
        limits = {"r": [1e6, 1e8], "h": [9000.0, 11000.0]}
        result = self.init_params_from_bounds(limits, params=["r", "h"])
        assert set(result.keys()) == {"r", "h"}

    def test_values_within_bounds(self):
        """Test that all sampled values are within their bounds."""
        limits = {
            "r": [1e6, 1e8],
            "h": [9000.0, 11000.0],
            "f": [0.004, 0.006],
        }
        for seed in range(10):
            result = self.init_params_from_bounds(limits, seed=seed)
            for param, (lo, hi) in limits.items():
                assert lo <= result[param] <= hi

    def test_reproducible_with_same_seed(self):
        """Test that same seed gives identical results."""
        limits = {"r": [1e6, 1e8], "h": [9000.0, 11000.0]}
        r1 = self.init_params_from_bounds(limits, seed=42)
        r2 = self.init_params_from_bounds(limits, seed=42)
        assert r1 == r2

    def test_different_seeds_give_different_values(self):
        """Test that different seeds give different sampled values."""
        limits = {"r": [1e6, 1e8], "h": [9000.0, 11000.0]}
        r1 = self.init_params_from_bounds(limits, seed=1)
        r2 = self.init_params_from_bounds(limits, seed=2)
        assert r1["r"] != r2["r"] or r1["h"] != r2["h"]

    def test_params_kwarg_limits_which_params_sampled(self):
        """Test that params kwarg restricts which keys are returned."""
        limits = {"r": [1e6, 1e8], "h": [9000.0, 11000.0], "f": [0.004, 0.006]}
        result = self.init_params_from_bounds(limits, params=["r"])
        assert set(result.keys()) == {"r"}

    def test_ref_values_respected_at_pf_zero(self):
        """Test that ref_values is used as the return value when pf=0."""
        limits = {"r": [1e6, 1e8], "h": [9000.0, 11000.0]}
        ref = {"r": 6371000.0, "h": 10000.0}
        result = self.init_params_from_bounds(
            limits, perturbation_factor=0.0, ref_values=ref, seed=0
        )
        assert result["r"] == 6371000.0
        assert result["h"] == 10000.0

    def test_r_draw_same_regardless_of_h_inclusion(self):
        """Test that r's draw is the same whether or not h is included."""
        limits = {"r": [1e6, 1e8], "h": [9000.0, 11000.0]}
        r_only = self.init_params_from_bounds(limits, seed=7, params=["r"])
        r_and_h = self.init_params_from_bounds(limits, seed=7, params=["r", "h"])
        assert r_only["r"] == r_and_h["r"]


class TestErrorHandling:
    """Test error handling and edge cases for file operations."""

    def test_extract_exif_with_io_error(self):
        """Test extract_exif error handling with non-existent file."""
        result = camera_module.extract_exif("nonexistent_file.jpg")
        assert result == {}


class TestCameraModelDetection:
    """Test camera model detection with various EXIF data configurations."""

    def test_get_camera_model_with_unknown_make_only(self):
        """Test get_camera_model with only Make field that's not in database."""
        exif = {"Make": "UnknownMake"}  # No Model field, Make not in DB
        result = get_camera_model(exif)
        assert result is None  # Should return None for unknown make

    def test_get_camera_model_with_lens_model_fallback(self):
        """Test get_camera_model using LensModel as fallback."""
        exif = {
            "Make": "Unknown",
            "Model": "Unknown Camera",
            "LensModel": "iPhone 13",  # Fallback to LensModel
        }
        result = get_camera_model(exif)
        assert result == "iPhone 13"

    def test_get_camera_model_with_partial_match_known_model_in_input(self):
        """Test partial matching where known model is found within input text."""
        exif = {
            "Make": "Canon",
            "Model": "Canon EOS 5D Mark III with extra text",  # Contains known model
        }
        result = get_camera_model(exif)
        # Should match the known model "Canon EOS 5D Mark III"
        assert result == "Canon EOS 5D Mark III"

    def test_infer_camera_type_mirrorless(self):
        """Test mirrorless camera type detection."""
        exif = {"Make": "Sony", "Model": "ILCE-7M3 Alpha"}
        camera_type = infer_camera_type(exif)
        assert camera_type == "mirrorless"


class TestParameterExtractionEdgeCases:
    """Test parameter extraction with missing or incomplete EXIF data."""

    def test_get_focal_length_mm_returns_none(self):
        """Test get_focal_length_mm when no focal length data."""
        exif = {"Make": "Test", "Model": "Test"}  # No FocalLength field
        result = get_focal_length_mm(exif)
        assert result is None

    def test_extract_camera_parameters_no_exif(self):
        """Test extract_camera_parameters with no EXIF data."""
        with patch.object(camera_module, "get_image_dimensions") as mock_dims:
            with patch.object(camera_module, "extract_exif") as mock_exif:
                mock_dims.return_value = (1920, 1080)
                mock_exif.return_value = {}  # No EXIF data

                params = extract_camera_parameters("dummy.jpg")

                assert params["image_width_px"] == 1920
                assert params["image_height_px"] == 1080
                assert params["camera_model"] is None
                assert params["confidence"] == "low"

    @patch.object(camera_module, "get_image_dimensions")
    @patch.object(camera_module, "extract_exif")
    def test_extract_camera_parameters_default_fallback(self, mock_exif, mock_dims):
        """Test fallback to default camera parameters."""
        mock_dims.return_value = (2000, 1500)
        mock_exif.return_value = {
            "Make": "VeryUnknown",
            "Model": "VeryUnknownCamera",
            # No focal length data, won't match any known patterns
        }

        params = extract_camera_parameters("dummy.jpg")

        assert params["confidence"] == "low"
        assert params["camera_type"] == "default"
        assert params["sensor_width_mm"] == 7.6  # Default sensor
        assert params["focal_length_mm"] == 6.0  # Default focal length


class TestConfigGeneration:
    """Test configuration generation with various edge cases."""

    def test_create_config_geometry_import_error(self):
        """Test create_config_from_image when geometry import fails."""
        with patch.object(camera_module, "extract_camera_parameters") as mock_params:
            with patch.object(camera_module, "get_gps_altitude") as mock_gps:
                mock_params.return_value = {
                    "focal_length_mm": 6.1,
                    "sensor_width_mm": 7.6,
                    "sensor_height_mm": 5.7,
                    "sensor_width_min": None,
                    "sensor_width_max": None,
                    "camera_model": "Test Camera",
                    "camera_type": "compact",
                    "confidence": "high",
                    "image_width_px": 4000,
                    "image_height_px": 3000,
                }
                mock_gps.return_value = 10000.0

                # Mock the geometry import to fail
                import sys

                original_modules = sys.modules.copy()
                if "planet_ruler.geometry" in sys.modules:
                    del sys.modules["planet_ruler.geometry"]

                try:
                    config = create_config_from_image(
                        "dummy.jpg", altitude_m=1000
                    )
                    from numpy import isclose

                    assert isclose(
                        config["init_parameter_values"]["theta_x"],
                        0.017716698486771397,
                        rtol=1e-5,
                        atol=1e-5,
                    )
                    assert config["init_parameter_values"]["theta_y"] == 0.0
                    assert config["init_parameter_values"]["theta_z"] == 0.0
                finally:
                    # Restore modules
                    sys.modules.update(original_modules)


# ============================================================================
# NEW TESTS FOR IMPROVED COVERAGE
# ============================================================================


class TestExifExtractionCoverage:
    """Tests to cover the EXIF extraction loop (lines 132-140)."""

    @patch.object(camera_module, "Image")
    def test_extract_exif_with_valid_data(self, mock_image):
        """Test extract_exif when EXIF data is properly extracted."""
        mock_img = MagicMock()
        mock_exif = MagicMock()
        mock_exif.__bool__ = Mock(return_value=True)
        mock_exif.items.return_value = {
            271: "Apple",
            272: "iPhone 13",
            37386: (4.2, 1),
        }.items()
        mock_exif.get_ifd.return_value = {}
        mock_img.getexif.return_value = mock_exif

        mock_image.open.return_value = mock_img

        result = camera_module.extract_exif("test.jpg")

        assert "Make" in result
        assert "Model" in result
        assert "FocalLength" in result
        assert result["Make"] == "Apple"
        assert result["Model"] == "iPhone 13"

    @patch.object(camera_module, "Image")
    def test_extract_exif_with_empty_exif(self, mock_image):
        """Test extract_exif when image has no EXIF data."""
        mock_img = MagicMock()
        mock_exif = MagicMock()
        mock_exif.__bool__ = Mock(return_value=False)
        mock_img.getexif.return_value = mock_exif

        mock_image.open.return_value = mock_img

        result = camera_module.extract_exif("test.jpg")

        assert result == {}

    @patch.object(camera_module, "Image")
    def test_extract_exif_with_no_gps(self, mock_image):
        """Test extract_exif when EXIF data exists but has no GPS sub-IFD."""
        mock_img = MagicMock()
        mock_exif = MagicMock()
        mock_exif.__bool__ = Mock(return_value=True)
        mock_exif.items.return_value = {271: "Apple"}.items()
        mock_exif.get_ifd.return_value = {}
        mock_img.getexif.return_value = mock_exif

        mock_image.open.return_value = mock_img

        result = camera_module.extract_exif("test.jpg")

        assert "Make" in result
        assert "GPSInfo" not in result


class TestFocalLength35mmCoverage:
    """Test coverage for 35mm equivalent focal length (line 179)."""

    def test_get_focal_length_35mm_equiv_present(self):
        """Test when FocalLengthIn35mmFilm is in EXIF data."""
        exif_data = {"FocalLengthIn35mmFilm": 28}

        result = get_focal_length_35mm_equiv(exif_data)

        assert result == 28.0
        assert isinstance(result, float)

    def test_get_focal_length_35mm_equiv_missing(self):
        """Test when FocalLengthIn35mmFilm is not in EXIF data."""
        exif_data = {"FocalLength": (6.1, 1)}  # Different field

        result = get_focal_length_35mm_equiv(exif_data)

        assert result is None


class TestThetaInitializationCoverage:
    """Test successful theta initialization with geometry import (lines 602-613)."""

    @patch.object(camera_module, "extract_camera_parameters")
    @patch.object(camera_module, "get_gps_altitude")
    @patch.object(camera_module, "get_initial_radius")
    def test_create_config_with_successful_geometry_import(
        self, mock_radius, mock_gps, mock_params
    ):
        """Test theta initialization when geometry import succeeds."""
        mock_params.return_value = {
            "focal_length_mm": 6.1,
            "sensor_width_mm": 7.6,
            "sensor_height_mm": 5.7,
            "sensor_width_min": None,
            "sensor_width_max": None,
            "camera_model": "Canon PowerShot G12",
            "camera_type": "compact",
            "confidence": "high",
            "image_width_px": 4000,
            "image_height_px": 3000,
        }
        mock_gps.return_value = 10000.0
        mock_radius.return_value = 6371000  # Earth radius

        # Mock the geometry module to exist and work
        mock_geometry = MagicMock()
        mock_geometry.limb_camera_angle = Mock(return_value=0.0157)  # ~0.9 degrees

        with patch.dict("sys.modules", {"planet_ruler.geometry": mock_geometry}):
            config = create_config_from_image("dummy.jpg", altitude_m=10000)

            # Verify theta values were set from geometry calculation
            assert "theta_x" in config["init_parameter_values"]
            assert "theta_y" in config["init_parameter_values"]
            assert "theta_z" in config["init_parameter_values"]
            # theta_x should be from limb_camera_angle
            assert config["init_parameter_values"]["theta_x"] == 0.0157
            # theta_y and theta_z should default to 0
            assert config["init_parameter_values"]["theta_y"] == 0.0
            assert config["init_parameter_values"]["theta_z"] == 0.0

    @patch.object(camera_module, "extract_camera_parameters")
    @patch.object(camera_module, "get_gps_altitude")
    @patch.object(camera_module, "get_initial_radius")
    def test_create_config_with_geometry_calculation_error(
        self, mock_radius, mock_gps, mock_params
    ):
        """Test theta initialization when geometry calculation raises error."""
        mock_params.return_value = {
            "focal_length_mm": 6.1,
            "sensor_width_mm": 7.6,
            "sensor_height_mm": 5.7,
            "sensor_width_min": None,
            "sensor_width_max": None,
            "camera_model": "Canon PowerShot G12",
            "camera_type": "compact",
            "confidence": "high",
            "image_width_px": 4000,
            "image_height_px": 3000,
        }
        mock_gps.return_value = 10000.0
        mock_radius.return_value = 6371000

        # Mock geometry to raise an exception during calculation
        mock_geometry = MagicMock()
        mock_geometry.limb_camera_angle = Mock(side_effect=ValueError("Invalid radius"))

        with patch.dict("sys.modules", {"planet_ruler.geometry": mock_geometry}):
            config = create_config_from_image("dummy.jpg", altitude_m=10000)

            # Should fall back to default theta values
            assert config["init_parameter_values"]["theta_x"] == 0.0
            assert config["init_parameter_values"]["theta_y"] == 0.0
            assert config["init_parameter_values"]["theta_z"] == 0.0


class TestAdditionalEdgeCases:
    """Additional edge cases to ensure robust coverage."""

    @patch.object(camera_module, "extract_exif")
    @patch.object(camera_module, "get_image_dimensions")
    def test_extract_params_with_35mm_equiv_calculation(self, mock_dims, mock_exif):
        """Test sensor dimension calculation from 35mm equivalent."""
        mock_dims.return_value = (4000, 3000)
        mock_exif.return_value = {
            "Make": "Test",
            "Model": "Test Camera",
            "FocalLength": (5.0, 1),
            "FocalLengthIn35mmFilm": 26,  # This triggers calculation path
        }

        params = extract_camera_parameters("test.jpg")

        # Should use calculated sensor dimensions
        assert params["confidence"] == "medium"
        assert params["camera_type"] == "calculated"
        assert params["sensor_width_mm"] is not None
        assert params["sensor_height_mm"] is not None

    def test_get_focal_length_mm_with_tuple(self):
        """Test focal length extraction when it's a tuple."""
        exif_data = {"FocalLength": (61, 10)}  # Tuple format: numerator/denominator

        result = get_focal_length_mm(exif_data)

        assert result == 6.1
        assert isinstance(result, float)

    def test_get_focal_length_mm_with_float(self):
        """Test focal length extraction when it's already a float."""
        exif_data = {"FocalLength": 6.1}

        result = get_focal_length_mm(exif_data)

        assert result == 6.1


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
