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
            assert "sensor_width" in specs
            assert "sensor_height" in specs
            assert "type" in specs
            assert specs["type"] in [
                "phone",
                "compact",
                "dslr",
                "mirrorless",
                "unknown",
            ]

    def test_database_has_known_cameras(self):
        """Test that database includes some known cameras."""
        assert "iPhone 13" in CAMERA_DB
        assert "Canon PowerShot G12" in CAMERA_DB
        assert "default" in CAMERA_DB

    def test_sensor_dimensions_positive(self):
        """Test that all sensor dimensions are positive."""
        for model, specs in CAMERA_DB.items():
            assert specs["sensor_width"] > 0
            assert specs["sensor_height"] > 0


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
        exif = {"Make": "Apple", "Model": "iPhone 13 Pro Max"}
        model = get_camera_model(exif)
        assert model == "iPhone 13 Pro"  # Should match iPhone 13 Pro in DB

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
    """Test initial radius generation with perturbation."""

    def test_earth_radius_perturbed(self):
        """Test Earth radius is perturbed within expected range."""
        r_init = get_initial_radius("earth", perturbation_factor=0.5)
        true_radius = PLANET_RADII["earth"]
        # Should be within ±50%
        assert 0.5 * true_radius <= r_init <= 1.5 * true_radius

    def test_mars_radius_perturbed(self):
        """Test Mars radius is perturbed."""
        r_init = get_initial_radius("mars", perturbation_factor=0.5)
        true_radius = PLANET_RADII["mars"]
        assert 0.5 * true_radius <= r_init <= 1.5 * true_radius

    def test_unknown_planet_returns_default(self):
        """Test unknown planet returns middle-range value."""
        r_init = get_initial_radius("unknown_planet")
        assert r_init == 10_000_000  # 10,000 km

    def test_custom_perturbation_factor(self):
        """Test custom perturbation factor."""
        r_init = get_initial_radius("earth", perturbation_factor=0.2)
        true_radius = PLANET_RADII["earth"]
        # Should be within ±20%
        assert 0.8 * true_radius <= r_init <= 1.2 * true_radius

    def test_multiple_calls_give_different_results(self):
        """Test that multiple calls give different perturbed values."""
        results = [get_initial_radius("earth") for _ in range(10)]
        # At least some should be different (very unlikely all are the same)
        assert len(set(results)) > 1


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
        mock_radius.return_value = 8892000  # Perturbed Earth radius

        config = create_config_from_image("dummy.jpg", planet="earth")

        assert config["init_parameter_values"]["r"] == 8892000
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
        mock_radius.return_value = 5084250  # Perturbed Mars radius

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

        config = create_config_from_image(
            "dummy.jpg", altitude_m=10000, param_tolerance=0.1
        )

        # Check r limits are wide
        assert config["parameter_limits"]["r"][0] == 1_000_000
        assert config["parameter_limits"]["r"][1] == 100_000_000

        # Check focal length has ±10% limits
        f_init = config["init_parameter_values"]["f"]
        assert abs(config["parameter_limits"]["f"][0] - f_init * 0.9) < 1e-6
        assert abs(config["parameter_limits"]["f"][1] - f_init * 1.1) < 1e-6

        # Check altitude has ±10% limits
        h_init = config["init_parameter_values"]["h"]
        assert abs(config["parameter_limits"]["h"][0] - h_init * 0.9) < 1
        assert abs(config["parameter_limits"]["h"][1] - h_init * 1.1) < 1


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
                        "dummy.jpg", altitude_m=1000, perturbation_factor=0.0
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


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
