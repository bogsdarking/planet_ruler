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
Tests for planet_ruler.crop module.

Tests the TkImageCropper class and parameter scaling logic without requiring
GUI interaction by directly manipulating internal state.
"""

import pytest
import numpy as np
import math
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, patch
import sys

# Set matplotlib to non-GUI backend FIRST
import matplotlib

matplotlib.use("Agg")

from PIL import Image
from planet_ruler.crop import TkImageCropper


@pytest.fixture
def mock_gui():
    """Mock only GUI dependencies, not entire methods."""
    with patch("planet_ruler.crop.tk.Tk") as mock_root, patch(
        "planet_ruler.crop.tk.Frame"
    ), patch("planet_ruler.crop.tk.Label"), patch("planet_ruler.crop.tk.Button"), patch(
        "planet_ruler.crop.tk.Canvas"
    ), patch(
        "planet_ruler.crop.tk.Scrollbar"
    ), patch(
        "planet_ruler.crop.ImageTk.PhotoImage"
    ):
        # Make root return a mock
        mock_root.return_value = MagicMock()
        yield


@pytest.fixture
def temp_image(tmp_path):
    """Create a temporary test image."""
    image_path = tmp_path / "test_image.jpg"
    # Create a simple test image
    img = Image.new("RGB", (4032, 3024), color="blue")
    img.save(image_path)
    return str(image_path)


@pytest.fixture
def smartphone_params() -> Dict:
    """Typical smartphone camera parameters."""
    return {
        "w": 0.0236,  # 23.6mm detector width (iPhone-like sensor)
        "f": 0.0042,  # 4.2mm focal length
        "n_pix_x": 4032,
        "n_pix_y": 3024,
        "x0": 2016,  # Principal point (centered)
        "y0": 1512,
    }


@pytest.fixture
def params_with_h_detector() -> Dict:
    """Parameters including detector height."""
    return {
        "w": 0.0236,
        "h_detector": 0.0177,  # 17.7mm detector height
        "f": 0.0042,
        "n_pix_x": 4032,
        "n_pix_y": 3024,
        "x0": 2016,
        "y0": 1512,
    }


@pytest.fixture
def cropper_no_gui(temp_image, smartphone_params, mock_gui):
    """Create a TkImageCropper instance with GUI methods mocked."""
    cropper = TkImageCropper(temp_image, smartphone_params, initial_zoom=1.0)
    return cropper


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================


def test_initialization_with_parameters(temp_image, smartphone_params, mock_gui):
    """Test TkImageCropper initializes with provided parameters."""
    cropper = TkImageCropper(temp_image, smartphone_params, initial_zoom=1.0)

    assert cropper.image_path == temp_image
    assert cropper.initial_parameters == smartphone_params
    assert cropper.width == 4032
    assert cropper.height == 3024
    assert cropper.zoom_level == 1.0
    assert cropper.crop_rect is None
    assert cropper.cropped_image is None
    assert cropper.scaled_parameters is None


def test_initialization_without_parameters(temp_image, mock_gui):
    """Test TkImageCropper initializes with empty parameters."""
    cropper = TkImageCropper(temp_image, initial_zoom=1.5)

    assert cropper.initial_parameters == {}
    assert cropper.zoom_level == 1.5


def test_initialization_loads_image(temp_image, mock_gui):
    """Test that image is loaded and dimensions are correct."""
    cropper = TkImageCropper(temp_image, {}, initial_zoom=1.0)

    assert cropper.original_image is not None
    assert cropper.width == 4032
    assert cropper.height == 3024


# ============================================================================
# CORE PARAMETER SCALING TESTS
# ============================================================================


def test_calculate_scaled_parameters_basic(cropper_no_gui):
    """Test basic parameter scaling calculation."""
    cropper = cropper_no_gui

    # Manually set crop region (simulating user selection)
    cropper.crop_rect = (1000, 500, 3000, 2750)

    scaled = cropper.calculate_scaled_parameters()

    # Check dimensions
    assert scaled["n_pix_x"] == 2000  # 3000 - 1000
    assert scaled["n_pix_y"] == 2250  # 2750 - 500

    # Check principal point shift
    assert scaled["x0"] == 1016  # 2016 - 1000
    assert scaled["y0"] == 1012  # 1512 - 500

    # Check detector width scaling
    scale_x = 2000 / 4032
    expected_w = 0.0236 * scale_x
    assert np.isclose(scaled["w"], expected_w)

    # Focal length unchanged
    assert scaled["f"] == 0.0042


def test_calculate_scaled_parameters_no_crop_raises(cropper_no_gui):
    """Test that calculating parameters without crop region raises error."""
    cropper = cropper_no_gui
    cropper.crop_rect = None

    with pytest.raises(ValueError, match="No crop region selected"):
        cropper.calculate_scaled_parameters()


def test_calculate_scaled_parameters_preserves_extra_params(cropper_no_gui):
    """Test that extra parameters are preserved in scaling."""
    cropper = cropper_no_gui
    # Add extra parameter
    cropper.initial_parameters["extra_param"] = "test_value"
    cropper.initial_parameters["another"] = 42

    cropper.crop_rect = (1000, 500, 3000, 2750)
    scaled = cropper.calculate_scaled_parameters()

    # Extra params should be copied
    assert scaled["extra_param"] == "test_value"
    assert scaled["another"] == 42


def test_calculate_scaled_parameters_integer_dimensions(cropper_no_gui):
    """Test that pixel dimensions are integers."""
    cropper = cropper_no_gui
    cropper.crop_rect = (1000, 500, 3000, 2750)

    scaled = cropper.calculate_scaled_parameters()

    assert isinstance(scaled["n_pix_x"], int)
    assert isinstance(scaled["n_pix_y"], int)


def test_calculate_scaled_parameters_does_not_modify_original(cropper_no_gui):
    """Test that scaling doesn't modify original parameters."""
    cropper = cropper_no_gui
    original_w = cropper.initial_parameters["w"]
    original_x0 = cropper.initial_parameters["x0"]

    cropper.crop_rect = (1000, 500, 3000, 2750)
    scaled = cropper.calculate_scaled_parameters()

    # Original should be unchanged
    assert cropper.initial_parameters["w"] == original_w
    assert cropper.initial_parameters["x0"] == original_x0

    # Scaled should be different
    assert scaled["w"] != original_w
    assert scaled["x0"] != original_x0


def test_pixel_size_remains_constant(cropper_no_gui):
    """Test that physical pixel size remains constant after cropping."""
    cropper = cropper_no_gui
    cropper.crop_rect = (1000, 500, 3000, 2750)

    scaled = cropper.calculate_scaled_parameters()

    # Calculate pixel size
    original_pixel_size = cropper.initial_parameters["w"] / cropper.width
    scaled_pixel_size = scaled["w"] / scaled["n_pix_x"]

    assert np.isclose(original_pixel_size, scaled_pixel_size)


def test_focal_length_unchanged(cropper_no_gui):
    """Test that focal length remains constant (lens property)."""
    cropper = cropper_no_gui
    cropper.crop_rect = (1000, 500, 3000, 2750)

    scaled = cropper.calculate_scaled_parameters()

    assert scaled["f"] == cropper.initial_parameters["f"]


def test_detector_height_scaling(temp_image, params_with_h_detector, mock_gui):
    """Test that h_detector scales correctly."""
    cropper = TkImageCropper(temp_image, params_with_h_detector, initial_zoom=1.0)
    cropper.crop_rect = (1000, 500, 3000, 2750)

    scaled = cropper.calculate_scaled_parameters()

    # Check h_detector scaling
    scale_y = 2250 / 3024
    expected_h = 0.0177 * scale_y
    assert "h_detector" in scaled
    assert np.isclose(scaled["h_detector"], expected_h)


def test_no_crop_is_identity(cropper_no_gui):
    """Test that full image crop leaves parameters unchanged."""
    cropper = cropper_no_gui
    # Crop entire image
    cropper.crop_rect = (0, 0, 4032, 3024)

    scaled = cropper.calculate_scaled_parameters()

    # Parameters should be identical
    assert scaled["n_pix_x"] == 4032
    assert scaled["n_pix_y"] == 3024
    assert np.isclose(scaled["w"], 0.0236)
    assert scaled["f"] == 0.0042
    assert np.isclose(scaled["x0"], 2016)
    assert np.isclose(scaled["y0"], 1512)


def test_principal_point_out_of_bounds(cropper_no_gui):
    """Test that out-of-bounds principal points are handled correctly."""
    cropper = cropper_no_gui
    # Crop right side only - principal point will be negative
    cropper.crop_rect = (2500, 0, 4032, 3024)

    scaled = cropper.calculate_scaled_parameters()

    # Principal point should be negative
    expected_x0 = 2016 - 2500  # -484
    assert scaled["x0"] == expected_x0
    assert scaled["x0"] < 0


def test_default_principal_point_when_missing(temp_image, mock_gui):
    """Test default principal point when not provided."""
    params = {
        "w": 0.0236,
        "f": 0.0042,
        "n_pix_x": 4032,
        "n_pix_y": 3024,
        # No x0, y0
    }

    cropper = TkImageCropper(temp_image, params, initial_zoom=1.0)
    cropper.crop_rect = (1000, 500, 3000, 2750)

    scaled = cropper.calculate_scaled_parameters()

    # Should default to center of crop
    assert scaled["x0"] == 1000  # (3000 - 1000) / 2
    assert scaled["y0"] == 1125  # (2750 - 500) / 2


# ============================================================================
# EDGE CASES
# ============================================================================


def test_very_small_crop(cropper_no_gui):
    """Test parameter scaling with very small crop."""
    cropper = cropper_no_gui
    # 100x100 pixel crop
    cropper.crop_rect = (0, 0, 100, 100)

    scaled = cropper.calculate_scaled_parameters()

    assert scaled["n_pix_x"] == 100
    assert scaled["n_pix_y"] == 100

    # Pixel size should still be constant
    original_px = cropper.initial_parameters["w"] / cropper.width
    scaled_px = scaled["w"] / scaled["n_pix_x"]
    assert np.isclose(original_px, scaled_px)


def test_single_pixel_crop(cropper_no_gui):
    """Test extreme case of 1x1 pixel crop."""
    cropper = cropper_no_gui
    cropper.crop_rect = (100, 100, 101, 101)

    scaled = cropper.calculate_scaled_parameters()

    assert scaled["n_pix_x"] == 1
    assert scaled["n_pix_y"] == 1
    assert scaled["w"] > 0


def test_asymmetric_crop(cropper_no_gui):
    """Test asymmetric crop (different x and y scaling)."""
    cropper = cropper_no_gui
    # 50% width, 75% height
    cropper.crop_rect = (0, 0, 2016, 2268)

    scaled = cropper.calculate_scaled_parameters()

    scale_x = 2016 / 4032
    scale_y = 2268 / 3024

    # Different scales
    assert not np.isclose(scale_x, scale_y)

    # Width scaled by x
    expected_w = 0.0236 * scale_x
    assert np.isclose(scaled["w"], expected_w)


def test_extreme_aspect_ratio(cropper_no_gui):
    """Test extreme aspect ratio crop."""
    cropper = cropper_no_gui
    # Very wide, very short
    cropper.crop_rect = (0, 1500, 4032, 1524)

    scaled = cropper.calculate_scaled_parameters()

    assert scaled["n_pix_x"] == 4032
    assert scaled["n_pix_y"] == 24


def test_corner_crop(cropper_no_gui):
    """Test crop in bottom-right corner."""
    cropper = cropper_no_gui
    cropper.crop_rect = (3032, 2024, 4032, 3024)

    scaled = cropper.calculate_scaled_parameters()

    # Principal point should be far out of bounds
    assert scaled["x0"] < 0
    assert scaled["y0"] < 0


# ============================================================================
# GETTER METHODS
# ============================================================================


def test_get_crop_bounds(cropper_no_gui):
    """Test get_crop_bounds method."""
    cropper = cropper_no_gui

    # No crop set
    assert cropper.get_crop_bounds() is None

    # Set crop
    cropper.crop_rect = (100, 200, 300, 400)
    assert cropper.get_crop_bounds() == (100, 200, 300, 400)


def test_get_scaled_parameters(cropper_no_gui):
    """Test get_scaled_parameters method."""
    cropper = cropper_no_gui

    # No parameters yet
    assert cropper.get_scaled_parameters() is None

    # Calculate parameters
    cropper.crop_rect = (1000, 500, 3000, 2750)
    cropper.scaled_parameters = cropper.calculate_scaled_parameters()

    params = cropper.get_scaled_parameters()
    assert params is not None
    assert "w" in params
    assert "f" in params


def test_get_status_text(cropper_no_gui):
    """Test status text generation."""
    cropper = cropper_no_gui

    # No crop
    status = cropper.get_status_text()
    assert "4032×3024px" in status
    assert "No selection" in status
    assert "100%" in status  # Zoom level

    # With crop
    cropper.crop_rect = (100, 200, 300, 400)
    status = cropper.get_status_text()
    assert "200×200px" in status  # crop size
    assert "Crop:" in status
    # Check percentages
    assert "5.0%" in status or "4.96%" in status  # 200/4032
    assert "6.6%" in status or "6.61%" in status  # 200/3024


def test_get_status_text_with_different_zoom(cropper_no_gui):
    """Test status text shows zoom level correctly."""
    cropper = cropper_no_gui

    cropper.zoom_level = 0.5
    status = cropper.get_status_text()
    assert "50%" in status

    cropper.zoom_level = 2.0
    status = cropper.get_status_text()
    assert "200%" in status


# ============================================================================
# ZOOM METHODS
# ============================================================================


def test_set_zoom(cropper_no_gui):
    """Test set_zoom with bounds enforcement."""
    cropper = cropper_no_gui

    # Normal zoom
    cropper.set_zoom(1.5)
    assert cropper.zoom_level == 1.5

    # Min bound (0.05)
    cropper.set_zoom(0.01)
    assert cropper.zoom_level == 0.05

    # Max bound (5.0)
    cropper.set_zoom(10.0)
    assert cropper.zoom_level == 5.0


def test_adjust_zoom(cropper_no_gui):
    """Test adjust_zoom multiplies current zoom."""
    cropper = cropper_no_gui

    # Start at 1.0
    cropper.zoom_level = 1.0

    # Zoom in 2x
    cropper.adjust_zoom(2.0)
    assert cropper.zoom_level == 2.0

    # Zoom out 0.5x
    cropper.adjust_zoom(0.5)
    assert cropper.zoom_level == 1.0


def test_adjust_zoom_respects_bounds(cropper_no_gui):
    """Test adjust_zoom respects min/max bounds."""
    cropper = cropper_no_gui

    # Try to zoom way out
    cropper.zoom_level = 1.0
    cropper.adjust_zoom(0.01)  # Would be 0.01
    assert cropper.zoom_level == 0.05  # Clamped to min

    # Try to zoom way in
    cropper.zoom_level = 1.0
    cropper.adjust_zoom(10.0)  # Would be 10.0
    assert cropper.zoom_level == 5.0  # Clamped to max


def test_zoom_levels_chain(cropper_no_gui):
    """Test multiple zoom adjustments chain correctly."""
    cropper = cropper_no_gui

    cropper.zoom_level = 1.0
    cropper.adjust_zoom(1.1)  # 1.1
    cropper.adjust_zoom(1.1)  # 1.21
    assert np.isclose(cropper.zoom_level, 1.21)


# ============================================================================
# SELECTION METHODS
# ============================================================================


def test_clear_selection(cropper_no_gui, mock_gui):
    """Test clear_selection resets crop state."""
    cropper = cropper_no_gui

    # Set up a crop
    cropper.crop_rect = (100, 200, 300, 400)
    cropper.drag_start = (50, 50)

    # Mock canvas and status_label
    cropper.canvas = MagicMock()
    cropper.status_label = MagicMock()

    # Clear it
    cropper.clear_selection()

    # Verify cleared
    assert cropper.crop_rect is None
    assert cropper.drag_start is None
    cropper.canvas.delete.assert_called_with("crop_rect")


# ============================================================================
# INTEGRATION WITH DIFFERENT PARAMETER SETS
# ============================================================================


def test_minimal_parameters(temp_image, mock_gui):
    """Test with minimal parameter set."""
    minimal_params = {
        "w": 0.0236,
        "f": 0.0042,
        "n_pix_x": 4032,
        "n_pix_y": 3024,
    }

    cropper = TkImageCropper(temp_image, minimal_params, initial_zoom=1.0)
    cropper.crop_rect = (1000, 500, 3000, 2750)

    scaled = cropper.calculate_scaled_parameters()

    # Should have required fields
    assert "w" in scaled
    assert "f" in scaled
    assert "n_pix_x" in scaled
    assert "n_pix_y" in scaled


def test_off_center_principal_point(temp_image, mock_gui):
    """Test with off-center principal point."""
    params = {
        "w": 0.0236,
        "f": 0.0042,
        "n_pix_x": 4032,
        "n_pix_y": 3024,
        "x0": 2500,  # Off-center
        "y0": 1200,
    }

    cropper = TkImageCropper(temp_image, params, initial_zoom=1.0)
    cropper.crop_rect = (1000, 500, 3000, 2000)

    scaled = cropper.calculate_scaled_parameters()

    # Check shift
    assert scaled["x0"] == 1500  # 2500 - 1000
    assert scaled["y0"] == 700  # 1200 - 500


# ============================================================================
# CONSISTENCY TESTS
# ============================================================================


def test_multiple_crops_consistency(cropper_no_gui):
    """Test that multiple crops maintain pixel size consistency."""
    cropper = cropper_no_gui

    original_px = cropper.initial_parameters["w"] / cropper.width

    crop_scenarios = [
        (0, 0, 2000, 2000),
        (500, 500, 3500, 2500),
        (1000, 1000, 3000, 2000),
        (100, 100, 4000, 3000),
        (0, 1000, 4032, 2024),
    ]

    for crop_bounds in crop_scenarios:
        cropper.crop_rect = crop_bounds
        scaled = cropper.calculate_scaled_parameters()

        scaled_px = scaled["w"] / scaled["n_pix_x"]
        assert np.isclose(
            original_px, scaled_px
        ), f"Pixel size inconsistent for crop {crop_bounds}"


def test_sequential_crops(temp_image, smartphone_params, mock_gui):
    """Test that sequential crops compose correctly."""
    # First crop
    cropper1 = TkImageCropper(temp_image, smartphone_params, initial_zoom=1.0)
    cropper1.crop_rect = (1016, 756, 3016, 2268)
    scaled1 = cropper1.calculate_scaled_parameters()

    # Verify first crop is ~50% of original
    ratio1 = scaled1["w"] / smartphone_params["w"]
    assert np.isclose(ratio1, 0.496, rtol=0.01)  # 2000/4032 ≈ 0.496

    # In practice, you'd crop the image and use the cropped version
    # Here we just verify the math would compose correctly
    # Second crop would be 50% of the 2000x1512 image = 1000x756
    second_crop_scale_x = 1000 / 2000  # 0.5

    # Final width would be: original * first_scale * second_scale
    # = 0.0236 * 0.496 * 0.5 ≈ 0.00585
    expected_final_w = smartphone_params["w"] * ratio1 * second_crop_scale_x

    # Which is ~25% of original
    final_ratio = expected_final_w / smartphone_params["w"]
    assert np.isclose(final_ratio, 0.25, rtol=0.02)


# ============================================================================
# REAL-WORLD SCENARIOS
# ============================================================================


def test_airplane_window_scenario(temp_image, mock_gui):
    """Test realistic airplane window photo with bottom crop."""
    params = {
        "w": 0.0236,  # iPhone sensor
        "f": 0.0026,  # Wide lens
        "n_pix_x": 4032,
        "n_pix_y": 3024,
        "x0": 2016,
        "y0": 1512,
    }

    cropper = TkImageCropper(temp_image, params, initial_zoom=1.0)

    # Crop bottom 30% to remove wing
    y_crop = int(3024 * 0.7)
    cropper.crop_rect = (0, 0, 4032, y_crop)

    scaled = cropper.calculate_scaled_parameters()

    # Width unchanged
    assert scaled["n_pix_x"] == 4032
    assert np.isclose(scaled["w"], params["w"])

    # Height reduced to 70%
    assert scaled["n_pix_y"] == y_crop


def test_floating_point_precision(cropper_no_gui):
    """Test numerical precision with non-integer scale factors."""
    cropper = cropper_no_gui
    # Use crop creating non-integer scale factors
    cropper.crop_rect = (333, 277, 3699, 2747)

    scaled = cropper.calculate_scaled_parameters()

    # Pixel size should be exactly equal (within floating point tolerance)
    original_px = cropper.initial_parameters["w"] / cropper.width
    scaled_px = scaled["w"] / scaled["n_pix_x"]

    assert np.isclose(original_px, scaled_px, rtol=1e-10)


# ============================================================================
# IMAGE DIMENSION HANDLING
# ============================================================================


def test_image_dimensions_read_correctly(temp_image, smartphone_params, mock_gui):
    """Test that image dimensions are read correctly from file."""
    cropper = TkImageCropper(temp_image, smartphone_params, initial_zoom=1.0)

    assert cropper.width == 4032
    assert cropper.height == 3024


def test_large_image_handling(tmp_path, mock_gui):
    """Test handling of large (8K) images."""
    # Create large image
    image_path = tmp_path / "large_image.jpg"
    img = Image.new("RGB", (7680, 4320), color="green")
    img.save(image_path)

    params = {
        "w": 0.0436,
        "f": 0.0050,
        "n_pix_x": 7680,
        "n_pix_y": 4320,
        "x0": 3840,
        "y0": 2160,
    }

    cropper = TkImageCropper(str(image_path), params, initial_zoom=1.0)
    cropper.crop_rect = (2000, 1000, 5680, 3320)

    scaled = cropper.calculate_scaled_parameters()

    assert scaled["n_pix_x"] == 3680
    assert scaled["n_pix_y"] == 2320
    assert scaled["w"] > 0
