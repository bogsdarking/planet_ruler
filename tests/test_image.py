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
Comprehensive tests for planet_ruler.image module.

Tests cover:
- load_image: Image loading with different formats and channels
- gradient_break: Horizon detection via gradient analysis
- ImageSegmentation: Segmentation-based limb detection with mocking
- smooth_limb: Limb smoothing with various methods
- fill_nans: NaN interpolation for limb data

Uses proper mocking for external dependencies (PIL, kagglehub, segment_anything).
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from PIL import Image

from planet_ruler.image import (
    load_image,
    gradient_break,
    ImageSegmentation,
    smooth_limb,
    fill_nans,
    directional_gradient_blur,
    bidirectional_gradient_blur,
    bilinear_interpolate,
    gradient_field,
)


class TestLoadImage:
    """Test image loading functionality."""

    @patch("planet_ruler.image.Image.open")
    def test_load_rgb_image(self, mock_open):
        """Test loading a standard RGB image."""
        # Mock RGB image (100x100x3)
        mock_img = Mock()
        mock_img.load.return_value = None
        mock_open.return_value = mock_img

        rgb_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mock_img.__array__ = lambda: rgb_array

        with patch("numpy.array") as mock_array:
            mock_array.return_value = rgb_array
            result = load_image("test.jpg")

        mock_open.assert_called_once_with("test.jpg")
        assert result.shape == (100, 100, 3)

    @patch("planet_ruler.image.Image.open")
    def test_load_grayscale_image(self, mock_open):
        """Test loading grayscale image gets converted to 3-channel."""
        mock_img = Mock()
        mock_open.return_value = mock_img

        # Mock grayscale image (100x100)
        gray_array = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        rgb_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        with patch("numpy.array") as mock_array:
            mock_array.return_value = gray_array
            with patch("numpy.dstack") as mock_dstack:
                mock_dstack.return_value = rgb_array

                result = load_image("test_gray.jpg")

                mock_dstack.assert_called_once()
                # Check that dstack was called with 3 copies of the grayscale image
                call_args = mock_dstack.call_args[0][0]
                assert len(call_args) == 3

    @patch("planet_ruler.image.Image.open")
    def test_load_rgba_image(self, mock_open):
        """Test loading RGBA image gets alpha channel removed."""
        mock_img = Mock()
        mock_open.return_value = mock_img

        # Mock RGBA image (100x100x4)
        rgba_array = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)

        with patch("numpy.array") as mock_array:
            mock_array.return_value = rgba_array
            result = load_image("test_rgba.png")

        # Should have removed alpha channel
        expected_result = rgba_array[:, :, :3]
        np.testing.assert_array_equal(result, expected_result)

    @patch("planet_ruler.image.Image.open")
    def test_load_image_file_not_found(self, mock_open):
        """Test proper error handling for missing files."""
        mock_open.side_effect = FileNotFoundError("No such file")

        with pytest.raises(FileNotFoundError):
            load_image("nonexistent.jpg")


class TestGradientBreak:
    """Test horizon detection via gradient analysis."""

    def test_gradient_break_basic(self):
        """Test basic gradient break detection."""
        # Create synthetic image with clear horizontal edge
        image = np.zeros((100, 50, 3), dtype=np.uint8)
        image[60:, :, :] = 255  # White bottom, black top

        breaks = gradient_break(image, window_length=11, polyorder=1)

        assert len(breaks) == 50  # One break per column
        assert all(50 <= b <= 70 for b in breaks)  # Should detect edge around row 60

    def test_gradient_break_auto_window(self):
        """Test basic gradient break detection."""
        # Create synthetic image with clear horizontal edge
        image = np.zeros((100, 50, 3), dtype=np.uint8)
        image[60:, :, :] = 255  # White bottom, black top

        breaks = gradient_break(image, window_length=None, polyorder=1)

        assert len(breaks) == 50  # One break per column
        assert np.median(breaks) == 54  # Should detect edge at row 54

    def test_gradient_break_with_log(self):
        """Test gradient break detection with log transformation."""
        image = np.zeros((100, 30, 3), dtype=np.uint8)
        image[50:, :, :] = 200

        breaks = gradient_break(image, log=True, window_length=11, polyorder=1)

        assert len(breaks) == 30
        assert all(40 <= b <= 60 for b in breaks)

    def test_gradient_break_y_limits(self):
        """Test gradient break with y-range limits."""
        image = np.zeros((100, 20, 3), dtype=np.uint8)
        image[30:, :, :] = 255

        # Only search in range [20, 80]
        breaks = gradient_break(image, y_min=20, y_max=80, window_length=11)

        assert len(breaks) == 20
        assert all(20 <= b <= 80 for b in breaks)

    def test_gradient_break_parameters(self):
        """Test different smoothing parameters."""
        image = np.random.randint(0, 255, (80, 25, 3), dtype=np.uint8)

        # Test different window lengths
        breaks1 = gradient_break(image, window_length=21, polyorder=1)
        breaks2 = gradient_break(image, window_length=41, polyorder=1)

        assert len(breaks1) == len(breaks2) == 25

        # Test different polynomial orders
        breaks3 = gradient_break(image, window_length=21, polyorder=2)
        assert len(breaks3) == 25

    def test_gradient_break_noisy_image(self):
        """Test gradient break on noisy synthetic image."""
        np.random.seed(42)
        image = np.random.randint(0, 100, (60, 15, 3), dtype=np.uint8)
        # Add clear edge
        image[35:, :, :] += 150
        image = np.clip(image, 0, 255)

        breaks = gradient_break(image, window_length=11, polyorder=1)

        assert len(breaks) == 15
        # Should still detect the edge despite noise
        assert np.mean(breaks) > 25  # Edge should be detected after row 25


class TestImageSegmentation:
    """Test segmentation-based limb detection."""

    @patch("planet_ruler.image.kagglehub.model_download")
    def test_init_segment_anything(self, mock_download):
        """Test ImageSegmentation initialization with segment-anything."""
        mock_download.return_value = "/fake/path"
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        seg = ImageSegmentation(image, segmenter="segment-anything")

        assert seg.image.shape == (100, 100, 3)
        assert seg.segmenter == "segment-anything"
        assert seg.model_path == "/fake/path"
        mock_download.assert_called_once()

    def test_init_invalid_segmenter(self):
        """Test initialization with invalid segmenter raises error."""
        image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="segmenter must be one of"):
            ImageSegmentation(image, segmenter="invalid")

    def test_limb_from_mask_basic(self):
        """Test limb extraction from segmentation mask."""
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        seg = ImageSegmentation.__new__(ImageSegmentation)  # Skip __init__
        seg.image = image

        # Create mask with clear limb at y=25
        mask = np.zeros((50, 50), dtype=bool)
        mask[25:, :] = True

        limb = seg.limb_from_mask(mask)

        assert len(limb) == 50
        assert all(l == 25.0 for l in limb)  # Should detect limb at y=25

    def test_limb_from_mask_with_gaps(self):
        """Test limb extraction handles gaps in mask."""
        image = np.zeros((40, 30, 3), dtype=np.uint8)
        seg = ImageSegmentation.__new__(ImageSegmentation)
        seg.image = image

        # Create mask with gap in middle columns
        mask = np.zeros((40, 30), dtype=bool)
        mask[20:, :15] = True  # Left side
        mask[20:, 25:] = True  # Right side
        # Columns 15-24 have no mask (gap)

        limb = seg.limb_from_mask(mask)

        assert len(limb) == 30
        # Gap should be interpolated
        assert not np.any(np.isnan(limb))

    def test_limb_from_mask_outlier_removal(self):
        """Test limb extraction removes outliers."""
        image = np.zeros((60, 20, 3), dtype=np.uint8)
        seg = ImageSegmentation.__new__(ImageSegmentation)
        seg.image = image

        # Create mask with one outlier column
        mask = np.zeros((60, 20), dtype=bool)
        mask[30:, :] = True
        mask[5:, 10] = True  # Outlier at column 10 starting much higher

        limb = seg.limb_from_mask(mask)

        assert len(limb) == 20
        # Outlier should be corrected via interpolation
        assert abs(limb[10] - 30.0) < 5.0  # Should be close to other values

    @patch("planet_ruler.image.sam_model_registry")
    @patch("planet_ruler.image.SamAutomaticMaskGenerator")
    @patch("planet_ruler.image.kagglehub.model_download")
    def test_segment_full_pipeline(
        self, mock_download, mock_generator_class, mock_registry
    ):
        """Test full segmentation pipeline with mocked SAM model."""
        mock_download.return_value = "/fake/path"

        # Mock SAM model
        mock_model = Mock()
        mock_registry.return_value = mock_model

        # Mock mask generator
        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator

        # Mock generated masks - simulate planet detection with valid limb
        fake_masks = [
            {"segmentation": np.ones((50, 50), dtype=bool)},  # Sky
            {"segmentation": np.zeros((50, 50), dtype=bool)},  # Planet
        ]
        # Create a proper limb by setting planet area in bottom half
        fake_masks[1]["segmentation"][30:, :] = True  # Planet in bottom half
        # Combine to create limb mask (sky AND NOT planet)
        combined_mask = fake_masks[0]["segmentation"] & ~fake_masks[1]["segmentation"]
        mock_generator.generate.return_value = fake_masks

        image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        seg = ImageSegmentation(image, segmenter="segment-anything")

        # Mock the limb_from_mask to return a predictable result
        with patch.object(seg, "limb_from_mask") as mock_limb:
            expected_limb = np.full(50, 30.0)  # Limb at y=30 for all columns
            mock_limb.return_value = expected_limb

            result = seg.segment()

            assert len(result) == 50
            np.testing.assert_array_equal(result, expected_limb)

        mock_generator.generate.assert_called_once_with(image)


class TestSmoothLimb:
    """Test limb position smoothing."""

    def test_smooth_limb_rolling_median(self):
        """Test rolling median smoothing."""
        # Create noisy limb data
        np.random.seed(42)
        limb = (
            30 + np.sin(np.linspace(0, 4 * np.pi, 100)) + np.random.normal(0, 0.5, 100)
        )

        smoothed = smooth_limb(limb, method="rolling-median", window_length=11)

        assert len(smoothed) == len(limb)
        # Smoothed version should have less variance
        assert np.var(smoothed) < np.var(limb)

    def test_smooth_limb_rolling_mean(self):
        """Test rolling mean smoothing."""
        limb = np.array([1, 2, 8, 2, 3, 2, 9, 2, 3, 2])  # With outliers

        smoothed = smooth_limb(limb, method="rolling-mean", window_length=3)

        assert len(smoothed) == len(limb)
        # Should reduce impact of outliers
        assert max(smoothed[2:]) < max(limb)  # Smoothed outliers

    def test_smooth_limb_savgol(self):
        """Test Savitzky-Golay smoothing."""
        # Polynomial signal with noise
        x = np.linspace(0, 10, 50)
        limb = 2 * x**2 - 3 * x + 1 + np.random.normal(0, 0.1, 50)

        smoothed = smooth_limb(limb, method="savgol", window_length=11, polyorder=2)

        assert len(smoothed) == len(limb)
        # Should preserve polynomial trend better than original noisy data
        assert np.var(np.diff(smoothed, 2)) < np.var(np.diff(limb, 2))

    def test_smooth_limb_bin_interpolate(self):
        """Test bin-interpolate smoothing."""
        limb = np.sin(np.linspace(0, 2 * np.pi, 60)) + np.random.normal(0, 0.1, 60)

        # Test linear interpolation
        smoothed = smooth_limb(
            limb, method="bin-interpolate", window_length=10, polyorder=1
        )

        assert len(smoothed) == len(limb)

        # Test quadratic interpolation
        smoothed_quad = smooth_limb(
            limb, method="bin-interpolate", window_length=10, polyorder=2
        )
        assert len(smoothed_quad) == len(limb)

    def test_smooth_limb_invalid_method(self):
        """Test error handling for invalid smoothing method."""
        limb = np.array([1, 2, 3, 4, 5])

        # The actual function uses AssertionError, not ValueError
        with pytest.raises(AssertionError):
            smooth_limb(limb, method="invalid")

    def test_smooth_limb_with_nans(self):
        """Test smoothing handles and fills NaN values."""
        # Create data where smoothing won't result in all NaNs
        limb = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

        smoothed = smooth_limb(limb, method="rolling-median", window_length=3)

        assert len(smoothed) == len(limb)
        # NaNs should be filled
        assert not np.any(np.isnan(smoothed))
        # Check that non-NaN values are reasonable
        valid_original = limb[~np.isnan(limb)]
        assert np.min(smoothed) >= np.min(valid_original) * 0.5
        assert np.max(smoothed) <= np.max(valid_original) * 1.5

    def test_smooth_limb_edge_cases(self):
        """Test smoothing with edge cases."""
        # Very short array
        short_limb = np.array([1.0, 2.0, 3.0])
        smoothed = smooth_limb(short_limb, method="rolling-mean", window_length=2)
        assert len(smoothed) == len(short_limb)

        # All same values
        flat_limb = np.ones(20)
        smoothed = smooth_limb(flat_limb, method="savgol", window_length=5, polyorder=1)
        np.testing.assert_array_almost_equal(smoothed, flat_limb)


class TestFillNans:
    """Test NaN interpolation for limb data."""

    def test_fill_nans_basic(self):
        """Test basic NaN filling via interpolation."""
        limb = np.array([1.0, 2.0, np.nan, 4.0, 5.0])

        filled = fill_nans(limb)

        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # Linear interpolation
        np.testing.assert_array_equal(filled, expected)

    def test_fill_nans_multiple_gaps(self):
        """Test filling multiple NaN gaps."""
        limb = np.array([1.0, np.nan, np.nan, 4.0, 5.0, np.nan, 7.0])

        filled = fill_nans(limb)

        assert len(filled) == len(limb)
        assert not np.any(np.isnan(filled))
        # Check interpolation makes sense
        assert 1 < filled[1] < 4
        assert 1 < filled[2] < 4
        assert 5 < filled[5] < 7

    def test_fill_nans_no_nans(self):
        """Test function handles data without NaNs."""
        limb = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        filled = fill_nans(limb)

        np.testing.assert_array_equal(filled, limb)

    def test_fill_nans_preserves_original(self):
        """Test original array is not modified."""
        original = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        limb_copy = original.copy()

        filled = fill_nans(limb_copy)

        # Original should be unchanged
        np.testing.assert_array_equal(original, limb_copy)
        # Result should be different
        assert not np.array_equal(filled, original)

    def test_fill_nans_edge_nans(self):
        """Test handling NaNs at array edges."""
        limb = np.array([np.nan, 2.0, 3.0, np.nan])

        filled = fill_nans(limb)

        assert not np.any(np.isnan(filled))
        # Edge NaNs should be extrapolated reasonably
        assert len(filled) == len(limb)

    def test_fill_nans_all_nans(self):
        """Test edge case where all values are NaN."""
        limb = np.array([np.nan, np.nan, np.nan])

        # This should handle gracefully (likely with zeros or error)
        # The exact behavior depends on numpy.interp implementation
        try:
            filled = fill_nans(limb)
            assert len(filled) == len(limb)
        except (ValueError, RuntimeError):
            # Some interpolation methods may fail with all NaNs
            pass


# Integration tests combining multiple functions
class TestImageIntegration:
    """Integration tests for image processing pipeline."""

    @patch("planet_ruler.image.Image.open")
    def test_image_processing_pipeline(self, mock_open):
        """Test complete image processing workflow."""
        # Mock image loading
        mock_img = Mock()
        mock_open.return_value = mock_img

        # Create synthetic planet image
        image_array = np.zeros((100, 80, 3), dtype=np.uint8)
        image_array[:40, :, :] = 50  # Dark space
        image_array[40:, :, :] = 200  # Bright planet

        with patch("numpy.array", return_value=image_array):
            loaded_image = load_image("test_planet.jpg")

        # Detect horizon using gradient break
        breaks = gradient_break(loaded_image, window_length=11, polyorder=1)

        # Add some noise to breaks
        noisy_breaks = breaks + np.random.normal(0, 0.5, len(breaks))

        # Smooth the detected limb
        smoothed_limb = smooth_limb(
            noisy_breaks, method="rolling-median", window_length=5
        )

        # Fill any potential NaNs
        final_limb = fill_nans(smoothed_limb)

        assert len(final_limb) == 80  # Same as image width
        assert not np.any(np.isnan(final_limb))
        assert all(30 <= l <= 50 for l in final_limb)  # Around the transition

    @patch("planet_ruler.image.kagglehub.model_download")
    def test_segmentation_workflow(self, mock_download):
        """Test segmentation-based limb detection workflow."""
        mock_download.return_value = "/fake/path"

        # Create test image
        image = np.random.randint(0, 255, (60, 40, 3), dtype=np.uint8)

        # Test initialization
        seg = ImageSegmentation(image, segmenter="segment-anything")

        # Test limb extraction with synthetic mask
        test_mask = np.zeros((60, 40), dtype=bool)
        test_mask[30:, :] = True

        limb = seg.limb_from_mask(test_mask)
        smoothed = smooth_limb(limb, method="savgol", window_length=7, polyorder=1)
        final = fill_nans(smoothed)

        assert len(final) == 40
        assert not np.any(np.isnan(final))


# Numerical and edge case tests
class TestNumericalStability:
    """Test numerical stability and edge cases."""

    def test_gradient_break_extreme_values(self):
        """Test gradient break with extreme pixel values."""
        # All black image
        black_image = np.zeros((50, 30, 3), dtype=np.uint8)
        breaks = gradient_break(black_image, window_length=11)
        assert len(breaks) == 30

        # All white image
        white_image = np.full((50, 30, 3), 255, dtype=np.uint8)
        breaks = gradient_break(white_image, window_length=11)
        assert len(breaks) == 30

    def test_smooth_limb_numerical_precision(self):
        """Test smoothing maintains reasonable numerical precision."""
        # Very small values
        small_limb = np.array([1e-10, 2e-10, 3e-10, 4e-10, 5e-10])
        smoothed = smooth_limb(small_limb, method="rolling-mean", window_length=3)
        assert all(s >= 0 for s in smoothed if not np.isnan(s))

        # Very large values
        large_limb = np.array([1e6, 2e6, 3e6, 4e6, 5e6])
        smoothed = smooth_limb(
            large_limb, method="savgol", window_length=5, polyorder=1
        )
        assert all(np.isfinite(s) for s in smoothed)


# Performance marker tests
class TestImagePerformance:
    """Test performance characteristics of image functions."""

    @pytest.mark.slow
    def test_gradient_break_large_image(self):
        """Test gradient break performance on larger images."""
        # Create large synthetic image
        large_image = np.random.randint(0, 255, (1000, 500, 3), dtype=np.uint8)

        breaks = gradient_break(large_image, window_length=101)

        assert len(breaks) == 500
        assert all(0 <= b < 1000 for b in breaks)

    @pytest.mark.slow
    def test_smooth_limb_long_array(self):
        """Test smoothing performance on long limb arrays."""
        # Long limb data
        long_limb = np.sin(np.linspace(0, 10 * np.pi, 10000)) + np.random.normal(
            0, 0.1, 10000
        )

        smoothed = smooth_limb(long_limb, method="rolling-median", window_length=101)

        assert len(smoothed) == len(long_limb)
        assert np.var(smoothed) < np.var(long_limb)


# New test classes for previously uncovered functions
class TestDirectionalGradientBlur:
    """Test directional gradient blur functionality."""

    def test_directional_gradient_blur_basic(self):
        """Test basic directional gradient blur."""
        # Create simple gradient image
        image = np.zeros((50, 50, 3), dtype=np.float32)
        image[:, 25:, :] = 255  # Vertical edge

        blurred_mag, grad_angle = directional_gradient_blur(
            image,
            sigma_base=1.0,
            streak_length=5,
            decay_rate=0.1,
            normalize_gradients=True,
        )

        assert blurred_mag.shape == (50, 50)
        assert grad_angle.shape == (50, 50)
        assert np.all(np.isfinite(blurred_mag))
        assert np.all(np.isfinite(grad_angle))

    def test_directional_gradient_blur_grayscale(self):
        """Test directional gradient blur on grayscale image."""
        # Grayscale image
        image = np.zeros((30, 30), dtype=np.float32)
        image[15:, :] = 100  # Horizontal edge

        blurred_mag, grad_angle = directional_gradient_blur(
            image, sigma_base=0.5, streak_length=10, normalize_gradients=True
        )

        assert blurred_mag.shape == (30, 30)
        assert grad_angle.shape == (30, 30)

    def test_directional_gradient_blur_no_normalize(self):
        """Test directional gradient blur without gradient normalization."""
        image = np.random.rand(20, 20, 3) * 255

        blurred_mag, grad_angle = directional_gradient_blur(
            image, sigma_base=2.0, streak_length=8, normalize_gradients=False
        )

        assert blurred_mag.shape == (20, 20)
        assert grad_angle.shape == (20, 20)

    def test_directional_gradient_blur_no_smoothing(self):
        """Test directional gradient blur with no initial smoothing."""
        image = np.ones((25, 25, 3)) * 128
        image[10:15, 10:15, :] = 200  # Square feature

        blurred_mag, grad_angle = directional_gradient_blur(
            image, sigma_base=0.0, streak_length=3, decay_rate=0.3
        )

        assert blurred_mag.shape == (25, 25)
        assert grad_angle.shape == (25, 25)

    def test_directional_gradient_blur_parameters(self):
        """Test directional gradient blur with various parameters."""
        image = np.random.rand(40, 40, 3) * 255

        # Different streak lengths
        for streak_length in [1, 5, 15]:
            blurred_mag, _ = directional_gradient_blur(
                image, streak_length=streak_length
            )
            assert blurred_mag.shape == (40, 40)

        # Different decay rates
        for decay_rate in [0.05, 0.2, 0.5]:
            blurred_mag, _ = directional_gradient_blur(image, decay_rate=decay_rate)
            assert blurred_mag.shape == (40, 40)


class TestBidirectionalGradientBlur:
    """Test bidirectional gradient blur functionality."""

    def test_bidirectional_gradient_blur_basic(self):
        """Test basic bidirectional gradient blur."""
        # Create simple gradient image
        image = np.zeros((40, 40, 3), dtype=np.float32)
        image[:, 20:, :] = 150  # Vertical edge

        blurred_mag, grad_angle = bidirectional_gradient_blur(
            image,
            sigma_base=1.5,
            streak_length=8,
            decay_rate=0.15,
            normalize_gradients=True,
        )

        assert blurred_mag.shape == (40, 40)
        assert grad_angle.shape == (40, 40)
        assert np.all(np.isfinite(blurred_mag))
        assert np.all(np.isfinite(grad_angle))

    def test_bidirectional_gradient_blur_grayscale(self):
        """Test bidirectional gradient blur on grayscale image."""
        image = np.zeros((35, 35), dtype=np.float32)
        image[15:20, :] = 180  # Horizontal stripe

        blurred_mag, grad_angle = bidirectional_gradient_blur(
            image, sigma_base=2.0, streak_length=6, normalize_gradients=True
        )

        assert blurred_mag.shape == (35, 35)
        assert grad_angle.shape == (35, 35)

    def test_bidirectional_gradient_blur_no_normalize(self):
        """Test bidirectional gradient blur without normalization."""
        image = np.random.rand(25, 25, 3) * 200

        blurred_mag, grad_angle = bidirectional_gradient_blur(
            image, sigma_base=1.0, streak_length=12, normalize_gradients=False
        )

        assert blurred_mag.shape == (25, 25)
        assert grad_angle.shape == (25, 25)

    def test_bidirectional_gradient_blur_no_smoothing(self):
        """Test bidirectional gradient blur with no initial smoothing."""
        image = np.ones((30, 30, 3)) * 100
        image[12:18, 12:18, :] = 255  # Bright square

        blurred_mag, grad_angle = bidirectional_gradient_blur(
            image, sigma_base=0.0, streak_length=5, decay_rate=0.25
        )

        assert blurred_mag.shape == (30, 30)
        assert grad_angle.shape == (30, 30)

    def test_bidirectional_gradient_blur_symmetry(self):
        """Test that bidirectional blur preserves edge locations better than unidirectional."""
        # Create image with clear edge
        image = np.zeros((50, 50, 3))
        image[:25, :, :] = 0  # Top half dark
        image[25:, :, :] = 255  # Bottom half bright

        blurred_mag, _ = bidirectional_gradient_blur(
            image, streak_length=10, decay_rate=0.2
        )

        # Peak should still be around row 25
        peak_row = np.argmax(blurred_mag[:, 25])
        assert abs(peak_row - 25) <= 2  # Should be close to original edge


class TestBilinearInterpolate:
    """Test bilinear interpolation functionality."""

    def test_bilinear_interpolate_exact_coordinates(self):
        """Test bilinear interpolation at exact integer coordinates."""
        array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)

        # Test at exact coordinates (non-boundary)
        result = bilinear_interpolate(array, np.array([1.0]), np.array([1.0]))
        np.testing.assert_array_almost_equal(result, [5.0], decimal=2)

        # Test coordinates that don't hit exact boundary (avoid clipping effects)
        result = bilinear_interpolate(array, np.array([0.0, 1.0]), np.array([0.0, 1.0]))
        np.testing.assert_array_almost_equal(result, [1.0, 5.0], decimal=2)

    def test_bilinear_interpolate_fractional_coordinates(self):
        """Test bilinear interpolation at fractional coordinates."""
        array = np.array([[1, 2], [3, 4]], dtype=np.float32)

        # Test at center (0.5, 0.5) - should be average
        result = bilinear_interpolate(array, np.array([0.5]), np.array([0.5]))
        expected = (1 + 2 + 3 + 4) / 4  # 2.5
        np.testing.assert_array_almost_equal(result, [expected])

    def test_bilinear_interpolate_edge_coordinates(self):
        """Test bilinear interpolation at array edges."""
        array = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.float32)

        # Test at edges (accounting for clipping behavior)
        result = bilinear_interpolate(array, np.array([0.0, 1.0]), np.array([0.0, 2.0]))
        # Due to clipping to avoid exact boundaries, expect close but not exact values
        assert len(result) == 2
        assert abs(result[0] - 10.0) < 0.1
        assert abs(result[1] - 60.0) < 0.1

    def test_bilinear_interpolate_out_of_bounds(self):
        """Test bilinear interpolation clips out-of-bounds coordinates."""
        array = np.array([[1, 2], [3, 4]], dtype=np.float32)

        # Test coordinates outside array bounds
        result = bilinear_interpolate(
            array, np.array([-1.0, 10.0]), np.array([-1.0, 10.0])
        )

        # Should clip to bounds and return corner values
        assert len(result) == 2
        assert np.all(np.isfinite(result))

    def test_bilinear_interpolate_large_array(self):
        """Test bilinear interpolation on larger arrays."""
        array = np.random.rand(100, 80).astype(np.float32)

        # Random coordinates within bounds
        y_coords = np.random.uniform(0, 99, 20)
        x_coords = np.random.uniform(0, 79, 20)

        result = bilinear_interpolate(array, y_coords, x_coords)

        assert len(result) == 20
        assert np.all(np.isfinite(result))
        assert np.all(result >= 0)  # Should be within original array range
        assert np.all(result <= 1)

    def test_bilinear_interpolate_single_pixel(self):
        """Test bilinear interpolation on single pixel array."""
        array = np.array([[42.0]], dtype=np.float32)

        result = bilinear_interpolate(array, np.array([0.0, 0.5]), np.array([0.0, 0.5]))
        np.testing.assert_array_almost_equal(result, [42.0, 42.0])


class TestGradientField:
    """Test gradient field computation functionality."""

    def test_gradient_field_basic(self):
        """Test basic gradient field computation."""
        # Create simple test image with clear edge
        image = np.zeros((50, 50, 3), dtype=np.float32)
        image[:, 25:, :] = 255  # Vertical edge at x=25

        field = gradient_field(
            image, kernel_smoothing=1.0, directional_smoothing=10, directional_decay_rate=0.1
        )

        # Check all required keys are present
        required_keys = [
            "grad_mag",
            "grad_angle",
            "grad_sin",
            "grad_cos",
            "grad_x",
            "grad_y",
            "grad_mag_dy",
            "grad_mag_dx",
            "grad_sin_dy",
            "grad_sin_dx",
            "grad_cos_dy",
            "grad_cos_dx",
            "image_height",
            "image_width",
        ]
        for key in required_keys:
            assert key in field

        # Check array shapes
        assert field["grad_mag"].shape == (50, 50)
        assert field["grad_angle"].shape == (50, 50)
        assert field["grad_sin"].shape == (50, 50)
        assert field["grad_cos"].shape == (50, 50)
        assert field["grad_x"].shape == (50, 50)
        assert field["grad_y"].shape == (50, 50)

        # Check derivatives shapes
        assert field["grad_mag_dy"].shape == (50, 50)
        assert field["grad_mag_dx"].shape == (50, 50)
        assert field["grad_sin_dy"].shape == (50, 50)
        assert field["grad_sin_dx"].shape == (50, 50)
        assert field["grad_cos_dy"].shape == (50, 50)
        assert field["grad_cos_dx"].shape == (50, 50)

        # Check dimensions
        assert field["image_height"] == 50
        assert field["image_width"] == 50

        # Check all arrays are finite
        for key in field:
            if isinstance(field[key], np.ndarray):
                assert np.all(np.isfinite(field[key]))

    def test_gradient_field_grayscale(self):
        """Test gradient field computation on grayscale image."""
        # Grayscale image
        image = np.zeros((30, 30), dtype=np.float32)
        image[15:, :] = 200  # Horizontal edge at y=15

        field = gradient_field(
            image, kernel_smoothing=2.0, directional_smoothing=15, directional_decay_rate=0.2
        )

        # Should work with grayscale input
        assert field["grad_mag"].shape == (30, 30)
        assert field["image_height"] == 30
        assert field["image_width"] == 30

        # Gradient magnitude should be non-zero near edge
        assert np.max(field["grad_mag"]) > 0

    def test_gradient_field_no_smoothing(self):
        """Test gradient field with no initial smoothing."""
        image = np.random.rand(40, 40, 3) * 255

        field = gradient_field(
            image, kernel_smoothing=0.0, directional_smoothing=5, directional_decay_rate=0.3
        )

        assert field["grad_mag"].shape == (40, 40)
        assert np.all(np.isfinite(field["grad_mag"]))

    def test_gradient_field_parameters(self):
        """Test gradient field with different parameters."""
        image = np.ones((35, 35, 3)) * 128
        image[15:20, 15:20, :] = 255  # Bright square in center

        # Test different streak lengths
        for streak_length in [1, 10, 25]:
            field = gradient_field(image, directional_smoothing=streak_length)
            assert field["grad_mag"].shape == (35, 35)
            assert np.all(np.isfinite(field["grad_mag"]))

        # Test different decay rates
        for decay_rate in [0.05, 0.2, 0.5]:
            field = gradient_field(image, directional_decay_rate=decay_rate)
            assert field["grad_mag"].shape == (35, 35)
            assert np.all(np.isfinite(field["grad_mag"]))

    def test_gradient_field_mathematical_properties(self):
        """Test mathematical properties of gradient field."""
        # Simple linear gradient
        image = np.zeros((20, 20, 3))
        for i in range(20):
            image[i, :, :] = i * 10  # Linear vertical gradient

        field = gradient_field(image, kernel_smoothing=0.5, directional_smoothing=5)

        # Sin^2 + Cos^2 should be close to 1 (within numerical precision)
        sin_cos_sum = field["grad_sin"] ** 2 + field["grad_cos"] ** 2
        np.testing.assert_array_almost_equal(
            sin_cos_sum, np.ones_like(sin_cos_sum), decimal=5
        )

        # Gradient x and y components should be consistent with magnitude and angle
        expected_grad_x = field["grad_mag"] * field["grad_cos"]
        expected_grad_y = field["grad_mag"] * field["grad_sin"]

        np.testing.assert_array_almost_equal(
            field["grad_x"], expected_grad_x, decimal=5
        )
        np.testing.assert_array_almost_equal(
            field["grad_y"], expected_grad_y, decimal=5
        )

    def test_gradient_field_edge_detection(self):
        """Test gradient field enhances edge detection."""
        # Create image with multiple edges
        image = np.ones((60, 60, 3)) * 50
        image[20:40, :, :] = 200  # Horizontal stripe
        image[:, 30:50, :] = 150  # Vertical stripe (overlapping)

        field = gradient_field(
            image, kernel_smoothing=1.0, directional_smoothing=12, directional_decay_rate=0.15
        )

        # Should have strong gradients at edges
        assert np.max(field["grad_mag"]) > np.mean(field["grad_mag"]) * 3

        # Check that edges are properly detected
        assert field["grad_mag"].shape == (60, 60)
        assert np.all(np.isfinite(field["grad_mag"]))


class TestImageImportHandling:
    """Test import error handling for optional dependencies."""

    @patch("planet_ruler.image.HAS_SEGMENT_ANYTHING", False)
    def test_segmentation_import_error(self):
        """Test ImageSegmentation raises ImportError when dependencies missing."""
        image = np.zeros((50, 50, 3))

        with pytest.raises(
            ImportError, match="segment-anything dependencies not available"
        ):
            ImageSegmentation(image, segmenter="segment-anything")

    @patch("planet_ruler.image.HAS_SEGMENT_ANYTHING", False)
    def test_segmentation_segment_import_error(self):
        """Test segment() method raises ImportError when dependencies missing."""
        image = np.zeros((50, 50, 3))
        seg = ImageSegmentation.__new__(ImageSegmentation)  # Skip __init__
        seg.image = image
        seg.segmenter = "segment-anything"

        with pytest.raises(
            ImportError, match="segment-anything dependencies not available"
        ):
            seg.segment()

    def test_smooth_limb_bin_interpolate_unsupported_polyorder(self):
        """Test bin-interpolate method with unsupported polynomial order."""
        limb = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        with pytest.raises(AttributeError, match="polyorder .* not supported"):
            smooth_limb(limb, method="bin-interpolate", window_length=2, polyorder=5)

    def test_smooth_limb_bin_interpolate_nearest(self):
        """Test bin-interpolate method with nearest neighbor interpolation."""
        limb = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        smoothed = smooth_limb(
            limb, method="bin-interpolate", window_length=2, polyorder=0
        )

        assert len(smoothed) == len(limb)
        assert np.all(np.isfinite(smoothed))


# Additional edge cases and integration tests
class TestImageEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_gradient_break_window_validation(self):
        """Test gradient break window length validation."""
        # Small image that would cause window issues
        small_image = np.ones((10, 5, 3)) * 128
        small_image[5:, :, :] = 200

        # Should handle small images gracefully
        breaks = gradient_break(
            small_image, window_length=15
        )  # Window larger than image

        assert len(breaks) == 5
        assert all(0 <= b < 10 for b in breaks)

    def test_gradient_break_even_window_adjustment(self):
        """Test gradient break adjusts even window lengths."""
        image = np.ones((50, 20, 3)) * 100
        image[25:, :, :] = 200

        # Even window length should be adjusted to odd
        breaks = gradient_break(image, window_length=20)  # Even window

        assert len(breaks) == 20
        assert all(0 <= b < 50 for b in breaks)

    def test_limb_from_mask_empty_columns(self):
        """Test limb extraction handles completely empty columns."""
        seg = ImageSegmentation.__new__(ImageSegmentation)

        # Mask with some completely empty columns
        mask = np.zeros((30, 20), dtype=bool)
        mask[15:, :10] = True  # Only left half has mask
        # Right half (columns 10-19) are completely empty

        limb = seg.limb_from_mask(mask)

        assert len(limb) == 20
        # Should interpolate missing values
        assert not np.any(np.isnan(limb))
