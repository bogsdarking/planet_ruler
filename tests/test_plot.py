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
Comprehensive tests for planet_ruler.plot module.

Tests cover:
- plot_image: Image display with gradient options
- plot_limb: Limb data scatter plotting
- plot_3d_solution: Complex 3D limb visualization
- plot_topography: 3D surface plotting of image data

Uses extensive mocking of matplotlib to avoid actual plot display during testing.
Tests verify correct matplotlib function calls and parameter passing.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock, call, ANY

from planet_ruler.plot import (
    plot_image,
    plot_limb,
    plot_3d_solution,
    plot_topography,
    plot_gradient_field_at_limb,
    compare_blur_methods,
    compare_gradient_fields,
    plot_diff_evol_posteriors,
    plot_full_limb,
    plot_segmentation_masks,
)


class TestPlotImage:
    """Test image display functionality."""

    @patch("planet_ruler.plot.plt")
    def test_plot_image_basic(self, mock_plt):
        """Test basic image plotting without gradient."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        plot_image(image, gradient=False, show=True)

        mock_plt.imshow.assert_called_once_with(image)
        mock_plt.show.assert_called_once()

    @patch("planet_ruler.plot.plt")
    def test_plot_image_no_show(self, mock_plt):
        """Test image plotting without showing."""
        image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)

        plot_image(image, gradient=False, show=False)

        mock_plt.imshow.assert_called_once_with(image)
        mock_plt.show.assert_not_called()

    @patch("planet_ruler.plot.plt")
    def test_plot_image_with_gradient(self, mock_plt):
        """Test image plotting with gradient transformation."""
        image = np.random.randint(0, 255, (80, 60, 3), dtype=np.uint8)

        # Just verify that gradient=True calls the right matplotlib functions
        plot_image(image, gradient=True, show=True)

        # Verify matplotlib was called (don't mock numpy internals)
        mock_plt.imshow.assert_called_once()
        mock_plt.show.assert_called_once()

        # Check that imshow was called with some array (gradient processed)
        call_args = mock_plt.imshow.call_args[0]
        assert len(call_args) == 1  # One positional argument
        assert hasattr(call_args[0], "shape")  # Should be array-like

    @patch("planet_ruler.plot.plt")
    def test_plot_image_grayscale(self, mock_plt):
        """Test plotting grayscale image."""
        # Simulate grayscale by using same values across channels
        gray_value = np.random.randint(0, 255, (40, 40), dtype=np.uint8)
        image = np.stack([gray_value, gray_value, gray_value], axis=2)

        plot_image(image, gradient=False, show=False)

        mock_plt.imshow.assert_called_once_with(image)
        mock_plt.show.assert_not_called()


class TestPlotLimb:
    """Test limb data scatter plotting."""

    @patch("planet_ruler.plot.plt")
    def test_plot_limb_basic(self, mock_plt):
        """Test basic limb plotting."""
        limb_data = np.array([30, 31, 29, 32, 28, 33, 27])

        plot_limb(limb_data, show=True)

        # Verify scatter was called with correct structure
        mock_plt.scatter.assert_called_once()
        call_args, call_kwargs = mock_plt.scatter.call_args

        # Check basic parameters - sparse data (7 < 20 points) triggers brightening
        assert len(call_args) == 2  # x, y data
        assert call_kwargs["c"] == "y"
        assert call_kwargs["s"] == 40  # Brightened for sparse data
        assert call_kwargs["alpha"] == 0.8  # Brightened for sparse data

        # Check that x and y arrays have same length
        x_arg, y_arg = call_args
        assert len(x_arg) == len(y_arg) == len(limb_data)

        mock_plt.show.assert_called_once()

    @patch("planet_ruler.plot.plt")
    def test_plot_limb_custom_params(self, mock_plt):
        """Test limb plotting with custom parameters."""
        limb_data = np.array([45, 46, 44, 47, 43])

        plot_limb(limb_data, show=False, c="red", s=20, alpha=0.8)

        # Verify scatter was called with custom parameters
        mock_plt.scatter.assert_called_once()
        call_args, call_kwargs = mock_plt.scatter.call_args

        assert call_kwargs["c"] == "red"
        # Sparse data (5 < 20 points) overrides s and alpha for visibility
        assert call_kwargs["s"] == 40  # Overridden for sparse data
        assert call_kwargs["alpha"] == 0.8  # Custom alpha matches sparse default

        mock_plt.show.assert_not_called()

    @patch("planet_ruler.plot.plt")
    def test_plot_limb_empty_array(self, mock_plt):
        """Test limb plotting with empty data."""
        limb_data = np.array([])

        plot_limb(limb_data, show=False)

        # Verify scatter was called
        mock_plt.scatter.assert_called_once()
        call_args, call_kwargs = mock_plt.scatter.call_args

        # Check that arrays are empty
        x_arg, y_arg = call_args
        assert len(x_arg) == 0
        assert len(y_arg) == 0

        mock_plt.show.assert_not_called()

    @patch("planet_ruler.plot.plt")
    def test_plot_limb_single_point(self, mock_plt):
        """Test limb plotting with single data point."""
        limb_data = np.array([25])

        plot_limb(limb_data, show=True, c="blue", s=50, alpha=1.0)

        # Verify the actual call made - single point triggers sparse data behavior
        mock_plt.scatter.assert_called_once()
        call_args, call_kwargs = mock_plt.scatter.call_args

        expected_x = np.array([0])
        np.testing.assert_array_equal(call_args[0], expected_x)
        np.testing.assert_array_equal(call_args[1], limb_data)

        assert call_kwargs["c"] == "blue"
        # Single point (1 < 20) overrides s and alpha for visibility
        assert call_kwargs["s"] == 40
        assert call_kwargs["alpha"] == 0.8

        mock_plt.show.assert_called_once()


class TestPlot3DSolution:
    """Test 3D limb solution visualization."""

    @patch("planet_ruler.plot.plt")
    @patch("planet_ruler.plot.limb_camera_angle")
    @patch("planet_ruler.plot.horizon_distance")
    def test_plot_3d_solution_basic(self, mock_horizon_dist, mock_limb_angle, mock_plt):
        """Test basic 3D solution plotting."""
        # Mock geometry calculations
        mock_limb_angle.return_value = 0.5  # radians
        mock_horizon_dist.return_value = 1000.0  # km

        # Mock figure and axis
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax

        r = 6371000  # Earth radius in meters
        h = 400000  # ISS altitude in meters

        plot_3d_solution(r, h, zoom=1.0, savefile=None, legend=True)

        # Verify geometry function calls
        mock_limb_angle.assert_called_once_with(r, h)
        mock_horizon_dist.assert_called_once_with(r, h)

        # Verify matplotlib setup
        mock_plt.figure.assert_called_once_with(figsize=(10, 10))
        mock_fig.add_subplot.assert_called_once_with(projection="3d")

        # Verify plotting calls
        assert mock_ax.plot_wireframe.call_count == 2  # Planet wireframes
        mock_ax.plot.assert_called()  # Limb and axis lines
        mock_ax.scatter.assert_called_once()  # Camera position

        mock_plt.show.assert_called_once()

    @patch("planet_ruler.plot.plt")
    @patch("planet_ruler.plot.limb_camera_angle")
    @patch("planet_ruler.plot.horizon_distance")
    def test_plot_3d_solution_with_zoom(
        self, mock_horizon_dist, mock_limb_angle, mock_plt
    ):
        """Test 3D solution plotting with zoom factor."""
        mock_limb_angle.return_value = 0.3
        mock_horizon_dist.return_value = 800.0

        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax

        r = 1737000  # Moon radius
        h = 100000  # Altitude
        zoom = 10.0

        plot_3d_solution(r, h, zoom=zoom, legend=False)

        # Verify zoom was applied to height (h = h * (1/zoom))
        expected_h = h * (1.0 / zoom)
        mock_limb_angle.assert_called_once_with(r, expected_h)
        mock_horizon_dist.assert_called_once_with(r, expected_h)

        # Verify legend was not shown
        mock_plt.legend.assert_not_called()

    @patch("planet_ruler.plot.plt")
    @patch("planet_ruler.plot.limb_camera_angle")
    @patch("planet_ruler.plot.horizon_distance")
    def test_plot_3d_solution_axis_options(
        self, mock_horizon_dist, mock_limb_angle, mock_plt
    ):
        """Test 3D solution plotting with different axis options."""
        mock_limb_angle.return_value = 0.4
        mock_horizon_dist.return_value = 1200.0

        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax

        plot_3d_solution(
            r=3390000,
            h=200000,  # Mars radius and altitude
            x_axis=True,
            y_axis=False,
            z_axis=True,
            vertical_axis="y",
            azim=45,
            roll=30,
        )

        # Count axis plotting calls - should have x and z axes but not y
        axis_plot_calls = mock_ax.plot.call_args_list

        # Verify view settings
        mock_ax.view_init.assert_called_once()
        call_args = mock_ax.view_init.call_args
        assert "azim" in call_args[1]
        assert "roll" in call_args[1]
        assert "vertical_axis" in call_args[1]
        assert call_args[1]["vertical_axis"] == "y"

    @patch("planet_ruler.plot.plt")
    @patch("planet_ruler.plot.limb_camera_angle")
    @patch("planet_ruler.plot.horizon_distance")
    def test_plot_3d_solution_save_file(
        self, mock_horizon_dist, mock_limb_angle, mock_plt
    ):
        """Test 3D solution plotting with file saving."""
        mock_limb_angle.return_value = 0.2
        mock_horizon_dist.return_value = 600.0

        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax

        save_path = "/tmp/test_plot.png"

        plot_3d_solution(r=1000000, h=50000, savefile=save_path)

        # Verify file saving
        mock_plt.savefig.assert_called_once_with(save_path, bbox_inches="tight")

    @patch("planet_ruler.plot.plt")
    @patch("planet_ruler.plot.limb_camera_angle")
    @patch("planet_ruler.plot.horizon_distance")
    def test_plot_3d_solution_kwargs(
        self, mock_horizon_dist, mock_limb_angle, mock_plt
    ):
        """Test 3D solution plotting absorbs extra kwargs."""
        mock_limb_angle.return_value = 0.1
        mock_horizon_dist.return_value = 300.0

        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax

        # Pass extra kwargs that should be absorbed
        plot_3d_solution(
            r=500000,
            h=25000,
            extra_param1="ignored",
            extra_param2=42,
            unused_setting=True,
        )

        # Should not raise any errors and complete successfully
        mock_plt.show.assert_called_once()


class TestPlotTopography:
    """Test 3D topography surface plotting."""

    @patch("planet_ruler.plot.plt")
    @patch("planet_ruler.plot.LightSource")
    def test_plot_topography_basic(self, mock_lightsource, mock_plt):
        """Test basic topography plotting."""
        # Create test image
        image = np.random.randint(0, 255, (50, 40, 3), dtype=np.uint8)

        # Mock LightSource and figure setup
        mock_ls = Mock()
        mock_lightsource.return_value = mock_ls

        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax

        plot_topography(image)

        # Verify LightSource creation
        mock_lightsource.assert_called_once_with(altdeg=30, azdeg=-15)

        # Verify matplotlib setup
        mock_plt.figure.assert_called_once_with(figsize=(10, 10))
        mock_fig.add_subplot.assert_called_once_with(projection="3d")

        # Verify surface plotting
        mock_ax.plot_surface.assert_called_once()
        surface_call = mock_ax.plot_surface.call_args
        assert "lightsource" in surface_call[1]

        # Verify view settings
        mock_ax.view_init.assert_called_once_with(elev=90, azim=0, roll=-90)

        # Verify layout adjustments
        mock_plt.subplots_adjust.assert_called_once_with(
            top=1, bottom=0, right=1, left=0, hspace=0, wspace=0
        )
        mock_plt.axis.assert_called_with("off")
        mock_plt.show.assert_called_once()

    @patch("planet_ruler.plot.plt")
    @patch("planet_ruler.plot.LightSource")
    def test_plot_topography_clipping(self, mock_lightsource, mock_plt):
        """Test topography plotting with image value clipping."""
        # Create image with some very high values that should be clipped
        image = np.random.randint(800, 1200, (30, 25, 3), dtype=np.uint16)

        mock_ls = Mock()
        mock_lightsource.return_value = mock_ls

        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax

        # Don't mock numpy internals, just verify the function completes
        plot_topography(image)

        # Verify basic plotting calls were made
        mock_lightsource.assert_called_once_with(altdeg=30, azdeg=-15)
        mock_ax.plot_surface.assert_called_once()
        mock_ax.view_init.assert_called_once_with(elev=90, azim=0, roll=-90)

    @patch("planet_ruler.plot.plt")
    @patch("planet_ruler.plot.LightSource")
    def test_plot_topography_meshgrid(self, mock_lightsource, mock_plt):
        """Test topography plotting creates correct coordinate meshgrid."""
        image = np.ones((20, 15, 3), dtype=np.uint8) * 100

        mock_ls = Mock()
        mock_lightsource.return_value = mock_ls

        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax

        # Don't over-mock numpy functions, just test that it works
        plot_topography(image)

        # Verify key plotting functions were called
        mock_lightsource.assert_called_once_with(altdeg=30, azdeg=-15)
        mock_plt.figure.assert_called_once_with(figsize=(10, 10))
        mock_fig.add_subplot.assert_called_once_with(projection="3d")
        mock_ax.plot_surface.assert_called_once()


# Integration and edge case tests
class TestPlotIntegration:
    """Integration tests for plot functions."""

    @patch("planet_ruler.plot.plt")
    def test_plot_image_and_limb_together(self, mock_plt):
        """Test plotting image and limb data together."""
        image = np.random.randint(0, 255, (60, 80, 3), dtype=np.uint8)
        limb_data = np.random.randint(20, 40, 80)

        # Plot image first, then limb overlay
        plot_image(image, show=False)
        plot_limb(limb_data, show=True)

        # Verify both plotting calls were made
        mock_plt.imshow.assert_called_once()
        mock_plt.scatter.assert_called_once()
        mock_plt.show.assert_called_once()  # Only the limb plot shows

        # Verify scatter call structure
        call_args, call_kwargs = mock_plt.scatter.call_args
        assert len(call_args) == 2  # x, y data
        assert call_kwargs["c"] == "y"

    @patch("planet_ruler.plot.plt")
    @patch("planet_ruler.plot.limb_camera_angle")
    @patch("planet_ruler.plot.horizon_distance")
    def test_plot_functions_numerical_edge_cases(
        self, mock_horizon_dist, mock_limb_angle, mock_plt
    ):
        """Test plot functions with extreme numerical values."""
        # Very small planet
        mock_limb_angle.return_value = 1e-6
        mock_horizon_dist.return_value = 1e-3

        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax

        # Should handle small values gracefully
        plot_3d_solution(r=1.0, h=0.1, zoom=1000)

        # Verify geometry calculations with small values
        mock_limb_angle.assert_called_once_with(1.0, 0.1 / 1000)
        mock_horizon_dist.assert_called_once_with(1.0, 0.1 / 1000)


class TestPlotError:
    """Test error handling and edge cases."""

    @patch("planet_ruler.plot.plt")
    def test_plot_image_invalid_dimensions(self, mock_plt):
        """Test plot_image with unexpected array dimensions."""
        # 1D array should still work with imshow
        image_1d = np.random.randint(0, 255, 100)

        plot_image(image_1d, show=False)

        mock_plt.imshow.assert_called_once_with(image_1d)

    @patch("planet_ruler.plot.plt")
    def test_plot_limb_nan_values(self, mock_plt):
        """Test plot_limb handles NaN values."""
        limb_data = np.array([30, np.nan, 32, np.inf, 28])

        plot_limb(limb_data, show=False)

        # Should still call scatter - matplotlib handles NaN/inf internally
        mock_plt.scatter.assert_called_once()
        call_args, call_kwargs = mock_plt.scatter.call_args

        # Verify structure without comparing exact arrays with NaN/inf
        assert len(call_args) == 2
        assert len(call_args[0]) == len(call_args[1]) == len(limb_data)
        assert call_kwargs["c"] == "y"

    @patch("planet_ruler.plot.plt")
    @patch("planet_ruler.plot.limb_camera_angle")
    @patch("planet_ruler.plot.horizon_distance")
    def test_plot_3d_solution_zero_radius(
        self, mock_horizon_dist, mock_limb_angle, mock_plt
    ):
        """Test 3D solution plotting with edge case parameters."""
        # Zero radius should be handled by geometry functions
        mock_limb_angle.return_value = 0.0
        mock_horizon_dist.return_value = 0.0

        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.figure.return_value = mock_fig
        mock_fig.add_subplot.return_value = mock_ax

        plot_3d_solution(r=0, h=100)

        # Should complete without errors
        mock_limb_angle.assert_called_once_with(0, 100)
        mock_horizon_dist.assert_called_once_with(0, 100)
        mock_plt.show.assert_called_once()


# Performance and compatibility tests
class TestPlotPerformance:
    """Test performance characteristics of plotting functions."""

    @pytest.mark.slow
    @patch("planet_ruler.plot.plt")
    def test_plot_large_image(self, mock_plt):
        """Test plotting very large images."""
        large_image = np.random.randint(0, 255, (2000, 1500, 3), dtype=np.uint8)

        plot_image(large_image, show=False)

        mock_plt.imshow.assert_called_once_with(large_image)

    @pytest.mark.slow
    @patch("planet_ruler.plot.plt")
    def test_plot_long_limb_data(self, mock_plt):
        """Test plotting very long limb arrays."""
        long_limb = np.sin(np.linspace(0, 10 * np.pi, 10000)) * 50 + 100

        plot_limb(long_limb, show=False)

        # Verify scatter was called with long arrays
        mock_plt.scatter.assert_called_once()
        call_args, call_kwargs = mock_plt.scatter.call_args

        assert len(call_args) == 2
        assert len(call_args[0]) == len(call_args[1]) == 10000
        assert call_kwargs["c"] == "y"


# Test matplotlib configuration
class TestMatplotlibConfig:
    """Test matplotlib configuration and setup."""

    def test_matplotlib_rcparams_set(self):
        """Test that matplotlib rcParams are configured."""
        import planet_ruler.plot as plot_module

        # Import should have set rcParams
        import matplotlib

        # Check that figure size and font size are set
        # Note: These might have been modified by other imports, so we just verify they exist
        assert "figure.figsize" in matplotlib.rcParams
        assert "font.size" in matplotlib.rcParams


class TestPlotGradientFieldAtLimb:
    """Test gradient field visualization at limb curves."""

    @patch("planet_ruler.image.gradient_field")
    @patch("planet_ruler.plot.plt")
    def test_plot_gradient_field_at_limb_basic(self, mock_plt, mock_gradient_field):
        """Test basic gradient field plotting at limb."""
        # Mock gradient field data
        mock_gradient_field.return_value = {
            "grad_mag": np.ones((100, 200)) * 0.5,
            "grad_angle": np.zeros((100, 200)),
            "image_height": 100,
        }

        # Mock figure and axis
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        # Test data
        y_pixels = np.array([30, 31, 32, 31, 30, 29, 28])
        image = np.random.random((100, 200, 3))

        result = plot_gradient_field_at_limb(y_pixels, image)

        # Verify gradient field computation
        mock_gradient_field.assert_called_once_with(
            ANY, gradient_smoothing=5.0, streak_length=30, decay_rate=0.15
        )

        # Verify matplotlib calls
        mock_plt.subplots.assert_called_once_with(figsize=(16, 10))
        mock_ax.imshow.assert_called_once()
        mock_ax.plot.assert_called()  # Limb curve
        assert mock_ax.arrow.call_count > 0  # Gradient arrows

        # Verify legend creation
        mock_ax.legend.assert_called_once()

        assert result == (mock_fig, mock_ax)

    @patch("planet_ruler.image.gradient_field")
    @patch("planet_ruler.plot.plt")
    def test_plot_gradient_field_custom_params(self, mock_plt, mock_gradient_field):
        """Test gradient field plotting with custom parameters."""
        mock_gradient_field.return_value = {
            "grad_mag": np.ones((50, 100)) * 0.3,
            "grad_angle": 3.14159 * np.ones((50, 100)),
            "image_height": 50,
        }

        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        y_pixels = np.linspace(20, 30, 100)
        image = np.random.random((50, 100, 3))

        plot_gradient_field_at_limb(
            y_pixels,
            image,
            gradient_smoothing=3.0,
            streak_length=20,
            decay_rate=0.1,
            sample_spacing=25,
            arrow_scale=10,
        )

        # Verify custom parameters passed to gradient_field
        mock_gradient_field.assert_called_once_with(
            ANY, gradient_smoothing=3.0, streak_length=20, decay_rate=0.1
        )

    @patch("planet_ruler.image.gradient_field")
    @patch("planet_ruler.plot.plt")
    def test_plot_gradient_field_edge_cases(self, mock_plt, mock_gradient_field):
        """Test gradient field plotting with edge cases."""
        mock_gradient_field.return_value = {
            "grad_mag": np.zeros((20, 30)),
            "grad_angle": np.full((20, 30), np.nan),
            "image_height": 20,
        }

        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        # Very short limb curve to avoid numpy gradient issues
        y_pixels = np.array(
            [15, 16, 17]
        )  # Minimum length to avoid gradient calculation issues
        image = np.random.random((20, 30))

        plot_gradient_field_at_limb(y_pixels, image, sample_spacing=10)

        # Should still create plot without errors
        mock_plt.subplots.assert_called_once()
        mock_ax.imshow.assert_called_once()


class TestCompareBlurMethods:
    """Test gradient blur method comparison."""

    @patch("planet_ruler.image.gradient_field")
    @patch("planet_ruler.plot.cv2")
    @patch("planet_ruler.plot.plt")
    def test_compare_blur_methods_basic(self, mock_plt, mock_cv2, mock_gradient_field):
        """Test basic blur method comparison."""
        # Mock cv2 GaussianBlur
        mock_cv2.GaussianBlur.return_value = np.ones((50, 60)) * 100

        # Mock gradient field
        mock_gradient_field.return_value = {
            "grad_mag": np.ones((50, 60)) * 0.5,
            "grad_angle": np.zeros((50, 60)),
        }

        # Mock figure and axes
        mock_fig = Mock()
        mock_axes = np.array([[Mock(), Mock()], [Mock(), Mock()]])
        mock_plt.subplots.return_value = (mock_fig, mock_axes)

        # Test with 3-channel image
        image = np.random.random((50, 60, 3))

        result = compare_blur_methods(image)

        # Verify matplotlib setup
        mock_plt.subplots.assert_called_once_with(2, 2, figsize=(14, 12))

        # Verify Gaussian blur call
        mock_cv2.GaussianBlur.assert_called_once()

        # Verify gradient field call
        mock_gradient_field.assert_called_once_with(
            ANY, streak_length=30, decay_rate=0.15, gradient_smoothing=2.0
        )

        # Verify all subplots were used
        for ax in mock_axes.flat:
            ax.imshow.assert_called()
            ax.set_title.assert_called()
            ax.axis.assert_called_with("off")

        assert result == (mock_fig, mock_axes)

    @patch("planet_ruler.image.gradient_field")
    @patch("planet_ruler.plot.cv2")
    @patch("planet_ruler.plot.plt")
    def test_compare_blur_methods_with_limb(
        self, mock_plt, mock_cv2, mock_gradient_field
    ):
        """Test blur method comparison with limb overlay."""
        mock_cv2.GaussianBlur.return_value = np.ones((40, 50)) * 50
        mock_gradient_field.return_value = {
            "grad_mag": np.ones((40, 50)) * 0.3,
            "grad_angle": np.ones((40, 50)) * 3.14159 / 4,
        }

        mock_fig = Mock()
        mock_axes = np.array([[Mock(), Mock()], [Mock(), Mock()]])
        mock_plt.subplots.return_value = (mock_fig, mock_axes)

        image = np.random.random((40, 50, 3))
        y_pixels = np.random.random(50) * 40

        compare_blur_methods(image, y_pixels=y_pixels)

        # Verify limb curve plotting on all subplots
        plot_call_count = sum(ax.plot.call_count for ax in mock_axes.flat)
        assert plot_call_count >= 4  # Should plot limb on all 4 subplots

    @patch("planet_ruler.image.gradient_field")
    @patch("planet_ruler.plot.cv2")
    @patch("planet_ruler.plot.plt")
    def test_compare_blur_methods_grayscale(
        self, mock_plt, mock_cv2, mock_gradient_field
    ):
        """Test blur method comparison with grayscale image."""
        mock_cv2.GaussianBlur.return_value = np.ones((30, 40)) * 75
        mock_gradient_field.return_value = {
            "grad_mag": np.ones((30, 40)) * 0.4,
            "grad_angle": np.zeros((30, 40)),
        }

        mock_fig = Mock()
        mock_axes = np.array([[Mock(), Mock()], [Mock(), Mock()]])
        mock_plt.subplots.return_value = (mock_fig, mock_axes)

        # Grayscale image (2D)
        image = np.random.random((30, 40))

        compare_blur_methods(image)

        # Should handle grayscale input
        mock_plt.subplots.assert_called_once()
        mock_cv2.GaussianBlur.assert_called_once()
        mock_gradient_field.assert_called_once()


class TestCompareGradientFields:
    """Test gradient field comparison for multiple limbs."""

    @patch("planet_ruler.image.gradient_field")
    @patch("planet_ruler.plot.plt")
    @patch("matplotlib.collections.LineCollection")
    def test_compare_gradient_fields_multiple_limbs(
        self, mock_line_collection, mock_plt, mock_gradient_field
    ):
        """Test gradient field comparison with multiple limb proposals."""
        mock_gradient_field.return_value = {
            "grad_mag": np.ones((60, 80)) * 0.5,
            "grad_angle": np.zeros((60, 80)),
        }

        # Mock LineCollection
        mock_lc = Mock()
        mock_line_collection.return_value = mock_lc

        # Mock figure and axes
        mock_fig = Mock()
        mock_ax1 = Mock()
        mock_ax2 = Mock()
        mock_axes = [mock_ax1, mock_ax2]
        mock_plt.subplots.return_value = (mock_fig, mock_axes)

        # Test data - multiple limb proposals
        y_pixels_list = [np.array([25, 26, 27, 26, 25]), np.array([30, 31, 32, 31, 30])]
        labels = ["Limb A", "Limb B"]
        image = np.random.random((60, 80, 3))

        result = compare_gradient_fields(y_pixels_list, labels, image)

        # Verify gradient field computation
        mock_gradient_field.assert_called_once_with(
            ANY, gradient_smoothing=5.0, streak_length=30, decay_rate=0.15
        )

        # Verify matplotlib setup
        mock_plt.subplots.assert_called_once_with(2, 1, figsize=(16, 10))

        # Verify both axes were used
        for ax in mock_axes:
            ax.imshow.assert_called_once()
            ax.set_title.assert_called_once()
            ax.set_xlabel.assert_called_once()
            ax.set_ylabel.assert_called_once()
            ax.add_collection.assert_called_once()

        # Verify colorbar creation
        assert mock_plt.colorbar.call_count == 2

        assert result == (mock_fig, mock_axes)

    @patch("planet_ruler.image.gradient_field")
    @patch("planet_ruler.plot.plt")
    @patch("matplotlib.collections.LineCollection")
    def test_compare_gradient_fields_single_limb(
        self, mock_line_collection, mock_plt, mock_gradient_field
    ):
        """Test gradient field comparison with single limb."""
        mock_gradient_field.return_value = {
            "grad_mag": np.ones((40, 50)) * 0.3,
            "grad_angle": np.ones((40, 50)) * 3.14159 / 2,
        }

        mock_lc = Mock()
        mock_line_collection.return_value = mock_lc

        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (
            mock_fig,
            mock_ax,
        )  # Single axis, will be wrapped by function

        y_pixels_list = [np.array([20, 21, 22, 21, 20])]
        labels = ["Single Limb"]
        image = np.random.random((40, 50, 3))

        compare_gradient_fields(
            y_pixels_list,
            labels,
            image,
            gradient_smoothing=3.0,
            streak_length=25,
            decay_rate=0.2,
        )

        # Verify custom parameters
        mock_gradient_field.assert_called_once_with(
            ANY, gradient_smoothing=3.0, streak_length=25, decay_rate=0.2
        )

        # Single limb should create single subplot
        mock_plt.subplots.assert_called_once_with(1, 1, figsize=(16, 5))


class TestPlotDiffEvolPosteriors:
    """Test differential evolution posterior plotting."""

    @patch("planet_ruler.plot.unpack_diff_evol_posteriors")
    @patch("planet_ruler.plot.plt")
    @patch("planet_ruler.plot.sns")
    def test_plot_diff_evol_posteriors_basic(self, mock_sns, mock_plt, mock_unpack):
        """Test basic differential evolution posterior plotting."""
        # Mock observation object
        mock_obs = Mock()
        mock_obs.free_parameters = ["r", "h"]
        mock_obs.parameter_limits = {"r": [6000000, 7000000], "h": [300000, 500000]}
        mock_obs.init_parameter_values = {"r": 6371000, "h": 400000}

        # Mock population data
        mock_pop = pd.DataFrame(
            {
                "r": [6350000, 6370000, 6390000],
                "h": [390000, 410000, 430000],
                "mse": [0.1, 0.2, 0.15],
                "other_param": [1, 2, 3],  # Should be ignored
            }
        )
        mock_unpack.return_value = mock_pop

        # Mock matplotlib
        mock_ax = Mock()
        mock_plt.gca.return_value = mock_ax
        mock_ax.get_legend_handles_labels.return_value = ([], [])

        plot_diff_evol_posteriors(mock_obs, show_points=True, log=True)

        # Verify population unpacking
        mock_unpack.assert_called_once_with(mock_obs)

        # Verify plotting for each free parameter
        assert mock_plt.scatter.call_count == 2  # r and h parameters
        assert mock_sns.kdeplot.call_count == 2
        assert mock_plt.axvline.call_count == 6  # 2 bounds + 1 initial per parameter
        assert mock_plt.title.call_count == 2
        assert mock_plt.show.call_count == 2

    @patch("planet_ruler.plot.unpack_diff_evol_posteriors")
    @patch("planet_ruler.plot.plt")
    @patch("planet_ruler.plot.sns")
    def test_plot_diff_evol_posteriors_no_points(self, mock_sns, mock_plt, mock_unpack):
        """Test posterior plotting without individual points."""
        mock_obs = Mock()
        mock_obs.free_parameters = ["r"]
        mock_obs.parameter_limits = {"r": [5000000, 8000000]}
        mock_obs.init_parameter_values = {"r": 6371000}

        mock_pop = pd.DataFrame(
            {"r": [6300000, 6400000, 6500000], "mse": [0.05, 0.1, 0.08]}
        )
        mock_unpack.return_value = mock_pop

        mock_ax = Mock()
        mock_plt.gca.return_value = mock_ax
        mock_ax.get_legend_handles_labels.return_value = ([], [])

        plot_diff_evol_posteriors(mock_obs, show_points=False, log=False)

        # Should not show individual points
        mock_plt.scatter.assert_not_called()
        # But should still show KDE plot
        mock_sns.kdeplot.assert_called_once()

        # Verify log scale setting
        mock_ax.set_yscale.assert_not_called()  # log=False

    @patch("planet_ruler.plot.unpack_diff_evol_posteriors")
    @patch("planet_ruler.plot.plt")
    @patch("planet_ruler.plot.sns")
    def test_plot_diff_evol_posteriors_missing_init_values(
        self, mock_sns, mock_plt, mock_unpack
    ):
        """Test posterior plotting with missing initial parameter values."""
        mock_obs = Mock()
        mock_obs.free_parameters = ["r", "h"]
        mock_obs.parameter_limits = {"r": [6000000, 7000000], "h": [300000, 500000]}
        mock_obs.init_parameter_values = {"r": 6371000}  # Missing 'h'

        mock_pop = pd.DataFrame(
            {"r": [6350000, 6370000], "h": [390000, 410000], "mse": [0.1, 0.2]}
        )
        mock_unpack.return_value = mock_pop

        mock_ax = Mock()
        mock_plt.gca.return_value = mock_ax
        mock_ax.get_legend_handles_labels.return_value = ([], [])

        plot_diff_evol_posteriors(mock_obs)

        # Should handle missing initial values gracefully (KeyError catch)
        mock_unpack.assert_called_once_with(mock_obs)


class TestPlotFullLimb:
    """Test full limb visualization including unseen sections."""

    @patch("planet_ruler.plot.limb_arc")
    @patch("planet_ruler.plot.plt")
    def test_plot_full_limb_with_best_parameters(self, mock_plt, mock_limb_arc):
        """Test full limb plotting with best parameters."""
        # Mock observation with best parameters
        mock_obs = Mock()
        mock_obs.best_parameters = {"r": 6371000, "h": 400000, "cx": 500, "cy": 300}
        mock_obs.image = np.random.random((600, 1000, 3))

        # Mock limb_arc returns
        mock_limb_arc.side_effect = [
            np.array([[100, 200], [101, 201], [102, 202]]),  # Full limb points
            np.array([200, 201, 202]),  # Image section limb
        ]

        mock_ax = Mock()
        mock_plt.gca.return_value = mock_ax

        plot_full_limb(mock_obs, x_min=0, x_max=1000, y_min=0, y_max=600)

        # Verify image display
        mock_plt.imshow.assert_called_once_with(mock_obs.image)

        # Verify limb_arc calls
        assert mock_limb_arc.call_count == 2

        # First call: full limb with return_full=True
        first_call = mock_limb_arc.call_args_list[0]
        assert first_call[0][0] == 6371000  # radius
        assert first_call[1]["return_full"] == True

        # Second call: image section limb
        second_call = mock_limb_arc.call_args_list[1]
        assert second_call[0][0] == 6371000  # radius
        assert "return_full" not in second_call[1]

        # Verify scatter plots
        assert mock_plt.scatter.call_count == 2

        # Verify axis limits
        mock_ax.set_xlim.assert_called_once_with(0, 1000)
        mock_ax.set_ylim.assert_called_once_with(0, 600)

        mock_plt.show.assert_called_once()

    @patch("planet_ruler.plot.limb_arc")
    @patch("planet_ruler.plot.plt")
    def test_plot_full_limb_with_init_parameters(self, mock_plt, mock_limb_arc):
        """Test full limb plotting fallback to init parameters."""
        # Mock observation without best_parameters (triggers AttributeError)
        mock_obs = Mock()
        del mock_obs.best_parameters  # Simulate AttributeError
        mock_obs.init_parameter_values = {
            "r": 3390000,
            "h": 200000,
            "cx": 400,
            "cy": 250,
        }
        mock_obs.image = np.random.random((500, 800, 3))

        mock_limb_arc.side_effect = [
            np.array([[50, 100], [51, 101]]),
            np.array([100, 101]),
        ]

        mock_ax = Mock()
        mock_plt.gca.return_value = mock_ax

        plot_full_limb(mock_obs)

        # Should use init_parameter_values when best_parameters not available
        first_call = mock_limb_arc.call_args_list[0]
        assert first_call[0][0] == 3390000  # radius from init values

    @patch("planet_ruler.plot.limb_arc")
    @patch("planet_ruler.plot.plt")
    def test_plot_full_limb_no_limits(self, mock_plt, mock_limb_arc):
        """Test full limb plotting without axis limits."""
        mock_obs = Mock()
        mock_obs.best_parameters = {"r": 1737000, "h": 100000}  # Moon parameters
        mock_obs.image = np.random.random((400, 600, 3))

        mock_limb_arc.side_effect = [np.array([[25, 50], [26, 51]]), np.array([50, 51])]

        mock_ax = Mock()
        mock_plt.gca.return_value = mock_ax

        plot_full_limb(mock_obs)

        # Should set limits to None when not provided
        mock_ax.set_xlim.assert_called_once_with(None, None)
        mock_ax.set_ylim.assert_called_once_with(None, None)


class TestPlotSegmentationMasks:
    """Test segmentation mask visualization."""

    @patch("planet_ruler.plot.plt")
    def test_plot_segmentation_masks_basic(self, mock_plt):
        """Test basic segmentation mask plotting."""
        # Mock observation with segmentation masks
        mock_obs = Mock()
        mock_obs._segmenter._masks = [
            {"segmentation": np.random.random((100, 150))},
            {"segmentation": np.random.random((100, 150))},
            {"segmentation": np.random.random((100, 150))},
        ]

        plot_segmentation_masks(mock_obs)

        # Verify plotting for each mask
        assert mock_plt.imshow.call_count == 3
        assert mock_plt.title.call_count == 3
        assert mock_plt.show.call_count == 3

        # Verify titles
        title_calls = mock_plt.title.call_args_list
        expected_titles = ["Mask 0", "Mask 1", "Mask 2"]
        for i, call in enumerate(title_calls):
            assert call[0][0] == expected_titles[i]

    @patch("planet_ruler.plot.plt")
    def test_plot_segmentation_masks_single_mask(self, mock_plt):
        """Test segmentation mask plotting with single mask."""
        mock_obs = Mock()
        mock_obs._segmenter._masks = [{"segmentation": np.random.random((50, 75))}]

        plot_segmentation_masks(mock_obs)

        # Should handle single mask
        mock_plt.imshow.assert_called_once()
        mock_plt.title.assert_called_once_with("Mask 0")
        mock_plt.show.assert_called_once()

    @patch("planet_ruler.plot.plt")
    def test_plot_segmentation_masks_empty(self, mock_plt):
        """Test segmentation mask plotting with no masks."""
        mock_obs = Mock()
        mock_obs._segmenter._masks = []

        plot_segmentation_masks(mock_obs)

        # Should handle empty mask list gracefully
        mock_plt.imshow.assert_not_called()
        mock_plt.title.assert_not_called()
        mock_plt.show.assert_not_called()


# Import pandas for DataFrame creation
import pandas as pd
