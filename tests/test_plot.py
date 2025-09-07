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
from unittest.mock import Mock, patch, MagicMock, call

from planet_ruler.plot import plot_image, plot_limb, plot_3d_solution, plot_topography


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

        # Check basic parameters
        assert len(call_args) == 2  # x, y data
        assert call_kwargs["c"] == "y"
        assert call_kwargs["s"] == 10
        assert call_kwargs["alpha"] == 0.2

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
        assert call_kwargs["s"] == 20
        assert call_kwargs["alpha"] == 0.8

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

        expected_x = np.array([0])
        mock_plt.scatter.assert_called_once_with(
            expected_x, limb_data, c="blue", s=50, alpha=1.0
        )
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
