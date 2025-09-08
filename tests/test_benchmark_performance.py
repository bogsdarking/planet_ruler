"""
Performance benchmark tests for planet_ruler critical functions.

These tests measure execution time and performance characteristics of
computationally intensive functions to track performance regressions
and identify optimization opportunities.
"""

import pytest
import numpy as np
from unittest.mock import patch

from planet_ruler.geometry import (
    horizon_distance,
    limb_camera_angle,
    limb_arc,
    intrinsic_transform,
    extrinsic_transform,
)
from planet_ruler.fit import CostFunction, unpack_parameters, pack_parameters
from planet_ruler.image import gradient_break, smooth_limb
from planet_ruler.observation import LimbObservation, PlanetObservation


class TestGeometryBenchmarks:
    """Benchmark tests for geometry module functions."""

    def test_horizon_distance_benchmark(self, benchmark):
        """Benchmark horizon distance calculation."""
        radius = 6371000  # Earth radius
        altitude = 35000  # Typical aircraft altitude

        result = benchmark(horizon_distance, radius, altitude)
        assert result > 0

    def test_limb_camera_angle_benchmark(self, benchmark):
        """Benchmark limb camera angle calculation."""
        radius = 6371000  # Earth radius
        altitude = 35000  # Typical aircraft altitude

        result = benchmark(limb_camera_angle, radius, altitude)
        assert 0 < result < np.pi / 2

    def test_horizon_distance_vectorized_benchmark(self, benchmark):
        """Benchmark vectorized horizon distance calculations."""
        radius = 6371000
        altitudes = np.linspace(1000, 100000, 1000)  # 1000 different altitudes

        def vectorized_horizon():
            return np.array([horizon_distance(radius, alt) for alt in altitudes])

        result = benchmark(vectorized_horizon)
        assert len(result) == 1000
        assert all(r > 0 for r in result)

    def test_limb_arc_benchmark_small(self, benchmark):
        """Benchmark limb_arc function with small image parameters."""
        result = benchmark(
            limb_arc,
            r=6371000,  # Earth radius
            n_pix_x=500,  # Small image width
            n_pix_y=500,  # Small image height
            h=35000,  # Altitude
            fov=30.0,  # Field of view
            w=0.01,  # Detector size
            num_sample=1000,  # Sample points
        )
        assert len(result) == 500

    def test_limb_arc_benchmark_large(self, benchmark):
        """Benchmark limb_arc function with large image parameters."""
        result = benchmark(
            limb_arc,
            r=6371000,  # Earth radius
            n_pix_x=2000,  # Large image width
            n_pix_y=2000,  # Large image height
            h=35000,  # Altitude
            fov=45.0,  # Field of view
            w=0.015,  # Detector size
            num_sample=5000,  # More sample points
        )
        assert len(result) == 2000

    def test_coordinate_transforms_benchmark(self, benchmark):
        """Benchmark coordinate transformation pipeline."""
        n_points = 1000

        def transform_pipeline():
            # Generate random world coordinates
            world_coords = np.random.standard_normal((n_points, 4))
            world_coords[:, 3] = 1  # Homogeneous coordinates

            # Apply extrinsic transform
            camera_coords = extrinsic_transform(
                world_coords,
                theta_x=0.1,
                theta_y=0.05,
                theta_z=0.02,
                origin_x=10,
                origin_y=100,
                origin_z=5,
            )

            # Apply intrinsic transform
            pixel_coords = intrinsic_transform(
                camera_coords, f=0.05, px=1e-5, py=1e-5, x0=500, y0=500
            )

            return pixel_coords

        result = benchmark(transform_pipeline)
        assert result.shape[0] == n_points


class TestFitBenchmarks:
    """Benchmark tests for fit module functions."""

    def test_parameter_packing_benchmark(self, benchmark):
        """Benchmark parameter packing operations."""
        params = {
            "radius": 6371000,
            "altitude": 35000,
            "focal_length": 0.05,
            "detector_width": 0.01,
            "fov": 30.0,
            "theta_x": 0.1,
            "theta_y": 0.05,
            "theta_z": 0.02,
        }

        # Create template with same keys
        template = list(params.keys())

        result = benchmark(pack_parameters, params, template)
        assert len(result) == len(params)

    def test_parameter_unpacking_benchmark(self, benchmark):
        """Benchmark parameter unpacking operations."""
        param_array = np.array([6371000, 35000, 0.05, 0.01, 30.0, 0.1, 0.05, 0.02])
        param_names = [
            "radius",
            "altitude",
            "focal_length",
            "detector_width",
            "fov",
            "theta_x",
            "theta_y",
            "theta_z",
        ]

        result = benchmark(unpack_parameters, param_array, param_names)
        assert len(result) == len(param_names)

    def test_cost_function_evaluation_benchmark(self, benchmark):
        """Benchmark CostFunction evaluation with synthetic data."""
        # Create synthetic observed and predicted data
        n_points = 1000
        observed = np.random.normal(500, 50, n_points)  # Pixel coordinates

        def dummy_function(**params):
            """Dummy function that simulates limb prediction."""
            noise = np.random.normal(0, params.get("noise", 1), n_points)
            return observed + noise

        cost_func = CostFunction(
            target=observed,
            function=dummy_function,
            free_parameters=["radius", "altitude", "noise"],
            init_parameter_values={"radius": 6371000, "altitude": 35000, "noise": 1.0},
            loss_function="l2",
        )

        params = {"radius": 6371000, "altitude": 35000, "noise": 2.0}

        result = benchmark(cost_func.evaluate, params)
        assert np.all(result >= 0)

    def test_cost_function_different_losses_benchmark(self, benchmark):
        """Benchmark different loss functions."""
        n_points = 500
        observed = np.random.normal(100, 10, n_points)
        predicted = observed + np.random.normal(0, 2, n_points)  # Small errors

        def benchmark_losses():
            results = {}
            init_params = {"param1": 1.0}
            for loss_type in ["l1", "l2", "log-l1"]:
                cost_func = CostFunction(
                    target=observed,
                    function=lambda **p: predicted,
                    free_parameters=["param1"],
                    init_parameter_values=init_params,
                    loss_function=loss_type,
                )
                results[loss_type] = cost_func.cost({})
            return results

        result = benchmark(benchmark_losses)
        assert all(cost >= 0 for cost in result.values())


class TestImageProcessingBenchmarks:
    """Benchmark tests for actual image processing functions."""

    def test_gradient_break_benchmark_small(self, benchmark):
        """Benchmark gradient_break with small synthetic image."""
        height, width = 300, 600
        # Create 3D synthetic image with gradual brightness change
        single_channel = np.linspace(0, 255, height * width).reshape((height, width))
        image = np.stack(
            [single_channel, single_channel, single_channel], axis=2
        ).astype(np.uint8)

        # Function now auto-calculates appropriate window size
        result = benchmark(gradient_break, image)
        assert len(result) == width

    def test_gradient_break_benchmark_large(self, benchmark):
        """Benchmark gradient_break with large synthetic image."""
        height, width = 1000, 2000
        # Create 3D synthetic image with gradual brightness change
        single_channel = np.random.uniform(0, 255, (height, width)).astype(np.uint8)
        image = np.stack([single_channel, single_channel, single_channel], axis=2)

        result = benchmark(gradient_break, image)
        assert len(result) == width

    def test_gradient_break_benchmark_realistic(self, benchmark):
        """Benchmark gradient_break with realistic horizon scenario."""
        height, width = 600, 900
        # Create realistic horizon image: sky above, ground below (3D for gradient_break)
        image = np.zeros((height, width, 3), dtype=np.uint8)

        # Sky (lighter, with gradient)
        sky_gradient = np.linspace(180, 220, height // 2)
        for i in range(height // 2):
            brightness = sky_gradient[i] + np.random.normal(0, 5, width)
            for c in range(3):  # RGB channels
                image[i, :, c] = brightness

        # Ground (darker, more uniform)
        for i in range(height // 2, height):
            brightness = np.random.normal(80, 10, width)
            for c in range(3):  # RGB channels
                image[i, :, c] = brightness

        result = benchmark(gradient_break, image)
        assert len(result) == width

    def test_smooth_limb_benchmark_rolling_median(self, benchmark):
        """Benchmark smooth_limb with rolling median method."""
        n_points = 1000
        # Create synthetic limb data with noise
        x = np.linspace(0, 500, n_points)
        y = 200 + 50 * np.sin(x / 100) + np.random.normal(0, 5, n_points)

        result = benchmark(smooth_limb, y, method="rolling-median", window_length=21)
        assert len(result) == n_points

    def test_smooth_limb_benchmark_savgol(self, benchmark):
        """Benchmark smooth_limb with Savitzky-Golay filter."""
        n_points = 2000
        # Create synthetic limb data with noise
        y = (
            300
            + 30 * np.cos(np.linspace(0, 4 * np.pi, n_points))
            + np.random.normal(0, 3, n_points)
        )

        result = benchmark(
            smooth_limb, y, method="savgol", window_length=51, polyorder=3
        )
        assert len(result) == n_points

    def test_smooth_limb_methods_comparison_benchmark(self, benchmark):
        """Benchmark comparison of different smoothing methods."""
        n_points = 1500
        # Create noisy limb data
        y = (
            250
            + 40 * np.sin(np.linspace(0, 2 * np.pi, n_points))
            + np.random.normal(0, 8, n_points)
        )

        def compare_methods():
            methods = {
                "rolling_median": smooth_limb(
                    y, method="rolling-median", window_length=25
                ),
                "rolling_mean": smooth_limb(y, method="rolling-mean", window_length=25),
                "savgol": smooth_limb(
                    y, method="savgol", window_length=25, polyorder=3
                ),
            }
            return methods

        result = benchmark(compare_methods)
        assert all(len(smoothed) == n_points for smoothed in result.values())


class TestObservationBenchmarks:
    """Benchmark tests for observation workflow functions."""

    @patch("planet_ruler.observation.load_image")
    def test_planet_observation_initialization_benchmark(
        self, mock_load_image, benchmark
    ):
        """Benchmark PlanetObservation initialization."""
        height, width = 1000, 1500
        image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

        # Mock the image loading to return our synthetic image
        mock_load_image.return_value = image

        result = benchmark(PlanetObservation, "mock_image_path.jpg")
        assert result.image.shape == (height, width, 3)

    @patch("planet_ruler.observation.load_image")
    def test_detect_limb_gradient_break_benchmark(self, mock_load_image, benchmark):
        """Benchmark limb detection using gradient break method."""
        height, width = 600, 900
        # Create realistic image with horizon
        image = np.zeros((height, width, 3), dtype=np.uint8)

        # Add sky and ground with realistic brightness difference
        for i in range(height // 2):
            image[i, :, :] = [150 + np.random.randint(-20, 20) for _ in range(3)]
        for i in range(height // 2, height):
            image[i, :, :] = [70 + np.random.randint(-15, 15) for _ in range(3)]

        # Mock the image loading to return our synthetic image
        mock_load_image.return_value = image
        obs = PlanetObservation("mock_image_path.jpg")

        # Use gradient_break directly since that's the actual implementation
        result = benchmark(gradient_break, image)  # Use full 3-channel image
        assert len(result) == width


class TestIntegratedWorkflowBenchmarks:
    """Benchmark tests for complete workflow scenarios."""

    def test_complete_geometry_pipeline_benchmark(self, benchmark):
        """Benchmark complete geometry calculation pipeline."""

        def complete_pipeline():
            # Step 1: Calculate horizon distance and camera angle
            radius = 6371000
            altitude = 35000
            distance = horizon_distance(radius, altitude)
            angle = limb_camera_angle(radius, altitude)

            # Step 2: Generate limb arc
            limb_y = limb_arc(
                r=radius,
                n_pix_x=1000,
                n_pix_y=1000,
                h=altitude,
                fov=30.0,
                w=0.01,
                num_sample=2000,
            )

            # Step 3: Apply coordinate transformations
            world_coords = np.random.standard_normal((100, 4))
            world_coords[:, 3] = 1

            camera_coords = extrinsic_transform(
                world_coords, theta_x=angle * 0.1, theta_y=0, theta_z=0
            )

            pixel_coords = intrinsic_transform(
                camera_coords, f=0.05, px=1e-5, py=1e-5, x0=500, y0=500
            )

            return {
                "distance": distance,
                "angle": angle,
                "limb_y": limb_y,
                "pixel_coords": pixel_coords,
            }

        result = benchmark(complete_pipeline)
        assert result["distance"] > 0
        assert result["angle"] > 0
        assert len(result["limb_y"]) == 1000
        assert result["pixel_coords"].shape[0] == 100

    def test_image_processing_pipeline_benchmark(self, benchmark):
        """Benchmark complete image processing pipeline."""

        def processing_pipeline():
            height, width = (
                400,
                800,
            )  # Increased width to work with gradient_break's savgol filter
            # Create realistic synthetic image with horizon (3D for gradient_break)
            image = np.zeros((height, width, 3), dtype=np.uint8)

            # Sky with gradient
            for i in range(height // 2):
                brightness = 200 - (i * 30 / (height // 2))  # Darker toward horizon
                for c in range(3):  # RGB channels
                    image[i, :, c] = brightness + np.random.normal(0, 10, width)

            # Ground
            for i in range(height // 2, height):
                for c in range(3):  # RGB channels
                    image[i, :, c] = np.random.normal(90, 15, width)

            # Step 1: Detect limb using gradient break (auto-calculates window size)
            limb_y = gradient_break(image)

            # Step 2: Smooth the detected limb
            smooth_limb_y = smooth_limb(
                limb_y, method="rolling-median", window_length=15
            )

            # Step 3: Further smooth with Savitzky-Golay (use appropriate window size)
            window_len = min(21, len(smooth_limb_y))
            if window_len % 2 == 0:  # Ensure odd window length
                window_len -= 1
            window_len = max(5, window_len)  # Minimum window size
            final_limb = smooth_limb(
                smooth_limb_y, method="savgol", window_length=window_len, polyorder=3
            )

            return {
                "original_limb": limb_y,
                "smooth_limb": smooth_limb_y,
                "final_limb": final_limb,
            }

        result = benchmark(processing_pipeline)
        assert len(result["original_limb"]) == 800
        assert len(result["final_limb"]) == 800

    def test_parameter_optimization_simulation_benchmark(self, benchmark):
        """Benchmark parameter optimization simulation."""

        def optimization_simulation():
            # Simulate parameter optimization process
            n_iterations = 50
            n_params = 8

            results = []
            for iteration in range(n_iterations):
                # Generate random parameters
                params = {
                    "radius": 6371000 + np.random.normal(0, 100000),
                    "altitude": 35000 + np.random.normal(0, 5000),
                    "focal_length": 0.05 + np.random.normal(0, 0.005),
                    "detector_width": 0.01 + np.random.normal(0, 0.001),
                    "fov": 30.0 + np.random.normal(0, 2.0),
                    "theta_x": np.random.normal(0, 0.1),
                    "theta_y": np.random.normal(0, 0.1),
                    "theta_z": np.random.normal(0, 0.1),
                }

                # Pack parameters
                # Create a template for packing
                template = {
                    "radius": 6371000,
                    "altitude": 35000,
                    "focal_length": 0.05,
                    "detector_width": 0.01,
                    "fov": 30.0,
                    "theta_x": 0.0,
                    "theta_y": 0.0,
                    "theta_z": 0.0,
                }
                param_array = pack_parameters(params, template)

                # Simulate cost calculation
                predicted_limb = np.random.normal(300, 20, 500)
                observed_limb = predicted_limb + np.random.normal(0, 5, 500)

                cost = np.sum((predicted_limb - observed_limb) ** 2) / len(
                    observed_limb
                )
                results.append(cost)

            return np.array(results)

        result = benchmark(optimization_simulation)
        assert len(result) == 50
        assert all(r >= 0 for r in result)


# Performance test markers
pytestmark = pytest.mark.benchmark
