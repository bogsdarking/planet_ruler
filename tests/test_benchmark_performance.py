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
Performance benchmark tests for planet_ruler critical functions.

These tests measure execution time and performance characteristics of
computationally intensive functions to track performance regressions
and identify optimization opportunities.

BENCHMARK CATEGORIES:
- Fast benchmarks (~microseconds to seconds): Default, no special markers
- Slow benchmarks (~minutes): Marked with @pytest.mark.slow

USAGE:
- Run all benchmarks: pytest tests/test_benchmark_performance.py
- Run only fast benchmarks: pytest tests/test_benchmark_performance.py -m "not slow"
- Run only slow benchmarks: pytest tests/test_benchmark_performance.py -m "slow"
- Skip slow during development: pytest tests/test_benchmark_performance.py -v -m "not slow"
"""

import pytest
import numpy as np
import json
from unittest.mock import patch

from planet_ruler.geometry import (
    horizon_distance,
    limb_camera_angle,
    limb_arc,
    intrinsic_transform,
    extrinsic_transform,
)
from planet_ruler.fit import CostFunction, unpack_parameters, pack_parameters
from planet_ruler.image import gradient_break, smooth_limb, MaskSegmenter
from planet_ruler.observation import LimbObservation, PlanetObservation
from planet_ruler.camera import (
    extract_camera_parameters,
    create_config_from_image,
    extract_exif,
    get_camera_model,
    get_focal_length_mm,
    get_gps_altitude,
)
from planet_ruler.validation import validate_limb_config
from planet_ruler.uncertainty import calculate_parameter_uncertainty


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


class TestCameraParameterBenchmarks:
    """Benchmark tests for camera parameter extraction pipeline."""

    def test_extract_exif_benchmark(self, benchmark):
        """Benchmark EXIF data extraction from image files."""
        # Use a test image from the demo directory
        image_path = "demo/images/2013-08-05_22-42-14_Wikimania.jpg"

        def extract_exif_data():
            return extract_exif(image_path)

        result = benchmark(extract_exif_data)
        assert result is not None

    def test_extract_camera_parameters_benchmark(self, benchmark):
        """Benchmark complete camera parameter extraction workflow."""
        image_path = "demo/images/2013-08-05_22-42-14_Wikimania.jpg"

        def extract_params():
            return extract_camera_parameters(image_path)

        result = benchmark(extract_params)
        assert result is not None
        assert "image_width_px" in result
        assert "image_height_px" in result

    def test_camera_model_detection_benchmark(self, benchmark):
        """Benchmark camera model detection from EXIF."""
        image_path = "demo/images/2013-08-05_22-42-14_Wikimania.jpg"

        def detect_camera_model():
            exif = extract_exif(image_path)
            return get_camera_model(exif)

        result = benchmark(detect_camera_model)
        # Result may be None for test images without camera info

    def test_focal_length_extraction_benchmark(self, benchmark):
        """Benchmark focal length extraction from EXIF."""
        image_path = "demo/images/2013-08-05_22-42-14_Wikimania.jpg"

        def extract_focal_length():
            exif = extract_exif(image_path)
            return get_focal_length_mm(exif)

        benchmark(extract_focal_length)

    def test_gps_altitude_extraction_benchmark(self, benchmark):
        """Benchmark GPS altitude extraction from EXIF."""
        image_path = "demo/images/2013-08-05_22-42-14_Wikimania.jpg"

        def extract_gps_altitude():
            return get_gps_altitude(image_path)

        benchmark(extract_gps_altitude)


class TestConfigurationBenchmarks:
    """Benchmark tests for configuration generation and validation."""

    def test_create_config_from_image_benchmark(self, benchmark):
        """Benchmark automatic configuration generation from image."""
        image_path = "demo/images/2013-08-05_22-42-14_Wikimania.jpg"

        def create_config():
            return create_config_from_image(
                image_path=image_path,
                altitude_m=10000,
                planet="earth",
                perturbation_factor=0.5,
            )

        result = benchmark(create_config)
        assert result is not None
        assert "init_parameter_values" in result
        assert "parameter_limits" in result

    def test_config_validation_benchmark(self, benchmark):
        """Benchmark configuration validation workflow."""
        image_path = "demo/images/2013-08-05_22-42-14_Wikimania.jpg"
        config = create_config_from_image(
            image_path=image_path, altitude_m=10000, planet="earth"
        )

        def validate_config():
            return validate_limb_config(config, strict=False)

        benchmark(validate_config)


class TestLimbDetectionBenchmarks:
    """Benchmark tests for different limb detection methods."""

    @pytest.fixture
    def test_observation_for_detection(self):
        """Create a test observation for limb detection benchmarks using real Pluto image."""
        # Use real Pluto image and configuration for realistic benchmark
        image_path = "demo/images/PIA19948.tif"
        config_path = "config/pluto-new-horizons.yaml"

        obs = LimbObservation(image_filepath=image_path, fit_config=config_path)

        return obs

    @pytest.mark.slow
    def test_gradient_field_detection_benchmark(
        self, benchmark, test_observation_for_detection
    ):
        """Benchmark gradient-field limb detection/optimization with real Pluto image."""
        obs = test_observation_for_detection
        obs.limb_detection = "gradient_field"

        def run_gradient_field():
            # Use a minimal fitting to benchmark the gradient field approach
            obs.fit_limb(
                loss_function="gradient_field",
                minimizer="dual-annealing",
                max_iter=150,  # More iterations for robust analysis
                verbose=False,
                resolution_stages=[2, 1],  # Quick multi-resolution
                image_smoothing=2.0,
                kernel_smoothing=8.0,
            )
            return obs.best_parameters

        result = benchmark(run_gradient_field)
        assert result is not None

        # Validate that we got a reasonable Pluto radius (true value ~1188 km)
        fitted_radius_km = result.get("r", 0) / 1000.0
        print(f"\nFitted Pluto radius: {fitted_radius_km:.1f} km (true: ~1188 km)")

        # Check that result is within reasonable bounds for Pluto
        assert (
            600 < fitted_radius_km < 1900
        ), f"Fitted radius {fitted_radius_km:.1f} km outside expected range [600, 1900] km"

        # Ideally should be close to true value, but allow some tolerance for benchmark speed
        # (using fewer iterations than a real analysis would)

    @pytest.mark.slow
    def test_manual_limb_detection_benchmark(
        self, benchmark, test_observation_for_detection
    ):
        """Benchmark manual limb detection workflow using stored annotations."""
        obs = test_observation_for_detection

        # Manually annotated Pluto limb points (embedded for test safety)
        # These are 19 points tracing Pluto's limb in the PIA19948.tif image
        pluto_limb_points = [
            [33.33874898456539, 1094.6222583265637],
            [150.02437043054425, 997.3842404549147],
            [302.8269699431356, 880.6986190089358],
            [450.0731112916328, 794.5735174654752],
            [677.8878960194963, 675.1096669374492],
            [858.4727863525588, 591.7627944760357],
            [1100.178716490658, 513.9723801787164],
            [1327.9935012185215, 452.8513403736799],
            [1472.4614134849714, 422.2908204711616],
            [1666.9374492282695, 397.2867587327376],
            [1880.8610885458975, 380.6173842404549],
            [2039.220146222583, 377.8391551584078],
            [2217.0268074735986, 386.17384240454913],
            [2364.2729488220957, 402.8432168968318],
            [2578.1965881397236, 427.84727863525586],
            [2800.4549147034927, 472.2989439480097],
            [2983.8180341186026, 525.0852965069049],
            [3181.0722989439478, 588.9845653939885],
            [3356.100731112916, 666.7749796913079],
        ]
        image_width = 3420  # PIA19948.tif width

        def run_manual_detection():
            # Create sparse target array following annotate.py pattern
            limb_target = np.full(image_width, np.nan)

            # Fill in the manually annotated points
            for x, y in pluto_limb_points:
                x_idx = int(round(x))
                if 0 <= x_idx < image_width:
                    limb_target[x_idx] = y

            # Register the limb with observation
            obs.register_limb(limb_target)

            # Fit using L1 loss (standard for manual detection)
            obs.fit_limb(
                loss_function="l1",
                minimizer="differential-evolution",
                max_iter=300,
                verbose=False,
                seed=42,
            )

            return obs.best_parameters

        result = benchmark(run_manual_detection)
        assert result is not None

        # Validate that we got a reasonable Pluto radius (true value ~1188 km)
        fitted_radius_km = result.get("r", 0) / 1000.0
        print(
            f"\nManual detection - Fitted Pluto radius: {fitted_radius_km:.1f} km (true: ~1188 km)"
        )

        # Check that result is within reasonable bounds for Pluto
        assert (
            600 < fitted_radius_km < 1900
        ), f"Fitted radius {fitted_radius_km:.1f} km outside expected range [600, 1900] km"


class TestUncertaintyBenchmarks:
    """Benchmark tests for parameter uncertainty calculations."""

    @pytest.fixture
    def fitted_observation(self):
        """Create a fitted observation for uncertainty benchmarks."""
        # Create minimal test setup
        height, width = 200, 300
        image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

        # Save image to temporary file for LimbObservation
        import tempfile
        import os
        from PIL import Image as PILImage

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            # Convert numpy array to PIL Image and save
            pil_image = PILImage.fromarray(image)
            pil_image.save(tmp.name)
            temp_path = tmp.name

        config = {
            "free_parameters": ["r", "h", "f", "w", "theta_x", "theta_y", "theta_z"],
            "init_parameter_values": {
                "r": 6371000,
                "h": 10000,
                "f": 0.024,
                "w": 0.036,
                "theta_x": 0.0,
                "theta_y": 0.0,
                "theta_z": 0.0,
            },
            "parameter_limits": {
                "r": [6000000, 7000000],
                "h": [5000, 15000],
                "f": [0.01, 0.1],
                "w": [0.02, 0.05],
                "theta_x": [-3.14, 3.14],
                "theta_y": [-3.14, 3.14],
                "theta_z": [-3.14, 3.14],
            },
            "loss_function": "L1",
            "minimizer": "differential-evolution",
        }

        obs = LimbObservation(image_filepath=temp_path, fit_config=config)

        # Mock fitted results
        obs.best_parameters = config["init_parameter_values"].copy()
        obs.fit_results = type(
            "MockResult",
            (),
            {
                "x": np.array([6371000, 10000, 0.024, 0.036, 0.0, 0.0, 0.0]),
                "success": True,
                "population": np.random.normal(
                    [6371000, 10000, 0.024, 0.036, 0.0, 0.0, 0.0],
                    [50000, 1000, 0.001, 0.001, 0.1, 0.1, 0.1],
                    (20, 7),
                ),
            },
        )()

        # Clean up temp file
        os.unlink(temp_path)

        return obs

    def test_parameter_uncertainty_calculation_benchmark(
        self, benchmark, fitted_observation
    ):
        """Benchmark parameter uncertainty calculation."""
        obs = fitted_observation

        def calculate_uncertainty():
            return calculate_parameter_uncertainty(
                obs, parameter="r", scale_factor=1000, method="auto"
            )

        result = benchmark(calculate_uncertainty)
        assert result is not None
        assert "value" in result
        assert "uncertainty" in result

    def test_multiple_parameter_uncertainties_benchmark(
        self, benchmark, fitted_observation
    ):
        """Benchmark calculating uncertainties for multiple parameters."""
        obs = fitted_observation

        def calculate_multiple_uncertainties():
            results = {}
            for param in ["r", "h", "theta_x"]:
                scale = 1000 if param in ["r", "h"] else 1
                results[param] = calculate_parameter_uncertainty(
                    obs, parameter=param, scale_factor=scale, method="auto"
                )
            return results

        result = benchmark(calculate_multiple_uncertainties)
        assert len(result) == 3


class TestFullPipelineBenchmarks:
    """Benchmark tests for complete end-to-end workflows."""

    @pytest.mark.slow
    def test_full_pipeline_earth_benchmark(self, benchmark):
        """Benchmark complete Earth measurement pipeline."""
        image_path = "demo/images/50644513538_56228a2027_o.jpg"

        def run_complete_pipeline():
            # Step 1: Extract camera parameters
            camera_info = extract_camera_parameters(image_path)

            # Step 2: Create configuration
            config = create_config_from_image(
                image_path=image_path,
                altitude_m=418_000,
                planet="earth",
            )

            # Step 3: Validate configuration
            validate_limb_config(config, strict=False)

            # Step 4: Create observation
            obs = LimbObservation(image_filepath=image_path, fit_config=config)

            # Step 5: Quick gradient-field fit (for speed)
            obs.fit_limb(
                loss_function="gradient_field",
                minimizer="differential-evolution",
                minimizer_preset="scipy-default",
                max_iter=300,
                verbose=False,
                resolution_stages=[8, 4],
                image_smoothing=2.0,  # Remove high-frequency image artifacts
                kernel_smoothing=8.0,  # Smooth gradient field for stability
                prefer_direction=None,
                seed=0,
            )

            # Step 6: Calculate uncertainty
            radius_uncertainty = calculate_parameter_uncertainty(
                obs, parameter="r", scale_factor=1000, method="auto"
            )

            return {
                "radius_km": obs.radius_km,
                "altitude_km": obs.altitude_km,
                "uncertainty": radius_uncertainty,
            }

        result = benchmark(run_complete_pipeline)
        assert result is not None
        assert "radius_km" in result
        fitted_radius_km = result["radius_km"]
        print(f"\nFitted Earth radius: {fitted_radius_km:.1f} km (true: ~6357 km)")
        assert (
            4000 < fitted_radius_km < 10000
        ), f"Fitted radius {fitted_radius_km:.1f} km outside expected range [4000, 10000] km"

    def test_configuration_generation_workflow_benchmark(self, benchmark):
        """Benchmark the workflow from image to ready-to-fit configuration."""
        image_path = "demo/images/2013-08-05_22-42-14_Wikimania.jpg"

        def config_workflow():
            # Extract camera parameters
            camera_info = extract_camera_parameters(image_path)

            # Generate configuration
            config = create_config_from_image(
                image_path=image_path, altitude_m=10000, planet="earth"
            )

            # Validate configuration
            validate_limb_config(config, strict=False)

            return config

        result = benchmark(config_workflow)
        assert result is not None
        assert "init_parameter_values" in result

    def test_observation_creation_and_setup_benchmark(self, benchmark):
        """Benchmark observation setup workflow."""
        image_path = "demo/images/2013-08-05_22-42-14_Wikimania.jpg"

        config = create_config_from_image(
            image_path=image_path, altitude_m=10000, planet="earth"
        )

        def obs_setup_workflow():
            # Create observation
            obs = LimbObservation(image_filepath=image_path, fit_config=config)

            # For benchmarking, use gradient-field which doesn't require explicit detection
            obs.limb_detection = "gradient_field"

            return obs

        result = benchmark(obs_setup_workflow)
        assert result is not None
        assert hasattr(result, "image")
        assert hasattr(result, "free_parameters")
        assert hasattr(result, "init_parameter_values")
        assert hasattr(result, "parameter_limits")


class TestSegmentationBenchmarks:
    """Benchmark tests for segmentation-based limb detection."""

    def test_custom_backend_segment_benchmark(self, benchmark):
        """Benchmark custom backend mask generation."""
        from planet_ruler.image import MaskSegmenter

        image = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)

        def simple_segmenter(img):
            """Simple threshold-based segmentation."""
            h, w = img.shape[:2]
            gray = img.sum(axis=2) / 3
            threshold = np.mean(gray)

            mask1 = gray < threshold
            mask2 = gray >= threshold

            return [
                {"mask": mask1, "area": np.sum(mask1)},
                {"mask": mask2, "area": np.sum(mask2)},
            ]

        def run_segmentation():
            segmenter = MaskSegmenter(
                image, method="custom", segment_fn=simple_segmenter, interactive=False
            )
            return segmenter.segment()

        result = benchmark(run_segmentation)

        assert len(result) == 500
        assert not np.all(np.isnan(result))

    def test_classify_automatic_benchmark(self, benchmark):
        """Benchmark automatic mask classification."""
        from planet_ruler.image import MaskSegmenter

        image = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)

        def multi_mask_segmenter(img):
            """Generate multiple masks for classification."""
            h, w = img.shape[:2]
            masks = []

            # Generate 10 masks at different positions
            for i in range(10):
                mask = np.zeros((h, w), dtype=bool)
                start_row = int(i * h / 10)
                end_row = int((i + 1) * h / 10)
                mask[start_row:end_row, :] = True
                masks.append({"mask": mask, "area": np.sum(mask)})

            return masks

        segmenter = MaskSegmenter(
            image, method="custom", segment_fn=multi_mask_segmenter, interactive=False
        )
        segmenter._masks = multi_mask_segmenter(image)

        result = benchmark(segmenter._classify_automatic)

        assert "sky" in result
        assert "planet" in result
        assert len(result["sky"]) == 1
        assert len(result["planet"]) == 1

    def test_combine_masks_benchmark(self, benchmark):
        """Benchmark limb extraction from classified masks."""
        from planet_ruler.image import MaskSegmenter

        image = np.random.randint(0, 255, (2000, 2000, 3), dtype=np.uint8)

        def horizon_segmenter(img):
            h, w = img.shape[:2]
            sky_mask = np.zeros((h, w), dtype=bool)
            sky_mask[: h // 2, :] = True
            planet_mask = np.zeros((h, w), dtype=bool)
            planet_mask[h // 2 :, :] = True

            return [
                {"mask": sky_mask, "area": np.sum(sky_mask)},
                {"mask": planet_mask, "area": np.sum(planet_mask)},
            ]

        segmenter = MaskSegmenter(
            image, method="custom", segment_fn=horizon_segmenter, interactive=False
        )
        segmenter._masks = horizon_segmenter(image)

        classified = {
            "sky": [segmenter._masks[0]],
            "planet": [segmenter._masks[1]],
            "exclude": [],
        }

        result = benchmark(segmenter._combine_masks, classified)

        assert "limb" in result
        assert len(result["limb"]) == 2000

    def test_downsampling_benchmark_2x(self, benchmark):
        """Benchmark segmentation with 2x downsampling."""
        from planet_ruler.image import MaskSegmenter

        image = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)

        def fast_segmenter(img):
            h, w = img.shape[:2]
            midpoint = h // 2

            sky = np.zeros((h, w), dtype=bool)
            sky[:midpoint, :] = True
            planet = np.zeros((h, w), dtype=bool)
            planet[midpoint:, :] = True

            return [
                {"mask": sky, "area": np.sum(sky)},
                {"mask": planet, "area": np.sum(planet)},
            ]

        def run_downsampled():
            segmenter = MaskSegmenter(
                image,
                method="custom",
                segment_fn=fast_segmenter,
                downsample_factor=2,
                interactive=False,
            )
            return segmenter.segment()

        result = benchmark(run_downsampled)

        assert len(result) == 1000  # Original size

    def test_downsampling_benchmark_4x(self, benchmark):
        """Benchmark segmentation with 4x downsampling."""
        from planet_ruler.image import MaskSegmenter

        image = np.random.randint(0, 255, (2000, 2000, 3), dtype=np.uint8)

        def fast_segmenter(img):
            h, w = img.shape[:2]
            midpoint = h // 2

            sky = np.zeros((h, w), dtype=bool)
            sky[:midpoint, :] = True
            planet = np.zeros((h, w), dtype=bool)
            planet[midpoint:, :] = True

            return [
                {"mask": sky, "area": np.sum(sky)},
                {"mask": planet, "area": np.sum(planet)},
            ]

        def run_downsampled():
            segmenter = MaskSegmenter(
                image,
                method="custom",
                segment_fn=fast_segmenter,
                downsample_factor=4,
                interactive=False,
            )
            return segmenter.segment()

        result = benchmark(run_downsampled)

        assert len(result) == 2000  # Original size

    def test_outlier_detection_benchmark(self, benchmark):
        """Benchmark outlier detection and interpolation in limb extraction."""
        from planet_ruler.image import MaskSegmenter

        image = np.random.randint(0, 255, (500, 1000, 3), dtype=np.uint8)

        def outlier_segmenter(img):
            h, w = img.shape[:2]
            sky_mask = np.zeros((h, w), dtype=bool)
            planet_mask = np.zeros((h, w), dtype=bool)

            # Create horizon with some outliers
            for col in range(w):
                if col % 50 == 0:  # Outlier every 50 columns
                    horizon_row = 100
                else:
                    horizon_row = 250

                sky_mask[:horizon_row, col] = True
                planet_mask[horizon_row:, col] = True

            return [
                {"mask": sky_mask, "area": np.sum(sky_mask)},
                {"mask": planet_mask, "area": np.sum(planet_mask)},
            ]

        segmenter = MaskSegmenter(
            image, method="custom", segment_fn=outlier_segmenter, interactive=False
        )
        segmenter._masks = outlier_segmenter(image)

        classified = {
            "sky": [segmenter._masks[0]],
            "planet": [segmenter._masks[1]],
            "exclude": [],
        }

        result = benchmark(segmenter._combine_masks, classified)

        assert "limb" in result
        assert len(result["limb"]) == 1000

    def test_sparse_mask_interpolation_benchmark(self, benchmark):
        """Benchmark interpolation with sparse masks."""
        from planet_ruler.image import MaskSegmenter

        image = np.random.randint(0, 255, (500, 1000, 3), dtype=np.uint8)

        def sparse_segmenter(img):
            h, w = img.shape[:2]
            sky_mask = np.zeros((h, w), dtype=bool)
            planet_mask = np.zeros((h, w), dtype=bool)

            # Only fill every 5th column
            for col in range(0, w, 5):
                sky_mask[:200, col] = True
                planet_mask[300:, col] = True

            return [
                {"mask": sky_mask, "area": np.sum(sky_mask)},
                {"mask": planet_mask, "area": np.sum(planet_mask)},
            ]

        segmenter = MaskSegmenter(
            image, method="custom", segment_fn=sparse_segmenter, interactive=False
        )
        segmenter._masks = sparse_segmenter(image)

        classified = {
            "sky": [segmenter._masks[0]],
            "planet": [segmenter._masks[1]],
            "exclude": [],
        }

        result = benchmark(segmenter._combine_masks, classified)

        assert "limb" in result
        assert not np.any(np.isnan(result["limb"]))

    @pytest.mark.slow
    def test_full_pipeline_benchmark_large_image(self, benchmark):
        """Benchmark full segmentation pipeline with large image."""
        from planet_ruler.image import MaskSegmenter

        image = np.random.randint(0, 255, (4000, 6000, 3), dtype=np.uint8)

        def comprehensive_segmenter(img):
            """More realistic segmentation with multiple regions."""
            h, w = img.shape[:2]
            masks = []

            # Sky region
            sky = np.zeros((h, w), dtype=bool)
            sky[: h // 3, :] = True
            masks.append({"mask": sky, "area": np.sum(sky)})

            # Planet region
            planet = np.zeros((h, w), dtype=bool)
            planet[h // 2 :, :] = True
            masks.append({"mask": planet, "area": np.sum(planet)})

            # Some noise regions
            for i in range(3):
                noise = np.zeros((h, w), dtype=bool)
                start_row = int((0.3 + i * 0.05) * h)
                end_row = int((0.35 + i * 0.05) * h)
                noise[start_row:end_row, ::10] = True
                masks.append({"mask": noise, "area": np.sum(noise)})

            return masks

        def run_full_pipeline():
            segmenter = MaskSegmenter(
                image,
                method="custom",
                segment_fn=comprehensive_segmenter,
                downsample_factor=2,  # Use downsampling for large image
                interactive=False,
            )
            return segmenter.segment()

        result = benchmark(run_full_pipeline)

        assert len(result) == 6000
        assert not np.all(np.isnan(result))


class TestSegmentationComparisonBenchmarks:
    """Compare performance of different segmentation configurations."""

    def test_downsampling_speedup_comparison(self, benchmark):
        """Benchmark segmentation with downsampling."""
        from planet_ruler.image import MaskSegmenter

        image = np.random.randint(0, 255, (2000, 2000, 3), dtype=np.uint8)

        def segmenter(img):
            h, w = img.shape[:2]
            sky = np.zeros((h, w), dtype=bool)
            sky[: h // 2, :] = True
            planet = np.zeros((h, w), dtype=bool)
            planet[h // 2 :, :] = True
            return [
                {"mask": sky, "area": np.sum(sky)},
                {"mask": planet, "area": np.sum(planet)},
            ]

        # Test with factor 2 (good balance of speed and accuracy)
        def run_with_downsampling():
            seg = MaskSegmenter(
                image,
                method="custom",
                segment_fn=segmenter,
                downsample_factor=2,
                interactive=False,
            )
            return seg.segment()

        result = benchmark(run_with_downsampling)

        # Verify result is valid
        assert len(result) == 2000  # Original width
        # Check that we don't have all NaN values (some valid limb points detected)
        import math

        assert not all(math.isnan(x) for x in result)


# Performance test markers
pytestmark = pytest.mark.benchmark
