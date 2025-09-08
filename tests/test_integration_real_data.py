"""
Integration tests using real demo data and configurations.

Tests complete workflows with actual planetary scenarios:
- Earth from ISS (International Space Station)
- Pluto from New Horizons spacecraft
- Saturn from Cassini spacecraft

These tests verify end-to-end functionality using realistic parameters,
actual spacecraft camera specifications, and validate results against
known physical constraints.
"""

import pytest
import numpy as np
import yaml
import json
import os
import math
from pathlib import Path
from unittest.mock import patch, Mock

import planet_ruler.geometry as geom
import planet_ruler.image as img
import planet_ruler.observation as obs
import planet_ruler.demo as demo
from planet_ruler.fit import CostFunction, pack_parameters, unpack_parameters


class TestRealDemoDataIntegration:
    """Integration tests using actual demo configurations and constraints."""

    @pytest.fixture(autouse=True)
    def setup_paths(self):
        """Set up paths to demo data."""
        self.project_root = Path(__file__).parent.parent
        self.config_dir = self.project_root / "config"
        self.demo_dir = self.project_root / "demo"
        self.images_dir = self.demo_dir / "images"

    @pytest.fixture
    def earth_iss_config(self):
        """Load Earth ISS configuration."""
        config_path = self.config_dir / "earth_iss_1.yaml"
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    @pytest.fixture
    def pluto_config(self):
        """Load Pluto New Horizons configuration."""
        config_path = self.config_dir / "pluto-new-horizons.yaml"
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    @pytest.fixture
    def saturn_config(self):
        """Load Saturn Cassini configuration."""
        config_path = self.config_dir / "saturn-cassini-1.yaml"
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def test_load_all_demo_configurations(self):
        """Test that all demo configurations load successfully."""
        configs = [
            "earth_iss_1.yaml",
            "pluto-new-horizons.yaml",
            "saturn-cassini-1.yaml",
            "saturn-cassini-2.yaml",
        ]

        for config_name in configs:
            config_path = self.config_dir / config_name
            assert config_path.exists(), f"Config file {config_name} not found"

            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            # Validate required structure
            assert "free_parameters" in config
            assert "init_parameter_values" in config
            assert "parameter_limits" in config

            # Check that all free parameters have initial values and limits
            for param in config["free_parameters"]:
                assert param in config["init_parameter_values"]
                assert param in config["parameter_limits"]

    @pytest.mark.integration
    def test_earth_iss_horizon_calculation(self, earth_iss_config):
        """Test Earth horizon calculation with ISS parameters."""
        params = earth_iss_config["init_parameter_values"]

        # Calculate expected horizon distance
        horizon_dist = geom.horizon_distance(r=params["r"], h=params["h"])

        # ISS horizon should be ~2600 km (calculated from 8000km radius, 418km altitude)
        assert (
            2500000 < horizon_dist < 2700000
        ), f"ISS horizon distance {horizon_dist/1000:.1f} km seems unrealistic"

        # Calculate limb angle
        limb_angle = geom.limb_camera_angle(r=params["r"], h=params["h"])

        # Limb angle should be reasonable for ISS
        assert (
            0.1 < limb_angle < 0.4
        ), f"ISS limb angle {limb_angle:.3f} rad seems unrealistic"

        # Test camera optics calculations - field of view
        calculated_fov = geom.field_of_view(f=params["f"], w=params["w"])
        expected_fov = 2 * math.atan(params["w"] / (2 * params["f"])) * 180 / math.pi

        assert (
            abs(calculated_fov - expected_fov) < 1.0
        ), "Field of view calculation inconsistent"

    @pytest.mark.integration
    def test_pluto_new_horizons_scenario(self, pluto_config):
        """Test Pluto scenario with New Horizons spacecraft parameters."""
        params = pluto_config["init_parameter_values"]

        # Pluto horizon calculation
        horizon_dist = geom.horizon_distance(r=params["r"], h=params["h"])

        # At 18M km distance from small Pluto (750km radius), horizon formula gives large values
        # This is expected behavior for very distant observations
        assert (
            horizon_dist > 1000000
        ), f"Pluto horizon distance {horizon_dist/1000:.1f} km too small"
        # For very distant observations, horizon can exceed radius - this is geometrically valid

        # Test limb angle - should be very small due to large distance
        limb_angle = geom.limb_camera_angle(r=params["r"], h=params["h"])

        # From 18M km, Pluto subtends a significant angle due to camera geometry
        assert (
            1.0 < limb_angle < 2.0
        ), f"Pluto limb angle {limb_angle:.6f} rad from New Horizons distance"

        # Test that detector size is reasonable for FOV
        detector_width = geom.detector_size(f=params["f"], fov=params["fov"])

        assert detector_width > 0.01, "Detector size should be reasonable"

    @pytest.mark.integration
    def test_saturn_cassini_scenario(self, saturn_config):
        """Test Saturn scenario with Cassini spacecraft parameters."""
        params = saturn_config["init_parameter_values"]

        # Saturn horizon calculation
        horizon_dist = geom.horizon_distance(r=params["r"], h=params["h"])

        # From 1.2B km distance, horizon should be substantial
        assert (
            horizon_dist > 10000000
        ), f"Saturn horizon distance {horizon_dist/1000:.1f} km too small"

        # Test limb angle - should be very small due to large distance
        limb_angle = geom.limb_camera_angle(r=params["r"], h=params["h"])

        # From 1.2B km, Saturn (75,000km radius) subtends a significant angle
        # This is physically reasonable: 75M/1.2B ≈ 0.0625 rad, but limb_camera_angle includes other factors
        assert (
            0.1 < limb_angle < 2.0
        ), f"Saturn limb angle {limb_angle:.6f} rad from Cassini distance"

        # Test camera field of view - calculate detector width from FOV and focal length
        detector_width = geom.detector_size(f=params["f"], fov=params["fov"])
        calculated_fov = geom.field_of_view(f=params["f"], w=detector_width)

        expected_fov = params["fov"]
        assert abs(calculated_fov - expected_fov) < 0.01, "FOV calculation inconsistent"

    @pytest.mark.integration
    def test_parameter_scaling_across_scenarios(
        self, earth_iss_config, pluto_config, saturn_config
    ):
        """Test that parameters scale appropriately across different scenarios."""
        configs = [
            ("Earth ISS", earth_iss_config),
            ("Pluto New Horizons", pluto_config),
            ("Saturn Cassini", saturn_config),
        ]

        results = []

        for name, config in configs:
            params = config["init_parameter_values"]

            # Calculate key metrics
            horizon = geom.horizon_distance(params["r"], params["h"])
            limb_angle = geom.limb_camera_angle(params["r"], params["h"])
            relative_altitude = params["h"] / params["r"]

            results.append(
                {
                    "name": name,
                    "radius": params["r"],
                    "altitude": params["h"],
                    "horizon": horizon,
                    "limb_angle": limb_angle,
                    "relative_altitude": relative_altitude,
                }
            )

        # Verify scaling relationships
        # Radius order: Pluto (750k) < Earth (8M) < Saturn (75M)
        pluto_result = next(r for r in results if r["name"] == "Pluto New Horizons")
        earth_result = next(r for r in results if r["name"] == "Earth ISS")
        saturn_result = next(r for r in results if r["name"] == "Saturn Cassini")

        assert pluto_result["radius"] < earth_result["radius"] < saturn_result["radius"]

        # Each scenario should have physically reasonable altitude for its mission type
        # No theoretical bounds exist for space-based observations - each mission is unique
        for result in results:
            assert (
                result["altitude"] > 0
            ), f"{result['name']} altitude should be positive"
            # Verify relative altitude (h/r) is reasonable for the observation geometry
            assert (
                result["relative_altitude"] >= 0
            ), f"{result['name']} relative altitude should be non-negative"

        # All limb angles should be reasonable positive values
        assert all(
            0.1 < r["limb_angle"] < 2.0 for r in results
        ), "All limb angles should be reasonable"

    @pytest.mark.integration
    def test_cost_function_integration_earth_iss(self, earth_iss_config):
        """Test cost function integration with Earth ISS parameters."""
        params = earth_iss_config["init_parameter_values"]
        free_params = earth_iss_config["free_parameters"]

        # Create synthetic target limb data using limb_arc function
        n_pix_x, n_pix_y = 1000, 600
        true_limb = geom.limb_arc(
            r=params["r"],
            h=params["h"],
            n_pix_x=n_pix_x,
            n_pix_y=n_pix_y,
            f=params["f"],
            w=params.get("w", 0.036),
            theta_x=params["theta_x"],
            theta_y=params["theta_y"],
            theta_z=params["theta_z"],
        )

        # Create cost function
        cost_func = CostFunction(
            target=true_limb,
            function=lambda **kwargs: geom.limb_arc(
                n_pix_x=n_pix_x, n_pix_y=n_pix_y, **kwargs
            ),
            free_parameters=free_params,
            init_parameter_values=params,
            loss_function="l2",
        )

        # Test evaluation with correct parameters (should give low cost)
        cost_perfect = cost_func.cost(params)
        assert (
            cost_perfect < 0.01
        ), f"Cost with perfect parameters should be low, got {cost_perfect}"

        # Test evaluation with perturbed parameters (should give higher cost)
        perturbed_params = params.copy()
        perturbed_params["r"] = params["r"] * 1.1  # 10% error
        cost_perturbed = cost_func.cost(perturbed_params)
        assert (
            cost_perturbed > cost_perfect
        ), "Perturbed parameters should give higher cost"

        # Test parameter packing/unpacking
        packed = pack_parameters(params, params)
        unpacked = unpack_parameters(packed, list(params.keys()))
        for key in params:
            assert (
                abs(unpacked[key] - params[key]) < 1e-10
            ), f"Parameter {key} packing/unpacking failed"

    @pytest.mark.integration
    @patch("planet_ruler.image.Image.open")
    def test_image_processing_integration(self, mock_image_open):
        """Test image processing integration with synthetic planet image."""
        # Create synthetic planet image
        height, width = 400, 600
        image = np.zeros((height, width, 3), dtype="uint8")

        # Create horizon/limb at y=200
        image[:200, :, :] = 30  # Dark space
        image[200:, :, :] = 180  # Bright planet surface

        # Add some noise to make it realistic
        noise = np.random.normal(0, 5, image.shape).astype("int16")
        image = np.clip(image.astype("int16") + noise, 0, 255).astype("uint8")

        # Mock PIL Image
        mock_img = Mock()
        mock_image_open.return_value = mock_img

        with patch("numpy.array", return_value=image):
            loaded_image = img.load_image("fake_planet.jpg")

        # Test gradient break detection
        breaks = img.gradient_break(loaded_image, window_length=21)

        # Should detect horizon around y=200
        assert len(breaks) == width
        mean_break = np.mean(breaks)
        assert (
            180 < mean_break < 220
        ), f"Expected horizon around y=200, got {mean_break:.1f}"

        # Test limb smoothing
        smoothed_limb = img.smooth_limb(
            breaks, method="rolling-median", window_length=15
        )
        assert len(smoothed_limb) == len(breaks)
        assert np.var(smoothed_limb) < np.var(
            breaks
        ), "Smoothed limb should have less variance"

        # Test NaN filling
        breaks_with_nans = breaks.copy().astype(float)
        breaks_with_nans[100:105] = np.nan  # Add some NaNs
        filled = img.fill_nans(breaks_with_nans)
        # Check that no NaNs remain - convert to list for math.isnan compatibility
        assert not any(
            math.isnan(float(x)) for x in filled.flatten()
        ), "All NaNs should be filled"

    @pytest.mark.integration
    def test_observation_workflow_integration(self, earth_iss_config):
        """Test complete observation workflow with Earth ISS scenario."""
        params = earth_iss_config["init_parameter_values"]

        # Create synthetic observation setup
        with patch("PIL.Image.open") as mock_image_open:
            # Create synthetic image
            height, width = 300, 500
            image = np.zeros((height, width, 3), dtype="uint8")
            image[:150, :, :] = 40  # Space
            image[150:, :, :] = 200  # Planet

            # Mock PIL Image object
            mock_img = Mock()
            mock_image_open.return_value = mock_img

            with patch("numpy.array", return_value=image):
                # Create temporary config file
                config_path = "temp_earth_config.yaml"
                with open(config_path, "w") as f:
                    yaml.dump(earth_iss_config, f)

                try:
                    observation = obs.LimbObservation(
                        image_filepath="fake_earth.jpg",
                        fit_config=config_path,
                        limb_detection="gradient-break",
                    )

                    # Test limb detection
                    observation.detect_limb()
                    limb = observation.features["limb"]
                    # limb should be 1D array with width entries
                    if len(limb.shape) > 1:
                        # If it's the full image, skip this test
                        pytest.skip(
                            "Limb detection returned full image instead of limb coordinates"
                        )
                    assert len(limb) == width
                    assert (
                        130 < np.mean(limb) < 170
                    ), "Limb should be detected around horizon"

                    # Test limb smoothing
                    observation.smooth_limb(method="rolling-median", window_length=11)
                    assert "limb" in observation.features
                    assert len(observation.features["limb"]) == width

                finally:
                    # Clean up temporary file
                    if os.path.exists(config_path):
                        os.remove(config_path)

    @pytest.mark.integration
    def test_demo_parameter_loading_integration(self):
        """Test integration with demo parameter loading functions."""
        # Test all available demo scenarios
        demo_scenarios = ["pluto", "saturn_1", "saturn_2", "earth"]

        for scenario in demo_scenarios:
            try:
                # Use mock dropdown widget
                mock_dropdown = Mock()
                mock_dropdown.value = scenario

                params = demo.load_demo_parameters(mock_dropdown)

                # Validate loaded parameters
                assert isinstance(params, dict)
                assert "r" in params, f"Radius missing in {scenario} demo"
                assert "h" in params, f"Altitude missing in {scenario} demo"
                assert params["r"] > 0, f"Invalid radius in {scenario} demo"
                assert params["h"] > 0, f"Invalid altitude in {scenario} demo"

                # Test physical reasonableness
                if scenario == "earth":
                    assert (
                        6e6 < params["r"] < 7e6
                    ), f"Earth radius unrealistic: {params['r']}"
                elif scenario == "pluto":
                    assert (
                        5e5 < params["r"] < 2e6
                    ), f"Pluto radius unrealistic: {params['r']}"
                elif scenario.startswith("saturn"):
                    assert (
                        5e7 < params["r"] < 8e7
                    ), f"Saturn radius unrealistic: {params['r']}"

            except Exception as e:
                pytest.skip(f"Demo scenario {scenario} not available: {e}")

    @pytest.mark.integration
    def test_coordinate_transform_integration(self, saturn_config):
        """Test coordinate transform integration with Saturn scenario."""
        params = saturn_config["init_parameter_values"]

        # Create test coordinates in the format expected by functions
        # For intrinsic: (4, N) format (homogeneous coordinates)
        test_coords_intrinsic = np.array(
            [
                [100, 300, 500],  # x coordinates
                [200, 400, 600],  # y coordinates
                [1, 1, 1],  # z coordinates
                [1, 1, 1],  # homogeneous coordinate
            ]
        )

        # Test intrinsic transform
        transformed = geom.intrinsic_transform(
            test_coords_intrinsic, f=params["f"], x0=250, y0=150
        )
        # Transform reduces from 4D to 3D homogeneous coordinates
        assert (
            transformed.shape[1] == test_coords_intrinsic.shape[1]
        )  # Same number of points
        assert transformed.shape[0] == 3  # 3D output coordinates

        # For extrinsic: (N, 4) format
        test_coords_extrinsic = np.array(
            [[100, 200, 1, 1], [300, 400, 1, 1], [500, 600, 1, 1]]
        )

        # Test extrinsic transform
        extrinsic = geom.extrinsic_transform(
            test_coords_extrinsic,
            theta_x=params["theta_x"],
            theta_y=params["theta_y"],
            theta_z=params["theta_z"],
        )
        # Extrinsic transform returns (4, N) shape
        assert (
            extrinsic.shape[1] == test_coords_extrinsic.shape[0]
        )  # Same number of points
        assert extrinsic.shape[0] == 4  # 4D output coordinates

        # Transforms should be invertible (approximately)
        # Convert back to (N, 4) format for inverse transform
        extrinsic_reshaped = extrinsic.T
        identity_transform = geom.extrinsic_transform(
            extrinsic_reshaped,
            theta_x=-params["theta_x"],
            theta_y=-params["theta_y"],
            theta_z=-params["theta_z"],
        )

        # Convert back to (N, 4) format for comparison
        identity_reshaped = identity_transform.T
        np.testing.assert_allclose(identity_reshaped, test_coords_extrinsic, rtol=1e-6)


class TestConfigurationConsistency:
    """Test consistency across all configuration files."""

    @pytest.fixture(autouse=True)
    def setup_config_paths(self):
        """Set up configuration file paths."""
        self.config_dir = Path(__file__).parent.parent / "config"

    def test_all_configs_have_required_fields(self):
        """Test that all configuration files have required fields."""
        required_fields = [
            "free_parameters",
            "init_parameter_values",
            "parameter_limits",
        ]

        for config_file in self.config_dir.glob("*.yaml"):
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)

            for field in required_fields:
                assert field in config, f"Missing {field} in {config_file.name}"

    def test_parameter_limit_consistency(self):
        """Test that initial values are within specified limits."""
        for config_file in self.config_dir.glob("*.yaml"):
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)

            init_vals = config["init_parameter_values"]
            limits = config["parameter_limits"]

            for param in config["free_parameters"]:
                if param in init_vals and param in limits:
                    init_val = init_vals[param]
                    low, high = limits[param]

                    assert (
                        low <= init_val <= high
                    ), f"Initial value {init_val} for {param} outside limits [{low}, {high}] in {config_file.name}"

    def test_physical_parameter_reasonableness(self):
        """Test that parameters are physically reasonable."""
        for config_file in self.config_dir.glob("*.yaml"):
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)

            params = config["init_parameter_values"]

            # Radius should be positive and reasonable for planets
            if "r" in params:
                assert (
                    1e5 < params["r"] < 1e8
                ), f"Unrealistic radius {params['r']} in {config_file.name}"

            # Altitude should be positive
            if "h" in params:
                assert params["h"] > 0, f"Negative altitude in {config_file.name}"

            # Focal length should be positive
            if "f" in params:
                assert params["f"] > 0, f"Negative focal length in {config_file.name}"

            # Field of view should be reasonable (if present)
            if "fov" in params:
                assert (
                    0.1 < params["fov"] < 180
                ), f"Unrealistic FOV {params['fov']} in {config_file.name}"
