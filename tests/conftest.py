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

# tests/conftest.py
"""
Pytest configuration and shared fixtures for planet_ruler tests.
"""
import pytest
import numpy as np

# Set numpy random seed for reproducible tests
np.random.seed(42)


@pytest.fixture(scope="session")
def earth_data():
    """Real Earth data for testing"""
    return {
        "radius": 6_371_000,  # meters
        "mass": 5.972e24,  # kg
        "surface_gravity": 9.81,  # m/s^2
    }


@pytest.fixture(scope="session")
def moon_data():
    """Real Moon data for testing"""
    return {
        "radius": 1_737_400,  # meters
        "mass": 7.342e22,  # kg
        "surface_gravity": 1.62,  # m/s^2
    }


@pytest.fixture(scope="session")
def mars_data():
    """Real Mars data for testing"""
    return {
        "radius": 3_389_500,  # meters
        "mass": 6.390e23,  # kg
        "surface_gravity": 3.71,  # m/s^2
    }


@pytest.fixture
def standard_camera():
    """Standard DSLR camera parameters"""
    return {
        "sensor_width": 0.036,  # 36mm (full frame)
        "sensor_height": 0.024,  # 24mm
        "focal_length": 0.050,  # 50mm lens
        "n_pix_x": 6000,  # 6K resolution
        "n_pix_y": 4000,
        "pixel_size": 6e-6,  # 6 micron pixels
    }


@pytest.fixture
def action_camera():
    """Action camera (GoPro-style) parameters"""
    return {
        "sensor_width": 0.006,  # Small sensor
        "sensor_height": 0.0045,
        "focal_length": 0.003,  # Wide angle
        "n_pix_x": 4000,
        "n_pix_y": 3000,
        "pixel_size": 1.5e-6,  # Small pixels
    }


@pytest.fixture
def flight_altitudes():
    """Common flight altitudes for testing"""
    return {
        "ground": 0,
        "drone": 120,  # meters (400 feet)
        "small_aircraft": 3_000,  # meters (10,000 feet)
        "airliner": 11_000,  # meters (36,000 feet)
        "u2_spy_plane": 21_000,  # meters (70,000 feet)
        "sr71": 26_000,  # meters (85,000 feet)
        "balloon": 40_000,  # meters (stratosphere)
    }


@pytest.fixture
def sample_horizon_image():
    """Generate a synthetic horizon image for testing"""

    def _generate_image(width=1920, height=1080, horizon_y=540, noise_level=0.1):
        """
        Generate a simple synthetic horizon image.

        Args:
            width, height: Image dimensions
            horizon_y: Y-coordinate of horizon line
            noise_level: Amount of random noise to add
        """
        image = np.zeros((height, width))

        # Sky (lighter)
        image[:horizon_y, :] = 0.7

        # Ground (darker)
        image[horizon_y:, :] = 0.3

        # Add some noise
        noise = np.random.normal(0, noise_level, (height, width))
        image = np.clip(image + noise, 0, 1)

        return image

    return _generate_image


@pytest.fixture
def known_test_cases():
    """Known test cases with expected results for validation"""
    return [
        {
            "name": "Earth_10km_50mm",
            "params": {
                "r": 6_371_000,
                "h": 10_000,
                "f": 0.050,
                "w": 0.036,
                "n_pix_x": 1920,
                "n_pix_y": 1080,
            },
            "expected_horizon_distance": 357_099,  # meters (calculated)
            "expected_limb_angle": 0.055992,  # radians (calculated)
        },
        {
            "name": "Moon_1km_28mm",
            "params": {
                "r": 1_737_400,
                "h": 1_000,
                "f": 0.028,
                "w": 0.036,
                "n_pix_x": 1920,
                "n_pix_y": 1080,
            },
            "expected_horizon_distance": 58_948,  # meters
            "expected_limb_angle": 0.00058,  # radians
        },
    ]


# Test data validation
def pytest_collection_modifyitems(config, items):
    """Add custom markers to tests"""
    for item in items:
        # Mark slow tests
        if "large_radius" in item.nodeid or "comprehensive" in item.nodeid:
            item.add_marker(pytest.mark.slow)

        # Mark integration tests
        if "integration" in item.nodeid.lower():
            item.add_marker(pytest.mark.integration)

        # Mark tests that require real data
        if "earth" in item.nodeid.lower() or "moon" in item.nodeid.lower():
            item.add_marker(pytest.mark.real_data)


# Custom markers
pytest_plugins = []


# Configure test markers
def pytest_configure(config):
    """Register custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line(
        "markers", "real_data: marks tests that use real planetary data"
    )
    config.addinivalue_line(
        "markers", "numerical: marks tests that focus on numerical precision"
    )
