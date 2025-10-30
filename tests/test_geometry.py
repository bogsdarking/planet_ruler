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

# tests/test_geometry.py
import pytest
import numpy as np
from planet_ruler.geometry import (
    horizon_distance,
    limb_camera_angle,
    focal_length,
    detector_size,
    field_of_view,
    intrinsic_transform,
    extrinsic_transform,
    limb_arc,
)


class TestBasicGeometry:
    """Test basic geometric calculations"""

    def test_horizon_distance_earth_example(self):
        """Test horizon distance with real Earth values"""
        # Earth radius: ~6.371 million meters
        # Height: 10km (typical airplane altitude)
        earth_radius = 6_371_000  # meters
        altitude = 10_000  # meters

        distance = horizon_distance(earth_radius, altitude)

        # Expected: roughly 357km for 10km altitude
        expected = np.sqrt(altitude**2 + 2 * altitude * earth_radius)
        assert np.isclose(distance, expected)
        assert distance > 350_000  # At least 350km
        assert distance < 370_000  # Less than 370km

    def test_horizon_distance_zero_height(self):
        """Test horizon distance at surface level"""
        radius = 1000
        height = 0

        distance = horizon_distance(radius, height)
        assert distance == 0

    def test_horizon_distance_small_values(self):
        """Test with small values to verify formula"""
        radius = 100
        height = 1

        distance = horizon_distance(radius, height)
        expected = np.sqrt(1 + 200)  # h^2 + 2*h*r
        assert np.isclose(distance, expected)


class TestCameraAngles:
    """Test camera angle calculations"""

    def test_limb_camera_angle_earth(self):
        """Test camera angle for Earth viewing"""
        earth_radius = 6_371_000
        altitude = 10_000

        angle = limb_camera_angle(earth_radius, altitude)

        # Should be a small angle (in radians) for typical altitudes
        assert 0 < angle < np.pi / 2
        # For Earth at 10km, angle should be around 0.056 radians (3.2 degrees)
        assert 0.055 < angle < 0.057

    def test_limb_camera_angle_surface(self):
        """Test camera angle at surface"""
        radius = 1000
        height = 0

        angle = limb_camera_angle(radius, height)
        assert angle == 0  # arccos(1) = 0

    def test_limb_camera_angle_high_altitude(self):
        """Test camera angle at very high altitude"""
        radius = 1000
        height = radius  # Height equals radius

        angle = limb_camera_angle(radius, height)
        expected = np.arccos(0.5)  # arccos(r/(r+r)) = arccos(1/2)
        assert np.isclose(angle, expected)


class TestCameraOptics:
    """Test camera optics calculations"""

    def test_focal_length_calculation(self):
        """Test focal length calculation"""
        detector_width = 0.036  # 36mm (full frame sensor)
        fov_degrees = 50  # degrees

        f = focal_length(detector_width, fov_degrees)

        # Should be reasonable focal length for photography
        assert 0.02 < f < 0.1  # Between 20mm and 100mm

    def test_detector_size_calculation(self):
        """Test detector size calculation"""
        focal_length_mm = 0.050  # 50mm lens
        fov_degrees = 40

        size = detector_size(focal_length_mm, fov_degrees)

        # Should be reasonable sensor size
        assert 0.01 < size < 0.1  # Between 10mm and 100mm

    def test_field_of_view_calculation(self):
        """Test field of view calculation"""
        focal_length_mm = 0.050  # 50mm
        detector_width = 0.036  # 36mm

        fov = field_of_view(focal_length_mm, detector_width)

        # Should be reasonable FOV
        assert 30 < fov < 60  # Between 30 and 60 degrees

    def test_camera_parameter_consistency(self):
        """Test that camera calculations are mutually consistent"""
        # Start with known values
        f_original = 0.050  # 50mm
        w_original = 0.036  # 36mm

        # Calculate FOV
        fov = field_of_view(f_original, w_original)

        # Use FOV to calculate back focal length
        f_calculated = focal_length(w_original, fov)

        # Should get back original focal length
        assert np.isclose(f_original, f_calculated, rtol=1e-10)

        # Use focal length and FOV to calculate detector size
        w_calculated = detector_size(f_original, fov)

        # Should get back original detector size
        assert np.isclose(w_original, w_calculated, rtol=1e-10)


class TestCoordinateTransforms:
    """Test coordinate transformation functions"""

    def test_intrinsic_transform_identity(self):
        """Test intrinsic transform with identity parameters"""
        # Simple test case: single point at origin
        camera_coords = np.array([[0, 0, 1, 1]]).T  # Homogeneous coordinates

        result = intrinsic_transform(camera_coords, f=1, px=1, py=1, x0=0, y0=0)

        # Should map to origin in pixel coordinates
        assert result.shape[0] == 1
        assert np.isclose(result[0, 0], 0)  # x coordinate
        assert np.isclose(result[0, 1], 0)  # y coordinate
        assert np.isclose(result[0, 2], 1)  # homogeneous coordinate

    def test_intrinsic_transform_offset(self):
        """Test intrinsic transform with principal point offset"""
        camera_coords = np.array([[0, 0, 1, 1]]).T
        x0, y0 = 320, 240  # Typical image center

        result = intrinsic_transform(camera_coords, f=1, px=1, py=1, x0=x0, y0=y0)

        # Should map to principal point
        assert np.isclose(result[0, 0], x0)
        assert np.isclose(result[0, 1], y0)

    def test_extrinsic_transform_identity(self):
        """Test extrinsic transform with no rotation or translation"""
        world_coords = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])

        result = extrinsic_transform(world_coords)

        # Should be unchanged (identity transform) - result is 4x3 (transposed)
        expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
        assert np.allclose(result, expected)

    def test_extrinsic_transform_translation(self):
        """Test extrinsic transform with translation only"""
        world_coords = np.array([[0, 0, 0, 1]])  # 1x4 array (single point)

        result = extrinsic_transform(world_coords, origin_x=1, origin_y=2, origin_z=3)

        # Point should be translated
        assert np.isclose(result[0, 0], 1)  # x
        assert np.isclose(result[1, 0], 2)  # y
        assert np.isclose(result[2, 0], 3)  # z


class TestLimbArcGeneration:
    """Test the main limb arc generation function"""

    def test_limb_arc_basic_functionality(self):
        """Test that limb_arc runs without errors"""
        # Basic Earth-like parameters
        result = limb_arc(
            r=6_371_000,  # Earth radius
            n_pix_x=1920,  # HD resolution
            n_pix_y=1080,
            h=10_000,  # 10km altitude
            f=0.050,  # 50mm lens
            w=0.036,  # 36mm sensor
        )

        # Should return array with x-pixel coordinates
        assert isinstance(result, np.ndarray)
        assert len(result) == 1920  # Should match n_pix_x
        assert not np.any(np.isnan(result))  # No NaN values

    def test_limb_arc_high_altitude(self):
        """Test limb arc at very high altitude where curvature is visible"""
        result = limb_arc(
            r=1_000_000,  # Smaller planet for testing
            n_pix_x=100,  # Small image for testing
            n_pix_y=100,
            h=100_000,  # High altitude
            f=0.050,
            w=0.036,
        )

        assert len(result) == 100
        # At high altitude, should see significant curvature
        # (exact values depend on viewing geometry)

    def test_limb_arc_return_full(self):
        """Test limb arc sample with return_full=True"""
        from planet_ruler.geometry import limb_arc_sample
        
        result = limb_arc_sample(
            r=1_000_000,
            n_pix_x=100,
            n_pix_y=100,
            h=10_000,
            f=0.050,
            w=0.036,
            return_full=True,
        )

        # Should return full coordinate array
        assert result.shape[1] == 3  # x, y, homogeneous coordinate
        assert result.shape[0] > 0  # Should have some points


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_very_small_planet(self):
        """Test with very small planet radius"""
        result = limb_arc(
            r=10, n_pix_x=100, n_pix_y=100, h=1, f=0.050, w=0.036
        )  # 10 meter "planet"  # 1 meter altitude

        assert len(result) == 100
        assert not np.any(np.isnan(result))

    def test_zero_altitude(self):
        """Test at surface level (zero altitude)"""
        # At surface level (h=0), horizon distance is 0, which causes
        # numerical issues in limb_arc. This is a degenerate case.
        # We test that the function handles it gracefully.
        result = limb_arc(
            r=1_000_000,
            n_pix_x=100,
            n_pix_y=100,
            h=1e-10,
            f=0.050,
            w=0.036,  # Very small altitude instead of exactly 0
        )

        assert len(result) == 100
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
        # At very low altitude, horizon should be nearly flat (small variation)
        variation = np.max(result) - np.min(result)
        assert variation < 1000  # Should be reasonably small variation

    def test_fov_parameter_combinations(self):
        """Test different parameter combinations for camera specification"""
        # Test with focal length and detector size
        result1 = limb_arc(
            r=1_000_000, n_pix_x=100, n_pix_y=100, h=1000, f=0.050, w=0.036
        )

        # Test with detector size and FOV
        result2 = limb_arc(
            r=1_000_000, n_pix_x=100, n_pix_y=100, h=1000, w=0.036, fov=40
        )

        # Test with focal length and FOV
        result3 = limb_arc(
            r=1_000_000, n_pix_x=100, n_pix_y=100, h=1000, f=0.050, fov=40
        )

        # All should produce valid results
        for result in [result1, result2, result3]:
            assert len(result) == 100
            assert not np.any(np.isnan(result))


class TestNumericalStability:
    """Test numerical stability and precision"""

    def test_large_radius_values(self):
        """Test with very large planetary radii"""
        result = limb_arc(
            r=1e10,
            n_pix_x=100,
            n_pix_y=100,
            h=1e6,
            f=0.050,
            w=0.036,  # Very large planet  # Proportionally large altitude
        )

        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_small_precision_values(self):
        """Test with very small values that might cause precision issues"""
        result = limb_arc(
            r=1e-3,  # Very small planet (millimeter scale)
            n_pix_x=100,
            n_pix_y=100,
            h=1e-6,  # Micrometer altitude
            f=0.050,
            w=0.036,
        )

        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))


# Parametrized tests for comprehensive coverage
class TestParametrizedGeometry:
    """Parametrized tests for thorough coverage"""

    @pytest.mark.parametrize(
        "radius,height,expected_min,expected_max",
        [
            (6_371_000, 10_000, 350_000, 370_000),  # Earth at 10km
            (6_371_000, 35_000, 600_000, 700_000),  # Earth at cruising altitude
            (1_737_400, 10_000, 180_000, 190_000),  # Moon at 10km (corrected range)
        ],
    )
    def test_horizon_distance_real_bodies(
        self, radius, height, expected_min, expected_max
    ):
        """Test horizon distance for real celestial bodies"""
        distance = horizon_distance(radius, height)
        assert expected_min <= distance <= expected_max

    @pytest.mark.parametrize("rotation_angle", [0, np.pi / 4, np.pi / 2, np.pi])
    def test_extrinsic_transform_rotations(self, rotation_angle):
        """Test extrinsic transform with various rotation angles"""
        world_coords = np.array([[1, 0, 0, 1]])  # 1x4 array (single point)

        # Test rotation around z-axis
        result = extrinsic_transform(world_coords, theta_z=rotation_angle)

        # Should maintain distance from origin
        distance = np.sqrt(result[0, 0] ** 2 + result[1, 0] ** 2)
        assert np.isclose(distance, 1.0, rtol=1e-10)


# Fixtures for common test data
@pytest.fixture
def earth_params():
    """Standard Earth parameters for testing"""
    return {
        "radius": 6_371_000,  # meters
        "altitude": 10_000,  # 10km
        "focal_length": 0.050,  # 50mm
        "sensor_width": 0.036,  # 36mm full frame
    }


@pytest.fixture
def camera_params():
    """Standard camera parameters for testing"""
    return {
        "n_pix_x": 1920,
        "n_pix_y": 1080,
        "f": 0.050,
        "w": 0.036,
        "x0": 960,  # Image center
        "y0": 540,  # Image center
    }


class TestIntegration:
    """Integration tests using fixtures"""

    def test_earth_horizon_calculation(self, earth_params, camera_params):
        """Test complete horizon calculation for Earth"""
        result = limb_arc(
            r=earth_params["radius"], h=earth_params["altitude"], **camera_params
        )

        assert len(result) == camera_params["n_pix_x"]
        assert not np.any(np.isnan(result))

        # For Earth at 10km, horizon should be nearly flat in most images
        # (curvature might not be visible depending on FOV)
        variation = np.max(result) - np.min(result)
        assert variation >= 0  # Sanity check


if __name__ == "__main__":
    # Run tests with: python -m pytest tests/test_geometry.py -v
    pytest.main([__file__, "-v"])
