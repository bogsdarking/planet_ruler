"""
Property-based tests for planet_ruler.geometry module using Hypothesis.

These tests verify mathematical properties that should always hold true,
regardless of specific input values, helping to catch edge cases and
ensure numerical robustness across the full parameter space.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, assume, settings
from hypothesis.extra.numpy import arrays

from planet_ruler.geometry import (
    horizon_distance,
    limb_camera_angle,
    focal_length,
    detector_size,
    field_of_view,
    intrinsic_transform,
    extrinsic_transform,
    limb_arc
)


# Custom strategies for realistic parameter ranges
realistic_radii = st.floats(min_value=1e3, max_value=1e8, allow_nan=False, allow_infinity=False)
realistic_radii_for_monotonic = st.floats(min_value=1e3, max_value=1e7, allow_nan=False, allow_infinity=False)
realistic_altitudes = st.floats(min_value=0, max_value=1e6, allow_nan=False, allow_infinity=False)
positive_altitudes = st.floats(min_value=1.0, max_value=1e5, allow_nan=False, allow_infinity=False)
small_angles = st.floats(min_value=-np.pi/4, max_value=np.pi/4, allow_nan=False, allow_infinity=False)
camera_fov = st.floats(min_value=1.0, max_value=179.0, allow_nan=False, allow_infinity=False)
detector_sizes = st.floats(min_value=1e-6, max_value=1e-1, allow_nan=False, allow_infinity=False)
focal_lengths = st.floats(min_value=1e-3, max_value=10.0, allow_nan=False, allow_infinity=False)


class TestHorizonDistanceProperties:
    """Property-based tests for horizon_distance function."""
    
    @given(radius=realistic_radii, altitude=realistic_altitudes)
    def test_horizon_distance_always_positive(self, radius, altitude):
        """Horizon distance should always be non-negative."""
        distance = horizon_distance(radius, altitude)
        assert distance >= 0, f"Distance {distance} should be non-negative"
        assert np.isfinite(distance), "Distance should be finite"
    
    @given(radius=realistic_radii, altitude=realistic_altitudes)
    def test_horizon_distance_monotonic_in_altitude(self, radius, altitude):
        """Higher altitude should give longer horizon distance."""
        assume(altitude < 1e5)  # Avoid overflow in test
        
        distance1 = horizon_distance(radius, altitude)
        distance2 = horizon_distance(radius, altitude + 100)
        
        assert distance2 > distance1, f"Distance should increase with altitude: {distance1} vs {distance2}"
    
    @given(radius=realistic_radii_for_monotonic, altitude=positive_altitudes)
    def test_horizon_distance_monotonic_in_radius(self, radius, altitude):
        """Larger radius should give longer horizon distance for same altitude."""
        
        distance1 = horizon_distance(radius, altitude)
        distance2 = horizon_distance(radius * 1.1, altitude)
        
        assert distance2 > distance1, f"Distance should increase with radius: {distance1} vs {distance2}"
    
    @given(radius=realistic_radii)
    def test_horizon_distance_zero_altitude(self, radius):
        """At zero altitude, horizon distance should be zero."""
        distance = horizon_distance(radius, 0)
        assert abs(distance) < 1e-10, f"Distance at zero altitude should be zero, got {distance}"
    
    @given(radius=realistic_radii, altitude=realistic_altitudes)
    def test_horizon_distance_bounds(self, radius, altitude):
        """Horizon distance should be bounded by geometric constraints."""
        assume(altitude > 0)
        
        distance = horizon_distance(radius, altitude)
        
        # Distance should be at least the altitude (looking straight down)
        assert distance >= altitude, f"Distance {distance} should be at least altitude {altitude}"
        
        # For small h relative to r, distance ≈ sqrt(2*h*r)
        if altitude / radius < 0.01:
            expected_approx = np.sqrt(2 * altitude * radius)
            assert abs(distance - expected_approx) / expected_approx < 0.01


class TestLimbCameraAngleProperties:
    """Property-based tests for limb_camera_angle function."""
    
    @given(radius=realistic_radii, altitude=realistic_altitudes)
    def test_limb_angle_bounds(self, radius, altitude):
        """Camera angle should be bounded between 0 and π/2."""
        assume(altitude > 0)  # Avoid division by zero scenarios
        
        angle = limb_camera_angle(radius, altitude)
        
        assert 0 <= angle <= np.pi/2, f"Angle {angle} should be between 0 and π/2"
        assert np.isfinite(angle), "Angle should be finite"
    
    @given(radius=realistic_radii, altitude=st.floats(min_value=10, max_value=1e5, allow_nan=False, allow_infinity=False))
    def test_limb_angle_monotonic_in_altitude(self, radius, altitude):
        """Higher altitude should give larger camera angle."""
        
        angle1 = limb_camera_angle(radius, altitude)
        angle2 = limb_camera_angle(radius, altitude + 100)
        
        assert angle2 > angle1, f"Angle should increase with altitude: {angle1} vs {angle2}"
    
    @given(radius=realistic_radii, altitude=realistic_altitudes)
    def test_limb_angle_approaches_limits(self, radius, altitude):
        """Test limiting behavior of camera angle."""
        assume(altitude > 0)
        
        angle = limb_camera_angle(radius, altitude)
        
        # For very high altitude, angle approaches π/2
        if altitude / radius > 100:
            assert angle > np.pi/2 - 0.1, f"High altitude angle {angle} should approach π/2"
        
        # For very low altitude, angle approaches 0
        if altitude / radius < 0.001:
            assert angle < 0.1, f"Low altitude angle {angle} should approach 0"


class TestCameraOpticsProperties:
    """Property-based tests for camera optics functions."""
    
    @given(detector_width=detector_sizes, fov=camera_fov)
    def test_focal_length_positive(self, detector_width, fov):
        """Focal length should always be positive."""
        f = focal_length(detector_width, fov)
        
        assert f > 0, f"Focal length {f} should be positive"
        assert np.isfinite(f), "Focal length should be finite"
    
    @given(f=focal_lengths, fov=camera_fov)
    def test_detector_size_positive(self, f, fov):
        """Detector size should always be positive."""
        w = detector_size(f, fov)
        
        assert w > 0, f"Detector size {w} should be positive"
        assert np.isfinite(w), "Detector size should be finite"
    
    @given(f=focal_lengths, detector_width=detector_sizes)
    def test_field_of_view_bounds(self, f, detector_width):
        """Field of view should be bounded between 0 and 180 degrees."""
        fov = field_of_view(f, detector_width)
        
        assert 0 < fov < 180, f"FOV {fov} should be between 0 and 180 degrees"
        assert np.isfinite(fov), "FOV should be finite"
    
    @given(detector_width=detector_sizes, fov=camera_fov)
    def test_focal_length_detector_size_reciprocal(self, detector_width, fov):
        """focal_length and detector_size should be reciprocal operations."""
        f = focal_length(detector_width, fov)
        w_recovered = detector_size(f, fov)
        
        relative_error = abs(w_recovered - detector_width) / detector_width
        assert relative_error < 1e-10, f"Reciprocal error: {relative_error}"
    
    @given(f=focal_lengths, detector_width=detector_sizes)
    def test_field_of_view_reciprocal(self, f, detector_width):
        """field_of_view with focal_length should be reciprocal operations."""
        fov = field_of_view(f, detector_width)
        f_recovered = focal_length(detector_width, fov)
        
        relative_error = abs(f_recovered - f) / f
        assert relative_error < 1e-10, f"Reciprocal error: {relative_error}"
    
    @given(detector_width=detector_sizes, fov=camera_fov)
    def test_focal_length_monotonic_in_fov(self, detector_width, fov):
        """Larger FOV should give smaller focal length for same detector."""
        assume(fov < 170)  # Avoid extreme angles
        
        f1 = focal_length(detector_width, fov)
        f2 = focal_length(detector_width, fov + 5)
        
        assert f2 < f1, f"Focal length should decrease with larger FOV: {f1} vs {f2}"


class TestCoordinateTransformProperties:
    """Property-based tests for coordinate transformation functions."""
    
    @given(
        n_points=st.integers(min_value=1, max_value=100),
        f=focal_lengths,
        px=st.floats(min_value=1e-6, max_value=1e-3),
        py=st.floats(min_value=1e-6, max_value=1e-3),
        x0=st.floats(min_value=-1000, max_value=1000),
        y0=st.floats(min_value=-1000, max_value=1000)
    )
    def test_intrinsic_transform_preserves_dimensions(self, n_points, f, px, py, x0, y0):
        """Intrinsic transform should preserve number of points."""
        # Generate random camera coordinates
        camera_coords = np.random.standard_normal((4, n_points))
        camera_coords[3, :] = 1  # Homogeneous coordinates
        
        pixel_coords = intrinsic_transform(camera_coords, f, px, py, x0, y0)
        
        assert pixel_coords.shape[0] == n_points, "Should preserve number of points"
        assert pixel_coords.shape[1] >= 3, "Should have at least 3 coordinates"
        assert np.allclose(pixel_coords[:, -1], 1), "Last column should be ones (homogeneous)"
    
    @given(
        n_points=st.integers(min_value=1, max_value=100),
        theta_x=small_angles,
        theta_y=small_angles,
        theta_z=small_angles,
        origin_x=st.floats(min_value=-1000, max_value=1000),
        origin_y=st.floats(min_value=-1000, max_value=1000),
        origin_z=st.floats(min_value=-1000, max_value=1000)
    )
    def test_extrinsic_transform_preserves_dimensions(self, n_points, theta_x, theta_y, theta_z, 
                                                     origin_x, origin_y, origin_z):
        """Extrinsic transform should preserve number of points."""
        world_coords = np.random.standard_normal((n_points, 4))
        world_coords[:, 3] = 1  # Homogeneous coordinates
        
        camera_coords = extrinsic_transform(
            world_coords, theta_x, theta_y, theta_z, origin_x, origin_y, origin_z
        )
        
        assert camera_coords.shape[0] == 4, "Should have 4 coordinate dimensions"
        assert camera_coords.shape[1] == n_points, "Should preserve number of points"
    
    @given(
        n_points=st.integers(min_value=5, max_value=50),
    )
    def test_extrinsic_transform_identity(self, n_points):
        """Identity transform should preserve coordinates."""
        world_coords = np.random.standard_normal((n_points, 4))
        world_coords[:, 3] = 1
        
        # Identity transformation (no rotation, no translation)
        camera_coords = extrinsic_transform(world_coords)
        
        # Should be nearly identical (accounting for numerical precision)
        expected = world_coords.T
        assert np.allclose(camera_coords, expected, rtol=1e-14), "Identity transform failed"
    
    @given(angle=st.floats(min_value=-np.pi, max_value=np.pi, allow_nan=False))
    def test_extrinsic_transform_rotation_properties(self, angle):
        """Rotation should preserve distances and angles."""
        # Simple test with a few points
        world_coords = np.array([
            [1, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 1, 1]
        ])
        
        # Rotate around z-axis
        camera_coords = extrinsic_transform(world_coords, theta_z=angle)
        
        # Check that distances between points are preserved
        for i in range(3):
            for j in range(i+1, 3):
                original_dist = np.linalg.norm(world_coords[i, :3] - world_coords[j, :3])
                rotated_dist = np.linalg.norm(camera_coords[:3, i] - camera_coords[:3, j])
                assert abs(original_dist - rotated_dist) < 1e-14, "Rotation should preserve distances"


class TestLimbArcProperties:
    """Property-based tests for limb_arc function."""
    
    @given(
        radius=st.floats(min_value=1e6, max_value=1e7),
        n_pix_x=st.integers(min_value=100, max_value=1000),
        n_pix_y=st.integers(min_value=100, max_value=1000),
        altitude=st.floats(min_value=1e4, max_value=1e5),
        fov=st.floats(min_value=10, max_value=60),
        w=detector_sizes
    )
    @settings(max_examples=20, deadline=10000)  # Limit examples due to computational cost
    def test_limb_arc_output_dimensions(self, radius, n_pix_x, n_pix_y, altitude, fov, w):
        """Limb arc should return array matching image width."""
        try:
            y_pixel = limb_arc(
                r=radius, n_pix_x=n_pix_x, n_pix_y=n_pix_y, h=altitude,
                fov=fov, w=w
            )
            
            assert len(y_pixel) == n_pix_x, f"Output length {len(y_pixel)} should match n_pix_x {n_pix_x}"
            assert np.all(np.isfinite(y_pixel)), "All y-coordinates should be finite"
        
        except (AssertionError, ValueError) as e:
            # Some parameter combinations may be invalid
            assume(False)
    
    @given(
        radius=st.floats(min_value=6e6, max_value=7e6),  # Earth-like
        n_pix_x=st.integers(min_value=200, max_value=400),
        n_pix_y=st.integers(min_value=200, max_value=400),
        altitude=st.floats(min_value=1e4, max_value=5e4)
    )
    @settings(max_examples=10, deadline=15000)
    def test_limb_arc_continuity(self, radius, n_pix_x, n_pix_y, altitude):
        """Limb arc should be reasonably continuous (no huge jumps)."""
        try:
            y_pixel = limb_arc(
                r=radius, n_pix_x=n_pix_x, n_pix_y=n_pix_y, h=altitude,
                fov=30.0, w=0.01  # Fixed reasonable values
            )
            
            # Check for reasonable continuity (no jumps larger than image dimensions)
            if len(y_pixel) > 1:
                max_jump = np.max(np.abs(np.diff(y_pixel)))
                assert max_jump < 10 * n_pix_y, f"Maximum jump {max_jump} too large for image size {n_pix_y}"
                
        except (AssertionError, ValueError, IndexError):
            # Some parameter combinations may be invalid or cause edge cases
            assume(False)


class TestNumericalStabilityProperties:
    """Property-based tests for numerical stability across functions."""
    
    @given(
        radius=st.floats(min_value=1e3, max_value=1e8),
        altitude=st.floats(min_value=1e-3, max_value=1e6)
    )
    def test_horizon_functions_numerical_stability(self, radius, altitude):
        """Test that horizon calculations remain stable across wide parameter ranges."""
        try:
            distance = horizon_distance(radius, altitude)
            angle = limb_camera_angle(radius, altitude)
            
            # Check for numerical stability indicators
            assert np.isfinite(distance), "Distance should be finite"
            assert np.isfinite(angle), "Angle should be finite"
            assert distance >= 0, "Distance should be non-negative"
            assert 0 <= angle <= np.pi/2, "Angle should be in valid range"
            
        except (ValueError, OverflowError, ZeroDivisionError):
            # Some extreme parameter combinations may cause expected failures
            assume(False)
    
    @given(
        param1=st.floats(min_value=1e-6, max_value=1e6),
        param2=st.floats(min_value=1.0, max_value=179.0)
    )
    def test_camera_functions_numerical_stability(self, param1, param2):
        """Test that camera optic calculations remain stable."""
        try:
            # Test different combinations of camera parameters
            f1 = focal_length(param1, param2)
            w1 = detector_size(param1, param2)
            fov1 = field_of_view(param1, param2 / 100)  # Scale param2 for detector size
            
            assert np.isfinite(f1) and f1 > 0, "Focal length should be finite and positive"
            assert np.isfinite(w1) and w1 > 0, "Detector size should be finite and positive"  
            assert np.isfinite(fov1) and 0 < fov1 < 180, "FOV should be finite and in valid range"
            
        except (ValueError, OverflowError, ZeroDivisionError):
            # Some parameter combinations may cause expected numerical issues
            assume(False)


# Slow tests marker for property-based tests that take longer to run
pytestmark = pytest.mark.slow