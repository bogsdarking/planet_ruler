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
from planet_ruler.fit import estimate_radius_via_sagitta, SagittaFitter


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


def _make_synthetic_limb(
    r=6_371_000,
    h=10_000,
    f=0.026,
    w=0.0173,
    W=4000,
    H=3000,
    n_points=12,
    noise_std=0.0,
    seed=42,
):
    """
    Generate a sparse limb array by sampling a synthetic arc at n_points
    evenly spaced columns (avoiding the outermost 10% to stay well within FOV).
    """
    theta_x = limb_camera_angle(r, h)
    y_arc = limb_arc(
        r=r, n_pix_x=W, n_pix_y=H, h=h, f=f, w=w,
        x0=W // 2, y0=H // 2,
        theta_x=theta_x, theta_y=0, theta_z=np.pi,
    )
    rng = np.random.default_rng(seed)
    margin = W // 10
    xs = np.linspace(margin, W - margin - 1, n_points, dtype=int)
    limb = np.full(W, np.nan)
    for x in xs:
        limb[x] = y_arc[x] + rng.normal(0, noise_std)
    return limb


class TestEstimateRadiusFromLimbArc:
    """Tests for estimate_radius_via_sagitta."""

    # --- accuracy across altitudes ---

    @pytest.mark.parametrize("h", [1_000, 10_000, 50_000])
    def test_clean_arc_within_5_percent(self, h):
        r_true = 6_371_000
        limb = _make_synthetic_limb(r=r_true, h=h, noise_std=0.0)
        f, w, W = 0.026, 0.0173, 4000
        f_px = f * W / w
        est = estimate_radius_via_sagitta(limb, h=h, f_px=f_px, sigma_px=1.0)
        assert est["status"] == "ok"
        assert abs(est["r"] - r_true) / r_true < 0.05

    def test_3sigma_bounds_contain_truth_under_noise(self):
        """3-sigma bounds (sigma_px = true noise) must bracket r_true."""
        r_true = 6_371_000
        h = 10_000
        noise_std = 1.0
        limb = _make_synthetic_limb(
            r=r_true, h=h, noise_std=noise_std, seed=42
        )
        f, w, W = 0.026, 0.0173, 4000
        f_px = f * W / w
        est = estimate_radius_via_sagitta(
            limb, h=h, f_px=f_px, sigma_px=noise_std, n_sigma=3.0
        )
        assert est["status"] == "ok"
        assert est["r_low"] <= r_true <= est["r_high"]

    # --- bounds sanity ---

    def test_bounds_bracket_estimate(self):
        limb = _make_synthetic_limb()
        f_px = 0.026 * 4000 / 0.0173
        est = estimate_radius_via_sagitta(
            limb, h=10_000, f_px=f_px, sigma_px=1.0
        )
        assert est["r_low"] < est["r"] < est["r_high"]

    def test_wider_sigma_gives_wider_bounds(self):
        limb = _make_synthetic_limb()
        f_px = 0.026 * 4000 / 0.0173
        narrow = estimate_radius_via_sagitta(
            limb, h=10_000, f_px=f_px, sigma_px=1.0
        )
        wide = estimate_radius_via_sagitta(
            limb, h=10_000, f_px=f_px, sigma_px=5.0
        )
        narrow_width = narrow["r_high"] - narrow["r_low"]
        wide_width = wide["r_high"] - wide["r_low"]
        assert wide_width > narrow_width

    # --- n_sigma scaling ---

    def test_n_sigma_scales_auto_bounds(self):
        limb = _make_synthetic_limb()
        f_px = 0.026 * 4000 / 0.0173
        one = estimate_radius_via_sagitta(
            limb, h=10_000, f_px=f_px, sigma_px="auto", n_sigma=1.0
        )
        two = estimate_radius_via_sagitta(
            limb, h=10_000, f_px=f_px, sigma_px="auto", n_sigma=2.0
        )
        assert (two["r_high"] - two["r_low"]) > (one["r_high"] - one["r_low"])

    def test_n_sigma_scales_bounds_for_explicit_sigma(self):
        """n_sigma widens bounds even when sigma_px is an explicit float."""
        limb = _make_synthetic_limb()
        f_px = 0.026 * 4000 / 0.0173
        a = estimate_radius_via_sagitta(
            limb, h=10_000, f_px=f_px, sigma_px=2.0, n_sigma=1.0
        )
        b = estimate_radius_via_sagitta(
            limb, h=10_000, f_px=f_px, sigma_px=2.0, n_sigma=3.0
        )
        # Point estimate and 1-sigma K_sigma are unchanged; bounds widen
        assert np.isclose(a["r"], b["r"])
        assert np.isclose(a["K_sigma"], b["K_sigma"])
        assert (b["r_high"] - b["r_low"]) > (a["r_high"] - a["r_low"])

    # --- auto sigma ---

    def test_auto_sigma_uses_residual_rms(self):
        limb = _make_synthetic_limb(noise_std=2.0)
        f_px = 0.026 * 4000 / 0.0173
        est = estimate_radius_via_sagitta(
            limb, h=10_000, f_px=f_px, sigma_px="auto"
        )
        assert est["status"] == "ok"
        assert est["residual_rms"] > 0
        # K_sigma = K_est * rms / sqrt(sum(s_i^2)) — must be finite and positive
        assert np.isfinite(est["K_sigma"])
        assert est["K_sigma"] > 0

    # --- failure modes ---

    def test_too_few_points(self):
        limb = np.full(4000, np.nan)
        limb[100] = 500.0
        limb[200] = 502.0
        limb[300] = 503.0  # only 3 points
        f_px = 0.026 * 4000 / 0.0173
        est = estimate_radius_via_sagitta(limb, h=10_000, f_px=f_px)
        assert est["status"] == "too_few_points"
        assert np.isnan(est["r"])

    def test_flat_arc_parabola(self):
        limb = np.full(400, np.nan)
        xs = np.linspace(40, 360, 10, dtype=int)
        for x in xs:
            limb[x] = 200.0  # perfectly flat line
        f_px = 0.026 * 400 / 0.0173
        est = estimate_radius_via_sagitta(
            limb, h=10_000, f_px=f_px, sigma_px=1.0
        )
        assert est["status"] == "flat_arc"

    # --- return dict completeness ---

    def test_return_dict_keys(self):
        limb = _make_synthetic_limb()
        f_px = 0.026 * 4000 / 0.0173
        est = estimate_radius_via_sagitta(limb, h=10_000, f_px=f_px)
        expected_keys = {
            "r", "r_low", "r_high", "r_sigma",
            "K", "K_sigma", "K_sigma_jack", "n_points", "residual_rms",
            "arc_angle_deg", "x_apex", "y_apex",
            "theta_x_est", "status", "warnings",
        }
        assert expected_keys == set(est.keys())

    def test_n_points_matches_annotated(self):
        limb = _make_synthetic_limb(n_points=8)
        f_px = 0.026 * 4000 / 0.0173
        est = estimate_radius_via_sagitta(limb, h=10_000, f_px=f_px)
        assert est["n_points"] == 8

    # --- hyperbola model accuracy ---

    def test_hyperbola_horizontal_camera_zero_bias(self):
        """theta_x=0: hyperbola fit is exact → bias < 0.01% on a clean arc.

        At theta_x=0 the arc is s = kappa*A(u), so the OLS s = s0 - c*A(u)
        recovers c = -kappa exactly and K = 1/|c| has zero residual.  This
        test uses the synth-benchmark camera (f=5.1mm, w=7.6mm) where the
        old chordspan formula gave -48.7 km bias at theta_x=0 due to the
        wide-angle (L/f_px≈0.56) parabola approximation error.
        """
        r_true = 6_371_000
        h = 10_000
        f_mm, w_mm = 5.1, 7.6
        n_pix_x = 4000
        f_px = f_mm / w_mm * n_pix_x
        x_coords = np.linspace(n_pix_x // 8, n_pix_x * 7 // 8, 11)
        y_arc = limb_arc(
            r=r_true, n_pix_x=n_pix_x, n_pix_y=3000, h=h,
            f=f_mm * 1e-3, w=w_mm * 1e-3,
            x0=n_pix_x / 2, y0=1500,
            theta_x=0, theta_z=np.pi,
            x_coords=x_coords,
        )
        limb = np.full(n_pix_x, np.nan)
        for xi, yi in zip(x_coords.astype(int), y_arc):
            limb[xi] = yi
        est = estimate_radius_via_sagitta(limb, h=h, f_px=f_px, sigma_px=1.0)
        assert est["status"] == "ok"
        assert abs(est["r"] - r_true) / r_true < 1e-4   # < 0.01% ≈ 600 m

    def test_hyperbola_at_dip_angle(self):
        """theta_x = dip angle: hyperbola fit gives < 0.1% error on clean arc.

        Synth images are generated at theta_x = limb_camera_angle(r, h) = alpha.
        The projected limb is algebraically a hyperbola at this angle (the true
        parabolic boundary is at arctan(K) = pi/2 - alpha ≈ 87°, not alpha).
        The hyperbola model has a small sin²(alpha)/K² systematic (~2-3 km).
        """
        r_true = 6_371_000
        h = 10_000
        f_mm, w_mm = 5.1, 7.6
        n_pix_x = 4000
        f_px = f_mm / w_mm * n_pix_x
        theta_x = limb_camera_angle(r_true, h)
        x_coords = np.linspace(n_pix_x // 8, n_pix_x * 7 // 8, 11)
        y_arc = limb_arc(
            r=r_true, n_pix_x=n_pix_x, n_pix_y=3000, h=h,
            f=f_mm * 1e-3, w=w_mm * 1e-3,
            x0=n_pix_x / 2, y0=1500,
            theta_x=theta_x, theta_z=np.pi,
            x_coords=x_coords,
        )
        limb = np.full(n_pix_x, np.nan)
        for xi, yi in zip(x_coords.astype(int), y_arc):
            limb[xi] = yi
        est = estimate_radius_via_sagitta(limb, h=h, f_px=f_px, sigma_px=1.0)
        assert est["status"] == "ok"
        assert abs(est["r"] - r_true) / r_true < 0.001  # < 0.1% ≈ 6 km


class TestSagittaEstimatorExtensions:
    """Tests for bias correction, jackknife, and SagittaFitter."""

    def test_bias_correct_reduces_error(self):
        """Bias correction via y0 reduces error when theta_x exceeds the
        theta_x=0 OLS approximation."""
        r_true = 6_371_000
        h = 10_000
        f, w, W, H = 0.026, 0.0173, 4000, 3000
        f_px = f * W / w
        theta_x = 0.1  # ~5.7° — large enough that OLS has residual bias

        y_full = limb_arc(
            r=r_true, n_pix_x=W, n_pix_y=H,
            h=h, f=f, w=w, x0=W // 2, y0=H // 2,
            theta_x=theta_x, theta_z=np.pi,
        )
        margin = W // 10
        limb = np.full(W, np.nan)
        for xi in np.linspace(margin, W - margin - 1, 20, dtype=int):
            limb[xi] = y_full[xi]

        y0 = H / 2.0
        est_raw = estimate_radius_via_sagitta(
            limb, h=h, f_px=f_px, sigma_px=1.0
        )
        est_bc = estimate_radius_via_sagitta(
            limb, h=h, f_px=f_px, sigma_px=1.0, y0=y0, bias_correct=True
        )
        err_raw = abs(est_raw["r"] - r_true)
        err_bc = abs(est_bc["r"] - r_true)
        assert err_bc < err_raw

    def test_bias_correct_skipped_without_y0(self):
        """bias_correct=True without y0 should not raise, just skip correction."""
        limb = _make_synthetic_limb()
        f_px = 0.026 * 4000 / 0.0173
        est = estimate_radius_via_sagitta(
            limb, h=10_000, f_px=f_px, bias_correct=True  # no y0
        )
        assert est["status"] == "ok"

    def test_jackknife_sigma_for_noisy_arc(self):
        """Jackknife sigma is finite and positive for a noisy arc."""
        limb = _make_synthetic_limb(n_points=20, noise_std=3.0)
        f_px = 0.026 * 4000 / 0.0173
        est = estimate_radius_via_sagitta(
            limb, h=10_000, f_px=f_px, uncertainty="jackknife"
        )
        assert np.isfinite(est["K_sigma_jack"])
        assert est["K_sigma_jack"] > 0

    def test_sagitta_fitter_dip_angle(self):
        """SagittaFitter recovers radius within 1% at a 5° dip angle."""
        r_true = 6_371_000
        h = 10_000
        f = 0.026
        w = 0.0173
        n_pix_x = 4000
        f_px = f / w * n_pix_x
        theta_x = 0.087  # ~5°

        x_coords = np.linspace(0, n_pix_x - 1, 60).astype(float)
        y_arc = limb_arc(
            r=r_true, n_pix_x=n_pix_x, n_pix_y=3000,
            h=h, f=f, w=w,
            theta_x=theta_x, theta_z=np.pi,
            x_coords=x_coords,
        )
        limb = np.full(n_pix_x, np.nan)
        for xi, yi in zip(x_coords.astype(int), y_arc):
            limb[xi] = yi

        init_vals = {
            "r": 7_000_000, "h": h, "f": f, "w": w,
            "theta_x": 0.0, "theta_y": 0.0, "theta_z": np.pi,
        }
        limits = {
            "r": [2_000_000, 14_000_000],
            "h": [5_000, 20_000],
            "f": [0.01, 0.1], "w": [0.005, 0.05],
            "theta_x": [-0.3, 0.3],
            "theta_y": [-0.3, 0.3],
            "theta_z": [-np.pi, np.pi],
        }

        fitter = SagittaFitter(
            limb=limb,
            h=h,
            f_px=f_px,
            y0=3000 / 2.0,
            free_parameters=["r", "theta_x"],
            init_parameter_values=init_vals,
            parameter_limits=limits,
            sigma_px="auto",
            n_sigma=2.0,
            uncertainty="ols",
        )
        result = fitter.fit()
        r_fitted = result["updated_init"]["r"]
        assert abs(r_fitted - r_true) / r_true < 0.01


if __name__ == "__main__":
    # Run tests with: python -m pytest tests/test_geometry.py -v
    pytest.main([__file__, "-v"])
