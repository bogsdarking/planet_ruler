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

"""Tests for synthetic benchmark data generation."""

import json

import numpy as np
import pytest

from planet_ruler.benchmarks.synthetic import (
    SYNTHETIC_CAMERAS,
    _build_name,
    _write_annotation,
    generate_synthetic_case,
)

# Small image dimensions keep unit tests fast.
SMALL_PARAMS = dict(
    r=6371000,
    h=10000,
    camera="iPhone 13",
    n_pix_x=400,
    n_pix_y=300,
    n_points=5,
    noise_sigma=0.0,
    seed=0,
)


# ---------------------------------------------------------------------------
# _build_name
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildName:
    def test_clean_iphone13_h10km(self):
        assert _build_name("iPhone 13", 10_000, 0.0) == "synth_iphone_13_h10km_clean"

    def test_noisy_iphone14_h15km(self):
        name = _build_name("iPhone 14 Pro", 15_000, 4.0)
        assert "noisy4px" in name
        assert "h15km" in name

    def test_camera_sm_s906e_slug(self):
        name = _build_name("SM-S906E", 5_000, 0.0)
        assert "sm-s906e" in name

    def test_theta_x_offset_appended(self):
        name = _build_name("iPhone 13", 10_000, 0.0, theta_x_offset=np.radians(10))
        assert "_txo10" in name

    def test_theta_y_appended(self):
        name = _build_name("iPhone 13", 10_000, 0.0, theta_y=np.radians(5))
        assert "_ty5" in name

    def test_theta_z_not_appended_when_pi(self):
        name = _build_name("iPhone 13", 10_000, 0.0, theta_z=np.pi)
        assert "_tz" not in name

    def test_theta_z_appended_when_nondefault(self):
        name = _build_name("iPhone 13", 10_000, 0.0, theta_z=np.pi / 2)
        assert "_tz90" in name

    def test_h_km_integer_truncated(self):
        # int(9999/1000) == 9, so result contains "h9km" not "h10km"
        name = _build_name("iPhone 13", 9_999, 0.0)
        assert "h9km" in name


# ---------------------------------------------------------------------------
# generate_synthetic_case — no I/O
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGenerateSyntheticCaseNoIO:
    def test_returns_dict_with_expected_keys(self):
        result = generate_synthetic_case(**SMALL_PARAMS)
        assert set(result.keys()) == {"image_path", "annotation_path", "params", "points"}

    def test_paths_none_without_output_dir(self):
        result = generate_synthetic_case(**SMALL_PARAMS)
        assert result["image_path"] is None
        assert result["annotation_path"] is None

    def test_points_is_list_of_xy_pairs(self):
        result = generate_synthetic_case(**SMALL_PARAMS)
        assert isinstance(result["points"], list)
        for pt in result["points"]:
            assert len(pt) == 2

    def test_at_least_3_points_generated(self):
        result = generate_synthetic_case(**SMALL_PARAMS)
        assert len(result["points"]) >= 3

    def test_params_has_ground_truth_keys(self):
        result = generate_synthetic_case(**SMALL_PARAMS)
        params = result["params"]
        for key in ("r", "h", "f_mm", "theta_x", "theta_y", "theta_z"):
            assert key in params

    def test_params_r_matches_input(self):
        result = generate_synthetic_case(**SMALL_PARAMS)
        assert result["params"]["r"] == 6371000

    def test_params_h_matches_input(self):
        result = generate_synthetic_case(**SMALL_PARAMS)
        assert result["params"]["h"] == 10000

    def test_noisy_variant_differs_from_clean(self):
        clean = generate_synthetic_case(**{**SMALL_PARAMS, "noise_sigma": 0.0, "seed": 0})
        noisy = generate_synthetic_case(**{**SMALL_PARAMS, "noise_sigma": 5.0, "seed": 0})
        # At least one y-value should differ due to added noise
        clean_ys = [pt[1] for pt in clean["points"]]
        noisy_ys = [pt[1] for pt in noisy["points"]]
        assert clean_ys != pytest.approx(noisy_ys)

    def test_deterministic_with_seed(self):
        r1 = generate_synthetic_case(**{**SMALL_PARAMS, "noise_sigma": 3.0, "seed": 42})
        r2 = generate_synthetic_case(**{**SMALL_PARAMS, "noise_sigma": 3.0, "seed": 42})
        assert r1["points"] == r2["points"]

    def test_different_seeds_differ(self):
        r1 = generate_synthetic_case(**{**SMALL_PARAMS, "noise_sigma": 3.0, "seed": 0})
        r2 = generate_synthetic_case(**{**SMALL_PARAMS, "noise_sigma": 3.0, "seed": 1})
        ys1 = [pt[1] for pt in r1["points"]]
        ys2 = [pt[1] for pt in r2["points"]]
        assert ys1 != pytest.approx(ys2)

    def test_invalid_camera_raises(self):
        with pytest.raises(ValueError, match="Unknown camera"):
            generate_synthetic_case(**{**SMALL_PARAMS, "camera": "bogus_camera_xyz"})

    def test_all_cameras_succeed(self):
        for camera in SYNTHETIC_CAMERAS:
            result = generate_synthetic_case(**{**SMALL_PARAMS, "camera": camera})
            assert len(result["points"]) >= 3


# ---------------------------------------------------------------------------
# generate_synthetic_case — with file I/O
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestGenerateSyntheticCaseWithIO:
    def test_writes_jpeg_file(self, tmp_path):
        generate_synthetic_case(**SMALL_PARAMS, output_dir=tmp_path, name="test_img")
        assert (tmp_path / "test_img.jpg").exists()

    def test_writes_json_annotation(self, tmp_path):
        generate_synthetic_case(**SMALL_PARAMS, output_dir=tmp_path, name="test_img")
        assert (tmp_path / "test_img_limb_points.json").exists()

    def test_annotation_json_has_correct_keys(self, tmp_path):
        generate_synthetic_case(**SMALL_PARAMS, output_dir=tmp_path, name="test_ann")
        with open(tmp_path / "test_ann_limb_points.json") as f:
            data = json.load(f)
        for key in ("image_path", "image_size", "points", "params"):
            assert key in data

    def test_annotation_points_match_returned_points(self, tmp_path):
        result = generate_synthetic_case(**SMALL_PARAMS, output_dir=tmp_path, name="test_pts")
        with open(tmp_path / "test_pts_limb_points.json") as f:
            data = json.load(f)
        assert data["points"] == result["points"]

    def test_jpeg_dimensions_match_params(self, tmp_path):
        from PIL import Image

        generate_synthetic_case(**SMALL_PARAMS, output_dir=tmp_path, name="test_dim")
        img = Image.open(tmp_path / "test_dim.jpg")
        assert img.width == SMALL_PARAMS["n_pix_x"]
        assert img.height == SMALL_PARAMS["n_pix_y"]

    def test_requires_name_when_output_dir_given(self, tmp_path):
        with pytest.raises(ValueError, match="name"):
            generate_synthetic_case(**SMALL_PARAMS, output_dir=tmp_path)


# ---------------------------------------------------------------------------
# _write_annotation (private helper)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestWriteAnnotation:
    def test_creates_json_file(self, tmp_path):
        path = tmp_path / "out.json"
        _write_annotation(path, tmp_path / "img.jpg", 400, 300, [[10.0, 200.0]])
        assert path.exists()

    def test_json_has_expected_structure(self, tmp_path):
        path = tmp_path / "out.json"
        _write_annotation(path, tmp_path / "img.jpg", 400, 300, [[10.0, 200.0]])
        with open(path) as f:
            data = json.load(f)
        assert "image_path" in data
        assert "image_size" in data
        assert "points" in data

    def test_params_included_when_provided(self, tmp_path):
        path = tmp_path / "out.json"
        _write_annotation(
            path, tmp_path / "img.jpg", 400, 300, [[10.0, 200.0]],
            params={"r": 6371000},
        )
        with open(path) as f:
            data = json.load(f)
        assert "params" in data
        assert data["params"]["r"] == 6371000

    def test_params_omitted_when_none(self, tmp_path):
        path = tmp_path / "out.json"
        _write_annotation(
            path, tmp_path / "img.jpg", 400, 300, [[10.0, 200.0]], params=None
        )
        with open(path) as f:
            data = json.load(f)
        assert "params" not in data

    def test_returns_path(self, tmp_path):
        path = tmp_path / "out.json"
        result = _write_annotation(path, tmp_path / "img.jpg", 400, 300, [[10.0, 200.0]])
        assert result == path
