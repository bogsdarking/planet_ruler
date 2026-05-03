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

"""Tests for benchmark annotation augmentation utilities."""

import pytest

from planet_ruler.benchmarks.augment_annotations import (
    generate_noisy_variants,
    generate_subset_variants,
)

# 5-point set with monotonically increasing x for easy reasoning
POINTS_5 = [
    [10.0, 200.0],
    [20.0, 205.0],
    [30.0, 210.0],
    [40.0, 215.0],
    [50.0, 220.0],
]

# 10-point set for subset tests
POINTS_10 = [[float(i * 10), 200.0 + i * 5.0] for i in range(10)]


@pytest.mark.unit
class TestGenerateNoisyVariants:
    def test_returns_correct_count(self):
        result = generate_noisy_variants(POINTS_5, n_variants=3, noise_sigma=1.0)
        assert len(result) == 3

    def test_each_variant_has_same_length(self):
        result = generate_noisy_variants(POINTS_5, n_variants=2, noise_sigma=1.0)
        for variant in result:
            assert len(variant) == len(POINTS_5)

    def test_zero_noise_returns_original_values(self):
        result = generate_noisy_variants(POINTS_5, n_variants=2, noise_sigma=0.0)
        for variant in result:
            for orig, perturbed in zip(POINTS_5, variant):
                assert perturbed[0] == pytest.approx(orig[0])
                assert perturbed[1] == pytest.approx(orig[1])

    def test_nonzero_noise_perturbs_values(self):
        result = generate_noisy_variants(POINTS_5, n_variants=1, noise_sigma=5.0, seed=42)
        variant = result[0]
        any_differ = any(
            abs(v[0] - p[0]) > 1e-9 or abs(v[1] - p[1]) > 1e-9
            for p, v in zip(POINTS_5, variant)
        )
        assert any_differ

    def test_deterministic_with_seed(self):
        r1 = generate_noisy_variants(POINTS_5, n_variants=3, noise_sigma=2.0, seed=7)
        r2 = generate_noisy_variants(POINTS_5, n_variants=3, noise_sigma=2.0, seed=7)
        assert r1 == r2

    def test_different_seeds_produce_different_results(self):
        r1 = generate_noisy_variants(POINTS_5, n_variants=1, noise_sigma=2.0, seed=0)
        r2 = generate_noisy_variants(POINTS_5, n_variants=1, noise_sigma=2.0, seed=1)
        # At least one value should differ between the two seeds
        flat1 = [v for pt in r1[0] for v in pt]
        flat2 = [v for pt in r2[0] for v in pt]
        assert flat1 != pytest.approx(flat2)

    def test_n_variants_zero_returns_empty(self):
        result = generate_noisy_variants(POINTS_5, n_variants=0, noise_sigma=1.0)
        assert result == []


@pytest.mark.unit
class TestGenerateSubsetVariants:
    def test_returns_correct_count(self):
        result = generate_subset_variants(POINTS_10, n_points=3, n_variants=4)
        assert len(result) == 4

    def test_each_subset_has_correct_size(self):
        result = generate_subset_variants(POINTS_10, n_points=4, n_variants=3)
        for variant in result:
            assert len(variant) == 4

    def test_subsets_are_valid_points(self):
        result = generate_subset_variants(POINTS_10, n_points=3, n_variants=5)
        pts_set = {(p[0], p[1]) for p in POINTS_10}
        for variant in result:
            for pt in variant:
                assert (pt[0], pt[1]) in pts_set

    def test_deterministic_with_seed(self):
        r1 = generate_subset_variants(POINTS_10, n_points=5, n_variants=3, seed=42)
        r2 = generate_subset_variants(POINTS_10, n_points=5, n_variants=3, seed=42)
        assert r1 == r2

    def test_different_seeds_produce_different_results(self):
        r1 = generate_subset_variants(POINTS_10, n_points=5, n_variants=1, seed=0)
        r2 = generate_subset_variants(POINTS_10, n_points=5, n_variants=1, seed=99)
        assert r1 != r2

    def test_sorted_x_order_preserved(self):
        # Points in POINTS_10 are already x-sorted; idx.sort() keeps that order.
        result = generate_subset_variants(POINTS_10, n_points=6, n_variants=3)
        for variant in result:
            xs = [pt[0] for pt in variant]
            assert xs == sorted(xs)

    def test_raises_when_n_points_exceeds_available(self):
        with pytest.raises(ValueError):
            generate_subset_variants(POINTS_5, n_points=len(POINTS_5) + 1, n_variants=1)

    def test_n_variants_zero_returns_empty(self):
        result = generate_subset_variants(POINTS_10, n_points=3, n_variants=0)
        assert result == []
