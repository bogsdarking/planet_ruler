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
Annotation augmentation utilities for Planet Ruler benchmarks.

Generates in-memory variants of limb point annotations to:
  - Sample human-annotation error distribution (noisy variants)
  - Test robustness to partial annotations (subset variants)

All functions are deterministic given a seed and produce list-of-lists
in the same [x, y] format used by *_limb_points.json files.
"""

from typing import List

import numpy as np


def generate_noisy_variants(
    points: List[List[float]],
    n_variants: int,
    noise_sigma: float,
    seed: int = 0,
) -> List[List[List[float]]]:
    """
    Return n_variants copies of points, each perturbed with isotropic Gaussian
    noise of standard deviation noise_sigma pixels.

    Args:
        points: List of [x, y] limb annotation points.
        n_variants: Number of noisy copies to generate.
        noise_sigma: Gaussian standard deviation in pixels.
        seed: RNG seed for reproducibility.

    Returns:
        List of n_variants point sets, each in the same [[x, y], ...] format.
    """
    rng = np.random.default_rng(seed)
    pts = np.array(points, dtype=float)  # (N, 2)
    variants = []
    for _ in range(n_variants):
        noise = rng.normal(0.0, noise_sigma, size=pts.shape)
        variants.append((pts + noise).tolist())
    return variants


def generate_subset_variants(
    points: List[List[float]],
    n_points: int,
    n_variants: int,
    seed: int = 0,
) -> List[List[List[float]]]:
    """
    Return n_variants subsets of size n_points drawn without replacement.

    Args:
        points: List of [x, y] limb annotation points.
        n_points: Number of points per subset.
        n_variants: Number of subsets to draw.
        seed: RNG seed for reproducibility.

    Returns:
        List of n_variants point sets, each containing n_points points.

    Raises:
        ValueError: If n_points > len(points).
    """
    if n_points > len(points):
        raise ValueError(
            f"n_points={n_points} exceeds number of available points ({len(points)})"
        )
    rng = np.random.default_rng(seed)
    pts = np.array(points, dtype=float)
    variants = []
    for _ in range(n_variants):
        idx = rng.choice(len(pts), size=n_points, replace=False)
        idx.sort()  # preserve x-ordering for consistency
        variants.append(pts[idx].tolist())
    return variants
