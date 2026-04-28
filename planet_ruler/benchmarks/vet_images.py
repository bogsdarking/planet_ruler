#!/usr/bin/env python
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
Image vetting tool for Planet Ruler benchmark data.

Determines whether each image+annotation pair produces a plausible fitted
altitude when the Earth radius is held fixed. This gates which images are
suitable for the benchmark suite.

Vetting logic:
  - Fix r = 6371000 m (known Earth radius)
  - Free h with wide bounds [500, 30000 m]
  - If fitted h lands in [1000, 20000 m]: PASS
  - If fitted h lands in [500, 1000) or (20000, 30000]: WARN
  - Otherwise (convergence failure or out-of-bounds): FAIL

Convergence notes:
  The search space (h, theta_x, theta_y, theta_z, f, w) is large. The default
  settings (preset=robust, max-iter=1000) are chosen to give the optimizer a
  realistic chance at convergence. Use --fast for a quick preview; use
  --max-iter 2000 or higher if an image borderline fails. A true FAIL after
  1000+ iterations is a strong signal that the image/annotation is problematic.

Usage:
    # Vet all images with annotations
    python -m planet_ruler.benchmarks.vet_images

    # Quick preview (less rigorous, faster)
    python -m planet_ruler.benchmarks.vet_images --fast

    # Vet a specific image
    python -m planet_ruler.benchmarks.vet_images --image pexels-claiton-17217951_exif_cropped

    # Fix camera params from EXIF (faster + potentially more accurate)
    python -m planet_ruler.benchmarks.vet_images --fix-camera

    # Save results to CSV
    python -m planet_ruler.benchmarks.vet_images --log planet_ruler/benchmarks/results/vetting_log.csv
"""

import argparse
import contextlib
import csv
import io
import json
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


import numpy as np


@contextlib.contextmanager
def _suppress_stdout():
    """Suppress stdout for chatty library calls (e.g. create_config_from_image prints)."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield

# Altitude range thresholds (meters).
#
# These also define the optimizer search bounds [_WARN_MIN_M, _WARN_MAX_M].
# Keeping the search space tight reduces the chance of the optimizer escaping
# to high-altitude local minima that are geometrically plausible but physically
# implausible for typical aircraft photography.
#
#   PASS  : core plausible altitude range (drone to commercial airliner)
#   WARN  : extended plausible range (very low aircraft / high-altitude edge)
#   FAIL  : outside WARN, or optimizer hit the boundary, or convergence error
#
# Commercial airliner ceiling: ~12,800 m (~42,000 ft).  Keep _PASS_MAX_M a bit
# above to allow for GPS/EXIF rounding and genuine high-cruise altitudes.
_PASS_MIN_M = 3_000.0    # 3 km  — low aircraft / regional jets
_PASS_MAX_M = 14_000.0   # 14 km — just above commercial airliner ceiling
_WARN_MIN_M = 1_500.0    # 1.5 km — very low (possible small aircraft)
_WARN_MAX_M = 18_000.0   # 18 km — high-altitude edge (SR-71, research planes)

# If the fitted altitude lands within this fraction of the search boundary,
# flag it as a boundary result — the true minimum may be outside the range.
_BOUNDARY_FRACTION = 0.03   # 3 % of search range ≈ 495 m for [1.5, 18] km

_DEFAULT_ALTITUDE_M = 10_000.0  # Fallback initial altitude for images missing GPS

EARTH_RADIUS_M = 6_371_000.0


@dataclass
class VetResult:
    image_name: str
    annotation_file: str
    fitted_altitude_km: Optional[float]
    gps_altitude_km: Optional[float]
    fitted_radius_km: Optional[float]
    radius_error_pct: Optional[float]
    convergence: str
    runtime_s: float
    verdict: str   # PASS, WARN, FAIL
    notes: str


def _find_benchmark_dir() -> Path:
    """Walk up from CWD to find the benchmarks directory inside the repo."""
    for parent in [Path.cwd(), *Path.cwd().parents]:
        candidate = parent / "planet_ruler" / "benchmarks"
        if (candidate / "images").exists():
            return candidate
    return Path(__file__).parent


def discover_pairs(benchmark_dir: Path, image_filter: Optional[list] = None) -> list:
    """
    Discover (image_path, annotation_path) pairs ready for vetting.

    Searches annotations/ and images/ directories for *_limb_points.json files,
    then matches each to its corresponding .jpg image.
    """
    annot_dirs = [
        benchmark_dir / "annotations",
        benchmark_dir / "images",  # some annotations live alongside images
    ]

    pairs = []
    seen_stems = set()

    for annot_dir in annot_dirs:
        if not annot_dir.exists():
            continue
        for annot_path in sorted(annot_dir.glob("*_limb_points.json")):
            if annot_path.name == "example.json":
                continue

            image_stem = annot_path.stem.replace("_limb_points", "")
            if image_stem in seen_stems:
                continue

            if image_filter and image_stem not in image_filter:
                continue

            image_path = benchmark_dir / "images" / f"{image_stem}.jpg"
            if not image_path.exists():
                continue

            seen_stems.add(image_stem)
            pairs.append((image_path, annot_path))

    return pairs


def _load_annotation_as_target(annot_path: Path, image_width: int) -> np.ndarray:
    """Load annotation JSON and convert to a sparse target array."""
    with open(annot_path) as f:
        data = json.load(f)

    if "limb_points" in data:
        points = data["limb_points"]["points"]
    elif "points" in data:
        points = data["points"]
    else:
        raise ValueError(f"Unrecognized annotation format in {annot_path}")

    target = np.full(image_width, np.nan)
    for x, y in points:
        x_idx = int(round(x))
        if 0 <= x_idx < image_width:
            target[x_idx] = y
    return target


def _build_vet_config(
    image_path: Path, fix_camera: bool = False
) -> Tuple[dict, Optional[float]]:
    """
    Build a fit config for vetting.

    r is fixed to EARTH_RADIUS_M; h is freed with wide bounds.  If
    ``fix_camera`` is True, f and w are also removed from free_parameters
    (useful when EXIF focal length and sensor width are trustworthy).

    The validator requires every entry in init_parameter_values to have a
    corresponding entry in parameter_limits.  When fixing r we satisfy this
    by adding tight (non-operative) bounds for r while keeping r out of
    free_parameters so the optimizer never moves it.
    """
    from planet_ruler.camera import create_config_from_image, get_gps_altitude

    gps_altitude_m = get_gps_altitude(str(image_path))

    # create_config_from_image raises ValueError if altitude is unavailable.
    # Fall back to a plausible default so vetting can proceed.
    # Suppress the informational prints from create_config_from_image so they
    # don't interleave with the progress output.
    try:
        with _suppress_stdout():
            config = create_config_from_image(
                image_path=str(image_path),
                altitude_m=gps_altitude_m,
                planet="earth",
            )
    except ValueError:
        with _suppress_stdout():
            config = create_config_from_image(
                image_path=str(image_path),
                altitude_m=_DEFAULT_ALTITUDE_M,
                planet="earth",
            )

    # Widen h bounds regardless of GPS constraint, and reset h init
    # to the GPS value (or midpoint of wide range if no GPS).
    h_init = gps_altitude_m if gps_altitude_m is not None else _DEFAULT_ALTITUDE_M
    config["init_parameter_values"]["h"] = float(h_init)
    config["parameter_limits"]["h"] = [float(_WARN_MIN_M), float(_WARN_MAX_M)]

    # Remove r from free_parameters so the optimizer cannot move it.
    # Keep r in init_parameter_values (as the fixed value) so it is
    # available to limb_arc during cost evaluation. Add matching
    # parameter_limits to satisfy the config validator.
    config["free_parameters"] = [p for p in config["free_parameters"] if p != "r"]
    config["init_parameter_values"]["r"] = EARTH_RADIUS_M
    config["parameter_limits"]["r"] = [
        EARTH_RADIUS_M * 0.999,
        EARTH_RADIUS_M * 1.001,
    ]

    if fix_camera:
        # Remove f and w from free_parameters; EXIF priors are used directly.
        # Tighter search space can improve accuracy when EXIF is trustworthy.
        config["free_parameters"] = [
            p for p in config["free_parameters"] if p not in ("f", "w")
        ]

    return config, gps_altitude_m


def _build_step2_config(step1_config: dict, h_fit_m: float, step1_obs) -> dict:
    """
    Build the step-2 config for the bootstrap self-consistency check.

    Pins h to the value recovered in step 1, frees r, and warm-starts
    orientation/camera parameters from the step-1 fitted values so the
    optimizer doesn't have to re-discover them.
    """
    import copy

    config = copy.deepcopy(step1_config)

    # Pin h: remove from free_parameters, use tight bounds to satisfy validator.
    config["free_parameters"] = [p for p in config["free_parameters"] if p != "h"]
    config["init_parameter_values"]["h"] = float(h_fit_m)
    h_tol = 0.001  # 0.1 % — effectively fixed
    config["parameter_limits"]["h"] = [
        h_fit_m * (1.0 - h_tol),
        h_fit_m * (1.0 + h_tol),
    ]

    # Free r with wide bounds; override the tight dummy bounds from step 1.
    # IMPORTANT: do NOT seed x0 at EARTH_RADIUS_M — observation.py passes
    # init_parameter_values directly as x0 to differential_evolution, which
    # would hand the optimizer one population member at the exact true answer.
    # Use the log-space midpoint of the search range instead (~10,000 km).
    _R_LO, _R_HI = 1_000_000.0, 100_000_000.0
    r_init_neutral = (_R_LO * _R_HI) ** 0.5  # geometric midpoint ≈ 10,000 km
    if "r" not in config["free_parameters"]:
        config["free_parameters"].append("r")
    config["init_parameter_values"]["r"] = r_init_neutral
    config["parameter_limits"]["r"] = [_R_LO, _R_HI]

    # Warm-start orientation and camera from step-1 fitted values so
    # the second optimizer run doesn't waste iterations re-discovering them.
    if step1_obs.best_parameters is not None:
        for param in ("theta_x", "theta_y", "theta_z", "f", "w"):
            if param in step1_obs.best_parameters:
                val = step1_obs.best_parameters[param]
                # Only update if still within the existing limits
                limits = config["parameter_limits"].get(param)
                if limits and limits[0] <= val <= limits[1]:
                    config["init_parameter_values"][param] = val

    return config


def vet_image(
    image_path: Path,
    annot_path: Path,
    max_iter: int,
    minimizer_preset: str,
    fix_camera: bool = False,
    two_step: bool = False,
) -> VetResult:
    """Run a vetting fit on a single image+annotation pair."""
    from planet_ruler.image import load_image
    from planet_ruler.observation import LimbObservation

    image_name = image_path.stem
    annot_name = annot_path.name
    t0 = time.time()

    try:
        img = load_image(str(image_path))
        config, gps_altitude_m = _build_vet_config(
            image_path, fix_camera=fix_camera
        )
        gps_alt_km = gps_altitude_m / 1000.0 if gps_altitude_m is not None else None

        obs = LimbObservation(
            image_filepath=str(image_path),
            fit_config=config,
            limb_detection="manual",
            minimizer="basinhopping",
        )

        target = _load_annotation_as_target(annot_path, img.shape[1])
        obs.register_limb(target)

        obs.fit_arc(
            loss_function="l2",
            max_iter=max_iter,
            seed=0,
            verbose=False,
            minimizer="basinhopping",
            minimizer_preset=minimizer_preset,
        )

        fitted_h_km = obs.altitude_km
        fitted_r_km = None

        # ── Step 2: bootstrap self-consistency check ──────────────────────────
        if two_step:
            h_fit_m = fitted_h_km * 1000.0
            config2 = _build_step2_config(config, h_fit_m, obs)

            obs2 = LimbObservation(
                image_filepath=str(image_path),
                fit_config=config2,
                limb_detection="manual",
                minimizer="basinhopping",
            )
            obs2.register_limb(target)
            obs2.fit_arc(
                loss_function="l2",
                max_iter=max_iter,
                seed=0,
                verbose=False,
                minimizer="basinhopping",
                minimizer_preset=minimizer_preset,
            )
            fitted_r_km = obs2.radius_km
        # ──────────────────────────────────────────────────────────────────────

        runtime = time.time() - t0
        r_err_pct = (
            abs(fitted_r_km - 6371.0) / 6371.0 * 100
            if fitted_r_km is not None
            else None
        )

        # Determine verdict based on fitted altitude
        h_m = fitted_h_km * 1000.0
        if _PASS_MIN_M <= h_m <= _PASS_MAX_M:
            verdict = "PASS"
        elif _WARN_MIN_M <= h_m < _PASS_MIN_M or _PASS_MAX_M < h_m <= _WARN_MAX_M:
            verdict = "WARN"
        else:
            verdict = "FAIL"

        # In two-step mode the radius recovery is the primary quality signal.
        if two_step and r_err_pct is not None:
            if r_err_pct > 20.0:
                verdict = "FAIL"
            elif r_err_pct > 5.0 and verdict == "PASS":
                verdict = "WARN"

        # Detect boundary-hitting: if the fit landed within _BOUNDARY_FRACTION of
        # either search boundary, the true minimum may lie outside the range.
        search_range = _WARN_MAX_M - _WARN_MIN_M
        boundary_tol = _BOUNDARY_FRACTION * search_range
        at_boundary = (
            h_m < _WARN_MIN_M + boundary_tol
            or h_m > _WARN_MAX_M - boundary_tol
        )

        # Build notes string
        note_parts = [f"h_fit={fitted_h_km:.1f} km"]
        if at_boundary:
            note_parts.append("BOUNDARY (true min may be outside search range)")
            if verdict == "PASS":
                verdict = "WARN"
            elif verdict == "WARN":
                verdict = "FAIL"
        if gps_alt_km is not None:
            diff = fitted_h_km - gps_alt_km
            note_parts.append(f"h_GPS={gps_alt_km:.1f} km (diff={diff:+.1f} km)")
        if fitted_r_km is not None:
            note_parts.append(f"r_fit={fitted_r_km:.0f} km ({r_err_pct:.1f}% err)")

        return VetResult(
            image_name=image_name,
            annotation_file=annot_name,
            fitted_altitude_km=fitted_h_km,
            gps_altitude_km=gps_alt_km,
            fitted_radius_km=fitted_r_km,
            radius_error_pct=r_err_pct,
            convergence="success",
            runtime_s=runtime,
            verdict=verdict,
            notes=", ".join(note_parts),
        )

    except Exception as e:
        runtime = time.time() - t0
        tb_last = traceback.format_exc().splitlines()[-1]
        return VetResult(
            image_name=image_name,
            annotation_file=annot_name,
            fitted_altitude_km=None,
            gps_altitude_km=None,
            fitted_radius_km=None,
            radius_error_pct=None,
            convergence=f"error: {e}",
            runtime_s=runtime,
            verdict="FAIL",
            notes=tb_last,
        )


def _print_results(results: list) -> None:
    """Print vetting results as a formatted summary table."""
    COL_NAME = 54
    print()
    print("=" * 110)
    print("Planet Ruler Image Vetting Results")
    print("=" * 110)
    print(
        f"{'Image':<{COL_NAME}} {'h_fit(km)':>9} {'h_GPS(km)':>9} {'Time(s)':>7}  {'Verdict':<8}  Notes"
    )
    print("-" * 110)

    for r in results:
        h_fit = f"{r.fitted_altitude_km:.1f}" if r.fitted_altitude_km is not None else "ERROR"
        h_gps = f"{r.gps_altitude_km:.1f}" if r.gps_altitude_km is not None else "N/A"
        verdict_str = {"PASS": "✓ PASS", "WARN": "~ WARN", "FAIL": "✗ FAIL"}.get(
            r.verdict, r.verdict
        )
        name = r.image_name[:COL_NAME]
        print(
            f"{name:<{COL_NAME}} {h_fit:>9} {h_gps:>9} {r.runtime_s:>7.1f}  {verdict_str:<8}  {r.notes}"
        )

    print("=" * 110)

    n_pass = sum(1 for r in results if r.verdict == "PASS")
    n_warn = sum(1 for r in results if r.verdict == "WARN")
    n_fail = sum(1 for r in results if r.verdict == "FAIL")
    print(f"Summary: {n_pass} PASS  {n_warn} WARN  {n_fail} FAIL  ({len(results)} total)")
    print()

    usable = n_pass + n_warn
    if usable >= 4:
        print(
            f"Decision: {usable} usable image(s). Proceed to configure manual.yaml and run augmentation."
        )
    else:
        print(
            f"Decision: Only {usable} usable image(s). Consider:\n"
            "  - Tweaking crops or annotations for failing images\n"
            "  - Sourcing additional real images from the web\n"
            "  - Pivoting to synthetic data generation (Track 3)"
        )
    print()


def _save_csv(results: list, log_path: Path) -> None:
    """Save vetting results to a CSV log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_name",
        "annotation_file",
        "fitted_altitude_km",
        "gps_altitude_km",
        "fitted_radius_km",
        "radius_error_pct",
        "convergence",
        "runtime_s",
        "verdict",
        "notes",
    ]
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "image_name": r.image_name,
                    "annotation_file": r.annotation_file,
                    "fitted_altitude_km": (
                        f"{r.fitted_altitude_km:.3f}"
                        if r.fitted_altitude_km is not None
                        else ""
                    ),
                    "gps_altitude_km": (
                        f"{r.gps_altitude_km:.3f}"
                        if r.gps_altitude_km is not None
                        else ""
                    ),
                    "fitted_radius_km": (
                        f"{r.fitted_radius_km:.1f}"
                        if r.fitted_radius_km is not None
                        else ""
                    ),
                    "radius_error_pct": (
                        f"{r.radius_error_pct:.2f}"
                        if r.radius_error_pct is not None
                        else ""
                    ),
                    "convergence": r.convergence,
                    "runtime_s": f"{r.runtime_s:.2f}",
                    "verdict": r.verdict,
                    "notes": r.notes,
                }
            )
    print(f"Results saved to: {log_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Vet Planet Ruler benchmark images for plausible altitude recovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Convergence notes:
  Basin hopping with 300 hops (default) is sufficient for most images.
  A FAIL after 300 iterations is a strong signal the image or annotation is
  problematic.  Use --fast for a quick preview (100 hops, ~3x faster).
  Use --fix-camera to reduce the search space when EXIF data is trustworthy.

Examples:
  python -m planet_ruler.benchmarks.vet_images
  python -m planet_ruler.benchmarks.vet_images --fast
  python -m planet_ruler.benchmarks.vet_images --image pexels-claiton-17217951_exif_cropped
  python -m planet_ruler.benchmarks.vet_images --fix-camera
  python -m planet_ruler.benchmarks.vet_images --max-iter 500 --preset robust
  python -m planet_ruler.benchmarks.vet_images --log planet_ruler/benchmarks/results/vetting_log.csv
        """,
    )

    parser.add_argument(
        "--image",
        action="append",
        metavar="IMAGE_STEM",
        dest="images",
        help="Image stem(s) to vet (without .jpg). Repeatable.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=None,
        metavar="N",
        help="Max optimizer iterations per image (default: 200 for --fast, 1000 otherwise)",
    )
    parser.add_argument(
        "--preset",
        choices=["fast", "balanced", "robust"],
        default=None,
        help="Minimizer preset (default: balanced for --fast, robust otherwise)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Quick preview mode: preset=balanced, max-iter=100 (~3x faster, less rigorous)",
    )
    parser.add_argument(
        "--fix-camera",
        action="store_true",
        help=(
            "Fix f and w from EXIF (remove from free_parameters). "
            "Can improve accuracy when EXIF focal length and sensor width are trustworthy."
        ),
    )
    parser.add_argument(
        "--two-step",
        action="store_true",
        help=(
            "Bootstrap self-consistency check: "
            "step 1 fixes r and finds h, "
            "step 2 pins h to that value and recovers r. "
            "Verdict is based on how well r recovers to 6371 km."
        ),
    )
    parser.add_argument(
        "--log",
        type=Path,
        metavar="CSV_PATH",
        help="Save results to a CSV file",
    )

    args = parser.parse_args()

    # Resolve preset and max_iter (--fast sets defaults, explicit flags override)
    if args.fast:
        preset = args.preset or "balanced"
        max_iter = args.max_iter or 100
    else:
        preset = args.preset or "robust"
        max_iter = args.max_iter or 300

    benchmark_dir = _find_benchmark_dir()
    pairs = discover_pairs(benchmark_dir, image_filter=args.images)

    if not pairs:
        if args.images:
            print(f"No pairs found for: {args.images}", file=sys.stderr)
        else:
            print(
                "No image+annotation pairs found. "
                "Check benchmarks/annotations/ directory.",
                file=sys.stderr,
            )
        sys.exit(1)

    if args.two_step:
        mode = "two-step bootstrap (step1: r fixed → h; step2: h pinned → r)"
    else:
        cam = " + camera fixed" if args.fix_camera else ""
        mode = f"r fixed at 6371 km, h free{cam}"
    print(f"Vetting {len(pairs)} image+annotation pair(s)")
    print(f"Fit mode  : {mode}")
    print(f"Preset    : {preset}  Max iters: {max_iter} per step")
    print(f"Benchmark : {benchmark_dir}")
    print()

    results = []
    for i, (image_path, annot_path) in enumerate(pairs, 1):
        print(f"[{i}/{len(pairs)}] {image_path.stem} ...", end="  ", flush=True)
        result = vet_image(
            image_path,
            annot_path,
            max_iter=max_iter,
            minimizer_preset=preset,
            fix_camera=args.fix_camera,
            two_step=args.two_step,
        )
        verdict_label = {"PASS": "PASS", "WARN": "WARN", "FAIL": "FAIL"}.get(
            result.verdict, result.verdict
        )
        print(f"{verdict_label}  ({result.runtime_s:.1f}s)")
        results.append(result)

    _print_results(results)

    if args.log:
        _save_csv(results, args.log)


if __name__ == "__main__":
    main()
