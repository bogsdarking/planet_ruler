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
Synthetic data generator for Planet Ruler benchmarks.

Creates JPEG images with embedded EXIF (focal length, GPS altitude,
camera make/model) and matching *_limb_points.json annotation files where every
point lies exactly on the theoretical limb arc (plus optional Gaussian noise).

Image appearance: white above the horizon (sky/space), black below (planet),
with the limb rendered as a sharp boundary — easy to visually verify and
potentially useful for edge-based detection methods.

These synthetic cases provide exact known ground truth (r, h, f, w, theta_x)
so the optimizer can be validated under controlled conditions independent of
real-image ambiguities.

Usage:
    python -m planet_ruler.benchmarks.synthetic single \\
        --r 6371000 --h 10000 --camera "iPhone 13" \\
        --n-points 11 --noise-sigma 0 \\
        --output-dir planet_ruler/benchmarks/images/ \\
        --name synth_iphone13_h10000_clean

    python -m planet_ruler.benchmarks.synthetic canonical \\
        --output-dir planet_ruler/benchmarks/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

# Camera DB entries used for synthetic data generation.
# (make, model, f_mm, sensor_width_mm)
SYNTHETIC_CAMERAS = {
    "iPhone 13": ("Apple", "iPhone 13", 5.1, 7.6),
    "iPhone 14 Pro": ("Apple", "iPhone 14 Pro", 6.9, 9.8),
    "SM-S906E": ("SAMSUNG", "SM-S906E", 5.4, 7.6),
}


def generate_synthetic_case(
    r: float,
    h: float,
    camera: str = "iPhone 13",
    n_pix_x: int = 4000,
    n_pix_y: int = 3000,
    n_points: int = 11,
    noise_sigma: float = 0.0,
    seed: int = 0,
    output_dir: Optional[Path] = None,
    name: Optional[str] = None,
) -> dict:
    """
    Generate a synthetic limb observation with exact ground truth.

    Camera orientation uses theta_x = limb_camera_angle(r, h) (the natural
    angle at which the horizon appears for the given altitude), theta_y = 0,
    theta_z = 0 — physically correct for a level, unrotated camera.

    Args:
        r: Planet radius (m).
        h: Camera altitude above surface (m).
        camera: Camera key in SYNTHETIC_CAMERAS dict.
        n_pix_x: Image width (pixels).
        n_pix_y: Image height (pixels).
        n_points: Number of annotation points to generate.
        noise_sigma: Gaussian noise std dev in pixels (applied to y).
        seed: RNG seed for reproducible noise.
        output_dir: If provided, write JPEG and JSON files here.
        name: File stem for output files (required when output_dir given).

    Returns:
        dict with keys:
            image_path (Path or None): Written JPEG path.
            annotation_path (Path or None): Written JSON path.
            params (dict): Ground-truth parameters.
            points (list): [[x, y], ...] annotation points (with noise if any).
    """
    from planet_ruler.geometry import limb_arc, limb_camera_angle

    if camera not in SYNTHETIC_CAMERAS:
        raise ValueError(
            f"Unknown camera '{camera}'. "
            f"Choose from: {list(SYNTHETIC_CAMERAS)}"
        )

    make, model, f_mm, w_mm = SYNTHETIC_CAMERAS[camera]
    f_m = f_mm * 1e-3   # meters
    w_m = w_mm * 1e-3   # meters

    # Natural tilt angle: camera points at horizon.
    # theta_z = π rotates 180° around the vertical axis so the visible limb
    # arc is the near side, producing a ∪-shaped arc (more planet at center
    # than edges) matching real horizon photos.
    # Note: theta_y has no effect — the limb circle is coaxial with the y-axis,
    # making y-rotation a pure phase shift in φ with no impact on projection.
    theta_x = limb_camera_angle(r, h)
    theta_z = np.pi

    # ---- annotation points (sparse, n_points) ----------------------------
    # x0/y0 must match LimbObservation: principal point = image centre.
    x0 = n_pix_x / 2.0
    y0 = n_pix_y / 2.0

    margin = n_pix_x // (n_points + 1)
    x_sparse = np.linspace(margin, n_pix_x - margin, n_points)

    y_sparse = limb_arc(
        r=r,
        n_pix_x=n_pix_x,
        n_pix_y=n_pix_y,
        h=h,
        f=f_m,
        w=w_m,
        x0=x0,
        y0=y0,
        theta_x=theta_x,
        theta_y=0.0,
        theta_z=theta_z,
        x_coords=x_sparse,
    )

    # Filter points where limb is not visible (NaN)
    valid = ~np.isnan(y_sparse)
    x_valid = x_sparse[valid]
    y_valid = y_sparse[valid]

    if len(x_valid) < 3:
        raise ValueError(
            f"Fewer than 3 visible limb points for r={r}, h={h}, "
            f"camera={camera}. Try increasing h or adjusting params."
        )

    # Apply optional noise to y only (x-coordinates are exact pixel indices)
    if noise_sigma > 0.0:
        rng = np.random.default_rng(seed)
        y_valid = y_valid + rng.normal(0.0, noise_sigma, size=len(y_valid))

    points = [[float(x), float(y)] for x, y in zip(x_valid, y_valid)]

    params = {
        "r": r,
        "h": h,
        "f_mm": f_mm,
        "w_mm": w_mm,
        "f_m": f_m,
        "w_m": w_m,
        "theta_x": float(theta_x),
        "theta_y": 0.0,
        "theta_z": float(theta_z),
        "n_pix_x": n_pix_x,
        "n_pix_y": n_pix_y,
        "camera": camera,
        "make": make,
        "model": model,
        "noise_sigma": noise_sigma,
        "seed": seed,
    }

    image_path = None
    annotation_path = None

    if output_dir is not None:
        if name is None:
            raise ValueError("'name' is required when output_dir is provided")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Full-width limb arc for image rendering (same x0/y0 as above)
        y_full = limb_arc(
            r=r,
            n_pix_x=n_pix_x,
            n_pix_y=n_pix_y,
            h=h,
            f=f_m,
            w=w_m,
            x0=x0,
            y0=y0,
            theta_x=theta_x,
            theta_y=0.0,
            theta_z=theta_z,
        )

        image_path = _write_jpeg(
            output_dir / f"{name}.jpg",
            n_pix_x=n_pix_x,
            n_pix_y=n_pix_y,
            limb_y=y_full,
            make=make,
            model=model,
            f_mm=f_mm,
            h_m=h,
        )
        annotation_path = _write_annotation(
            output_dir / f"{name}_limb_points.json",
            image_path=image_path,
            n_pix_x=n_pix_x,
            n_pix_y=n_pix_y,
            points=points,
            params=params,
        )

    return {
        "image_path": image_path,
        "annotation_path": annotation_path,
        "params": params,
        "points": points,
    }


def _write_jpeg(
    path: Path,
    n_pix_x: int,
    n_pix_y: int,
    limb_y: np.ndarray,
    make: str,
    model: str,
    f_mm: float,
    h_m: float,
) -> Path:
    """
    Write a JPEG with embedded EXIF metadata.

    Pixels above the limb (sky/space) are white; pixels below (planet
    surface) are black. Where limb_y is NaN (limb not visible), the entire
    column is filled with the sky colour (white).
    """
    import piexif

    # Build pixel array: white above, black below
    arr = np.zeros((n_pix_y, n_pix_x), dtype=np.uint8)
    row_indices = np.arange(n_pix_y, dtype=float)  # (n_pix_y,)

    for col in range(n_pix_x):
        ly = limb_y[col]
        if np.isnan(ly):
            # No limb visible in this column — all sky (white)
            arr[:, col] = 255
        else:
            # rows with row_idx < limb_y → above limb → sky → white
            arr[:, col] = np.where(row_indices < ly, 255, 0)

    img = Image.fromarray(arr, mode="L").convert("RGB")

    exif_dict = {
        "0th": {},
        "Exif": {},
        "GPS": {},
        "1st": {},
        "thumbnail": None,
    }

    # Camera make / model — used by get_camera_model() for sensor DB lookup
    exif_dict["0th"][piexif.ImageIFD.Make] = make.encode("utf-8")
    exif_dict["0th"][piexif.ImageIFD.Model] = model.encode("utf-8")

    # Focal length as rational: use 10000 denominator to avoid truncation
    # (e.g. int(5.1*100)=509 → 5.09mm; round(5.1*10000)=51000 → exact)
    exif_dict["Exif"][piexif.ExifIFD.FocalLength] = (
        round(f_mm * 10000),
        10000,
    )

    # GPS altitude (metres above sea level)
    exif_dict["GPS"][piexif.GPSIFD.GPSAltitude] = (int(h_m * 100), 100)
    exif_dict["GPS"][piexif.GPSIFD.GPSAltitudeRef] = 0  # 0 = above sea level

    exif_bytes = piexif.dump(exif_dict)
    img.save(str(path), exif=exif_bytes, quality=95)
    return path


def _write_annotation(
    path: Path,
    image_path: Path,
    n_pix_x: int,
    n_pix_y: int,
    points: list,
    params: Optional[dict] = None,
) -> Path:
    """Write a *_limb_points.json annotation file."""
    annotation = {
        "image_path": str(image_path),
        "image_size": [n_pix_x, n_pix_y],
        "points": points,
    }
    if params is not None:
        annotation["params"] = params
    with open(path, "w") as f:
        json.dump(annotation, f, indent=2)
    return path


def _build_name(camera: str, h_m: float, noise_sigma: float) -> str:
    """Build a canonical file stem from camera + altitude + noise."""
    cam_key = camera.lower().replace(" ", "_").replace("/", "_")
    h_km = int(h_m / 1000)
    noise_tag = f"_noisy{int(noise_sigma)}px" if noise_sigma > 0 else "_clean"
    return f"synth_{cam_key}_h{h_km}km{noise_tag}"


def generate_canonical_dataset(output_dir: Path, r: float = 6371000.0) -> list:
    """
    Generate the canonical synthetic benchmark dataset.

    3 cameras × 4 altitude/noise cases = 12 scenarios:
      - h=5 km,  clean
      - h=10 km, clean
      - h=10 km, noisy 4px
      - h=15 km, clean

    Images and annotations are written under output_dir/images/ and
    output_dir/annotations/ respectively.

    Args:
        output_dir: Root benchmark dir (parent of images/ and annotations/).
        r: Planet radius in meters (default: Earth).

    Returns:
        List of result dicts from generate_synthetic_case().
    """
    images_dir = Path(output_dir) / "images"
    annot_dir = Path(output_dir) / "annotations"

    cases = [
        (5_000,  0.0),   # low alt, no noise
        (10_000, 0.0),   # mid alt, no noise
        (10_000, 4.0),   # mid alt, realistic annotation noise
        (15_000, 0.0),   # high alt, no noise
    ]

    results = []
    for camera in SYNTHETIC_CAMERAS:
        for h, sigma in cases:
            name = _build_name(camera, h, sigma)
            print(f"  Generating {name} ...")

            # Generate image in images/
            result = generate_synthetic_case(
                r=r,
                h=h,
                camera=camera,
                noise_sigma=sigma,
                seed=0,
                output_dir=images_dir,
                name=name,
            )

            # Move annotation to annotations/
            annot_src = images_dir / f"{name}_limb_points.json"
            annot_dir.mkdir(parents=True, exist_ok=True)
            annot_dst = annot_dir / f"{name}_limb_points.json"
            if annot_src.exists():
                annot_src.rename(annot_dst)
                result["annotation_path"] = annot_dst

                # Update image_path reference in annotation file
                with open(annot_dst) as f:
                    ann = json.load(f)
                ann["image_path"] = str(result["image_path"])
                with open(annot_dst, "w") as f:
                    json.dump(ann, f, indent=2)

            results.append(result)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate synthetic Planet Ruler benchmark data."
    )
    sub = p.add_subparsers(dest="command")

    # --- single case ---
    single = sub.add_parser("single", help="Generate one synthetic case.")
    single.add_argument(
        "--r", type=float, default=6371000.0, help="Planet radius (m)"
    )
    single.add_argument("--h", type=float, required=True, help="Altitude (m)")
    single.add_argument(
        "--camera",
        default="iPhone 13",
        choices=list(SYNTHETIC_CAMERAS),
        help="Camera key",
    )
    single.add_argument("--n-points", type=int, default=11)
    single.add_argument("--noise-sigma", type=float, default=0.0)
    single.add_argument("--seed", type=int, default=0)
    single.add_argument("--output-dir", type=Path, default=None)
    single.add_argument("--name", type=str, default=None)

    # --- canonical dataset ---
    canon = sub.add_parser(
        "canonical", help="Generate the full canonical dataset."
    )
    canon.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Benchmark root dir (parent of images/ and annotations/)",
    )
    canon.add_argument("--r", type=float, default=6371000.0)

    return p.parse_args()


def main():
    args = _parse_args()

    if args.command == "canonical":
        print(f"Generating canonical dataset under {args.output_dir} ...")
        results = generate_canonical_dataset(args.output_dir, r=args.r)
        print(f"\nDone. Generated {len(results)} cases.")
        for res in results:
            print(f"  {res['image_path']}  ({len(res['points'])} points)")

    elif args.command == "single":
        name = args.name or _build_name(
            args.camera, args.h, args.noise_sigma
        )
        result = generate_synthetic_case(
            r=args.r,
            h=args.h,
            camera=args.camera,
            n_points=args.n_points,
            noise_sigma=args.noise_sigma,
            seed=args.seed,
            output_dir=args.output_dir,
            name=name,
        )
        print(f"Generated {len(result['points'])} limb points.")
        print(f"Ground truth: {result['params']}")
        if result["image_path"]:
            print(f"Image:        {result['image_path']}")
            print(f"Annotation:   {result['annotation_path']}")

    else:
        print("Specify a subcommand: single or canonical")
        print("  python -m planet_ruler.benchmarks.synthetic --help")


if __name__ == "__main__":
    main()
