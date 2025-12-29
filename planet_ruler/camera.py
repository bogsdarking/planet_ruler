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

# planet_ruler/camera.py
"""
Automatic extraction of camera parameters from smartphone images.
Uses EXIF data and a phone camera database.
"""

from PIL import Image
from PIL.ExifTags import TAGS
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# Camera database - sensor dimensions in mm
# Includes phones, point-and-shoot, DSLRs, etc.
CAMERA_DB = {
    # ========================================================================
    # Multi-Camera Phones
    # ========================================================================
    
    # Samsung Galaxy S22+ (SM-S906E/B/U variants)
    "SM-S906E": {
        "type": "phone",
        "cameras": [
            {
                "name": "main",
                "focal_length_mm": 5.4,
                "aperture": 1.8,
                "sensor_width": 7.6,
                "sensor_height": 5.7,
                "notes": "50MP main camera",
            },
            {
                "name": "telephoto",
                "focal_length_mm": 7.0,
                "aperture": 2.4,
                "sensor_width": 5.16,
                "sensor_height": 3.87,
                "notes": "10MP 3x optical zoom",
            },
            {
                "name": "ultrawide",
                "focal_length_mm": 2.2,
                "aperture": 2.2,
                "sensor_width": 6.4,
                "sensor_height": 4.8,
                "notes": "12MP ultra-wide",
            },
        ],
    },
    
    # Samsung Galaxy S22 Ultra (SM-S908E/B/U)
    "SM-S908E": {
        "type": "phone",
        "cameras": [
            {
                "name": "main",
                "focal_length_mm": 5.4,
                "aperture": 1.8,
                "sensor_width": 8.0,
                "sensor_height": 6.0,
                "notes": "108MP main camera",
            },
            {
                "name": "telephoto_3x",
                "focal_length_mm": 6.9,
                "aperture": 2.4,
                "sensor_width": 5.16,
                "sensor_height": 3.87,
                "notes": "10MP 3x optical zoom",
            },
            {
                "name": "telephoto_10x",
                "focal_length_mm": 23.0,
                "aperture": 4.9,
                "sensor_width": 4.32,
                "sensor_height": 3.24,
                "notes": "10MP 10x optical zoom",
            },
            {
                "name": "ultrawide",
                "focal_length_mm": 2.2,
                "aperture": 2.2,
                "sensor_width": 6.4,
                "sensor_height": 4.8,
                "notes": "12MP ultra-wide",
            },
        ],
    },
    
    # iPhone 13 (base model)
    "iPhone 13": {
        "type": "phone",
        "cameras": [
            {
                "name": "main",
                "focal_length_mm": 5.1,
                "aperture": 1.6,
                "sensor_width": 7.6,
                "sensor_height": 5.7,
                "notes": "12MP main camera",
            },
            {
                "name": "ultrawide",
                "focal_length_mm": 1.54,
                "aperture": 2.4,
                "sensor_width": 5.68,
                "sensor_height": 4.26,
                "notes": "12MP ultra-wide",
            },
        ],
    },
    
    # iPhone 13 Pro/Pro Max
    "iPhone 13 Pro": {
        "type": "phone",
        "cameras": [
            {
                "name": "main",
                "focal_length_mm": 5.7,
                "aperture": 1.5,
                "sensor_width": 8.47,
                "sensor_height": 6.35,
                "notes": "12MP main camera",
            },
            {
                "name": "telephoto",
                "focal_length_mm": 9.0,
                "aperture": 2.8,
                "sensor_width": 5.68,
                "sensor_height": 4.26,
                "notes": "12MP 3x optical zoom",
            },
            {
                "name": "ultrawide",
                "focal_length_mm": 2.0,
                "aperture": 1.8,
                "sensor_width": 5.68,
                "sensor_height": 4.26,
                "notes": "12MP ultra-wide",
            },
        ],
    },
    
    # iPhone 14 Pro/Pro Max
    "iPhone 14 Pro": {
        "type": "phone",
        "cameras": [
            {
                "name": "main",
                "focal_length_mm": 6.9,
                "aperture": 1.78,
                "sensor_width": 9.8,
                "sensor_height": 7.35,
                "notes": "48MP main camera",
            },
            {
                "name": "telephoto",
                "focal_length_mm": 9.0,
                "aperture": 2.8,
                "sensor_width": 5.68,
                "sensor_height": 4.26,
                "notes": "12MP 3x optical zoom",
            },
            {
                "name": "ultrawide",
                "focal_length_mm": 2.0,
                "aperture": 2.2,
                "sensor_width": 5.68,
                "sensor_height": 4.26,
                "notes": "12MP ultra-wide",
            },
        ],
    },
    
    # iPhone 15 Pro/Pro Max
    "iPhone 15 Pro": {
        "type": "phone",
        "cameras": [
            {
                "name": "main",
                "focal_length_mm": 6.86,
                "aperture": 1.78,
                "sensor_width": 9.8,
                "sensor_height": 7.35,
                "notes": "48MP main camera",
            },
            {
                "name": "telephoto",
                "focal_length_mm": 9.0,  # Pro
                "aperture": 2.8,
                "sensor_width": 5.68,
                "sensor_height": 4.26,
                "notes": "12MP 3x optical zoom (Pro)",
            },
            {
                "name": "ultrawide",
                "focal_length_mm": 2.0,
                "aperture": 2.2,
                "sensor_width": 5.68,
                "sensor_height": 4.26,
                "notes": "12MP ultra-wide",
            },
        ],
    },
    
    # iPhone 15 Pro Max (different telephoto)
    "iPhone 15 Pro Max": {
        "type": "phone",
        "cameras": [
            {
                "name": "main",
                "focal_length_mm": 6.86,
                "aperture": 1.78,
                "sensor_width": 9.8,
                "sensor_height": 7.35,
                "notes": "48MP main camera",
            },
            {
                "name": "telephoto",
                "focal_length_mm": 12.7,  # Pro Max
                "aperture": 2.8,
                "sensor_width": 5.68,
                "sensor_height": 4.26,
                "notes": "12MP 5x optical zoom (Pro Max)",
            },
            {
                "name": "ultrawide",
                "focal_length_mm": 2.0,
                "aperture": 2.2,
                "sensor_width": 5.68,
                "sensor_height": 4.26,
                "notes": "12MP ultra-wide",
            },
        ],
    },
    
    # Google Pixel 7 Pro
    "Pixel 7 Pro": {
        "type": "phone",
        "cameras": [
            {
                "name": "main",
                "focal_length_mm": 6.81,
                "aperture": 1.85,
                "sensor_width": 8.0,
                "sensor_height": 6.0,
                "notes": "50MP main camera",
            },
            {
                "name": "telephoto",
                "focal_length_mm": 12.0,
                "aperture": 3.5,
                "sensor_width": 5.76,
                "sensor_height": 4.29,
                "notes": "48MP 5x optical zoom",
            },
            {
                "name": "ultrawide",
                "focal_length_mm": 2.2,
                "aperture": 2.2,
                "sensor_width": 6.4,
                "sensor_height": 4.8,
                "notes": "12MP ultra-wide",
            },
        ],
    },
    # Realme 9 Pro+ (RMX3393)
    "RMX3393": {
        "type": "phone",
        "cameras": [
            {
                "name": "main",
                "focal_length_mm": 5.7,  # Calculated from 24mm equiv. and sensor size
                "aperture": 1.8,
                "sensor_width": 8.19,
                "sensor_height": 6.14,
                "notes": "50MP Sony IMX766 with OIS",
            },
            {
                "name": "ultrawide",
                "focal_length_mm": 2.2,  # Estimated from 15mm equiv.
                "aperture": 2.2,
                "sensor_width": 6.4,  # Typical 8MP ultrawide
                "sensor_height": 4.8,
                "notes": "8MP ultra-wide, 119° FOV",
            },
            {
                "name": "macro",
                "focal_length_mm": 3.1,  # Estimated from 21.8mm equiv.
                "aperture": 2.4,
                "sensor_width": 2.76,  # Typical 2MP macro
                "sensor_height": 2.07,
                "notes": "2MP macro, fixed focus 4cm",
            },
        ],
    },
    # ========================================================================
    # Traditional Digital
    # ========================================================================
    # Canon PowerShot series
    "Canon PowerShot G12": {
        "sensor_width": 7.6,
        "sensor_height": 5.7,
        "type": "compact",
    },  # 1/1.7" sensor
    "Canon PowerShot G15": {
        "sensor_width": 7.6,
        "sensor_height": 5.7,
        "type": "compact",
    },
    "Canon PowerShot G16": {
        "sensor_width": 7.6,
        "sensor_height": 5.7,
        "type": "compact",
    },
    "Canon PowerShot SX50 HS": {
        "sensor_width": 6.17,
        "sensor_height": 4.55,
        "type": "compact",
    },  # 1/2.3"
    # Canon DSLRs
    "Canon EOS 5D Mark III": {
        "sensor_width": 36.0,
        "sensor_height": 24.0,
        "type": "dslr",
    },
    "Canon EOS 6D": {"sensor_width": 35.8, "sensor_height": 23.9, "type": "dslr"},
    "Canon EOS Rebel T7i": {
        "sensor_width": 22.3,
        "sensor_height": 14.9,
        "type": "dslr",
    },
    "Canon EOS RP": {"sensor_width": 36, "sensor_height": 24, "type": "dslr"},
    # Nikon cameras
    "NIKON D850": {"sensor_width": 35.9, "sensor_height": 23.9, "type": "dslr"},
    "NIKON D750": {"sensor_width": 35.9, "sensor_height": 24.0, "type": "dslr"},
    "NIKON D4": {"sensor_width": 36.0, "type": "dslr"},
    # Sony cameras
    "ILCE-7RM3": {
        "sensor_width": 35.9,
        "sensor_height": 24.0,
        "type": "mirrorless",
    },  # A7R III
    "ILCE-7M3": {
        "sensor_width": 35.6,
        "sensor_height": 23.8,
        "type": "mirrorless",
    },  # A7 III
    # Common sensor sizes (by sensor type name)
    "1/2.3": {
        "sensor_width": 6.17,
        "sensor_height": 4.55,
        "type": "compact",
    },  # Most compact cameras
    "1/1.7": {
        "sensor_width": 7.6,
        "sensor_height": 5.7,
        "type": "compact",
    },  # Premium compacts
    "1": {"sensor_width": 13.2, "sensor_height": 8.8, "type": "compact"},  # 1" sensor
    "APS-C": {
        "sensor_width": 23.6,
        "sensor_height": 15.7,
        "type": "dslr",
    },  # Canon APS-C
    "Full Frame": {"sensor_width": 36.0, "sensor_height": 24.0, "type": "dslr"},
    # Fallback for unknown cameras
    "default": {"sensor_width": 7.6, "sensor_height": 5.7, "type": "unknown"},
}


def extract_exif(image_path: str) -> Dict:
    """Extract EXIF data from image."""
    try:
        image = Image.open(image_path)
        exif_data = {}

        if hasattr(image, "_getexif") and image._getexif() is not None:
            exif = image._getexif()
            for tag_id, value in exif.items():
                tag = TAGS.get(tag_id, tag_id)
                exif_data[tag] = value

        return exif_data
    except Exception as e:
        logger.warning(f"Failed to extract EXIF: {e}")
        return {}


def get_camera_model(exif_data: Dict) -> Optional[str]:
    """Extract camera model from EXIF data."""
    # Try to build full camera name from Make + Model
    make = exif_data.get("Make", "").strip()
    model = exif_data.get("Model", "").strip()

    # Build full model name
    if make and model:
        # Check if model already contains make
        if make.lower() not in model.lower():
            full_model = f"{make} {model}"
        else:
            full_model = model
    elif model:
        full_model = model
    elif make:
        full_model = make
    else:
        return None

    # Direct match in database
    if full_model in CAMERA_DB:
        return full_model

    # Try just the model name
    if model and model in CAMERA_DB:
        return model

    # Try partial matches (avoid matching sensor size entries like "1", "1/1.7", etc.)
    # Sort by length descending to prefer longer, more specific matches
    candidate_models = []
    for known_model in CAMERA_DB:
        # Skip sensor size entries and very short entries that might cause false matches
        if len(known_model) <= 3 or known_model in ["1", "1/1.7", "1/2.3", "APS-C", "Full Frame", "default"]:
            continue
            
        if known_model.lower() in full_model.lower():
            candidate_models.append(known_model)
        elif known_model.lower() in model.lower():
            candidate_models.append(known_model)
    
    if candidate_models:
        # Return the longest match (most specific)
        return max(candidate_models, key=len)

    # Check LensModel as fallback (for some cameras)
    if "LensModel" in exif_data:
        lens_model = exif_data["LensModel"].strip()
        if lens_model in CAMERA_DB:
            return lens_model

    logger.info(f"Camera model '{full_model}' not in database")
    return None


def get_aperture(exif_data: Dict) -> Optional[float]:
    """
    Extract aperture (f-number) from EXIF data.
    
    Args:
        exif_data: EXIF dictionary
    
    Returns:
        Aperture as float (e.g., 2.4 for f/2.4), or None if not found
    """
    if not exif_data:
        return None
    
    try:
        from PIL.ExifTags import TAGS
        
        # Look for FNumber or ApertureValue tags
        for tag_id, value in exif_data.items():
            tag_name = TAGS.get(tag_id, tag_id)
            
            if tag_name == "FNumber":
                # FNumber is typically stored as rational (numerator, denominator)
                if isinstance(value, tuple):
                    return value[0] / value[1]
                else:
                    return float(value)
            
            elif tag_name == "ApertureValue":
                # ApertureValue is stored as APEX value: f-number = 2^(value/2)
                if isinstance(value, tuple):
                    apex = value[0] / value[1]
                else:
                    apex = float(value)
                return 2 ** (apex / 2)
        
    except Exception as e:
        logger.debug(f"Could not extract aperture: {e}")
    
    return None


def match_camera_module(camera_data: Dict, focal_length_mm: float, aperture: Optional[float] = None) -> Dict:
    """
    Match specific camera module for multi-camera phones.
    
    Uses focal length as primary matcher, aperture as secondary confirmation.
    Falls back to main camera if no match found.
    
    Args:
        camera_data: Camera entry from CAMERA_DB
        focal_length_mm: Actual focal length from EXIF
        aperture: F-number from EXIF (optional)
    
    Returns:
        Camera module dict with sensor_width, sensor_height, etc.
    """
    # Legacy format: single camera (no "cameras" array)
    if "cameras" not in camera_data:
        return camera_data
    
    # Multi-camera phone: match based on focal length + aperture
    cameras = camera_data["cameras"]
    
    # Tolerance for focal length matching (±0.3mm accounts for manufacturing variance)
    FOCAL_LENGTH_TOLERANCE = 0.3
    
    best_match = None
    best_score = -1
    
    for cam in cameras:
        score = 0
        
        # Primary: Focal length match
        focal_diff = abs(cam["focal_length_mm"] - focal_length_mm)
        if focal_diff <= FOCAL_LENGTH_TOLERANCE:
            score += 100  # Strong match
            score -= focal_diff * 10  # Prefer closer matches
        else:
            continue  # Skip if focal length doesn't match
        
        # Secondary: Aperture match (if available)
        if aperture is not None and "aperture" in cam:
            aperture_diff = abs(cam["aperture"] - aperture)
            if aperture_diff < 0.1:
                score += 50  # Confirms correct camera
            elif aperture_diff < 0.5:
                score += 20  # Close enough
            # Larger diff doesn't disqualify, just doesn't add points
        
        if score > best_score:
            best_score = score
            best_match = cam
    
    if best_match:
        logger.info(
            f"Matched '{best_match['name']}' camera: "
            f"f={best_match['focal_length_mm']:.1f}mm, f/{best_match['aperture']}"
        )
        return best_match
    
    # No match found - fall back to main camera with warning
    main_camera = cameras[0]  # Convention: main camera is first
    logger.warning(
        f"Could not match camera module (f={focal_length_mm:.1f}mm"
        + (f", f/{aperture}" if aperture else "")
        + f") - using '{main_camera['name']}' camera as fallback"
    )
    return main_camera


def get_focal_length_mm(exif_data: Dict) -> Optional[float]:
    """Extract focal length in mm from EXIF."""
    if "FocalLength" in exif_data:
        focal_length = exif_data["FocalLength"]

        # Handle tuple format (numerator, denominator)
        if isinstance(focal_length, tuple):
            focal_length = focal_length[0] / focal_length[1]

        return float(focal_length)

    return None


def get_focal_length_35mm_equiv(exif_data: Dict) -> Optional[float]:
    """Get 35mm equivalent focal length (more reliable for phones)."""
    if "FocalLengthIn35mmFilm" in exif_data:
        return float(exif_data["FocalLengthIn35mmFilm"])
    return None


def get_image_dimensions(image_path: str) -> Tuple[int, int]:
    """Get image width and height in pixels."""
    with Image.open(image_path) as img:
        return img.size  # (width, height)


def get_image_orientation_from_exif(exif_data: Optional[Dict] = None) -> Optional[str]:
    """
    Determine orientation from EXIF Orientation tag only.
    
    Args:
        exif_data: Pre-extracted EXIF data
    
    Returns:
        'portrait', 'landscape', or None if can't determine
    """
    if exif_data is None:
        return None
    
    try:
        from PIL.ExifTags import TAGS
        
        # Find Orientation tag
        orientation = None
        for tag_id, value in exif_data.items():
            tag_name = TAGS.get(tag_id, tag_id)
            if tag_name == "Orientation":
                orientation = value
                break
        
        if orientation is not None:
            # Values 5, 6, 7, 8 indicate 90° or 270° rotation
            if orientation in [5, 6, 7, 8]:
                logger.info(f"EXIF Orientation={orientation}: portrait mode detected")
                return 'portrait'
            else:
                logger.info(f"EXIF Orientation={orientation}: landscape mode detected")
                return 'landscape'
    except Exception as e:
        logger.debug(f"Could not read EXIF Orientation tag: {e}")
    
    return None


def get_image_orientation_from_dimensions(width: int, height: int) -> str:
    """
    Determine orientation from image dimensions (heuristic).
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
    
    Returns:
        'portrait' or 'landscape'
    """
    if height > width:
        logger.info(f"Image dimensions {width}×{height}: portrait mode (heuristic)")
        return 'portrait'
    else:
        logger.info(f"Image dimensions {width}×{height}: landscape mode (heuristic)")
        return 'landscape'


def apply_orientation_correction(params: Dict, exif_data: Optional[Dict] = None) -> Dict:
    """
    Apply orientation correction to camera parameters.
    
    This modifies params in-place and returns it.
    Call this RIGHT BEFORE each return statement in extract_camera_parameters().
    
    Args:
        params: Camera parameters dict (must have image_width_px, image_height_px)
        exif_data: Optional pre-extracted EXIF data
    
    Returns:
        params: Modified parameters (same object, modified in-place)
    """
    # Detect orientation (try EXIF first, fall back to dimensions)
    orientation = get_image_orientation_from_exif(exif_data)
    
    if orientation is None:
        # Fall back to dimensions (already in params, no file I/O needed!)
        if params.get("image_width_px") and params.get("image_height_px"):
            orientation = get_image_orientation_from_dimensions(
                params["image_width_px"],
                params["image_height_px"]
            )
        else:
            # Can't determine orientation, skip correction
            logger.debug("Cannot determine orientation - skipping sensor correction")
            return params
    
    # Correct sensor dimensions
    if params["sensor_width_mm"] is not None and params["sensor_height_mm"] is not None:
        sensor_is_landscape = params["sensor_width_mm"] > params["sensor_height_mm"]
        
        # If orientations don't match, swap dimensions
        if orientation == 'portrait' and sensor_is_landscape:
            logger.info(
                f"Portrait image with landscape sensor: swapping dimensions "
                f"{params['sensor_width_mm']:.1f}mm × {params['sensor_height_mm']:.1f}mm → "
                f"{params['sensor_height_mm']:.1f}mm × {params['sensor_width_mm']:.1f}mm"
            )
            params["sensor_width_mm"], params["sensor_height_mm"] = (
                params["sensor_height_mm"], 
                params["sensor_width_mm"]
            )
        elif orientation == 'landscape' and not sensor_is_landscape:
            logger.info(
                f"Landscape image with portrait sensor: swapping dimensions "
                f"{params['sensor_width_mm']:.1f}mm × {params['sensor_height_mm']:.1f}mm → "
                f"{params['sensor_height_mm']:.1f}mm × {params['sensor_width_mm']:.1f}mm"
            )
            params["sensor_width_mm"], params["sensor_height_mm"] = (
                params["sensor_height_mm"],
                params["sensor_width_mm"]
            )
    
    # Also correct min/max ranges if they exist
    if params.get("sensor_width_min") and params.get("sensor_height_min"):
        sensor_min_is_landscape = params["sensor_width_min"] > params["sensor_height_min"]
        if (orientation == 'portrait' and sensor_min_is_landscape) or \
           (orientation == 'landscape' and not sensor_min_is_landscape):
            params["sensor_width_min"], params["sensor_height_min"] = (
                params["sensor_height_min"],
                params["sensor_width_min"]
            )
    
    if params.get("sensor_width_max") and params.get("sensor_height_max"):
        sensor_max_is_landscape = params["sensor_width_max"] > params["sensor_height_max"]
        if (orientation == 'portrait' and sensor_max_is_landscape) or \
           (orientation == 'landscape' and not sensor_max_is_landscape):
            params["sensor_width_max"], params["sensor_height_max"] = (
                params["sensor_height_max"],
                params["sensor_width_max"]
            )
    
    return params


def calculate_sensor_dimensions(
    focal_length_mm: float, focal_length_35mm: float
) -> Tuple[float, float]:
    """
    Calculate sensor dimensions from focal length ratio.
    Uses the relationship: focal_length_mm / sensor_width = focal_length_35mm / 36mm
    """
    # 35mm film dimensions
    full_frame_width = 36.0  # mm
    full_frame_height = 24.0  # mm

    # Calculate crop factor
    crop_factor = focal_length_35mm / focal_length_mm

    # Calculate sensor dimensions
    sensor_width = full_frame_width / crop_factor
    sensor_height = full_frame_height / crop_factor

    return sensor_width, sensor_height


def get_sensor_statistics_by_type(camera_type: str) -> Optional[Dict]:
    """
    Calculate sensor dimension statistics for a given camera type.
    Returns median, min, and max for sensor width and height.
    """
    cameras_of_type = [cam for cam in CAMERA_DB.values() if cam["type"] == camera_type]

    if not cameras_of_type:
        return None

    # Collect sensor dimensions, handling both old and new camera database formats
    widths = []
    heights = []
    
    for cam in cameras_of_type:
        # Multi-camera format
        if "cameras" in cam:
            for camera_module in cam["cameras"]:
                if "sensor_width" in camera_module:
                    widths.append(camera_module["sensor_width"])
                if "sensor_height" in camera_module:
                    heights.append(camera_module["sensor_height"])
        # Single-camera format
        else:
            if "sensor_width" in cam:
                widths.append(cam["sensor_width"])
            if "sensor_height" in cam:
                heights.append(cam["sensor_height"])

    if not widths or not heights:
        return None

    import numpy as np

    stats = {
        "sensor_width_median": float(np.median(widths)),
        "sensor_width_min": float(np.min(widths)),
        "sensor_width_max": float(np.max(widths)),
        "sensor_height_median": float(np.median(heights)),
        "sensor_height_min": float(np.min(heights)),
        "sensor_height_max": float(np.max(heights)),
        "count": len(widths),  # Count individual camera modules, not camera models
    }

    return stats


def infer_camera_type(exif_data: Dict) -> Optional[str]:
    """
    Try to infer camera type from EXIF data even if exact model unknown.
    """
    make = exif_data.get("Make", "").lower()
    model = exif_data.get("Model", "").lower()

    # Phone indicators
    phone_indicators = [
        "iphone",
        "pixel",
        "samsung",
        "galaxy",
        "sm-",
        "oneplus",
        "xiaomi",
        "huawei",
    ]
    if any(indicator in model or indicator in make for indicator in phone_indicators):
        return "phone"

    # DSLR indicators
    dslr_indicators = ["eos", "nikon d", "rebel", "mark"]
    if any(indicator in model for indicator in dslr_indicators):
        return "dslr"

    # Mirrorless indicators
    mirrorless_indicators = ["ilce", "alpha", "a7", "z 6", "z 7", "eos r", "eos m"]
    if any(indicator in model for indicator in mirrorless_indicators):
        return "mirrorless"

    # Compact/point-and-shoot indicators
    compact_indicators = ["powershot", "coolpix", "cyber-shot", "dmc-", "lumix"]
    if any(indicator in model or indicator in make for indicator in compact_indicators):
        return "compact"

    return None


def extract_camera_parameters(image_path: str) -> Dict:
    """
    Automatically extract all camera parameters from any camera image.
    Handles phones, point-and-shoot cameras, DSLRs, mirrorless, etc.

    Returns:
        dict: Camera parameters including:
            - focal_length_mm: focal length in millimeters
            - sensor_width_mm: sensor width in millimeters
            - sensor_height_mm: sensor height in millimeters
            - sensor_width_min: minimum sensor width (if using type statistics)
            - sensor_width_max: maximum sensor width (if using type statistics)
            - image_width_px: image width in pixels
            - image_height_px: image height in pixels
            - camera_model: detected camera model
            - camera_type: 'phone', 'compact', 'dslr', 'mirrorless', or 'unknown'
            - confidence: 'high', 'medium', or 'low'
    """
    params = {
        "focal_length_mm": None,
        "sensor_width_mm": None,
        "sensor_height_mm": None,
        "sensor_width_min": None,
        "sensor_width_max": None,
        "sensor_height_min": None,
        "sensor_height_max": None,
        "image_width_px": None,
        "image_height_px": None,
        "camera_model": None,
        "camera_type": None,
        "confidence": "low",
    }

    # Get image dimensions (always available)
    params["image_width_px"], params["image_height_px"] = get_image_dimensions(
        image_path
    )

    # Extract EXIF data
    exif_data = extract_exif(image_path)

    if not exif_data:
        logger.warning("No EXIF data found - using defaults")
        return params

    # Get camera model
    camera_model = get_camera_model(exif_data)
    params["camera_model"] = camera_model

    # Get focal length (almost always available in EXIF)
    focal_length_mm = get_focal_length_mm(exif_data)
    focal_length_35mm = get_focal_length_35mm_equiv(exif_data)
    params["focal_length_mm"] = focal_length_mm

    # Strategy 1: Known camera model (highest confidence)
    if camera_model and camera_model in CAMERA_DB:
        camera_data = CAMERA_DB[camera_model]
        
        # For multi-camera phones, match specific camera module
        aperture = get_aperture(exif_data)
        if focal_length_mm is not None:
            camera_module = match_camera_module(camera_data, focal_length_mm, aperture)
        else:
            # If no focal length, use first camera (main) as fallback
            camera_module = camera_data.get("cameras", [camera_data])[0] if "cameras" in camera_data else camera_data
        
        params["sensor_width_mm"] = camera_module.get("sensor_width", None)
        params["sensor_height_mm"] = camera_module.get("sensor_height", None)

        if (params["sensor_width_mm"]) is None and (params["sensor_height_mm"] is None):
            logger.warning(
                f"Known camera missing critical parameter -- supply sensor_width and/or sensor_height for {camera_model}"
            )
        else:
            if (
                params["sensor_height_mm"] is None
                and params["sensor_width_mm"] is not None
            ):
                params["sensor_height_mm"] = (
                    params["sensor_width_mm"]
                    * params["image_height_px"]
                    / params["image_width_px"]
                )

            if (
                params["sensor_width_mm"] is None
                and params["sensor_height_mm"] is not None
            ):
                params["sensor_width_mm"] = (
                    params["sensor_height_mm"]
                    / params["image_height_px"]
                    * params["image_width_px"]
                )

            params["camera_type"] = camera_data["type"]
            params["confidence"] = "high"
            logger.info(
                f"Detected known camera: {camera_model} ({camera_data['type']})"
            )
            return apply_orientation_correction(params, exif_data)

    # Strategy 2: Calculate from focal length ratio (medium-high confidence)
    if focal_length_mm and focal_length_35mm:
        sensor_width, sensor_height = calculate_sensor_dimensions(
            focal_length_mm, focal_length_35mm
        )
        params["sensor_width_mm"] = sensor_width
        params["sensor_height_mm"] = sensor_height
        params["camera_type"] = "calculated"
        params["confidence"] = "medium"
        logger.info(
            f"Calculated sensor dimensions from focal length ratio: {sensor_width:.1f}mm × {sensor_height:.1f}mm"
        )
        return apply_orientation_correction(params, exif_data)

    # Strategy 3: Infer from camera type (medium-low confidence)
    inferred_type = infer_camera_type(exif_data)
    if inferred_type:
        stats = get_sensor_statistics_by_type(inferred_type)
        if stats:
            params["sensor_width_mm"] = stats["sensor_width_median"]
            params["sensor_height_mm"] = stats["sensor_height_median"]
            params["sensor_width_min"] = stats["sensor_width_min"]
            params["sensor_width_max"] = stats["sensor_width_max"]
            params["sensor_height_min"] = stats["sensor_height_min"]
            params["sensor_height_max"] = stats["sensor_height_max"]
            params["camera_type"] = inferred_type
            params["confidence"] = "medium-low"
            logger.info(
                f"Inferred camera type '{inferred_type}' - using median sensor: {stats['sensor_width_median']:.1f}mm × {stats['sensor_height_median']:.1f}mm"
            )
            logger.info(
                f"  Sensor width range: [{stats['sensor_width_min']:.1f}, {stats['sensor_width_max']:.1f}] mm (from {stats['count']} cameras)"
            )
            return apply_orientation_correction(params, exif_data)

    # Strategy 4: Use defaults (low confidence)
    logger.warning(
        f"Using default camera parameters for '{camera_model}' - accuracy may be reduced"
    )
    default_data = CAMERA_DB["default"]
    params["sensor_width_mm"] = default_data["sensor_width"]
    params["sensor_height_mm"] = default_data["sensor_height"]
    params["camera_type"] = "default"
    params["focal_length_mm"] = (
        focal_length_mm or 6.0
    )  # typical compact camera focal length
    params["confidence"] = "low"

    return apply_orientation_correction(params, exif_data)


def get_gps_altitude(image_path: str) -> Optional[float]:
    """
    Extract GPS altitude from EXIF data if available.

    Returns:
        float: Altitude in meters, or None if not available
    """
    exif_data = extract_exif(image_path)

    # GPS altitude is in GPSInfo tag
    if "GPSInfo" in exif_data:
        gps_info = exif_data["GPSInfo"]

        # Altitude is usually tag 6
        if 6 in gps_info:
            altitude = gps_info[6]
            if isinstance(altitude, tuple):
                altitude = altitude[0] / altitude[1]

            # Check altitude reference (0 = above sea level, 1 = below)
            altitude_ref = gps_info.get(5, 0)
            if altitude_ref == 1:
                altitude = -altitude

            return float(altitude)

    return None


# Planet radius database (in meters) - TRUE VALUES
# NOTE: These are perturbed during initialization to avoid local minima
PLANET_RADII = {
    "earth": 6371000,
    "mars": 3389500,
    "jupiter": 69911000,
    "saturn": 58232000,
    "uranus": 25362000,
    "neptune": 24622000,
    "venus": 6051800,
    "mercury": 2439700,
    "moon": 1737400,
    "pluto": 1188300,
}


def get_initial_radius(
    planet: str = "earth", perturbation_factor: float = 0.5, seed: Optional[int] = None
) -> float:
    """
    Get initial radius guess with perturbation.

    Args:
        planet: Planet name
        perturbation_factor: Relative perturbation (default: 0.5 = ±50%)
        seed: Random seed for reproducibility (default: None = unseeded)
    """
    import random

    if seed is not None:
        random.seed(seed)

    if planet.lower() in PLANET_RADII:
        true_value = PLANET_RADII[planet.lower()]
        min_factor = 1.0 - perturbation_factor
        max_factor = 1.0 + perturbation_factor
        perturbation = random.uniform(min_factor, max_factor)
        r_init = true_value * perturbation
        logger.info(
            f"Perturbed {planet} radius: {true_value/1000:.0f} km → {r_init/1000:.0f} km ({perturbation:.2f}x)"
        )
        return r_init
    else:
        logger.warning(f"Unknown planet '{planet}', using middle-range guess")
        return 10_000_000  # 10,000 km


def create_config_from_image(
    image_path: str,
    altitude_m: Optional[float] = None,
    planet: str = "earth",
    param_tolerance: float = 0.1,
    perturbation_factor: float = 0.5,
    seed: Optional[int] = None,
) -> Dict:
    """
    Create a complete planet_ruler configuration from an image.
    Works with any camera - phones, point-and-shoot, DSLRs, etc.

    Args:
        image_path: Path to the image
        altitude_m: Altitude in meters (REQUIRED if not in GPS data)
        planet: Planet name for initial radius guess (default: 'earth')
        param_tolerance: Fractional tolerance for parameter limits (default: 0.1 = ±10%)
        perturbation_factor: Initial radius perturbation (default: 0.5 = ±50%)
        seed: Random seed for reproducibility (default: None = unseeded)

    Returns:
        dict: Configuration ready for planet_ruler

    Raises:
        ValueError: If altitude cannot be determined from GPS and not provided manually

    Notes:
        - Initial radius is randomly perturbed to avoid local minima and prove data-driven results
        - For uncertainty quantification, consider running multiple times (multi-start optimization)
        - Theta parameters (orientation) have wide default limits to handle r-h coupling
    """
    # Extract camera parameters
    camera_params = extract_camera_parameters(image_path)

    # Try to get GPS altitude if not provided
    if altitude_m is None:
        altitude_m = get_gps_altitude(image_path)

    # Altitude is REQUIRED - no placeholders
    if altitude_m is None:
        raise ValueError(
            "Altitude is required but could not be extracted from image GPS data. "
            "Please provide altitude_m parameter explicitly:\n"
            "  config = create_config_from_image('photo.jpg', altitude_m=10668)\n"
            "Altitude is critical for accurate planetary radius measurement."
        )

    # Get initial planet radius with perturbation to avoid local minima
    r_init = get_initial_radius(planet, perturbation_factor, seed=seed)

    # Determine which camera parameters to use (pick 2 of 3)
    # Priority: focal_length > sensor_width > field_of_view
    free_params = ["r", "h", "theta_x", "theta_y", "theta_z"]  # Always free
    init_values = {"r": r_init, "h": altitude_m}
    param_limits = {"r": [1000000, 100000000]}  # Wide range: 1000 km to 100,000 km

    # Always include focal length if available (highest priority)
    if camera_params["focal_length_mm"]:
        focal_m = camera_params["focal_length_mm"] / 1000
        free_params.append("f")
        init_values["f"] = focal_m
        # Tight constraint: ±param_tolerance
        param_limits["f"] = [
            focal_m * (1 - param_tolerance),
            focal_m * (1 + param_tolerance),
        ]

    # Include sensor width if available (second priority)
    if camera_params["sensor_width_mm"]:
        sensor_m = camera_params["sensor_width_mm"] / 1000
        free_params.append("w")
        init_values["w"] = sensor_m

        # Use data-driven limits if available (from camera type statistics)
        if camera_params["sensor_width_min"] and camera_params["sensor_width_max"]:
            param_limits["w"] = [
                camera_params["sensor_width_min"] / 1000,
                camera_params["sensor_width_max"] / 1000,
            ]
            logger.info(
                f"Using data-driven sensor width limits: [{param_limits['w'][0]*1000:.1f}, {param_limits['w'][1]*1000:.1f}] mm"
            )
        else:
            # Use tight constraint for known sensors
            param_limits["w"] = [
                sensor_m * (1 - param_tolerance),
                sensor_m * (1 + param_tolerance),
            ]

    # Constrain altitude with tolerance since precision altitude is rare
    param_limits["h"] = [
        float(altitude_m * (1 - param_tolerance)),
        float(altitude_m * (1 + param_tolerance)),
    ]

    # Calculate initial theta_x using geometry
    # theta_y and theta_z default to 0 (user doesn't know orientation)
    try:
        from planet_ruler.geometry import limb_camera_angle

        theta_x_init = limb_camera_angle(r_init, altitude_m)
        init_values["theta_x"] = theta_x_init
        init_values["theta_y"] = 0.0
        init_values["theta_z"] = 0.0
        logger.info(f"Initialized theta_x = {theta_x_init:.6f} rad from geometry")
    except ImportError:
        logger.warning(
            "Could not import planet_ruler.geometry.limb_camera_angle - using default theta_x=0"
        )
        init_values["theta_x"] = 0.0
        init_values["theta_y"] = 0.0
        init_values["theta_z"] = 0.0
    except Exception as e:
        logger.warning(f"Error calculating initial theta_x: {e} - using default=0")
        init_values["theta_x"] = 0.0
        init_values["theta_y"] = 0.0
        init_values["theta_z"] = 0.0

    # Be very generous with directional limits by default
    param_limits["theta_x"] = [-3.14, 3.14]
    param_limits["theta_y"] = [-3.14, 3.14]
    param_limits["theta_z"] = [-3.14, 3.14]

    # Build config
    config = {
        "description": f"Auto-generated from {Path(image_path).name} (planet: {planet})",
        "free_parameters": free_params,
        "init_parameter_values": init_values,
        "parameter_limits": param_limits,
        "camera_info": {
            "camera_model": camera_params["camera_model"],
            "camera_type": camera_params["camera_type"],
            "confidence": camera_params["confidence"],
            "image_dimensions": [
                camera_params["image_width_px"],
                camera_params["image_height_px"],
            ],
            "planet": planet,
            "altitude_source": "gps" if get_gps_altitude(image_path) else "manual",
        },
        "notes": {
            "r_init_perturbed": True,
            "perturbation_info": "Initial radius randomly perturbed ±50% to avoid local minima",
            "multi_start_ready": "Config suitable for multi-start optimization if desired",
        },
    }

    return config


# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        planet = sys.argv[2] if len(sys.argv) > 2 else "earth"
        altitude = float(sys.argv[3]) if len(sys.argv) > 3 else None

        print("Extracting camera parameters...")
        params = extract_camera_parameters(image_path)

        print("\nCamera Parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")

        print(f"\nGenerating config for planet: {planet}")
        try:
            config = create_config_from_image(
                image_path, altitude_m=altitude, planet=planet
            )

            print("\nConfig:")
            import json

            print(json.dumps(config, indent=2))
        except ValueError as e:
            print(f"\n❌ Error: {e}")
            print("\nTo fix this:")
            print(
                f"  python camera_params.py {image_path} {planet} <altitude_in_meters>"
            )
            print("\nExample:")
            print(f"  python camera_params.py {image_path} earth 10668")
    else:
        print("Usage: python camera_params.py <image_path> [planet] [altitude_m]")
        print("  planet: earth (default), mars, jupiter, saturn, moon, etc.")
        print("  altitude_m: altitude in meters (optional if GPS data available)")
        print("\nExamples:")
        print("  python camera_params.py airplane.jpg earth 10668")
        print("  python camera_params.py mars_rover.jpg mars 4500")
        print("\nMulti-start optimization example:")
        print("  # For uncertainty quantification, run multiple times:")
        print("  for i in range(10):")
        print("      config = create_config_from_image('photo.jpg', altitude_m=10000)")
        print("      result = fit_limb(config)  # Run your optimization")
        print("      print(f'Run {i}: r = {result.r}')")
        print("  # Consistency across runs proves data-driven measurement")
