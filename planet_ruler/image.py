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

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from tqdm.notebook import tqdm
import cv2
from PIL import Image
import logging

# Optional ML dependencies - gracefully handle import failures
try:
    import kagglehub
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    HAS_SEGMENT_ANYTHING = True
except ImportError:
    HAS_SEGMENT_ANYTHING = False
    kagglehub = None
    SamAutomaticMaskGenerator = None
    sam_model_registry = None
    logging.warning(
        "segment-anything and/or kagglehub not available. AI-powered horizon detection will be disabled."
    )


def load_image(filepath: str) -> np.ndarray:
    """
    Load a 3 or 4-channel image from filepath into an array.

    Args:
        filepath (str): Path to image file.
    Returns:
        image array (np.ndarray)
    """
    img = Image.open(filepath)
    im_arr = np.array(img)

    # for segmentation to work we need 3 channels
    if len(im_arr.shape) == 2:
        im_arr = np.dstack([im_arr] * 3)
    if im_arr.shape[2] == 4:
        im_arr = im_arr[:, :, :3]

    return im_arr


def directional_gradient_blur(
    image, sigma_base=2.0, streak_length=20, decay_rate=0.2, normalize_gradients=True
):
    """
    Blur along gradient directions to create smooth 'streaks' toward edges.

    For each pixel, follow its gradient direction outward, averaging pixels
    along that ray with exponential decay. This:
    - Smooths noisy/weak gradients (striations with inconsistent directions)
    - Strengthens coherent gradients (limb where all gradients align)
    - Creates smooth 'valleys' guiding optimizer toward strong edges

    Args:
        image (np.ndarray): Input image (H x W or H x W x 3)
        sigma_base (float): Initial gradient smoothing for direction estimation
        streak_length (int): How far to follow gradient direction (pixels)
        decay_rate (float): Exponential decay rate (higher = faster decay)
        normalize_gradients (bool):
            - True: Use unit vectors (direction only) - RECOMMENDED
              All pixels sample same distance regardless of gradient strength.
              Consistent, predictable behavior.
            - False: Use full gradient magnitude as direction
              Strong gradients sample further. Can create artifacts.

    Returns:
        blurred_grad_mag (np.ndarray): Blurred gradient magnitude field
        grad_angle (np.ndarray): Gradient angle field

    Note: This is UNI-directional (samples in one direction only).
          Use bidirectional_gradient_blur() to preserve peak locations.
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = image.sum(axis=2).astype(np.float32)
    else:
        gray = image.copy().astype(np.float32)

    # Initial gradient computation with small smoothing
    if sigma_base > 0:
        gray_smooth = cv2.GaussianBlur(gray, (0, 0), sigma_base)
    else:
        gray_smooth = gray

    grad_y, grad_x = np.gradient(gray_smooth)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_angle = np.arctan2(grad_y, grad_x)

    # Normalize gradient directions (optionally normalize magnitudes too)
    if normalize_gradients:
        # Unit vectors in gradient direction
        grad_mag_norm = np.copy(grad_mag)
        grad_mag_norm[grad_mag_norm < 1e-6] = 1.0  # Avoid division by zero
        grad_x_unit = grad_x / grad_mag_norm
        grad_y_unit = grad_y / grad_mag_norm
    else:
        grad_x_unit = grad_x
        grad_y_unit = grad_y

    height, width = gray.shape
    blurred_mag = np.zeros_like(grad_mag)

    # For each pixel, integrate along gradient direction
    y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")

    for step in range(streak_length):
        # Weight for this step (exponential decay)
        weight = np.exp(-decay_rate * step)

        # Follow gradient direction by 'step' pixels
        # Note: gradient points toward increasing intensity
        # We want to blur "outward" from edges, so follow gradient direction
        offset_x = grad_x_unit * step
        offset_y = grad_y_unit * step

        # Sample coordinates (with subpixel precision)
        sample_x = x_coords + offset_x
        sample_y = y_coords + offset_y

        # Bilinear interpolation for sub-pixel sampling
        sample_mag = bilinear_interpolate(grad_mag, sample_y, sample_x)

        # Accumulate weighted contribution
        blurred_mag += weight * sample_mag

    # Normalize by total weight
    total_weight = (1 - np.exp(-decay_rate * streak_length)) / (1 - np.exp(-decay_rate))
    blurred_mag /= total_weight

    return blurred_mag, grad_angle


def bilinear_interpolate(array, y, x):
    """
    Bilinear interpolation for 2D array at non-integer coordinates.

    Args:
        array: 2D array to sample from
        y, x: Arrays of y and x coordinates (can be non-integer)

    Returns:
        Interpolated values at (y, x) positions
    """
    height, width = array.shape

    # Clip to valid range
    x = np.clip(x, 0, width - 1.001)
    y = np.clip(y, 0, height - 1.001)

    # Integer parts
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)

    # Fractional parts
    fx = x - x0
    fy = y - y0

    # Bilinear weights
    w00 = (1 - fx) * (1 - fy)
    w01 = (1 - fx) * fy
    w10 = fx * (1 - fy)
    w11 = fx * fy

    # Sample and interpolate
    result = (
        w00 * array[y0, x0]
        + w01 * array[y1, x0]
        + w10 * array[y0, x1]
        + w11 * array[y1, x1]
    )

    return result


def bidirectional_gradient_blur(
    image, sigma_base=2.0, streak_length=20, decay_rate=0.2, normalize_gradients=True
):
    """
    Blur in BOTH directions along gradient (forward and backward).

    This creates symmetric streaks on both sides of edges, preserving the
    location of gradient maxima while smoothing the field.

    Args:
        image: Input image
        sigma_base: Initial smoothing for gradient estimation
        streak_length: How far to sample in each direction
        decay_rate: Exponential decay (higher = faster falloff)
        normalize_gradients: Use unit vectors (direction only) vs full magnitude

    Returns:
        blurred_mag: Smoothed gradient magnitude field
        grad_angle: Original gradient angle field
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = image.sum(axis=2).astype(np.float32)
    else:
        gray = image.copy().astype(np.float32)

    # Initial gradient computation
    if sigma_base > 0:
        gray_smooth = cv2.GaussianBlur(gray, (0, 0), sigma_base)
    else:
        gray_smooth = gray

    grad_y, grad_x = np.gradient(gray_smooth)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_angle = np.arctan2(grad_y, grad_x)

    # Normalize to unit vectors (direction only, ignore magnitude)
    # This ensures we sample the same distance in both directions
    if normalize_gradients:
        grad_mag_safe = np.copy(grad_mag)
        grad_mag_safe[grad_mag_safe < 1e-6] = 1.0
        grad_x_unit = grad_x / grad_mag_safe
        grad_y_unit = grad_y / grad_mag_safe
    else:
        # Keep magnitude - will sample further in strong gradient regions
        grad_x_unit = grad_x
        grad_y_unit = grad_y

    height, width = gray.shape
    blurred_mag = np.zeros_like(grad_mag)
    y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")

    # Sample in BOTH directions symmetrically
    total_weight = 0.0

    for step in range(-streak_length, streak_length + 1):
        # Weight (symmetric, peak at step=0)
        weight = np.exp(-decay_rate * abs(step))
        total_weight += weight

        # Sample at offset along gradient direction
        # Positive step = follow gradient (dark → bright)
        # Negative step = opposite (bright → dark)
        offset_x = grad_x_unit * step
        offset_y = grad_y_unit * step

        sample_x = x_coords + offset_x
        sample_y = y_coords + offset_y

        # Bilinear interpolation
        sample_mag = bilinear_interpolate(grad_mag, sample_y, sample_x)

        # Accumulate
        blurred_mag += weight * sample_mag

    # Normalize by total weight
    blurred_mag /= total_weight

    return blurred_mag, grad_angle


def gradient_break(
    im_arr: np.ndarray,
    log: bool = False,
    y_min: int = 0,
    y_max: int = -1,
    window_length: int = None,
    polyorder: int = 1,
    deriv: int = 0,
    delta: int = 1,
):
    """
    Scan each vertical line of an image for the maximum change
    in brightness gradient -- usually corresponds to a horizon.

    Args:
        im_arr (np.ndarray): Image array.
        log (bool): Use the log(gradient). Sometimes good for
            smoothing.
        y_min (int): Minimum y-position to consider.
        y_max (int): Maximum y-position to consider.
        window_length (int): Width of window to apply smoothing
            for each vertical. Larger means less noise but less
            sensitivity.
        polyorder (int): Polynomial order for smoothing.
        deriv (int): Derivative level for smoothing.
        delta (int): Delta for smoothing.
    Returns:
        image array (np.ndarray)
    """
    # Auto-calculate window_length if not provided
    if window_length is None:
        # Use ~10% of image height, with reasonable bounds
        window_length = min(501, max(5, int(0.1 * im_arr.shape[0])))
        # Ensure odd window length for savgol_filter
        if window_length % 2 == 0:
            window_length -= 1
    else:
        # Validate provided window_length against image dimensions
        max_window = im_arr.shape[0] - 1
        if max_window % 2 == 0:
            max_window -= 1
        if window_length > max_window:
            window_length = max(5, max_window)
            if window_length % 2 == 0:
                window_length -= 1

    grad = abs(np.gradient(im_arr.sum(axis=2), axis=0))

    if log:
        grad[grad > 0] = np.log10(grad[grad > 0])
        grad = np.log10(grad)
        grad[np.isinf(grad)] = 0
        grad[grad < 0] = 0

    breaks = []
    for i in range(im_arr.shape[1]):
        y = grad[:, i]
        yhat = savgol_filter(
            y,
            window_length=window_length,
            polyorder=polyorder,
            deriv=deriv,
            delta=delta,
        )

        yhathat = np.diff(yhat)
        m = np.argmax(yhathat[y_min:y_max])
        breaks += [m + y_min]
    breaks = np.array(breaks)

    return breaks


class MaskSegmenter:
    """
    Method-agnostic mask-based segmentation.

    Supports pluggable backends (SAM, custom algorithms, etc.) with
    optional downsampling and interactive classification.
    """

    def __init__(
        self,
        image: np.ndarray,
        method: str = "sam",
        downsample_factor: int = 1,
        interactive: bool = True,
        **backend_kwargs,
    ):
        """
        Initialize mask segmenter.

        Args:
            image: Input image (H x W x 3)
            method: Backend method ('sam' or 'custom')
            downsample_factor: Downsample factor for speed (1 = no downsampling)
            interactive: Use interactive GUI for classification
            **backend_kwargs: Backend-specific arguments
        """
        self.image = image
        self.original_shape = image.shape[:2]  # (H, W)
        self.downsample_factor = downsample_factor
        self.interactive = interactive

        # Create backend
        self._backend = self._create_backend(method, **backend_kwargs)

        # Will hold masks after segmentation
        self._masks = None

    def _create_backend(self, method: str, **kwargs):
        """Create segmentation backend."""
        if method == "sam":
            return SAMBackend(**kwargs)
        elif method == "custom":
            if "segment_fn" not in kwargs:
                raise ValueError("custom method requires 'segment_fn' argument")
            return CustomBackend(kwargs["segment_fn"])
        else:
            raise ValueError(f"Unknown segmentation method: {method}")

    def segment(self) -> np.ndarray:
        """
        Run segmentation pipeline.

        Returns:
            Limb coordinates (1D array of y-values)
        """
        # Downsample if requested
        if self.downsample_factor > 1:
            work_height = self.original_shape[0] // self.downsample_factor
            work_width = self.original_shape[1] // self.downsample_factor
            work_image = cv2.resize(
                self.image, (work_width, work_height), interpolation=cv2.INTER_AREA
            )
            logging.info(f"Downsampled: {self.original_shape} → {work_image.shape[:2]}")
        else:
            work_image = self.image

        # Generate masks using backend
        self._masks = self._backend.segment(work_image)

        # Scale masks back to original size if needed
        if self.downsample_factor > 1:
            scaled_masks = []
            for mask_dict in self._masks:
                mask_array = mask_dict.get("mask", mask_dict.get("segmentation"))

                # Scale mask
                scaled_mask = cv2.resize(
                    mask_array.astype(np.uint8),
                    (self.original_shape[1], self.original_shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)

                # Update dict
                new_dict = mask_dict.copy()
                new_dict["mask"] = scaled_mask
                new_dict["area"] = int(np.sum(scaled_mask))
                scaled_masks.append(new_dict)

            self._masks = scaled_masks
            logging.info(f"Scaled {len(self._masks)} masks back to original size")

        # Classify masks (interactive or automatic)
        if self.interactive:
            classified = self._classify_interactive()
        else:
            classified = self._classify_automatic()

        # Extract limb directly (simplified v6 method)
        result = self._combine_masks(classified)

        # Return limb coordinates
        return result["limb"]

    def _classify_interactive(self) -> dict:
        """
        Classify masks interactively using GUI.

        Returns:
            Dict with 'planet', 'sky', 'exclude' keys
        """
        from planet_ruler.annotate import TkMaskSelector

        logging.info("Opening interactive mask classifier...")

        # Create and run selector
        selector = TkMaskSelector(self.image, self._masks)
        selector.run()

        # Get classifications
        classified = selector.get_classified_masks()

        logging.info(
            f"Classified: {len(classified['planet'])} planet, "
            f"{len(classified['sky'])} sky, {len(classified['exclude'])} exclude"
        )

        return classified

    def _classify_automatic(self) -> dict:
        """
        Classify masks automatically (simple heuristic).

        Assumes largest mask in lower half is planet, largest in upper half is sky.

        Returns:
            Dict with 'planet', 'sky', 'exclude' keys
        """
        logging.info("Automatic classification...")

        if len(self._masks) < 2:
            raise ValueError("Need at least 2 masks for automatic classification")

        # Find center row of each mask
        image_height = self.original_shape[0]
        midpoint = image_height // 2

        upper_masks = []
        lower_masks = []

        for mask_dict in self._masks:
            mask_array = mask_dict.get("mask", mask_dict.get("segmentation"))
            rows = np.where(mask_array.any(axis=1))[0]

            if len(rows) > 0:
                center_row = (rows[0] + rows[-1]) // 2

                if center_row < midpoint:
                    upper_masks.append(mask_dict)
                else:
                    lower_masks.append(mask_dict)

        # Sort by area
        upper_masks.sort(key=lambda x: x["area"], reverse=True)
        lower_masks.sort(key=lambda x: x["area"], reverse=True)

        # Largest in upper half = sky, largest in lower half = planet
        classified = {
            "planet": [lower_masks[0]] if lower_masks else [],
            "sky": [upper_masks[0]] if upper_masks else [],
            "exclude": [],
        }

        logging.info(
            f"Auto-classified: planet area={classified['planet'][0]['area'] if classified['planet'] else 0}, "
            f"sky area={classified['sky'][0]['area'] if classified['sky'] else 0}"
        )

        return classified

    def _combine_masks(self, classified_masks: dict) -> dict:
        """
        Combine classified masks and extract horizon directly.

        Simplified v6 approach: Returns the actual limb coordinates.

        Args:
            classified_masks: Dict with 'planet', 'sky', 'exclude' keys

        Returns:
            dict with 'limb' array directly
        """
        planet_masks = classified_masks.get("planet", [])
        sky_masks = classified_masks.get("sky", [])

        logging.info(
            f"Combining masks: {len(planet_masks)} planet, {len(sky_masks)} sky"
        )

        # Fallback if no classifications
        if not planet_masks or not sky_masks:
            logging.warning(
                "Incomplete classification. "
                f"Planet: {len(planet_masks)}, Sky: {len(sky_masks)}"
            )
            if len(self._masks) >= 2:
                if not planet_masks:
                    planet_masks = [self._masks[0]]
                    logging.info("Using mask #0 as planet (fallback)")
                if not sky_masks:
                    sky_masks = [self._masks[1]]
                    logging.info("Using mask #1 as sky (fallback)")

        # Combine masks
        planet_mask = np.zeros(self.original_shape, dtype=bool)
        for mask_obj in planet_masks:
            mask_array = mask_obj.get("mask", mask_obj.get("segmentation"))
            if isinstance(mask_obj, np.ndarray):
                mask_array = mask_obj
            planet_mask |= mask_array

        sky_mask = np.zeros(self.original_shape, dtype=bool)
        for mask_obj in sky_masks:
            mask_array = mask_obj.get("mask", mask_obj.get("segmentation"))
            if isinstance(mask_obj, np.ndarray):
                mask_array = mask_obj
            sky_mask |= mask_array

        # Check extents for debugging
        planet_rows = np.where(planet_mask.any(axis=1))[0]
        sky_rows = np.where(sky_mask.any(axis=1))[0]

        if len(planet_rows) > 0:
            logging.info(f"Planet: rows {planet_rows[0]} to {planet_rows[-1]}")
        if len(sky_rows) > 0:
            logging.info(f"Sky: rows {sky_rows[0]} to {sky_rows[-1]}")

        # Extract horizon directly: column by column
        limb = []

        for col_idx in range(self.original_shape[1]):
            sky_col = sky_mask[:, col_idx]
            planet_col = planet_mask[:, col_idx]

            # Find where sky is
            sky_indices = np.where(sky_col)[0]
            # Find where planet is
            planet_indices = np.where(planet_col)[0]

            if len(sky_indices) > 0 and len(planet_indices) > 0:
                # Last row with sky
                last_sky = sky_indices[-1]
                # First row with planet
                first_planet = planet_indices[0]

                # Horizon is between them
                horizon_y = (last_sky + first_planet) / 2.0
                limb.append(horizon_y)

            elif len(planet_indices) > 0:
                # Only planet - use top edge
                limb.append(float(planet_indices[0]))

            elif len(sky_indices) > 0:
                # Only sky - use bottom edge
                limb.append(float(sky_indices[-1]))

            else:
                # No mask at all
                limb.append(np.nan)

        limb = np.array(limb)

        # Apply robust outlier detection (your original code)
        # Set big jumps to NaN
        diff = np.abs(np.diff(limb, n=1, append=limb[-1]))
        mean_diff = np.nanmean(diff)

        if mean_diff > 0:
            outliers = diff > 10 * mean_diff
            limb[outliers] = np.nan

            # Set their immediate neighbors to NaN
            limb[np.isnan(np.diff(limb, n=1, append=limb[-1]))] = np.nan

        # Interpolate NaNs
        nan_mask = np.isnan(limb)
        if np.any(~nan_mask) and np.sum(~nan_mask) > 1:
            limb[nan_mask] = np.interp(
                np.flatnonzero(nan_mask), np.flatnonzero(~nan_mask), limb[~nan_mask]
            )

        logging.info(
            f"Extracted limb: min={np.nanmin(limb):.1f}, "
            f"max={np.nanmax(limb):.1f}, mean={np.nanmean(limb):.1f}"
        )

        # Return limb directly
        return {"limb": limb, "planet_mask": planet_mask, "sky_mask": sky_mask}


class SegmentationBackend:
    """Base class for segmentation backends."""

    def segment(self, image: np.ndarray) -> list:
        """
        Segment image into masks.

        Args:
            image: Input image (H x W x 3)

        Returns:
            List of mask objects (format can vary by backend)
        """
        raise NotImplementedError


class SAMBackend(SegmentationBackend):
    """Segment Anything Model backend."""

    def __init__(self, model_size: str = "vit_b"):
        """
        Initialize SAM backend.

        Args:
            model_size: SAM model variant ('vit_b', 'vit_l', 'vit_h')
        """
        if not HAS_SEGMENT_ANYTHING:
            raise ImportError(
                "segment-anything dependencies not available. "
                "Install with: pip install 'planet_ruler[ml]'"
            )

        self.model_size = model_size

        # Convert underscores to hyphens for Kaggle download path
        kaggle_model_name = model_size.replace("_", "-")

        # Download model (cached after first time)
        self.model_path = kagglehub.model_download(
            f"metaresearch/segment-anything/pyTorch/{kaggle_model_name}"
        )

        logging.info(f"SAM model ready: {model_size}")

    def segment(self, image: np.ndarray) -> list:
        """Run SAM on image."""
        sam = sam_model_registry[self.model_size](
            checkpoint=f"{self.model_path}/model.pth"
        )
        mask_generator = SamAutomaticMaskGenerator(sam)
        masks = mask_generator.generate(image)

        logging.info(f"SAM generated {len(masks)} masks")
        return masks


class CustomBackend(SegmentationBackend):
    """Custom user-provided segmentation backend."""

    def __init__(self, segment_fn):
        """
        Initialize with custom segmentation function.

        Args:
            segment_fn: Function that takes image and returns list of masks
        """
        self.segment_fn = segment_fn

    def segment(self, image: np.ndarray) -> list:
        """Run custom segmentation function."""
        masks = self.segment_fn(image)

        # Normalize to expected format
        normalized = []
        for idx, mask in enumerate(masks):
            if isinstance(mask, np.ndarray):
                normalized.append(
                    {"mask": mask.astype(bool), "area": int(np.sum(mask)), "id": idx}
                )
            elif isinstance(mask, dict):
                normalized.append(mask)
            else:
                raise TypeError(f"Mask must be ndarray or dict, got {type(mask)}")

        logging.info(f"Custom backend generated {len(normalized)} masks")
        return normalized


# Removed vestigial ImageSegmentation class - replaced by MaskSegmenter


def smooth_limb(
    y: np.ndarray,
    method: str = "rolling-median",
    window_length: int = 50,
    polyorder: int = 1,
    deriv: int = 0,
    delta=1,
) -> np.ndarray:
    """
    Smooth the limb position values.

    Args:
        y (np.ndarray): Y-locations of the string for each column.
        method (str): Smoothing method. Must be one of ['bin-interpolate', 'savgol',
            'rolling-mean', 'rolling-median'].
        window_length (int): The length of the filter window (i.e., the number of coefficients).
            If mode is ‘interp’, window_length must be less than or equal to the size of x.
        polyorder (float): The order of the polynomial used to fit the samples.
            polyorder must be less than window_length.
        deriv (int): The order of the derivative to compute. This must be a non-negative integer.
            The default is 0, which means to filter the data without differentiating.
        delta (int): The spacing of the samples to which the filter will be applied.
            This is only used if deriv > 0. Default is 1.0.

    Returns:
        position (np.ndarray): Y-locations of the smoothed string for each column.
    """
    assert method in ["bin-interpolate", "savgol", "rolling-mean", "rolling-median"]

    if method == "bin-interpolate":
        binned = []
        x = []
        for i in range(len(y[::window_length])):
            binned += [
                np.mean(y[i * window_length : i * window_length + window_length])
            ]
            x += [i * window_length + int(0.5 * window_length)]
        binned = np.array(binned)
        x = np.array(x)

        if polyorder == 1:
            kind = "linear"
        elif polyorder == 2:
            kind = "quadratic"
        elif polyorder == 0:
            kind = "nearest"
        else:
            raise AttributeError(
                f"polyorder {polyorder} not supported for bin-interpolate"
            )
        interp = interp1d(x, binned, kind=kind, fill_value="extrapolate")

        limb = interp(np.arange(len(y)))
    elif method == "savgol":
        limb = savgol_filter(
            y,
            window_length=window_length,
            polyorder=polyorder,
            deriv=deriv,
            delta=delta,
        )
    elif method == "rolling-mean":
        limb = pd.Series(y).rolling(window_length).mean()
    elif method == "rolling-median":
        limb = pd.Series(y).rolling(window_length).median()
    else:
        raise ValueError(f"Did not recognize smoothing method {method}")

    mask = np.isnan(limb)
    limb[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), limb[~mask])

    return limb


def fill_nans(limb: np.ndarray) -> np.ndarray:
    """
    Fill NaNs for the limb position values.

    Args:
        limb (np.ndarray): Y-locations of the limb on the image.

    Returns:
        limb (np.ndarray): Y-locations of the limb on the image.
    """
    fixed = limb.copy()
    mask = np.isnan(fixed)
    fixed[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), fixed[~mask])
    return fixed


def gradient_field(
    image: np.ndarray,
    kernel_smoothing: float = 5.0,
    directional_smoothing: int = 50,
    directional_decay_rate: float = 0.15,
) -> dict:
    """
    Pre-compute gradient field AND its derivatives for fast sub-pixel interpolation.
    Uses directional blur for enhanced edge detection.

    Args:
        image: Input image
        kernel_smoothing: Sigma for initial gradient direction estimation
        directional_smoothing: Distance to sample in each direction (±pixels).
                              Set to 0 to bypass directional smoothing entirely.
        directional_decay_rate: Exponential decay rate for directional blur

    Returns:
        dict: Dictionary containing gradient field components:
            - grad_mag: Gradient magnitude array
            - grad_angle: Gradient angle array
            - grad_sin: Sin of gradient angle (for interpolation)
            - grad_cos: Cos of gradient angle (for interpolation)
            - grad_mag_dy, grad_mag_dx: Derivatives of gradient magnitude
            - grad_sin_dy, grad_sin_dx: Derivatives of sine component
            - grad_cos_dy, grad_cos_dx: Derivatives of cosine component
            - image_height, image_width: Dimensions
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = image.sum(axis=2).astype(np.float32)
    else:
        gray = image.copy().astype(np.float32)

    # Use directional blur for enhanced edge detection
    # First compute initial gradients for direction
    if kernel_smoothing > 0:
        gray_smooth = cv2.GaussianBlur(gray, (0, 0), min(kernel_smoothing, 2.0))
    else:
        gray_smooth = gray

    grad_y_init, grad_x_init = np.gradient(gray_smooth)
    grad_mag_init = np.sqrt(grad_x_init**2 + grad_y_init**2)
    grad_angle = np.arctan2(grad_y_init, grad_x_init)

    # Apply directional smoothing if enabled (directional_smoothing > 0)
    if directional_smoothing > 0:
        # Normalize to unit vectors for sampling
        grad_mag_safe = np.copy(grad_mag_init)
        grad_mag_safe[grad_mag_safe < 1e-6] = 1.0
        grad_x_unit = grad_x_init / grad_mag_safe
        grad_y_unit = grad_y_init / grad_mag_safe

        # Bidirectional blur along gradient
        height, width = gray.shape
        blurred_mag = np.zeros_like(grad_mag_init)
        y_coords, x_coords = np.meshgrid(
            np.arange(height), np.arange(width), indexing="ij"
        )

        total_weight = 0.0
        for step in range(-directional_smoothing, directional_smoothing + 1):
            weight = np.exp(-directional_decay_rate * abs(step))
            total_weight += weight

            offset_x = grad_x_unit * step
            offset_y = grad_y_unit * step

            sample_x = x_coords + offset_x
            sample_y = y_coords + offset_y

            # Bilinear interpolation
            sample_mag = bilinear_interpolate(grad_mag_init, sample_y, sample_x)
            blurred_mag += weight * sample_mag

        blurred_mag /= total_weight

        # Use blurred magnitude with original angles
        grad_mag = blurred_mag
        grad_x = blurred_mag * np.cos(grad_angle)
        grad_y = blurred_mag * np.sin(grad_angle)
    else:
        # Bypass directional smoothing - use original gradients
        grad_mag = grad_mag_init
        grad_x = grad_x_init
        grad_y = grad_y_init

    # Convert to sin/cos for angle interpolation (handles wraparound)
    grad_sin = np.sin(grad_angle)
    grad_cos = np.cos(grad_angle)

    # PRE-COMPUTE DERIVATIVES for Taylor expansion (one-time cost!)
    grad_mag_dy, grad_mag_dx = np.gradient(grad_mag)
    grad_sin_dy, grad_sin_dx = np.gradient(grad_sin)
    grad_cos_dy, grad_cos_dx = np.gradient(grad_cos)

    # Store dimensions
    image_height, image_width = grad_mag.shape

    return {
        "grad_mag": grad_mag,
        "grad_angle": grad_angle,
        "grad_sin": grad_sin,
        "grad_cos": grad_cos,
        "grad_x": grad_x,
        "grad_y": grad_y,
        "grad_mag_dy": grad_mag_dy,
        "grad_mag_dx": grad_mag_dx,
        "grad_sin_dy": grad_sin_dy,
        "grad_sin_dx": grad_sin_dx,
        "grad_cos_dy": grad_cos_dy,
        "grad_cos_dx": grad_cos_dx,
        "image_height": image_height,
        "image_width": image_width,
    }
