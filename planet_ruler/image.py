import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from tqdm.notebook import tqdm
from PIL import Image
import kagglehub
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


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


def gradient_break(im_arr: np.ndarray,
                   log: bool = False,
                   y_min: int = 0,
                   y_max: int = -1,
                   window_length: int = None,
                   polyorder: int = 1,
                   deriv: int = 0,
                   delta: int = 1):
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
        yhat = savgol_filter(y, window_length=window_length,
                             polyorder=polyorder, deriv=deriv, delta=delta)

        yhathat = np.diff(yhat)
        m = np.argmax(yhathat[y_min:y_max])
        breaks += [m+y_min]
    breaks = np.array(breaks)

    return breaks


class ImageSegmentation:
    def __init__(self,
                 image: np.ndarray,
                 segmenter: str = 'segment-anything'):
        self.image = image
        self.segmenter = segmenter
        self._masks = None

        if segmenter == 'segment-anything':
            self.model_path = kagglehub.model_download("metaresearch/segment-anything/pyTorch/vit-b")
        else:
            self.model_path = None
            raise ValueError(f"segmenter must be one of [segment-anything]")

    def limb_from_mask(self, mask: np.ndarray) -> np.ndarray:
        limb = []
        for y in mask.T:
            try:
                pt = np.arange(len(y))[np.where((y == True))][0]
            except IndexError:
                pt = np.nan
            limb.append(pt)
        limb = np.array(limb).astype(float)
        # handling of blips (sometimes the image edges confuse segmenters)
        # set big jumps to NaN
        limb[abs(np.diff(limb, n=1, append=limb[-1])) > 10 * abs(
            np.nanmean(np.diff(limb, n=1, append=limb[-1])))] = np.nan
        # set their immediate neighbors to NaN
        limb[np.isnan(np.diff(limb, n=1, append=limb[-1]))] = np.nan
        nan_mask = np.isnan(limb)
        # interpolate them back in
        limb[nan_mask] = np.interp(np.flatnonzero(nan_mask), np.flatnonzero(~nan_mask), limb[~nan_mask])

        return np.array(limb)

    def segment(self) -> np.ndarray:
        if self.segmenter == 'segment-anything':
            sam = sam_model_registry["vit_b"](checkpoint=f"{self.model_path}/model.pth")
            mask_generator = SamAutomaticMaskGenerator(sam)
            masks = mask_generator.generate(self.image)
            self._masks = masks
            # combine first two (should be above/below)
            mask = (1 - masks[0]['segmentation']) * masks[1]['segmentation']

            return self.limb_from_mask(mask)


def smooth_limb(y: np.ndarray,
                method: str = 'rolling-median',
                window_length: int = 50,
                polyorder: int = 1,
                deriv: int = 0,
                delta=1) -> np.ndarray:
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
    assert method in ['bin-interpolate', 'savgol', 'rolling-mean', 'rolling-median']

    if method == 'bin-interpolate':
        binned = []
        x = []
        for i in range(len(y[::window_length])):
            binned += [np.mean(y[i*window_length:i*window_length+window_length])]
            x += [i*window_length+int(0.5*window_length)]
        binned = np.array(binned)
        x = np.array(x)

        if polyorder == 1:
            kind = 'linear'
        elif polyorder == 2:
            kind = 'quadratic'
        elif polyorder == 0:
            kind = 'nearest'
        else:
            raise AttributeError(f'polyorder {polyorder} not supported for bin-interpolate')
        interp = interp1d(x, binned, kind=kind, fill_value='extrapolate')

        limb = interp(np.arange(len(y)))
    elif method == 'savgol':
        limb =  savgol_filter(y, window_length=window_length, polyorder=polyorder,
                              deriv=deriv, delta=delta)
    elif method == 'rolling-mean':
        limb = pd.Series(y).rolling(window_length).mean()
    elif method == 'rolling-median':
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
