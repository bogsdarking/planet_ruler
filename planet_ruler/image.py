import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from tqdm.notebook import tqdm
from PIL import Image


def load_image(filepath: str):
    """
    Load a 3 or 4-channel image from filepath into an array.

    Args:
        filepath (str): Path to image file.
    Returns:
        image array (np.ndarray)
    """
    img = Image.open(filepath)
    im_arr = np.fromstring(img.tobytes(), dtype=np.uint8)
    try:
        im_arr = im_arr.reshape((img.size[1], img.size[0], 3))
    except ValueError:
        # todo add capacity for auto band detection
        im_arr = im_arr.reshape((img.size[1], img.size[0], 4))
    return im_arr


def gradient_break(im_arr: np.ndarray,
                   log: bool = False,
                   y_min: int = 0,
                   y_max: int = -1,
                   window_length: int = 501,
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


class StringDrop:
    def __init__(self, image):
        self.image = image
        self.gradient = None
        self.topography = None
        self.force_map = None
        self.tilt = None
        self.smoothing_window = None
        self.string_positions = None

    def transverse_force(self, x, y):
        try:
            m = np.diff(y)[x]
        except IndexError:
            m = 0
        theta = np.arctan(m)
        return -np.sin(theta)

    def compute_force_map(self, tilt=0.05, smoothing_window=50):
        # skip if already run with same parameters
        if self.tilt == tilt and smoothing_window == smoothing_window:
            return None
        else:
            self.tilt = tilt
            self.smoothing_window = smoothing_window

        self.gradient = abs(np.gradient(self.image.sum(axis=2), axis=0))

        self.topography = np.zeros_like(self.gradient)
        self.force_map = np.zeros_like(self.gradient)
        for j in range(self.gradient.shape[1]):
            y = self.gradient[:, j]
            x = np.arange(len(y))
            # smooth the topography along y-axis lines, adding a minor tilt to establish movement towards basin
            # todo automate tilt level
            self.topography[:, j] = pd.Series(y).rolling(smoothing_window).mean() - tilt*x
            for i in range(self.gradient.shape[0]):
                self.force_map[i, j] = self.transverse_force(i, self.topography[:, j])
        # back-fill smoothed image with repeats of first viable row
        self.force_map[:smoothing_window-1, :] = np.median(self.force_map[smoothing_window-1, :])

    def drop_string(self, start=10, steps=150000, g=100, m=1, k=3e-1,
                    t_step=0.03, max_acc=2, max_vel=2):
        # todo add parameter sets by name 'safe' etc.
        position = np.ones(self.gradient.shape[1]) * start
        vel = np.zeros(self.gradient.shape[1])
        f_spring_left = np.zeros(self.gradient.shape[1])
        f_spring_right = np.zeros(self.gradient.shape[1])

        positions = []
        # todo automate steps to convergence criterion
        for _ in tqdm(range(steps)):

            f_grav = self.force_map[position.astype(int),
                                    np.arange(self.force_map.shape[1])] * g
            f_spring_left[1:] = np.diff(position) * k
            f_spring_right[:-1] = np.diff(position) * k
            acc = f_grav + f_spring_left + f_spring_right
            acc = np.clip(acc / m, -max_acc, max_acc)

            position += vel*t_step + acc*0.5*t_step**2
            position = np.clip(position, 0, self.gradient.shape[0]-1)
            vel += acc * t_step
            vel = np.clip(vel, -max_vel, max_vel)
            positions.append(position)

        self.string_positions = positions
        return position


def smooth_limb(y, method='moving-mean', window_length=101, polyorder=1, deriv=0, delta=1):

    assert method in ['moving-mean', 'savgol', 'rolling-mean', 'rolling-median']

    if method == 'moving-mean':
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
            raise AttributeError(f'polyorder {polyorder} not supported for moving-mean')
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

def fill_nans(limb):
    fixed = limb.copy()
    mask = np.isnan(fixed)
    fixed[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), fixed[~mask])
    return fixed
