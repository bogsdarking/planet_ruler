import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from tqdm.notebook import tqdm
from PIL import Image


def load_image(filepath):
    img = Image.open(filepath)
    im_arr = np.fromstring(img.tobytes(), dtype=np.uint8)
    try:
        im_arr = im_arr.reshape((img.size[1], img.size[0], 3))
    except ValueError:
        # todo add capacity for auto band detection
        im_arr = im_arr.reshape((img.size[1], img.size[0], 4))
    return im_arr


def gradient_break(image, log=False, y_min=0, y_max=-1,
                   window_length=501, polyorder=1, deriv=0, delta=1):

    grad = abs(np.gradient(image.sum(axis=2), axis=0))

    if log:
        grad[grad > 0] = np.log10(grad[grad > 0])
        grad = np.log10(grad)
        grad[np.isinf(grad)] = 0
        grad[grad < 0] = 0

    breaks = []
    for i in range(image.shape[1]):
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

        return interp(np.arange(len(y)))
    elif method == 'savgol':
        return savgol_filter(y, window_length=window_length, polyorder=polyorder,
                             deriv=deriv, delta=delta)
    elif method == 'rolling-mean':
        return pd.Series(y).rolling(window_length).mean()
    elif method == 'rolling-median':
        return pd.Series(y).rolling(window_length).median()
