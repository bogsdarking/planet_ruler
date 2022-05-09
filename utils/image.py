import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from PIL import Image


def load_image(filepath):
    img = Image.open(filepath)
    im_arr = np.fromstring(img.tobytes(), dtype=np.uint8)
    im_arr = im_arr.reshape((img.size[1], img.size[0], 3))
    return im_arr


def detect_limb(image, y_min=0, y_max=-1,
                window_length=501, polyorder=1, deriv=0, delta=1):

    grad = abs(np.gradient(image.sum(axis=2), axis=0))
    # grad[grad > 0] = np.log10(grad[grad > 0])
    grad = np.log10(grad)
    grad[np.isinf(grad)] = 0
    # grad[grad < 0] = 0

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


def smooth_limb(y, method='moving-mean', window_length=101, polyorder=1, deriv=0, delta=1):

    assert method in ['moving-mean', 'savgol']

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

