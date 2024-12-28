import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['figure.figsize'] = (16, 10)
matplotlib.rcParams.update({'font.size': 18})


def plot_image(
        im_arr: np.ndarray,
        gradient: bool = False,
        show: bool = True) -> None:
    """
    Display an image using matplotlib.pyplot.imshow.

    Args:
        im_arr (np.ndarray): Array of image values.
        gradient (bool): Display as gradient (y-axis).
        show (bool): Display the image.
    """
    if gradient:
        grad = abs(np.gradient(im_arr.sum(axis=2), axis=0))
        grad[grad > 0] = np.log10(grad[grad > 0])
        grad[grad < 0] = 0
        plt.imshow(grad)
    else:
        plt.imshow(im_arr)
    if show:
        plt.show()


def plot_limb(
        y: np.ndarray,
        show: bool = True,
        c: str = 'y',
        s: int = 10,
        alpha: float = 0.2) -> None:
    """
    Display the limb (usually on top of an image).

    Args:
        y (np.ndarray): Array of image values.
        show (bool): Display the image.
        c (str): Color of the limb.
        s (int): Size of marker.
        alpha (float): Opacity of markers.
    """
    plt.scatter(np.arange(len(y)), y, c=c, s=s, alpha=alpha)
    if show:
        plt.show()
    