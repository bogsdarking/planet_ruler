import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['figure.figsize'] = (16, 10)
matplotlib.rcParams.update({'font.size': 18})


def plot_image(im_arr, gradient=False, show=True):
    if gradient:
        grad = abs(np.gradient(im_arr.sum(axis=2), axis=0))
        grad[grad > 0] = np.log10(grad[grad > 0])
        grad[grad < 0] = 0
        plt.imshow(grad)
    else:
        plt.imshow(im_arr)
    if show:
        plt.show()


def plot_limb(y, show=True, c='y', s=10, alpha=0.2):
    plt.scatter(np.arange(len(y)), y, c=c, s=s, alpha=alpha)
    if show:
        plt.show()

    