import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from planet_ruler.geometry import limb_camera_angle, horizon_distance

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


def plot_3d_solution(
        r: float,
        h: float = 1,
        zoom: float = 1,
        savefile: str = None,
        legend: bool = True,
        vertical_axis: str = 'z',
        azim: float = None,
        roll: float = None,
        x_axis: bool = False,
        y_axis: bool = True,
        z_axis: bool = False,
        **kwargs) -> None:
    """
    Plot a limb solution in 3D.

    Args:
        r (float): Radius of the body in question.
        h (float): Height above surface (units should match radius).
        zoom (float): Shrink the height according to a zoom factor to make viewing easier.
        savefile (str): Path to optionally save figure.
        legend (bool): Display the legend.
        vertical_axis (str): Which axis will be used as the vertical (x, y, or z).
        azim (float): Viewing azimuth.
        roll (float): Viewing roll.
        x_axis (bool): Plot the x-axis.
        y_axis (bool): Plot the y-axis.
        z_axis (bool): Plot the z-axis.
        kwargs (dict): Absorbs other solution kwargs that don't matter for physical space.
     Returns:
         None
     """
    h = h * (1. / zoom)
    limb_theta = np.pi / 2 - limb_camera_angle(r, h)
    d = horizon_distance(r, h)
    horizon_radius = d * np.cos(limb_theta)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')

    theta = np.linspace(-limb_theta, limb_theta, 100)
    phi = np.linspace(0, np.pi, 50)
    theta, phi = np.meshgrid(theta, phi)

    x_world = r * np.sin(theta) * np.cos(phi)
    y_world = -(h + r) + r * np.cos(theta)
    z_world = r * np.sin(theta) * np.sin(phi)

    ax.plot_wireframe(x_world, y_world, z_world, color='y', alpha=0.1, label='planet')

    theta = np.ones(1) * limb_theta
    phi = np.linspace(-np.pi, np.pi, num=5000)
    theta, phi = np.meshgrid(theta, phi)

    x_world = r * np.sin(theta) * np.cos(phi)
    y_world = -(h + r) + r * np.cos(theta)
    z_world = r * np.sin(theta) * np.sin(phi)
    ax.plot(x_world, y_world, z_world, c='k', label='limb')

    ax.scatter([0], [0], [0], marker='.', s=100,
               label=f'camera/origin [elevation = {int(h / 1000)} km]')

    if y_axis:
        ax.plot([0, 0], [-h, 0], [0, 0], c='g', ls='--', alpha=0.7, label='y-axis')
    if z_axis:
        ax.plot([0, 0], [0, 0], [-horizon_radius, horizon_radius],
                c='b', ls='--', alpha=0.7, label='z-axis')
    if x_axis:
        ax.plot([-horizon_radius, horizon_radius], [0, 0], [0, 0],
                c='k', ls='--', alpha=0.7, label='x-axis')

    theta = limb_theta
    phi = 0

    x_world = r * np.sin(theta) * np.cos(phi)
    y_world = -(h + r) + r * np.cos(theta)
    z_world = r * np.sin(theta) * np.sin(phi)

    ax.plot([0, x_world], [0, y_world], [0, z_world], c='purple', ls='--', alpha=0.7,
            label=f'line of sight [distance = {int(d / 1000)} km]')

    plt.autoscale(False)
    theta = np.linspace(0, 2 * np.pi, 1000)
    phi = np.linspace(0, np.pi, 500)
    theta, phi = np.meshgrid(theta, phi)

    x_world = r * np.sin(theta) * np.cos(phi)
    y_world = -(h + r) + r * np.cos(theta)
    z_world = r * np.sin(theta) * np.sin(phi)

    ax.plot_wireframe(x_world, y_world, z_world, color='b', alpha=0.01)

    ax.view_init(elev=limb_theta * 180 / np.pi, azim=azim, roll=roll, vertical_axis=vertical_axis)
    plt.axis('off')
    if legend:
        plt.legend(fontsize=12)
    plt.axis('equal')
    if savefile is not None:
        plt.savefig(savefile, bbox_inches='tight')
    plt.show()


def plot_topography(image: np.ndarray) -> None:
    """
    Display the full limb, including the section not seen in the image.

    Args:
        image (np.ndarray): Image array.
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')

    image = np.clip(image, None, 1000)
    n_rows, n_cols = image.shape
    x = np.arange(n_cols)[::-1]
    y = np.arange(n_rows)

    x, y = np.meshgrid(x, y)

    ls = LightSource(altdeg=30, azdeg=-15)
    ax.plot_surface(x, y, image, lightsource=ls)
    ax.view_init(elev=90, azim=0, roll=-90)

    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0)

    plt.axis('off')
    plt.show()

