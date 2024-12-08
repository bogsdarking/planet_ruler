import numpy as np
from scipy.interpolate import UnivariateSpline


def horizon_distance(r: float,
                     h: float):
    return r * np.arctan(np.sqrt(2 * r * h) / r)


def intrinsic_transform(world_coords, f=1, px=1, py=1, x0=0, y0=0):
    """
    Transform from camera coordinates into

    Args:
        world_coords (np.ndarray):
        f (float):
        px (float):
        py (float)
        x0 (float):
        y0 (float):
    Returns:
        playlist_ids (pd.DataFrame)
    """
    Mi = np.array([
        [float(f)/px, 0, x0, 0],
        [0, float(f)/py, y0, 0],
        [0, 0, 1, 0]
    ])

    # todo allow for shear/etc.

    camera_coords = []
    for i in range(len(world_coords)):
        camera_coords += [Mi @ world_coords[i, :]]
    camera_coords = np.array(camera_coords)

    return camera_coords


def extrinsic_transform(world_coords,
                        theta_x=0, theta_y=0, theta_z=0,
                        origin_x=0, origin_y=0, origin_z=0):

    """
    origin_x (world coords center in camera coords)
    """

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x)],
        [0, np.sin(theta_x), np.cos(theta_x)]
    ])

    Ry = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y)],
        [0, 1, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y)]
    ])

    Rz = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0],
        [np.sin(theta_z), np.cos(theta_z), 0],
        [0, 0, 1]
    ])

    rotation = Rx @ Ry @ Rz

    translation = np.array([origin_x, origin_y, origin_z])

    # see https://en.wikipedia.org/wiki/Camera_resectioning
    transform = np.zeros((4, 4))
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    transform[3, 3] = 1

    camera_coords = []
    for i in range(len(world_coords)):
        camera_coords += [transform @ world_coords[i, :]]
    camera_coords = np.array(camera_coords)

    return camera_coords


def limb_arc(x, r, h=1,
             method='resection', screen_dist=1,
             f=0.035, pxy=1, px=1, py=1, x0=0, y0=0,
             theta_x=0, theta_y=0, theta_z=0,
             origin_x=0, origin_y=0, origin_z=0
             ):

    # diffraction correction?
    #     r = r * 1.2

    d = np.sqrt(h**2 + 2*h*r)
    theta = np.arccos(r / (r + h))

    xdcostheta = np.clip(x / (d * np.cos(theta)), -1, 1) # to avoid out of bounds
    #     xdcostheta = x / (d * np.cos(theta))
    y_world = d * np.cos(np.arcsin(xdcostheta)) * np.sin(theta)

    if method == 'screen':
        y_camera = y_world * screen_dist / np.sqrt(d**2 - y_world**2)
    elif method == 'resection':
        x_world = d * np.sin(np.arcsin(xdcostheta)) * np.cos(theta)
        z_world = d * np.cos(np.arcsin(xdcostheta)) * np.cos(theta)

        world_coords = np.ones((len(x), 4))
        world_coords[:, 0] = x_world
        world_coords[:, 1] = y_world
        world_coords[:, 2] = z_world
        # todo worth a 3d scatter demo of world coords?

        camera_coords = extrinsic_transform(world_coords,
                                            theta_x=theta_x, theta_y=theta_y, theta_z=theta_z,
                                            origin_x=origin_x, origin_y=origin_y, origin_z=origin_z)
        camera_coords = intrinsic_transform(camera_coords, f=f, px=pxy*px, py=pxy*py,
                                            x0=x0, y0=y0)
        x_camera = camera_coords[:, 0]
        y_camera = camera_coords[:, 1]
        # spline needed to map back to pixel coordinates (and extrapolate to fill)
        order = np.argsort(x_camera)  # spline requires monotonic increase
        # print('x', min(x_camera), max(x_camera))
        # print('y', min(y_camera), max(y_camera))
        interp = UnivariateSpline(x_camera[order], y_camera[order]) #, ext='zeros') # ext='const')
        y_camera = interp(x)

    return y_camera
