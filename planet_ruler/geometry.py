import numpy as np
from scipy.interpolate import UnivariateSpline


def horizon_distance(r: float,
                     h: float) -> float:
    """
    Estimate the distance to the horizon (limb) given a height.

    Args:
        r (float): Radius of the body in question.
        h (float): Height above surface (units should match radius).
    Returns:
        d (float): Distance in same units as inputs.
    """
    return r * np.arctan(np.sqrt(2 * r * h) / r)


def intrinsic_transform(camera_coords: np.ndarray,
                        f: float = 1,
                        px: float = 1,
                        py: float = 1,
                        x0: float = 0,
                        y0: float = 0) -> np.ndarray:
    """
    Transform from camera coordinates into image coordinates.

    Args:
        camera_coords (np.ndarray): Coordinates of the limb in camera space.
            Array has Nx4 shape where N is the number of x-axis pixels
            in the image.
        f (float): Focal length of the camera (m).
        px (float): The scale of x pixels.
        py (float): The scale of y pixels.
        x0 (float): The x-axis principle point.
        y0 (float): The y-axis principle point.
    Returns:
        pixel_coords (np.ndarray): Coordinates in image space.
    """
    transform = np.array([
        [float(f)/px, 0, x0, 0],
        [0, float(f)/py, y0, 0],
        [0, 0, 1, 0]
    ])

    # todo allow for shear/etc.

    pixel_coords = []
    for i in range(len(camera_coords)):
        pixel_coords += [transform @ camera_coords[i, :]]
    pixel_coords = np.array(pixel_coords)

    return pixel_coords


def extrinsic_transform(world_coords,
                        theta_x: float = 0,
                        theta_y: float = 0,
                        theta_z: float = 0,
                        origin_x: float = 0,
                        origin_y: float = 0,
                        origin_z: float = 0) -> np.ndarray:
    """
    Transform from world coordinates into camera coordinates.

    Args:
        world_coords (np.ndarray): Coordinates of the limb in the world.
            Array has Nx4 shape where N is the number of x-axis pixels
            in the image.
        theta_x (float): Rotation around the x (horizontal) axis,
            AKA pitch. (radians)
        theta_y (float): Rotation around the y (toward the limb) axis,
            AKA roll. (radians)
        theta_z (float): Rotation around the z (vertical) axis,
            AKA yaw. (radians)
        origin_x (float): Horizontal offset from the object in question
            to the camera (m).
        origin_y (float): Distance from the object in question to the
            camera (m).
        origin_z (float): Height difference from the object in question
            to the camera (m).
    Returns:
        camera_coords (np.ndarray): Coordinates in camera space.
    """

    x_rotation = np.array([
        [1, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x)],
        [0, np.sin(theta_x), np.cos(theta_x)]
    ])

    y_rotation = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y)],
        [0, 1, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y)]
    ])

    z_rotation = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0],
        [np.sin(theta_z), np.cos(theta_z), 0],
        [0, 0, 1]
    ])

    rotation = x_rotation @ y_rotation @ z_rotation

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


def limb_arc(x: np.ndarray,
             r: float,
             h: float = 1,
             f: float = 0.035,
             pxy: float = 1,
             px: float = 1,
             py: float = 1,
             x0: float = 0,
             y0: float = 0,
             theta_x: float = 0,
             theta_y: float = 0,
             theta_z: float = 0,
             origin_x: float = 0,
             origin_y: float = 0,
             origin_z: float = 0
             ) -> np.ndarray:
    """
    Calculate the limb orientation in an image given the physical
    parameters of the system.

    Args:
        x (np.ndarray): X-coords in image. Should just be an integer
            range from [0, image width in pixels].
        r (float): Radius of the body in question.
        h (float): Height above surface (units should match radius).
        f (float): Focal length of the camera (m).
        pxy (float): The scale of both x and y pixels.
        px (float): The relative scale of x pixels.
        py (float): The relative scale of y pixels (redundant with px).
        x0 (float): The x-axis principle point.
        y0 (float): The y-axis principle point.
        theta_x (float): Rotation around the x (horizontal) axis,
            AKA pitch. (radians)
        theta_y (float): Rotation around the y (toward the limb) axis,
            AKA roll. (radians)
        theta_z (float): Rotation around the z (vertical) axis,
            AKA yaw. (radians)
        origin_x (float): Horizontal offset from the object in question
            to the camera (m).
        origin_y (float): Distance from the object in question to the
            camera (m).
        origin_z (float): Height difference from the object in question
            to the camera (m).
     Returns:
         camera_coords (np.ndarray): Coordinates in camera space --
            will be a set of y positions to correspond to the given x.
     """
    # todo diffraction correction?
    #     r = r * 1.2

    d = np.sqrt(h**2 + 2*h*r)
    theta = np.arccos(r / (r + h))

    xdcostheta = np.clip(x / (d * np.cos(theta)), -1, 1) # to avoid out of bounds
    y_world = d * np.cos(np.arcsin(xdcostheta)) * np.sin(theta)

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
    interp = UnivariateSpline(x_camera[order], y_camera[order]) #, ext='zeros') # ext='const')
    y_camera = interp(x)

    return y_camera
