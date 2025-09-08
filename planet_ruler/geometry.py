import numpy as np


def horizon_distance(r: float, h: float) -> float:
    """
    Estimate the distance to the horizon (limb) given a height
    and radius.

    Args:
        r (float): Radius of the body in question.
        h (float): Height above surface (units should match radius).
    Returns:
        d (float): Distance in same units as inputs.
    """
    return np.sqrt(h**2 + 2 * h * r)


def limb_camera_angle(r: float, h: float) -> float:
    """
    The angle the camera must tilt in theta_x or theta_y
    to center the limb. Complement of theta (angle of limb
    down from the x-y plane).

    Args:
        r (float): Radius of the body in question.
        h (float): Height above surface (units should match radius).
    Returns:
        theta_c (float): Angle of camera (radians).
    """
    theta = np.arccos(r / (r + h))
    return theta


def focal_length(w: float, fov: float) -> float:
    """
    The size of the CCD (inferred) based on focal length and
    field of view.

    Args:
        w (float): detector size (float): Width of CCD (m).
        fov (float): Field of view, assuming square (degrees).
    Returns:
        f (float): Focal length of the camera (m).
    """

    return w / (2 * np.tan(0.5 * fov * np.pi / 180))


def detector_size(f: float, fov: float) -> float:
    """
    The size of the CCD (inferred) based on focal length and
    field of view.

    Args:
        f (float): Focal length of the camera (m).
        # todo really need to pick either degrees or radians
        fov (float): Field of view, assuming square (degrees).
    Returns:
        detector size (float): Width of CCD (m).
    """

    return 2 * f * np.tan(fov * np.pi / 180.0 / 2)


def field_of_view(f: float, w: float) -> float:
    """
    The size of the CCD (inferred) based on focal length and
    field of view.

    Args:
        f (float): Focal length of the camera (m).
        w (float): Width of detector (m).
    Returns:
        fov (float): Field of view, assuming square (degrees).
    """

    return 2 * np.arctan(w / (2 * f)) * 180.0 / np.pi


def intrinsic_transform(
    camera_coords: np.ndarray,
    f: float = 1,
    px: float = 1,
    py: float = 1,
    x0: float = 0,
    y0: float = 0,
) -> np.ndarray:
    """
    Transform from camera coordinates into image coordinates.

    Args:
        camera_coords (np.ndarray): Coordinates of the limb in camera space.
            Array has Nx4 shape where N is the number of x-axis pixels
            in the image.
        f (float): Focal length of the camera (m).
        px (float): The scale of x pixels.
        py (float): The scale of y pixels.
        x0 (float): The x-axis principle point (should be center of image in
            pixel coordinates).
        y0 (float): The y-axis principle point. (should be center of image in
            pixel coordinates).
    Returns:
        pixel_coords (np.ndarray): Coordinates in image space.
    """
    # note the intentional extension to 3x4 (for homogenous coords)
    transform = np.array(
        [[float(f) / px, 0, x0, 0], [0, float(f) / py, y0, 0], [0, 0, 1, 0]]
    )

    # todo allow for shear/etc.

    pixel_coords = transform @ camera_coords

    # rescale back to homogenous coords (last dim == 1)
    pixel_coords = pixel_coords.T
    pixel_coords = pixel_coords / pixel_coords[:, -1].reshape((len(pixel_coords), 1))

    return pixel_coords


def extrinsic_transform(
    world_coords,
    theta_x: float = 0,
    theta_y: float = 0,
    theta_z: float = 0,
    origin_x: float = 0,
    origin_y: float = 0,
    origin_z: float = 0,
) -> np.ndarray:
    """
    Transform from world coordinates into camera coordinates.
    Note that for a limb calculation we will define origin_x/y/z
    as the camera position -- these should all be set to zero.

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

    x_rotation = np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta_x), -np.sin(theta_x)],
            [0, np.sin(theta_x), np.cos(theta_x)],
        ]
    )

    y_rotation = np.array(
        [
            [np.cos(theta_y), 0, np.sin(theta_y)],
            [0, 1, 0],
            [-np.sin(theta_y), 0, np.cos(theta_y)],
        ]
    )

    z_rotation = np.array(
        [
            [np.cos(theta_z), -np.sin(theta_z), 0],
            [np.sin(theta_z), np.cos(theta_z), 0],
            [0, 0, 1],
        ]
    )

    rotation = x_rotation @ y_rotation @ z_rotation

    translation = np.array([origin_x, origin_y, origin_z])

    # homogenous coords
    # see https://en.wikipedia.org/wiki/Camera_resectioning
    transform = np.zeros((4, 4))
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    transform[3, 3] = 1

    # camera_coords = world_coords @ transform.T
    camera_coords = transform @ world_coords.T

    return camera_coords


def limb_arc(
    r: float,
    n_pix_x: int,
    n_pix_y: int,
    h: float = 1,
    f: float = None,
    fov: float = None,
    w: float = None,
    x0: float = 0,
    y0: float = 0,
    theta_x: float = 0,
    theta_y: float = 0,
    theta_z: float = 0,
    origin_x: float = 0,
    origin_y: float = 0,
    origin_z: float = 0,
    return_full: bool = False,
    num_sample: int = 5000,
) -> np.ndarray:
    """
    Calculate the limb orientation in an image given the physical
    parameters of the system.

    Args:
        n_pix_x (int): Width of image (pixels).
        n_pix_y (int): Height of image (pixels).
        r (float): Radius of the body in question.
        h (float): Height above surface (units should match radius).
        f (float): Focal length of the camera (m).
        fov (float): Field of view, assuming square (degrees).
        w (float): detector size (float): Width of CCD (m).
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
        return_full (bool): Return both the x and y coordinates of the limb
            in camera space. Note these will *not* be interpolated back on
            to the pixel grid.
        num_sample (int): The number of points sampled from the simulated
            limb -- will be interpolated onto pixel grid. [default 1000]
     Returns:
         camera_coords (np.ndarray): Coordinates in camera space --
            will be a set of y positions to correspond to the given x.
    """

    # origin_* is the position of the origin of the world coordinate system
    # expressed in coordinates of the camera-centered coordinate system

    # here the origin of the world coordinates is the camera (why not)
    # the z-axis is vertical, going up from the center of the planet
    # the y-axis is horizontal, tangent to the surface toward the horizon
    # the x-axis is horizontal, tangent to the surface and orthogonal the y-axis

    # todo diffraction correction?
    #     r = r * 1.2

    assert (
        f is None or fov is None or w is None
    ), "Cannot specify focal length, field of view, and detector size. Set one of them to None."

    if f is None:
        f = focal_length(w, fov)
    if fov is None:
        fov = field_of_view(f, w)

    # distance to limb
    d = horizon_distance(r, h)
    # angle below x-z plane that points to horizon (same in all directions)
    limb_theta = limb_camera_angle(r, h)

    # using field of view and distance we can get linear
    # size of pixels in the projection plane (note: uses thin lens)
    pxy = 2 * (1 / f - 1 / d) ** -1 * np.tan(0.5 * fov * np.pi / 180) / n_pix_x

    # todo allow for auto-calculation of sample density
    # num_sample = int(np.pi / dphi)
    theta = np.ones(1) * limb_theta
    phi = np.linspace(-np.pi, np.pi, num=num_sample)
    theta, phi = np.meshgrid(theta, phi)

    x_world = r * np.sin(theta) * np.cos(phi)
    y_world = -(h + r) + r * np.cos(theta)
    z_world = r * np.sin(theta) * np.sin(phi)

    world_coords = np.ones((num_sample, 4))
    world_coords[:, 0] = x_world[:, 0]
    world_coords[:, 1] = y_world[:, 0]
    world_coords[:, 2] = z_world[:, 0]

    camera_coords = extrinsic_transform(
        world_coords=world_coords,
        theta_x=theta_x,
        theta_y=theta_y,
        theta_z=theta_z,
        origin_x=origin_x,
        origin_y=origin_y,
        origin_z=origin_z,
    )

    all_in_front = all(camera_coords[2, :] > 0)
    all_behind = all(camera_coords[2, :] < 0)

    # the limb can be both behind and in front of the camera
    # if it is, remove the behind part, or it will cause weirdness
    if not all_behind and not all_in_front:
        cut = np.where((camera_coords[2, :] > 0))[0]
        camera_coords = camera_coords[:, cut]

    pixel_coords = intrinsic_transform(
        camera_coords=camera_coords, f=f, px=pxy, py=pxy, x0=x0, y0=y0
    )

    if return_full:
        return pixel_coords

    x = pixel_coords[:, 0]
    y = pixel_coords[:, 1]

    x_pixel = np.arange(n_pix_x)

    x_reg = np.digitize(x, x_pixel)

    # get whatever actually landed in the FOV
    y_reg = y[(x_reg > 0) & (x_reg < n_pix_x)]
    x_reg = x_reg[(x_reg > 0) & (x_reg < n_pix_x)]

    # grab half of the arc (arbitrary) when the whole
    # circle is visible
    if all_in_front:
        try:
            diff = np.diff(x_reg, append=x_reg[-1])
            x_reg = x_reg[diff > 0]
            y_reg = y_reg[diff > 0]
        except IndexError:
            pass

    # if nothing is in the FOV, draw the proposed limb
    # as a flat line at the signed (in y-axis) Euclidean
    # distance between limb apex. this is purely to keep
    # the minimization space continuous

    arc_min = np.argmin(abs(np.gradient(y)))
    x_min = x[arc_min]
    y_min = y[arc_min]
    # assume the limb is centered in the image
    limb_x_min = int(n_pix_x * 0.5)
    limb_y_min = int(n_pix_y * 0.5)
    y_proxy = np.sqrt((limb_x_min - x_min) ** 2 + (limb_y_min - y_min) ** 2)
    # sign is compound but should be continuous when it enters the frame
    sign = (1 - 2 * (limb_x_min > x_min)) * (1 - 2 * (limb_y_min > y_min))
    if len(x_reg) == 0:
        y_pixel = sign * np.ones_like(x_pixel) * y_proxy
    else:
        # interp goes really wrong if things are not sorted
        order = np.argsort(x_reg)
        y_pixel = np.interp(x_pixel, x_reg[order], y_reg[order])
    return y_pixel
