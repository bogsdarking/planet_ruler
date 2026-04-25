# Copyright 2025 Brandon Anderson
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional
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
        theta_x (float): Rotation around the x (horizontal lateral) axis,
            AKA pitch — tilts the camera up/down. (radians)
        theta_y (float): Rotation around the y (toward-limb) axis, AKA roll.
            When theta_z=0, acts as a pure phase shift in φ with no effect on
            the projected arc shape. When theta_z≠0, the z-rotation breaks
            that symmetry and theta_y has a genuine effect on the arc. (radians)
        theta_z (float): Rotation around the z (vertical) axis, AKA yaw.
            Use theta_z=π for the physically correct ∪-shaped horizon arc
            (near limb visible, more planet at image center). (radians)
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


def limb_arc_sample(
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
        theta_x (float): Rotation around the x (horizontal lateral) axis,
            AKA pitch — tilts the camera up/down. (radians)
        theta_y (float): Rotation around the y (toward-limb) axis, AKA roll.
            When theta_z=0, acts as a pure phase shift in φ with no effect on
            the projected arc shape. When theta_z≠0, the z-rotation breaks
            that symmetry and theta_y has a genuine effect on the arc. (radians)
        theta_z (float): Rotation around the z (vertical) axis, AKA yaw.
            Use theta_z=π for the physically correct ∪-shaped horizon arc
            (near limb visible, more planet at image center). (radians)
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


def get_rotation_matrix(theta_x: float, theta_y: float, theta_z: float) -> np.ndarray:
    """
    Compute combined rotation matrix from Euler angles.
    Extracted from extrinsic_transform for reuse.

    Returns:
        R: 3x3 rotation matrix
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

    return x_rotation @ y_rotation @ z_rotation


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
    x_coords: Optional[np.ndarray] = None,
    **kwargs,  # Ignore num_sample - not needed!
) -> np.ndarray:
    """
    Calculate limb position analytically at each pixel x-coordinate.

    No sampling or interpolation - directly solves for phi at each column.
    This eliminates edge artifacts and is sometimes faster than sampling methods.

    Mathematical approach:
    1. Limb is parameterized by angle phi around circle
    2. For each x_pixel, solve: x_pixel = f(phi) for phi
    3. This reduces to: a·cos(phi) + b·sin(phi) = c
    4. Standard analytical solution exists!

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
        theta_x (float): Rotation around the x (horizontal lateral) axis,
            AKA pitch — tilts the camera up/down. (radians)
        theta_y (float): Rotation around the y (toward-limb) axis, AKA roll.
            When theta_z=0, acts as a pure phase shift in φ with no effect on
            the projected arc shape. When theta_z≠0, the z-rotation breaks
            that symmetry and theta_y has a genuine effect on the arc. (radians)
        theta_z (float): Rotation around the z (vertical) axis, AKA yaw.
            Use theta_z=π for the physically correct ∪-shaped horizon arc
            (near limb visible, more planet at image center). (radians)
        origin_x (float): Horizontal offset from the object in question
            to the camera (m).
        origin_y (float): Distance from the object in question to the
            camera (m).
        origin_z (float): Height difference from the object in question
            to the camera (m).
        return_full (bool): Return both the x and y coordinates of the limb
            in camera space. Note these will *not* be interpolated back on
            to the pixel grid.
        x_coords: Optional array of x-coordinates to compute (default: all pixels).
                  For sparse computation (e.g., manual annotation), pass only
                  the x-coordinates where you have data. Dramatically speeds up
                  fitting when only a few points are annotated.

    Returns:
        y_pixel: Array of y-coordinates for each x-pixel column
                 Length matches len(x_coords) if provided, else n_pix_x
    """
    # Setup (same as original)
    assert (
        f is None or fov is None or w is None
    ), "Cannot specify focal length, field of view, and detector size."

    if f is None:
        f = focal_length(w, fov)
    if fov is None:
        fov = field_of_view(f, w)

    d = horizon_distance(r, h)
    limb_theta = limb_camera_angle(r, h)
    pxy = 2 * (1 / f - 1 / d) ** -1 * np.tan(0.5 * fov * np.pi / 180) / n_pix_x

    # Get rotation matrix
    R = get_rotation_matrix(theta_x, theta_y, theta_z)

    # Limb geometry: circle in x-z plane
    # x_world = r * sin(limb_theta) * cos(phi)
    # y_world = -(h + r) + r * cos(limb_theta)  [constant!]
    # z_world = r * sin(limb_theta) * sin(phi)

    A = r * np.sin(limb_theta)  # Circle radius in x-z plane
    B = -(h + r) + r * np.cos(limb_theta)  # Constant y_world

    # After rotation + translation, camera coordinates are:
    # X_cam = R[0,0]*A*cos(phi) + R[0,1]*B + R[0,2]*A*sin(phi) + origin_x
    # Y_cam = R[1,0]*A*cos(phi) + R[1,1]*B + R[1,2]*A*sin(phi) + origin_y
    # Z_cam = R[2,0]*A*cos(phi) + R[2,1]*B + R[2,2]*A*sin(phi) + origin_z

    # Rewrite as: X_cam = C1*cos(phi) + C2 + C3*sin(phi)
    C1 = R[0, 0] * A
    C2 = R[0, 1] * B + origin_x
    C3 = R[0, 2] * A

    E1 = R[1, 0] * A
    E2 = R[1, 1] * B + origin_y
    E3 = R[1, 2] * A

    D1 = R[2, 0] * A
    D2 = R[2, 1] * B + origin_z
    D3 = R[2, 2] * A

    # Perspective projection: x_pixel = f * X_cam / Z_cam / px + x0
    # Rearranging: (x_pixel - x0) * px * Z_cam = f * X_cam
    # Substituting: (x_pixel - x0) * px * (D1*cos + D2 + D3*sin) = f * (C1*cos + C2 + C3*sin)
    # Collecting: a*cos(phi) + b*sin(phi) = c
    # where:
    #   a = (x_pixel - x0)*px*D1 - f*C1
    #   b = (x_pixel - x0)*px*D3 - f*C3
    #   c = f*C2 - (x_pixel - x0)*px*D2

    if x_coords is None:
        x_pixel_arr = np.arange(n_pix_x)
    else:
        x_pixel_arr = x_coords  # Use sparse coordinates!

    y_pixel = np.zeros(len(x_pixel_arr))
    phi_solutions = np.zeros(len(x_pixel_arr))

    # Vectorize over all x_pixel values
    a = (x_pixel_arr - x0) * pxy * D1 - f * C1
    b = (x_pixel_arr - x0) * pxy * D3 - f * C3
    c = f * C2 - (x_pixel_arr - x0) * pxy * D2

    # Solve: a*cos(phi) + b*sin(phi) = c
    # Standard solution: phi = atan2(b, a) ± acos(c / sqrt(a² + b²))

    discriminant = np.sqrt(a**2 + b**2)

    # Check if solution exists (with small numerical tolerance)
    eps = 1e-6
    valid_mask = (discriminant > eps) & (np.abs(c) <= discriminant * (1 + eps))

    # For valid pixels, compute phi
    phi_base = np.arctan2(b[valid_mask], a[valid_mask])
    phi_offset = np.arccos(c[valid_mask] / discriminant[valid_mask])

    # Two solutions (± offset)
    phi1 = phi_base + phi_offset
    phi2 = phi_base - phi_offset

    # VECTORIZED: Pick the solution with Z > 0 (in front of camera)
    eps_z = 1e-6

    # Compute Z for both solutions (vectorized)
    Z1 = D1 * np.cos(phi1) + D2 + D3 * np.sin(phi1)
    Z2 = D1 * np.cos(phi2) + D2 + D3 * np.sin(phi2)

    # Choose phi1 if its Z is positive, otherwise phi2
    use_phi1 = Z1 > -eps_z
    phi_chosen = np.where(use_phi1, phi1, phi2)
    Z_chosen = np.where(use_phi1, Z1, Z2)

    # Check if chosen solution is valid (in front of camera)
    solution_valid = Z_chosen > -eps_z

    # Compute Y_cam for all valid pixels (vectorized)
    Y_cam = E1 * np.cos(phi_chosen) + E2 + E3 * np.sin(phi_chosen)

    # Safe division (avoid Z near zero)
    Z_safe = np.where(np.abs(Z_chosen) > eps_z, Z_chosen, np.sign(Z_chosen) * eps_z)

    # Compute y_pixel for all valid solutions
    valid_indices = np.where(valid_mask)[0]
    y_pixel[valid_indices] = f * Y_cam / Z_safe / pxy + y0
    phi_solutions[valid_indices] = phi_chosen

    # Update valid_mask to exclude pixels where both solutions are behind camera
    valid_mask[valid_indices[~solution_valid]] = False

    # Handle invalid pixels (no solution or behind camera)
    if not np.any(valid_mask):
        # Nothing in FOV - find apex analytically
        # Apex is where dy/dφ = 0
        # This reduces to solving: a_apex*sin(φ) + b_apex*cos(φ) = c_apex

        a_apex = E2 * D1 - E1 * D2
        b_apex = E3 * D2 - E2 * D3
        c_apex = E1 * D3 - E3 * D1

        discriminant_apex = np.sqrt(a_apex**2 + b_apex**2)

        if discriminant_apex > 1e-10 and abs(c_apex) <= discriminant_apex:
            # Standard solution for a*sin(φ) + b*cos(φ) = c
            phi_base_apex = np.arctan2(a_apex, b_apex)
            phi_offset_apex = np.arccos(c_apex / discriminant_apex)

            # Two solutions - test both
            candidates = [
                phi_base_apex + phi_offset_apex,
                phi_base_apex - phi_offset_apex,
            ]

            x_apex = None
            y_apex = None

            for phi_apex in candidates:
                X_apex = C1 * np.cos(phi_apex) + C2 + C3 * np.sin(phi_apex)
                Y_apex = E1 * np.cos(phi_apex) + E2 + E3 * np.sin(phi_apex)
                Z_apex = D1 * np.cos(phi_apex) + D2 + D3 * np.sin(phi_apex)

                # Project (even if behind camera - matches old code)
                if abs(Z_apex) > 1e-10:
                    x_apex = f * X_apex / Z_apex / pxy + x0
                    y_apex = f * Y_apex / Z_apex / pxy + y0
                    break  # Take first valid solution

            # Fallback if both candidates have Z ≈ 0
            if x_apex is None:
                x_apex = n_pix_x * 0.5
                y_apex = n_pix_y * 0.5
        else:
            # No solution exists - use image center
            x_apex = n_pix_x * 0.5
            y_apex = n_pix_y * 0.5

        # Signed distance (matches old code exactly)
        limb_x_min = int(n_pix_x * 0.5)
        limb_y_min = int(n_pix_y * 0.5)
        y_proxy = np.sqrt((limb_x_min - x_apex) ** 2 + (limb_y_min - y_apex) ** 2)
        sign = (1 - 2 * (limb_x_min > x_apex)) * (1 - 2 * (limb_y_min > y_apex))

        y_pixel = sign * np.ones_like(x_pixel_arr, dtype=float) * y_proxy

    if return_full:
        # Return both x and y (for compatibility)
        full_coords = np.column_stack([x_pixel_arr, y_pixel])
        return full_coords

    return y_pixel


def limb_arc_sagitta(
    u: np.ndarray,
    theta_x: float,
    f_px: float,
    r: float,
    h: float,
) -> np.ndarray:
    """Exact projected sagitta of the planetary limb arc at pixel offset u.

    Derivation (perspective geometry, theta_y=0, theta_z=pi for ∪-arc):
      The camera is at altitude h above a sphere of radius r.  In camera
      coordinates the visible horizon is the set of tangent points satisfying
      X·r̂ = r²/(r+h).  After rotation R_x(theta_x)·R_z(pi) the horizon circle
      projects as a conic; solving the x-pixel equation for the azimuth angle
      phi and back-substituting for the y-pixel yields the closed form below.

      The formula is verified to machine precision against limb_arc() for
      theta_x ∈ {-alpha, 0, alpha, 2*alpha} and is invariant to theta_y
      (verified to 2.7e-7 px residuals for theta_y ∈ {0, 0.01, 0.05, 0.1}).

    At theta_x=0 this reduces exactly to
        s(u) = kappa * (sqrt(f_px**2 + u**2) - f_px)
    where kappa = sqrt(h*(2r+h)) / r = 1/K.  Consequently, the OLS fit
    s = s0 - c*A(u) (A = sqrt(f²+u²) - f) recovers K = 1/|c| with zero
    residual — the fundamental reason the hyperbola model is exact at theta_x=0.

    Args:
        u:        Horizontal pixel offsets from the image centre (u = x - x0).
        theta_x:  Camera tilt in radians (0 = horizontal, positive = looking
                  down toward the horizon).
        f_px:     Focal length in pixels (f_mm / sensor_width_mm * n_pix_x).
        r:        Planetary radius [m].
        h:        Camera altitude above surface [m].

    Returns:
        Sagitta at each u [pixels].  Positive values are below the arc apex
        for ∪-arcs (theta_z=pi).

    Exact formulas:
        kappa = sqrt(h * (2*r + h)) / r
        g(u)  = sqrt(f_px**2 + u**2 * (cos(theta_x)**2 - kappa**2*sin(theta_x)**2))
        s(u)  = (f_px**2 + u**2*cos(theta_x)*(cos(theta_x) + kappa*sin(theta_x)) - f_px*g(u))
                / ((f_px*sin(theta_x) + cos(theta_x)*g(u)/kappa) * (kappa*sin(theta_x) + cos(theta_x)))
    """
    u = np.asarray(u, dtype=float)
    kappa = np.sqrt(h * (2.0 * r + h)) / r
    ct = np.cos(theta_x)
    st = np.sin(theta_x)
    g = np.sqrt(f_px**2 + u**2 * (ct**2 - kappa**2 * st**2))
    numer = f_px**2 + u**2 * ct * (ct + kappa * st) - f_px * g
    denom = (f_px * st + ct * g / kappa) * (kappa * st + ct)
    return numer / denom


def estimate_radius_from_limb_arc(
    limb: np.ndarray,
    h: float,
    f_px: float,
    x0: Optional[float] = None,
    sigma_px: "float | str" = "auto",
    n_sigma: float = 1.0,
) -> dict:
    """
    Estimate planetary radius from a manually annotated limb arc.

    **Hyperbola K estimator** — direct fit, exact for horizontal cameras.

    The perspective projection of the horizon circle is algebraically a
    hyperbola for all practical aerial views (theta_x < arctan(K) = π/2−α,
    where α = arccos(r/(r+h)) is the dip angle and K = r/√(h·(2r+h))).
    It transitions to a parabola at theta_x = arctan(K) and to an ellipse
    only for near-nadir orientations.  In all cases the conic is
    characterised by the single dimensionless ratio K.  Radius follows from
    K and altitude via r = h·(K² + K·√(K²+1)).

    The arc sagitta is fit to the model s = s₀ − c·A(u), where
    A(u) = √(f²+u²) − f.  At theta_x=0 this is exact (the arc shape IS
    κ·A(u) where κ = 1/K), so K = 1/|c| recovers the true radius with
    zero residual.  For theta_x near the dip angle α (typical real photos)
    it is an approximation whose bias scales as sin²(theta_x)/K² ≈ 10⁻⁵.
    The earlier chordspan formula (K = A_L / (|a|·L²), where |a| is the
    OLS parabola curvature) has a larger systematic bias that scales with
    (L/f_px)²; for wide-angle cameras this can exceed 40 km.

    Uncertainty bounds use OLS covariance: σ_c = σ_px·√((XᵀX)⁻¹₁₁) where X
    is the [1, −A(u)] design matrix, and σ_K = σ_c / c².  The bounds are
    then mapped through the nonlinear r(K) map.

    Steps:
        1. Conic fit (algebraic, 5-DOF) to determine principal-axis direction.
        2. Rotate annotation into principal frame (u=chord, s=sagitta).
        3. Degree-2 OLS fit for curvature a_rot (still needed for apex coords).
        4. Hyperbola fit s = s₀ − c·A(u); K = 1/|c|.
        5. OLS covariance → σ_K → r_low / r_high via r(K) map.

    Use the returned r_low / r_high as parameter_limits["r"] in a fit_config
    dict before running differential evolution.

    Args:
        limb:     Sparse target array of shape (W,) with np.nan for unannotated
                  x-positions.  Direct output of load_limb_points_from_json().
        h:        Observer altitude above the surface (metres).
        f_px:     Focal length in pixels = f * W / w.
        x0:       Principal point x (pixels).  Reserved for future use.
        sigma_px: Per-point annotation noise (pixels), or "auto" (default) to
                  derive it from the RMS of sagitta-direction residuals.
                  Propagated analytically to K_sigma and r_low/r_high.
        n_sigma:  Bound width in units of sigma (default 1.0).  K_sigma and
                  r_sigma in the return dict are always 1-sigma values.

    Returns:
        dict with keys:
            r             – best-estimate radius (metres)
            r_low         – n_sigma lower bound (metres)
            r_high        – n_sigma upper bound (metres)
            r_sigma       – 1-sigma uncertainty, linearised (metres)
            K             – hyperbola K = 1/|c| from fit s = s₀ − c·A(u)
            K_sigma       – propagated 1-sigma uncertainty on K
            n_points      – number of valid annotated points
            residual_rms  – RMS of sagitta-direction model residuals (pixels)
            arc_angle_deg – rotation of arc principal axis from image x-axis
            x_apex        – apex x coordinate in image pixels (visualisation)
            y_apex        – apex y coordinate in image pixels (visualisation)
            status        – "ok" | "flat_arc" | "too_few_points"
            warnings      – list of warning strings
    """
    warnings_list = []
    W = len(limb)
    if x0 is None:
        x0 = W / 2.0

    valid = ~np.isnan(limb)
    n_valid = int(np.sum(valid))

    def _nan_result(status, extra_warnings=None):
        return {
            "r": np.nan, "r_low": np.nan, "r_high": np.nan,
            "r_sigma": np.nan, "K": np.nan, "K_sigma": np.nan,
            "n_points": n_valid, "residual_rms": np.nan,
            "arc_angle_deg": np.nan, "x_apex": np.nan, "y_apex": np.nan,
            "status": status,
            "warnings": extra_warnings or [],
        }

    if n_valid < 4:
        return _nan_result(
            "too_few_points",
            [f"Only {n_valid} annotated point(s); need at least 4"],
        )

    x_px = np.where(valid)[0].astype(float)
    y_px = limb[valid]

    # ── Step 1: principal-axis rotation via conic fit ─────────────────────────
    # Bookstein algebraic form: a·x²+b·xy+c·y²+d·x+e·y = 1  (5 DOF, linear).
    # Eigendecompose the quadratic part to get the chord/sagitta directions
    # directly: the eigenvector with the *smaller* |eigenvalue| is the chord
    # (low curvature) and the one with the *larger* |eigenvalue| is the
    # sagitta (high curvature).  This is robust for flat arcs where the
    # cross-term comparison is numerically unreliable.
    # Fall back to the image x-axis as chord when < 5 points are available.
    u_dir = np.array([1.0, 0.0])
    theta = 0.0

    if n_valid >= 5:
        M = np.column_stack(
            [x_px**2, x_px * y_px, y_px**2, x_px, y_px]
        )
        p, _, rank, _ = np.linalg.lstsq(M, np.ones(n_valid), rcond=None)
        if rank >= 5:
            C_mat = np.array([[p[0], p[1] / 2], [p[1] / 2, p[2]]])
            eigvals, eigvecs = np.linalg.eigh(C_mat)
            idx_u = int(np.argmin(np.abs(eigvals)))  # chord = min curvature
            u_dir = eigvecs[:, idx_u]
            if u_dir[0] < 0:     # canonical: chord points in +x half
                u_dir = -u_dir
            theta = float(np.arctan2(u_dir[1], u_dir[0]))
        else:
            warnings_list.append(
                "Conic fit rank-deficient; using axis-aligned frame"
            )

    # ── Step 2: rotate annotation points into principal frame ─────────────────
    cos_t, sin_t = u_dir[0], u_dir[1]
    x_cen, y_cen = np.mean(x_px), np.mean(y_px)
    dx, dy = x_px - x_cen, y_px - y_cen
    u_rot = cos_t * dx + sin_t * dy    # chord direction (low curvature)
    s_rot = -sin_t * dx + cos_t * dy   # sagitta direction (high curvature)

    span_rot = float(np.max(u_rot) - np.min(u_rot))
    if span_rot < W / 4:
        warnings_list.append(
            f"Annotated points span only {span_rot:.0f} px along the arc"
            " — K estimate may be unreliable"
        )

    # ── Step 3: parabola fit for curvature and residuals ─────────────────────
    coeffs_rot = np.polyfit(u_rot, s_rot, 2)
    a_rot = float(coeffs_rot[0])

    if abs(a_rot) < 1e-12:
        return _nan_result(
            "flat_arc", ["Arc is effectively flat in principal frame"]
        )

    s_model = np.polyval(coeffs_rot, u_rot)
    residual_rms = float(np.sqrt(np.mean((s_rot - s_model) ** 2)))

    sigma_px_val = (
        (residual_rms if residual_rms > 0 else 1.0)
        if sigma_px == "auto"
        else float(sigma_px)
    )

    # ── Step 4: direct hyperbola fit  s = s₀ − c·A(u) ───────────────────────
    # K = 1/|c| is exact for a horizontal camera (θ=0); residuals are machine-
    # precision zero.  For θ near the dip angle α (typical real photos) it is
    # an approximation, but better than the chordspan formula especially for
    # wide-angle cameras where (L/f_px)² degradation of the parabola is large.
    # |c| handles both ∪-arcs (c<0, theta_z=π) and ∩-arcs (c>0) identically.
    A_u = np.sqrt(f_px**2 + u_rot**2) - f_px
    X_hyp = np.column_stack([np.ones_like(u_rot), -A_u])
    coeff_hyp, _, _, _ = np.linalg.lstsq(X_hyp, s_rot, rcond=None)
    abs_c = abs(float(coeff_hyp[1]))

    L   = (np.max(u_rot) - np.min(u_rot)) / 2.0
    A_L = float(np.sqrt(f_px**2 + L**2) - f_px)
    s_est = abs_c * A_L

    s_resid = s_rot - X_hyp @ coeff_hyp
    residual_rms = float(np.sqrt(np.mean(s_resid**2)))
    sigma_px_val = (residual_rms if sigma_px == "auto" and residual_rms > 0
                    else sigma_px_val)

    K_est = 1.0 / abs_c if abs_c > 0 else np.inf

    status = "flat_arc" if s_est < sigma_px_val else "ok"
    if status == "flat_arc":
        warnings_list.append(
            f"Estimated sagitta ({s_est:.2f} px) is below noise floor "
            f"({sigma_px_val:.1f} px) — radius bound will be very wide"
        )

    # ── Step 5: σ_K from OLS covariance of c ─────────────────────────────────
    # Var(ĉ) = σ²·(XᵀX)⁻¹₁₁  where X = [1, −A(u)].
    # σ_K = σ_c / c²  because K = 1/|c|.
    try:
        XtX_inv = np.linalg.inv(X_hyp.T @ X_hyp)
        sigma_c = sigma_px_val * np.sqrt(float(XtX_inv[1, 1]))
    except np.linalg.LinAlgError:
        sigma_c = abs_c * 0.1
        warnings_list.append("XᵀX singular; using fallback σ_c = 0.1·|c|")
    K_sigma = sigma_c / (abs_c**2) if abs_c > 0 else np.inf

    # ── Step 6: K → r ─────────────────────────────────────────────────────────
    def _r_from_K(K):
        K = max(float(K), 1e-12)
        return h * (K**2 + K * np.sqrt(K**2 + 1))

    r      = _r_from_K(K_est)
    r_low  = _r_from_K(max(K_est - n_sigma * K_sigma, 1e-9))
    r_high = _r_from_K(K_est + n_sigma * K_sigma)

    sq = np.sqrt(K_est**2 + 1)
    drdK = h * (2 * K_est + sq + K_est**2 / sq)
    r_sigma = abs(drdK) * K_sigma

    # Apex in image coordinates (for visualisation only — not used by K)
    u_apex = -coeffs_rot[1] / (2.0 * a_rot)
    s_apex = float(np.polyval(coeffs_rot, u_apex))
    x_apex_img = x_cen + cos_t * u_apex - sin_t * s_apex
    y_apex_img = y_cen + sin_t * u_apex + cos_t * s_apex

    return {
        "r": r,
        "r_low": r_low,
        "r_high": r_high,
        "r_sigma": r_sigma,
        "K": K_est,
        "K_sigma": K_sigma,
        "n_points": n_valid,
        "residual_rms": residual_rms,
        "arc_angle_deg": float(np.degrees(theta)),
        "x_apex": float(x_apex_img),
        "y_apex": float(y_apex_img),
        "status": status,
        "warnings": warnings_list,
    }
