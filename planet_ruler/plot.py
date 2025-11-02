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

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib.lines import Line2D
from planet_ruler.geometry import limb_camera_angle, horizon_distance, limb_arc
from planet_ruler.fit import unpack_parameters, unpack_diff_evol_posteriors
from typing import Optional
import pandas as pd
import seaborn as sns
import cv2

matplotlib.rcParams["figure.figsize"] = (16, 10)
matplotlib.rcParams.update({"font.size": 18})


def plot_image(im_arr: np.ndarray, gradient: bool = False, show: bool = True) -> None:
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
    y: np.ndarray, show: bool = True, c: str = "y", s: int = 10, alpha: float = 0.2
) -> None:
    """
    Display the limb (usually on top of an image).

    Args:
        y (np.ndarray): Array of image values.
        show (bool): Display the image.
        c (str): Color of the limb.
        s (int): Size of marker.
        alpha (float): Opacity of markers.
    """
    # brighten if the annotations are sparse
    if len(y[y == y]) < 20:
        s = 40
        alpha = 0.8
    plt.scatter(np.arange(len(y)), y, c=c, s=s, alpha=alpha)
    if show:
        plt.show()


def plot_3d_solution(
    r: float,
    h: float = 1,
    zoom: float = 1,
    savefile: Optional[str] = None,
    legend: bool = True,
    vertical_axis: str = "z",
    azim: Optional[float] = None,
    roll: Optional[float] = None,
    x_axis: bool = False,
    y_axis: bool = True,
    z_axis: bool = False,
    **kwargs,
) -> None:
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
    h = h * (1.0 / zoom)
    limb_theta = limb_camera_angle(r, h)
    d = horizon_distance(r, h)
    horizon_radius = d * np.cos(limb_theta)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection="3d")

    theta = np.linspace(-limb_theta, limb_theta, 100)
    phi = np.linspace(0, np.pi, 50)
    theta, phi = np.meshgrid(theta, phi)

    x_world = r * np.sin(theta) * np.cos(phi)
    y_world = -(h + r) + r * np.cos(theta)
    z_world = r * np.sin(theta) * np.sin(phi)

    ax.plot_wireframe(x_world, y_world, z_world, color="y", alpha=0.1, label="planet")

    theta = np.ones(1) * limb_theta
    phi = np.linspace(-np.pi, np.pi, num=5000)
    theta, phi = np.meshgrid(theta, phi)

    x_world = r * np.sin(theta) * np.cos(phi)
    y_world = -(h + r) + r * np.cos(theta)
    z_world = r * np.sin(theta) * np.sin(phi)
    ax.plot(x_world, y_world, z_world, c="k", label="limb")

    ax.scatter(
        [0],
        [0],
        [0],
        marker=".",
        s=100,
        label=f"camera/origin [elevation = {int(h / 1000)} km]",
    )

    if y_axis:
        ax.plot([0, 0], [-h, 0], [0, 0], c="g", ls="--", alpha=0.7, label="y-axis")
    if z_axis:
        ax.plot(
            [0, 0],
            [0, 0],
            [-horizon_radius, horizon_radius],
            c="b",
            ls="--",
            alpha=0.7,
            label="z-axis",
        )
    if x_axis:
        ax.plot(
            [-horizon_radius, horizon_radius],
            [0, 0],
            [0, 0],
            c="k",
            ls="--",
            alpha=0.7,
            label="x-axis",
        )

    theta = limb_theta
    phi = 0

    x_world = r * np.sin(theta) * np.cos(phi)
    y_world = -(h + r) + r * np.cos(theta)
    z_world = r * np.sin(theta) * np.sin(phi)

    ax.plot(
        [0, x_world],
        [0, y_world],
        [0, z_world],
        c="purple",
        ls="--",
        alpha=0.7,
        label=f"line of sight [distance = {int(d / 1000)} km]",
    )

    plt.autoscale(False)
    theta = np.linspace(0, 2 * np.pi, 1000)
    phi = np.linspace(0, np.pi, 500)
    theta, phi = np.meshgrid(theta, phi)

    x_world = r * np.sin(theta) * np.cos(phi)
    y_world = -(h + r) + r * np.cos(theta)
    z_world = r * np.sin(theta) * np.sin(phi)

    ax.plot_wireframe(x_world, y_world, z_world, color="b", alpha=0.01)

    # Handle matplotlib version compatibility
    try:
        ax.view_init(
            elev=limb_theta * 180 / np.pi,
            azim=azim,
            roll=roll,
            vertical_axis=vertical_axis,
        )
    except TypeError:
        # Fallback for older matplotlib versions
        ax.view_init(elev=limb_theta * 180 / np.pi, azim=azim)
    plt.axis("off")
    if legend:
        plt.legend(fontsize=12)
    plt.axis("equal")
    if savefile is not None:
        plt.savefig(savefile, bbox_inches="tight")
    plt.show()


def plot_topography(image: np.ndarray) -> None:
    """
    Display the full limb, including the section not seen in the image.

    Args:
        image (np.ndarray): Image array.
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection="3d")

    image = np.clip(image.sum(axis=-1), None, 1000)
    n_rows, n_cols = image.shape
    x = np.arange(n_cols)[::-1]
    y = np.arange(n_rows)

    x, y = np.meshgrid(x, y)

    ls = LightSource(altdeg=30, azdeg=-15)
    ax.plot_surface(x, y, image, lightsource=ls)
    # Handle matplotlib version compatibility
    try:
        ax.view_init(elev=90, azim=0, roll=-90)
    except TypeError:
        # Fallback for older matplotlib versions
        ax.view_init(elev=90, azim=0)

    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

    plt.axis("off")
    plt.show()


def plot_gradient_field_at_limb(
    y_pixels,
    image,
    image_smoothing=None,
    gradient_smoothing=5.0,
    streak_length=30,
    decay_rate=0.15,
    sample_spacing=50,
    arrow_scale=20,
):
    """
    Visualize gradient field along a proposed limb curve.

    Args:
        y_pixels (np.ndarray): Y-coordinates of limb at each x-position
        image (np.ndarray): Input image (H x W x 3 or H x W)
        image_smoothing (int): For gradient_field - Gaussian blur sigma applied to image
            before gradient computation. Removes high-frequency artifacts (crater rims,
            striations) that could mislead optimization. Different from gradient_smoothing.
        gradient_smoothing (float): Initial smoothing for gradient direction estimation
        streak_length (int): How far to sample along gradients
        decay_rate (float): Exponential decay rate for sampling
        sample_spacing (int): Sample every N pixels along x-axis
        arrow_scale (float): Scale factor for arrow lengths

    Returns:
        fig, ax: Matplotlib figure and axis objects
    """
    # Import here to avoid circular import issues
    from planet_ruler.image import gradient_field

    if image_smoothing is not None:
        working_image = cv2.GaussianBlur(
                    image.astype(np.float32),
                    (0, 0),  # Kernel size auto-determined from sigma
                    sigmaX=image_smoothing,
                    sigmaY=image_smoothing
                )
    else:
        working_image = image.copy()

    # Compute gradients using directional blur (now the default method)
    grad_data = gradient_field(
        working_image,
        gradient_smoothing=gradient_smoothing,
        streak_length=streak_length,
        decay_rate=decay_rate,
    )

    grad_mag = grad_data["grad_mag"]
    grad_angle = grad_data["grad_angle"]
    title = f"Gradient Field with Directional Blur (streak={streak_length}, decay={decay_rate})"

    # Compute curve tangent and normal at each point
    # Tangent direction: (1, dy/dx) in direction of increasing x
    dy_dx = np.gradient(y_pixels)

    # Normal perpendicular to tangent: rotate tangent 90° counter-clockwise
    # Tangent (1, dy/dx) → Normal (-dy/dx, 1)
    # This gives a vector perpendicular to curve, pointing "upward" (positive y-component)
    tangent_x = np.ones_like(dy_dx)
    tangent_y = dy_dx
    tangent_mag = np.sqrt(tangent_x**2 + tangent_y**2)

    # Normalize tangent
    tangent_x_unit = tangent_x / tangent_mag
    tangent_y_unit = tangent_y / tangent_mag

    # Normal: rotate tangent 90° CCW: (x,y) → (-y, x)
    normal_x_unit = -tangent_y_unit  # = -dy/dx / sqrt(1 + (dy/dx)²)
    normal_y_unit = tangent_x_unit  # = 1 / sqrt(1 + (dy/dx)²)

    normal_angle = np.arctan2(normal_y_unit, normal_x_unit)

    # Verify perpendicularity (dot product should be ~0)
    # dot = tangent_x_unit * normal_x_unit + tangent_y_unit * normal_y_unit
    # Should equal 0 for perpendicular vectors

    # Sample points along the curve
    x_samples = np.arange(0, len(y_pixels), sample_spacing)

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.imshow(image, cmap="gray", aspect="auto")
    ax.plot(
        np.arange(len(y_pixels)),
        y_pixels,
        "r-",
        linewidth=2,
        label="Proposed Limb",
        alpha=0.8,
    )

    # For each sample point, draw gradient arrow
    for x_idx in x_samples:
        if x_idx >= len(y_pixels):
            continue

        y_idx = int(np.clip(y_pixels[x_idx], 0, grad_data["image_height"] - 1))
        x_pos = x_idx
        y_pos = y_pixels[x_idx]

        # Get gradient at this location
        mag = grad_mag[y_idx, x_idx]
        angle = grad_angle[y_idx, x_idx]
        norm_angle = normal_angle[x_idx]  # Fixed: index directly, not as array

        # Compute alignment with normal
        angle_diff = np.arctan2(np.sin(angle - norm_angle), np.cos(angle - norm_angle))
        alignment = np.cos(angle_diff)

        # Arrow components (gradient direction)
        dx = mag * np.cos(angle) * arrow_scale
        dy = mag * np.sin(angle) * arrow_scale

        # Color based on alignment
        if abs(alignment) > 0.7:
            color = "lime"  # Good alignment
        elif abs(alignment) < 0.3:
            color = "yellow"  # Perpendicular
        else:
            color = "orange"  # Medium alignment

        # Draw gradient arrow
        ax.arrow(
            x_pos,
            y_pos,
            dx,
            dy,
            head_width=10,
            head_length=10,
            fc=color,
            ec=color,
            alpha=0.7,
            linewidth=2,
        )

        # # Draw tangent direction (magenta, dashed)
        # tangent_x_at_point = tangent_x_unit[x_idx]
        # tangent_y_at_point = tangent_y_unit[x_idx]
        # tan_dx = 30 * tangent_x_at_point
        # tan_dy = 30 * tangent_y_at_point
        # ax.arrow(
        #     x_pos,
        #     y_pos,
        #     tan_dx,
        #     tan_dy,
        #     head_width=8,
        #     head_length=8,
        #     fc="magenta",
        #     ec="magenta",
        #     alpha=0.6,
        #     linestyle="--",
        #     linewidth=1.5,
        # )

        # Draw normal direction (cyan, dashed)
        norm_dx = 30 * normal_x_unit[x_idx]
        norm_dy = 30 * normal_y_unit[x_idx]
        ax.arrow(
            x_pos,
            y_pos,
            norm_dx,
            norm_dy,
            head_width=8,
            head_length=8,
            fc="cyan",
            ec="cyan",
            alpha=0.5,
            linewidth=1.5,
        )

        # # Verify perpendicularity (for debugging)
        # dot_product = (
        #     tangent_x_at_point * normal_x_unit[x_idx]
        #     + tangent_y_at_point * normal_y_unit[x_idx]
        # )
        # if abs(dot_product) > 0.01:
        #     print(
        #         f"WARNING at x={x_idx}: dot product = {dot_product:.4f} (should be ~0)"
        #     )

    # Add legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color="r", linewidth=2, label="Proposed Limb"),
        Line2D(
            [0],
            [0],
            color="lime",
            marker=">",
            markersize=10,
            linestyle="",
            label="Gradient (well aligned)",
        ),
        Line2D(
            [0],
            [0],
            color="orange",
            marker=">",
            markersize=10,
            linestyle="",
            label="Gradient (medium aligned)",
        ),
        Line2D(
            [0],
            [0],
            color="yellow",
            marker=">",
            markersize=10,
            linestyle="",
            label="Gradient (perpendicular)",
        ),
        # Line2D(
        #     [0],
        #     [0],
        #     color="magenta",
        #     marker=">",
        #     markersize=10,
        #     linestyle="--",
        #     label="Curve Tangent",
        # ),
        Line2D(
            [0],
            [0],
            color="cyan",
            marker=">",
            markersize=10,
            linestyle="",
            label="Curve Normal (⊥ to tangent)",
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=12)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("X (pixels)", fontsize=12)
    ax.set_ylabel("Y (pixels)", fontsize=12)
    ax.set_aspect("equal")  # Force equal scaling (or normal appears angled)
    plt.tight_layout()
    return fig, ax


def compare_blur_methods(image, y_pixels=None):
    """
    Compare gradient magnitude with different blur methods.

    Args:
        image: Input image
        y_pixels: Optional limb curve to overlay
    """
    # Gaussian blur
    if len(image.shape) == 3:
        gray = image.sum(axis=2).astype(np.float32)
    else:
        gray = image.copy().astype(np.float32)

    gray_gauss = cv2.GaussianBlur(gray, (0, 0), 15)
    grad_y_g, grad_x_g = np.gradient(gray_gauss)
    mag_gauss = np.sqrt(grad_x_g**2 + grad_y_g**2)

    # Directional blur
    from planet_ruler.image import gradient_field

    grad_data = gradient_field(
        image, streak_length=30, decay_rate=0.15, gradient_smoothing=2.0
    )
    mag_directional = grad_data["grad_mag"]
    angle_dir = grad_data["grad_angle"]

    # For comparison, also compute without blur
    if len(image.shape) == 3:
        gray_orig = image.sum(axis=2).astype(np.float32)
    else:
        gray_orig = image.copy().astype(np.float32)
    grad_y_orig, grad_x_orig = np.gradient(gray_orig)
    mag_original = np.sqrt(grad_x_orig**2 + grad_y_orig**2)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    axes[0, 0].imshow(image, cmap="gray")
    axes[0, 0].set_title("Original Image")
    if y_pixels is not None:
        axes[0, 0].plot(
            np.arange(len(y_pixels)), y_pixels, "r-", linewidth=2, alpha=0.7
        )

    axes[0, 1].imshow(mag_original, cmap="hot")
    axes[0, 1].set_title("No Blur\nGradient Magnitude")
    if y_pixels is not None:
        axes[0, 1].plot(
            np.arange(len(y_pixels)), y_pixels, "cyan", linewidth=2, alpha=0.7
        )

    axes[1, 0].imshow(mag_gauss, cmap="hot")
    axes[1, 0].set_title("Gaussian Blur (σ=15)\nGradient Magnitude")
    if y_pixels is not None:
        axes[1, 0].plot(
            np.arange(len(y_pixels)), y_pixels, "cyan", linewidth=2, alpha=0.7
        )

    axes[1, 1].imshow(mag_directional, cmap="hot")
    axes[1, 1].set_title("Directional Blur (streak=30)\nGradient Magnitude")
    if y_pixels is not None:
        axes[1, 1].plot(
            np.arange(len(y_pixels)), y_pixels, "cyan", linewidth=2, alpha=0.7
        )

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    return fig, axes


def compare_gradient_fields(
    y_pixels_list,
    labels,
    image,
    image_smoothing=None,
    gradient_smoothing=5.0,
    streak_length=30,
    decay_rate=0.15,
):
    """
    Compare gradient alignment for multiple proposed limbs.

    Args:
        y_pixels_list (list): List of y-coordinate arrays (different limb proposals)
        labels (list): Labels for each limb
        image (np.ndarray): Input image
        gradient_smoothing (float): Initial smoothing for gradient direction estimation
        streak_length (int): How far to sample along gradients
        decay_rate (float): Exponential decay rate for sampling
    """
    # Import here to avoid circular import issues
    from planet_ruler.image import gradient_field

    if image_smoothing is not None:
        working_image = cv2.GaussianBlur(
                    image.astype(np.float32),
                    (0, 0),  # Kernel size auto-determined from sigma
                    sigmaX=image_smoothing,
                    sigmaY=image_smoothing
                )
    else:
        working_image = image.copy()

    # Compute gradients using directional blur (now the default method)
    grad_data = gradient_field(
        working_image,
        gradient_smoothing=gradient_smoothing,
        streak_length=streak_length,
        decay_rate=decay_rate,
    )

    grad_mag = grad_data["grad_mag"]
    grad_angle = grad_data["grad_angle"]

    # Compute alignment scores for each limb
    fig, axes = plt.subplots(
        len(y_pixels_list), 1, figsize=(16, 5 * len(y_pixels_list))
    )
    if len(y_pixels_list) == 1:
        axes = [axes]

    for idx, (y_pixels, label) in enumerate(zip(y_pixels_list, labels)):
        ax = axes[idx]
        ax.imshow(image, cmap="gray", aspect="auto")

        # Compute alignment
        dy_dx = np.gradient(y_pixels)
        normal_angle = np.arctan2(1, -dy_dx)

        x_idx = np.arange(len(y_pixels)).astype(int)
        y_idx = np.clip(y_pixels.astype(int), 0, grad_mag.shape[0] - 1)

        mag = grad_mag[y_idx, x_idx]
        angle = grad_angle[y_idx, x_idx]

        angle_diff = np.arctan2(
            np.sin(angle - normal_angle), np.cos(angle - normal_angle)
        )
        alignment = np.abs(np.cos(angle_diff))  # Use absolute for flux

        # Color the limb by alignment
        from matplotlib.collections import LineCollection

        points = np.array([x_idx, y_pixels]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, cmap="RdYlGn", linewidth=3)
        lc.set_array(alignment)
        lc.set_clim(0, 1)
        ax.add_collection(lc)

        # Compute flux (like the cost function)
        flux = np.sum(mag * alignment)
        mean_mag = np.mean(mag)
        mean_global = np.mean(grad_mag)

        ax.set_title(
            f"{label} | Flux: {flux:.0f} | Avg Mag: {mean_mag:.1f} "
            f"(global: {mean_global:.1f})",
            fontsize=12,
        )
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")

        # Add colorbar
        cbar = plt.colorbar(lc, ax=ax, orientation="vertical", pad=0.01)
        cbar.set_label("Gradient Alignment", fontsize=10)

    plt.tight_layout()
    return fig, axes


def plot_diff_evol_posteriors(observation, show_points: bool = False, log: bool = True):
    """
    Extract and display the final state population of a differential evolution
    minimization.

    Args:
        observation (object): Instance of LimbObservation (must have
            used differential evolution minimizer).
        show_points (bool): Show the individual population members in
            addition to the contour.
        log (bool): Set the y-scale to log.

    Returns:
        None
    """
    pop = unpack_diff_evol_posteriors(observation)

    for col in pop.columns:
        if col not in observation.free_parameters:
            continue
        if show_points:
            plt.scatter(pop[col], pop["mse"])
        sns.kdeplot(
            x=pop[col],
            y=pop["mse"],
            color="blue",
            warn_singular=False,
            label="posterior",
        )
        plt.axvline(
            observation.parameter_limits[col][0],
            ls="--",
            c="k",
            alpha=0.5,
            label="bounds",
        )
        plt.axvline(observation.parameter_limits[col][1], ls="--", c="k", alpha=0.5)
        try:
            plt.axvline(
                observation.init_parameter_values[col],
                ls="-",
                c="y",
                alpha=0.5,
                label="initial value",
            )
        except KeyError:
            pass
        plt.title(col)
        plt.grid(which="both", ls="--", alpha=0.2)
        ax = plt.gca()
        if log:
            ax.set_yscale("log")

        handles, labels = ax.get_legend_handles_labels()
        h_plus, l_plus = [Line2D([0], [0], color="blue", lw=2)], ["posterior"]
        plt.legend(handles + h_plus, labels + l_plus)

        plt.show()


def plot_full_limb(
    observation,
    x_min: int = None,
    x_max: int = None,
    y_min: int = None,
    y_max: int = None,
) -> None:
    """
    Display the full limb, including the section not seen in the image.

    Args:
        observation (object): Instance of LimbObservation.
        x_min (int): Left edge in pixels.
        x_max (int): Right edge in pixels.
        y_min (int): Bottom edge in pixels.
        y_max (int): Top edge in pixels.
    """
    try:
        params = observation.best_parameters.copy()
    except AttributeError:
        params = observation.init_parameter_values.copy()

    plt.imshow(observation.image)

    # Get image dimensions and radius
    n_pix_y, n_pix_x = observation.image.shape[:2]
    r = params.pop("r")  # Extract r and remove from params

    # Add required positional arguments
    pix = limb_arc(r, n_pix_x, n_pix_y, return_full=True, **params)
    x = pix[:, 0]
    y = pix[:, 1]
    plt.scatter(x, y)

    x = np.arange(observation.image.shape[1])
    y = limb_arc(r, n_pix_x, n_pix_y, **params)
    plt.scatter(x, y)

    ax = plt.gca()
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    plt.show()


def plot_segmentation_masks(observation) -> None:
    """
    Display all the classes/masks generated by the segmentation.

    Args:
        observation (object): Instance of LimbObservation.
    """
    for i, mask in enumerate(observation._segmenter._masks):
        mask = mask["segmentation"]
        plt.imshow(mask)
        plt.title(f"Mask {i}")
        plt.show()
