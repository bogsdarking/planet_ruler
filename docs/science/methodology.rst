Physical Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We establish a world coordinate system with origin at Earth's center, z-axis pointing toward 
the North Pole, and x-y plane defining the equator. An observer at altitude :math:`h` above 
the surface sits at position :math:`(0, 0, R+h)` where :math:`R` is Earth's radius. The horizon 
circle consists of points on the surface where sight lines from the observer become tangent to the sphere.

Camera Coordinate System
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The camera has its own coordinate system with origin at the lens, z-axis along the 
optical axis (viewing direction), y-axis pointing upward in the image, and x-axis pointing 
rightward. The camera's orientation in world space is described by three rotation angles:

- **Pitch** (:math:`\theta_p`): Rotation about the x-axis (looking up/down)
- **Roll** (:math:`\theta_r`): Rotation about the z-axis (image tilt)  
- **Yaw** (:math:`\theta_y`): Rotation about the y-axis (compass heading)

The rotation matrix :math:`R_{\text{cam}}` combines these three rotations:

.. math::

   R_{\text{cam}} = R_z(\theta_y) R_x(\theta_p) R_y(\theta_r)

This matrix transforms world coordinates into camera coordinates 
through :math:`\mathbf{p}_{\text{cam}} = R_{\text{cam}} \mathbf{p}_{\text{world}}`.

Projection to Image Coordinates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The pinhole camera model projects 3D camera coordinates onto the 2D image plane. 
A point at camera coordinates :math:`(x_c, y_c, z_c)` projects to image coordinates :math:`(u, v)` through:

.. math::

   u = f_x \frac{x_c}{z_c} + c_x, \quad v = f_y \frac{y_c}{z_c} + c_y

where :math:`f_x`, :math:`f_y` are focal lengths in pixels (typically equal), and :math:`(c_x, c_y)` is 
the principal point (image center). These four parameters define the intrinsic matrix :math:`K`.

Complete Measurement Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Given a set of free parameters :math:`\{R, h, \theta_p, \theta_r, \theta_y, f_x, c_x, c_y\}` 
(some may be fixed if known), we can predict where the horizon circle should appear in the image:

1. Generate points on the horizon circle in world coordinates
2. Transform to camera coordinates using :math:`R_{\text{cam}}`
3. Project to image coordinates using the intrinsic matrix :math:`K`
4. Compare predicted limb arc to observed horizon

The inverse problem—estimating unknown parameters from an observed horizon—requires optimization. 
We define a cost function measuring the mismatch between predicted and observed horizon geometry, 
then search parameter space to minimize this cost.

Parameter Bounds and Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Physical constraints provide bounds on free parameters. Earth's radius is approximately 6371 km
(±50 km for various reference models). Altitude for aircraft photography ranges from 5-20 km.
Camera pitch typically ranges from -90° to +90°, roll from -180° to +180°. Focal length can
often be constrained from EXIF metadata to within ±10%.

Global optimization algorithms (differential evolution, basin-hopping, dual annealing) search this
bounded parameter space to find the best-fit solution. Multi-resolution strategies for gradient-field
optimization start with coarse image resolution (fast evaluation) and progressively refine to full
resolution, helping avoid local minima in the cost landscape.

Cost Function Structure and Curvature Disambiguation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The L2 annotation cost function measures the mean squared pixel distance between predicted
and observed horizon points. However, because a planetary limb arc and its mirror image
(with the curvature flipped) can produce nearly identical pixel residuals, the optimizer
can sometimes converge to a physically wrong solution: a predicted arc that curves in the
opposite direction from the true limb.

Planet Ruler resolves this ambiguity with an additive *concavity penalty* that fires when
the predicted arc curves the wrong way. The penalty evaluates the predicted arc at the
annotated x-coordinates, draws a chord between the leftmost and rightmost predictions,
and checks whether the interior of the arc lies above or below that chord:

* **∩-shaped arc** (interior above the chord): the limb peaks toward the top of the image —
  the correct shape for standard high-altitude photography (e.g. an airplane window).
  Penalty = 0.
* **∪-shaped arc** (interior below the chord): the limb sags toward the bottom of the image
  — the wrong-curvature mirror solution. Penalty fires, adding a large cost that steers
  differential evolution out of that basin.

The penalty is enabled by default. It can be disabled with ``concavity_penalty=False`` in
:meth:`~planet_ruler.observation.LimbObservation.fit_arc` if needed (for example, when
the viewing geometry is inverted and the ∪ shape is genuinely correct).

Uncertainty Quantification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Measurement uncertainty arises from pixel discretization, camera parameter uncertainty, and 
atmospheric effects. Bootstrap resampling provides empirical confidence intervals: we repeatedly 
resample the observed horizon points with replacement, refit the model to each bootstrap sample, 
and examine the distribution of fitted parameters. The standard deviation of this distribution 
estimates parameter uncertainty.

Alternatively, the Hessian matrix (second derivatives of the cost function at the solution) provides 
an approximate covariance matrix for the parameters, yielding uncertainty estimates from the optimization 
geometry itself. Both methods typically agree within a factor of two and provide order-of-magnitude 
uncertainty characterization.