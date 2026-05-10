The gradient-field method optimizes model parameters by aligning predicted limb position with 
image gradients across the entire image, eliminating the need for explicit horizon detection. 
This approach uses all available gradient information rather than reducing it to a 1D detected curve.

Mathematical Foundation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Image gradients :math:`\nabla I(x,y) = (\partial I/\partial x, \partial I/\partial y)` point perpendicular 
to edges—at horizon pixels, gradients point from sky (dark) toward ground (bright), or vice versa depending 
on scene lighting. The gradient magnitude :math:`|\nabla I|` indicates edge strength, while the normalized 
direction :math:`\hat{g} = \nabla I / |\nabla I|` specifies the orientation.

For a given set of model parameters, we predict where the horizon should appear in the image, generating 
a model limb curve :math:`\mathbf{r}_{\text{model}}(t)` in image coordinates. At each point along this 
curve, we compute the inward-pointing normal vector :math:`\hat{n}_{\text{model}}`.

The cost function measures misalignment between image gradients and model normals:

.. math::

   C = -\sum_{\text{pixels}} |\nabla I| \cdot (\hat{g} \cdot \hat{n}_{\text{model}})^+

where :math:`(\cdot)^+` denotes positive part (max(0, ·)). This flux-based cost automatically weights pixels 
by gradient strength—strong edges contribute more to the cost than weak edges, providing natural robustness 
to noise in uniform regions.

Directional blur enhancement applies Gaussian smoothing along the predicted limb tangent direction while 
preserving sharpness perpendicular to it. This distinguishes atmospheric gradients (which blur in all directions) 
from the true planetary edge (which remains sharp perpendicular to the limb), improving detection accuracy in hazy conditions.

Multi-Resolution Strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Multi-resolution optimization solves the cost landscape's rugged structure. We begin at coarse resolution 
(image downsampled 8× or 16×), quickly finding the approximate solution. This coarse solution initializes 
optimization at medium resolution (4× downsample), which refines the parameters. Finally, full-resolution 
optimization achieves pixel-level precision.

This coarse-to-fine strategy has two benefits: (1) coarse resolution blurs away local minima, making global 
optimization more reliable, and (2) cost function evaluation is ~64× faster at 8× downsampling, accelerating 
the expensive initial search phase.

Advantages and Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The gradient-field method requires no detection preprocessing—it works directly with raw image gradients. 
By using all pixels rather than a sparse detected curve, it potentially extracts more information from the 
image. The flux-based weighting naturally emphasizes strong edges without requiring manual threshold selection.

However, computational cost is substantial (10-30 seconds vs 1-5 seconds for detection-based methods) because 
the cost function must evaluate the predicted limb across many image pixels for each parameter set during 
optimization. The method also struggles with very low contrast scenes where gradient magnitude is weak everywhere, 
as it has no signal to align with.

The method performs best when the horizon is the dominant edge in the image. Complex scenes with strong 
non-horizon edges (terrain features, cloud boundaries) can confuse the optimization, though directional 
blur helps mitigate this issue.

Theoretical Connection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The gradient-field approach relates to active contour methods in computer vision, where a curve evolves 
to align with image features. It also connects to variational formulations of edge detection, which find 
curves maximizing a functional involving image gradients. The key difference is that our model curve has 
geometric constraints (it must be a valid horizon projection given camera parameters), whereas active 
contours have no such constraint.

This geometric constraint is both a strength and a limitation. It prevents the solution from collapsing 
to arbitrary edges in the image (a common problem for unconstrained active contours) but requires that 
the true horizon actually match the assumed spherical Earth geometry. Deviations from this geometry 
(atmospheric refraction, terrain roughness) introduce systematic errors.