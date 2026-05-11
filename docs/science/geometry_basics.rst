For photographic measurement, Earth can be approximated as a sphere of radius :math:`R \approx 6371` km. 
While Earth is actually an oblate spheroid (equatorial radius ~21 km larger than polar radius), the spherical 
approximation introduces errors of only ~0.3%, typically smaller than measurement uncertainty.

The Horizon Circle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When an observer is elevated at altitude :math:`h` above the surface, the visible horizon forms a circle 
where lines of sight become tangent to the spherical surface. This horizon circle has a specific radius 
and angular extent as seen from the observer's position.

The horizon distance (surface distance from the point directly below the observer to the horizon) is:

.. math::

   d = \sqrt{2Rh + h^2} \approx \sqrt{2Rh}

For typical aircraft altitudes (10-20 km), this simplifies to :math:`d \approx \sqrt{2Rh}` 
since :math:`h \ll R`. At :math:`h = 10` km, the horizon is approximately 357 km away.

The Limb Arc in Images
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When photographed, the horizon circle projects onto the image as a curved arc—the "limb arc." The 
angular radius of this arc (measured from the camera's viewing direction) depends on both altitude and camera orientation.

For a camera pointing at angle :math:`\theta_p` below the horizontal (pitch angle), the limb arc appears as 
an elliptical segment in the image. The geometry is characterized by:

.. math::

   \alpha = \arcsin\left(\frac{R}{R+h}\right)

where :math:`\alpha` is the angular radius of the horizon circle as seen from 
altitude :math:`h`. For :math:`h = 10` km, :math:`\alpha \approx 2.0°`. At :math:`h = 100` km, :math:`\alpha \approx 6.4°`.

Measuring the Curvature
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The measurable quantity in an image is the vertical extent (sagitta) of the curved horizon relative to a 
straight line. For a horizon arc spanning width :math:`w` pixels with curvature radius :math:`r` pixels 
(in image coordinates), the sagitta :math:`s` is:

.. math::

   s \approx \frac{w^2}{8r}

This sagitta scales as :math:`s \propto \sqrt{h}` for a fixed field of view—doubling the altitude 
increases the sagitta by only :math:`\sqrt{2} \approx 1.4\times`. This square-root scaling means 
that high altitude provides only modest improvements in measurement precision compared to moderate altitude.

The key insight is that the limb arc's shape encodes the observer's altitude :math:`h` through the
relationship between :math:`R`, :math:`h`, and the observed angular geometry. By measuring the arc's
curvature in an image with known camera parameters, we can infer :math:`h` (if :math:`R` is known)
or :math:`R` (if :math:`h` is known).

Parameter Identifiability and the R–h Degeneracy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The arc curvature constrains the ratio :math:`R/(R+h)`, not :math:`R` and :math:`h` independently.
Any pair :math:`(R', h')` satisfying the same ratio produces an identical limb arc in the image.
Freeing both parameters simultaneously therefore creates a degenerate ridge in the optimization
landscape: the optimizer can trade off radius against altitude without changing the predicted arc
shape. This is why altitude must be supplied from an external source—GPS, barometric sensor, or
flight telemetry—rather than inferred from the image alongside radius.

When altitude :math:`h` is known, the sensitivity of the inferred radius to altitude uncertainty
follows directly from differentiating :math:`R = \rho h\,/\,(1 - \rho)` where
:math:`\rho = R/(R+h)`:

.. math::

   \frac{dR}{dh} = \frac{R}{h}

The amplification factor :math:`R/h` is large at aircraft altitudes. At :math:`h = 10` km, a 1 km
error in altitude propagates to a ~637 km (~10%) error in the inferred radius. However, civilian GPS
altitude is accurate to roughly 10–30 m, which propagates to only 6–19 km (~0.1–0.3%)—well within
useful measurement bounds.

Three independent sources contribute to overall radius uncertainty:

- **Altitude** (:math:`\sigma_h`): :math:`\sigma_R \approx (R/h)\,\sigma_h`. Reduced by better GPS
  or barometric sensing. Typically dominates with good annotations.

- **Annotation noise** (:math:`\sigma_\alpha`): uncertainty in the fitted arc position, driven by
  the number and precision of horizon points. Reduced by denser or automated annotation.

- **Camera parameters** (:math:`\sigma_{f/w}`): errors in focal length or sensor width shift the
  pixel-to-angle mapping, biasing the inferred curvature. Reduced by camera calibration;
  usually 1–2% for smartphones with accurate EXIF metadata.