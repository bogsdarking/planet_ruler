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