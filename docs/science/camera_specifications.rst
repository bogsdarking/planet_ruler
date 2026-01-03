Minimum Detectable Altitude
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A key scientific question is: what is the minimum altitude at which planetary curvature 
becomes measurable? The answer is surprisingly low—modern cameras can detect Earth's curvature 
from as little as 1-34 meters above the surface, depending on focal length and sensor resolution.

This calculation considers only geometric visibility: at what altitude does the horizon's vertical 
extent (sagitta) exceed one pixel? For a camera with :math:`N_y` vertical pixels, focal 
length :math:`f`, and pixel size :math:`p`, the minimum detectable altitude :math:`h_{\text{min}}` occurs when 
the difference between the minimum and maximum y-coordinates of the horizon within the image equals one pixel.

Through numerical calculation of the limb arc geometry, we find:

**Smartphone main cameras** (f=26mm equiv, 4000px): :math:`h_{\text{min}} \approx 1\text{-}2` m

**Ultrawide cameras** (f=13mm equiv, 4000px): :math:`h_{\text{min}} \approx 1` m  

**Telephoto 3×** (f=77mm equiv, 4000px): :math:`h_{\text{min}} \approx 2\text{-}5` m

**Telephoto 5×** (f=130mm equiv, 4000px): :math:`h_{\text{min}} \approx 4\text{-}5` m

**Telephoto 10×** (f=240mm equiv, 4000px): :math:`h_{\text{min}} \approx 34` m

These values assume perfect optical quality and measurement precision. Real limitations arise 
from atmospheric refraction, optical aberrations, scene complexity, and measurement noise—not geometric constraints.

The key insight: Earth's curvature is geometrically detectable from ground level with modern cameras. 
The "sweet spot" of 50,000-120,000 feet for horizon photography exists because curvature becomes 
*obvious* and *easily measurable* at these altitudes, not because it's invisible below them.

Camera Parameters for Measurement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Three camera parameters determine the limb arc's appearance in an image: sensor dimensions, 
focal length, and pixel resolution. The sensor converts incoming light into a digital image 
through an array of photosensitive elements. Typical sensors range from 1/2.55" (~5.6×4.2 mm) 
in smartphones to full-frame (36×24 mm) in professional cameras. The pixel count determines measurement 
resolution—modern cameras provide 3000-8000 pixels along the sensor's long axis.

The lens focal length controls the field of view—shorter focal lengths (wide-angle lenses) 
capture more of the horizon circle but with lower angular resolution, while longer focal lengths 
(telephoto) magnify a smaller region. For horizon measurement, the critical quantity is the 
angular size of one pixel, given by :math:`\theta_{\text{pixel}} = \arctan(p/f)` where :math:`p` is 
pixel size and :math:`f` is focal length. Smaller pixel angles enable finer curvature discrimination.

The camera's orientation in 3D space—particularly the pitch angle relative to horizontal—determines which 
portion of the horizon circle appears in the image. A camera pointed slightly downward captures 
more of the curved arc, while one pointed at the horizon shows primarily the tangent point with minimal 
visible curvature. This orientation must be determined during analysis as part of the measurement problem.

Signal Scaling with Altitude
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The measurable signal (sagitta) scales as :math:`s \propto \sqrt{h}`, meaning curvature grows 
slowly with altitude. Doubling altitude increases the signal by only 41%. This square-root scaling 
has important implications: measurements at 10 km altitude are not vastly superior to those at 5 km, 
and measurements at 100 km are only :math:`\sqrt{10} \approx 3.2\times` stronger than at 10 km.

For a fixed camera, the ratio of signal (sagitta in pixels) to measurement uncertainty (typically ~1 pixel) 
determines measurement quality. While higher altitude always improves this ratio, the improvement is gradual 
rather than dramatic. This explains why the method remains viable across a wide altitude range rather than 
requiring specific threshold altitudes.