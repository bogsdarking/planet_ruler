Now for the hard part. We need to deduce the radius of the object below us by leveraging what we know and fitting what we do not.

So -- what do we know?

Let's start with a list of the free parameters in the fit. These are:
- **[r]** Radius of the planet (m)
- **[h]** Height of the camera above the surface (m)
- **[f]** Focal length of the camera (m)
- **[fov]** The angle observed by the camera (degrees)
- **[w]** The width of the detector (m)
- **[x0]** The x-axis principle point (center of the image in pixel space)
- **[y0]** Same as x0 but for the y-axis
- **[theta_x]** Rotation around the x (horizontal) axis, AKA pitch. (radians)
- **[theta_y]** Rotation around the y (vertical) axis, AKA yaw. (radians)
- **[theta_z]** Rotation around the z (toward the limb) axis, AKA roll. (radians)
- **[origin_x]** Horizontal offset from the coordinate origin to the camera (m)
- **[origin_y]** Distance from the coordinate origin to the camera (m)
- **[origin_z]** Height difference from the coordinate origin to the camera (m)

To help the fit we can give initial guesses and boundaries to each of these features. This is a tough optimization with a lot of parameter space, degeneracies, and weird inflection points, so the more help we can give the more likely we are to get somewhere meaningful. Let's step through the parameters.

- **[r]** Radius of the planet (m)
  
> Obviously we are trying to find out **r** so we might not have much to go on here, but that doesn't mean we can't put in some reasonable bounds. We know for example that we are looking at a rocky dwarf planet that is quite spherical which is a clue to the minimum radius. I leave it to the user to do that napkin math -- today let's just put in a guess for the radius at 750 km -- in the ballpark of the truth (1188 km) but not so close to give away the answer. We set the bounds loosely to 600-1900 km.

- **[h]** Height of the camera above the surface (m)

> Here we can make a guess using the known parameters of NASA's New Horizons mission. The NASA image data claims a distance of 18,000 km so we'll start there, plus or minus a few thousand km as boundaries.

- **[f]** Focal length of the camera (m)
- **[fov]** The angle observed by the camera (degrees)
- **[w]** The width of the detector (m)

> These three are grouped together because they are mathematically tied -- see [angle of view](https://en.wikipedia.org/wiki/Angle_of_view_(photography)). By specifying any two, the third is calculable. For that reason we will restrict ourselves to specifying two at a time (to avoid nonsensical combinations). Any two are sufficient, so here we will let f and fov be our free parameters since their initial values were readily available: see the [mission parameters](https://www.dpreview.com/articles/5293237047/nasa-new-horizons-probe-cameras-ralph). We set the focal length to 0.6575 m, and since we're pretty sure about this one, let's give it a very small (0.5cm) tolerance. Field of view will be 5.7 degrees with a 1% uncertainty.

- **[x0]** The x-axis principle point (center of the image in pixel space)

> This one will be set automatically and fixed, assuming the camera has a CCD centered on the optical axis.

- **[y0]** Same as x0 but for the y-axis

> Same as y0.

- **[theta_x]** Rotation around the x (horizontal) axis, AKA pitch. (radians)
- **[theta_y]** Rotation around the y (vertical) axis, AKA yaw. (radians)
- **[theta_z]** Rotation around the z (toward the limb) axis, AKA roll. (radians)

> These describe where the camera is pointing. We start it in the rough direction the limb should be given our assumptions so far about the height and radius, but don't give any real constraints -- i.e., they are loose enough that it can point wherever it wants.

- **[origin_x]** Horizontal offset from the object in question to the camera (m)
- **[origin_y]** Distance from the object in question to the camera (m)
- **[origin_z]** Height difference from the object in question to the camera (m)

> These tell us where the camera is in space relative to the coordinate origin. But since we decide where the origin is, we can just set it to equal the camera position. Now these are all zero by definition -- they can be manipulated for demonstration but don't change these if you want to get the correct answer.

You can see the current initial parameter set [here](https://github.com/bogsdarking/planet_ruler/blob/c8c0a39cae7712363491bc60c861d1a2e410b745/config/pluto-new-horizons.yaml).

Let's try that fit!