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
  
> Obviously we are trying to find out **r** so we might not have much to go on here, but that doesn't mean we can't put in some reasonable bounds. Today let's just put in a guess for the radius at 8,000 km -- in the ballpark of the truth (6,371 km) but not so close to give away the answer. We set the bounds loosely to 1,000-20,000 km.

- **[h]** Height of the camera above the surface (m)

> The approximate altitude of the ISS is 418 km. We allow for some uncertainty ranging from 400 - 450 km.

- **[f]** Focal length of the camera (m)
- **[fov]** The angle observed by the camera (degrees)
- **[w]** The width of the detector (m)

> These three are grouped together because they are mathematically tied -- see [angle of view](https://en.wikipedia.org/wiki/Angle_of_view_(photography)). By specifying any two, the third is calculable. For that reason we will restrict ourselves to specifying two at a time (to avoid nonsensical combinations). Any two are sufficient, so here we will let f and w be our free parameters since their initial values were readily available: see pages on the [image](https://www.flickr.com/photos/nasa2explore/50644513538/) and [camera](https://en.nikon.ca/p/d4/25482/overview#tech-specs). We allow a bit of give -- 3mm and 1mm, respectively.

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

You can see the current initial parameter set [here](https://github.com/bogsdarking/planet_ruler/blob/b908bd94601ba4f4cb4b3e9453fcd1a503042364/config/saturn-cassini-2.yaml).

Let's try that fit!