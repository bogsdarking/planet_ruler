Now for the hard part. We need to deduce the radius of the object below us by leveraging what we know and fitting what we do not.

So -- what do we know?

Let's start with a list of the free parameters in the fit. These are:
- **[r]** Radius of the planet (m)
- **[h]** Height of the camera above the surface (m)
- **[f]** Focal length of the camera (m)
- **[pxy]** The scale of pixel width/height on the projection plane (m)
- **[px]** The relative scale of x pixels (vs. y)
- **[x0]** The x-axis principle point (center of the image in pixel space -- really should be close to 0)
- **[y0]** Same as x0 but for the y-axis
- **[theta_x]** Rotation around the x (horizontal) axis, AKA pitch. (radians)
- **[theta_y]** Rotation around the y (toward the limb) axis, AKA roll. (radians)
- **[theta_z]** Rotation around the z (vertical) axis, AKA yaw. (radians)
- **[origin_x]** Horizontal offset from the object in question to the camera (m)
- **[origin_y]** Distance from the object in question to the camera (m)
- **[origin_z]** Height difference from the object in question to the camera (m)

To help the fit we can give initial guesses and boundaries to each of these features. This is a tough optimization with a lot of parameter space, degeneracies, and weird inflection points, so the more help we can give the more likely we are to get somewhere meaningful. Let's step through the parameters.

- **[r]** Radius of the planet (m)
  
> Obviously we are trying to find out **r** so we might not have much to go on here, but that doesn't mean we can't put in some reasonable bounds. We know for example that we are looking at a rocky dwarf planet that is quite spherical which is a clue to the minimum radius. I leave it to the user to do that napkin math -- today let's just put in a guess for the radius at 750 km -- in the ballpark of the truth (1188 km) but not so close to give away the answer. We set the bounds loosely to 600-1600 km.

- **[h]** Height of the camera above the surface (m)

> Here we can make a guess using the known parameters of NASA's New Horizons mission. The point of closest approach to Pluto was 12,500 km, so we'll start there, again plus or minus a few thousand km as boundaries.

- **[f]** Focal length of the camera (m)

> From the stated mission parameters, the [RALPH](https://www.boulder.swri.edu/pkb/ssr/ssr-ralph.pdf) camera has a 75mm focal length. Since we're pretty sure about this one, let's give it a very small (0.1mm) tolerance.

- **[pxy]** The scale of pixel width/height on the projection plane (m)

> This one is a bit of a mystery to me what it should be. After running the optimization a few times I was able to give it a rough set of limits to where it seems to settle -- currently between 1 and 300.

- **[px]** The relative scale of x pixels (vs. y)

> This might be in the description of the CCD for the RALPH camera somewhere but I just set it widely to between 0.1 and 2.0.

- **[x0]** The x-axis principle point (center of the image in pixel space -- really should be close to 0)
- **[y0]** Same as x0 but for the y-axis

> I believe these are basically tolerance parameters for the camera as it's hard to imagine someone designing a CCD plane on purpose with an off-center focal point. Set to a very small tolerance (-5e-4 to 5e-4) as they actually have a tremendous effect on the image if they are far out of alignment.

- **[theta_x]** Rotation around the x (horizontal) axis, AKA pitch. (radians)
- **[theta_y]** Rotation around the y (toward the limb) axis, AKA roll. (radians)
- **[theta_z]** Rotation around the z (vertical) axis, AKA yaw. (radians)

> These describe where the camera is pointing. 'Straight ahead' depends a bit on how you define your coordinates, but we set these in that range with ranges of around 20 degrees either way.

- **[origin_x]** Horizontal offset from the object in question to the camera (m)
- **[origin_y]** Distance from the object in question to the camera (m)
- **[origin_z]** Height difference from the object in question to the camera (m)

> These tell us where the camera is in space relative to the target. For a far-off limb, you can set origin_y to something like h. Otherwise you can use geometry.horizon_distance to calculate it based on your h and estimate of r (that's actually what was done here). The other two origins are a bit of a toss-up. I set the limits to something like 1000 km either direction and then refined a little after seeing some fit posteriors.

You can see the current initial parameter set [here](https://github.com/bogsdarking/planet_ruler/blob/c8c0a39cae7712363491bc60c861d1a2e410b745/config/pluto-new-horizons.yaml).

Let's try that fit!