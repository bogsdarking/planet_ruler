# PLANET RULER

A tool to infer the radius of the planet you are sitting on, just by taking a picture of the horizon! At least that's the idea...

# How?

Planets are round. This might not be obvious from the ground, but it becomes more so the higher you go. This is because the planet's horizon, or 'limb', 
is getting further away from you. As it does, its apparent curvature must increase, transforming from a nearly straight line to a curve, a circle, and eventually a point.
Imagine looking at a basketball. If your eye is almost touching it, looking across the surface, you see an arc. Sit across the room from it and it becomes a circle. After
a hundred meters or so it becomes a dot.

If you have some idea how high you are and a picture of the horizon, you should be able to use the apparent curvature to infer the size (or radius) of the object
in question.

# Why?

No real reason besides it is fun. One potential application would be to reverse the whole process and guess your altitude if you knew the dimensions of the planet you
are looking at, but we generally have better tools for that available.

# Does it work?

Right now, sort of. The hardest part of this problem is that cameras are complicated. Exactly what happens between the horizon you are looking at and the flattened image 
we save as pixels _really_ matters to what result we get. The current solution I've been using is to look for a best fit for all the camera parameters at the same time 
I look for the planet dimensions. Unfortunately this is a fairly degenerate system and is therefore quite sensitive to the priors you put on said parameters (in English, 
the results are not stable). But hey, it's still a fun idea.

Check out [this notebook](https://github.com/bogsdarking/planet-ruler/blob/39b4bf0ae97b40e8936090acdab3ddfc523a98ac/notebooks/limb_demo-pluto-lite.ipynb) for a tour.

# Where can I learn more?

Here are a few references I used:

- [Horizon Wiki](https://en.wikipedia.org/wiki/Horizon)
- [earth science stack exchange thread](https://earthscience.stackexchange.com/questions/7283/how-high-must-one-be-for-the-curvature-of-the-earth-to-be-visible-to-the-eye)
- [debunking flat earth page (claims this can't work)](https://flatearth.ws/standing-on-a-beach)
- [some camera basics](https://www.cambridgeincolour.com/tutorials/image-projections.htm)
- [camera resectioning](https://en.wikipedia.org/wiki/Camera_resectioning)
- [more camera](https://courses.cs.washington.edu/courses/cse455/09wi/Lects/lect5.pdf)
- [intrisic matrix](https://ksimek.github.io/2013/08/13/intrinsic/)
- [camera calibration lecture](https://people.cs.rutgers.edu/~elgammal/classes/cs534/lectures/Calibration.pdf)
- [example camera specs](https://www.devicespecifications.com/en/model/36ea45ae)
