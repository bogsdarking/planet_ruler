Planet Ruler Documentation
=========================

**A tool to infer the radius of the planet you are sitting on, just by taking a picture of the horizon!**

.. image:: ../demo/images/beach-ocean-panorama-187916.jpg
   :alt: Beach horizon panorama
   :class: with-border
   :width: 600

Overview
--------

Planet Ruler uses horizon curvature analysis to determine planetary radius from photographs. The method leverages the relationship between observation altitude, apparent horizon curvature, and planetary geometry to extract dimensional information.

The tool combines:

* **Geometric Analysis**: Mathematical relationships between horizon distance, limb angles, and planetary curvature
* **Computer Vision**: Automated horizon detection and image processing techniques  
* **Camera Modeling**: Intrinsic and extrinsic camera parameter estimation
* **Optimization**: Cost function minimization for parameter fitting
* **Uncertainty Quantification**: Statistical analysis of parameter confidence intervals

Key Features
------------

* **Multi-planetary Support**: Earth, Pluto, Saturn scenarios with spacecraft camera specifications
* **Advanced Image Processing**: Gradient-based horizon detection, AI-powered segmentation, and manual annotation
* **Interactive Annotation**: Manual limb point selection with intuitive GUI interface
* **Robust Optimization**: Differential evolution and cost function analysis
* **Comprehensive Testing**: 200+ tests including integration tests with real mission data
* **Performance Benchmarking**: Optimized computational workflows
* **Uncertainty Analysis**: Statistical confidence intervals and parameter distributions

How It Works
------------

Planets are round, and this becomes apparent at higher altitudes. The horizon's curvature increases with observation height, transforming from a nearly straight line to a visible arc. By analyzing this curvature in photographs along with camera parameters and altitude estimates, we can infer the planetary radius.

The mathematical foundation relies on:

.. math::

   d_h = \sqrt{2rh + h^2}

where :math:`d_h` is horizon distance, :math:`r` is planetary radius, and :math:`h` is observation altitude.

Quick Start
-----------

Here's how to determine planetary radius from a horizon photograph with uncertainty estimates:

.. code-block:: python

   import planet_ruler.observation as obs
   from planet_ruler.fit import calculate_parameter_uncertainty, format_parameter_result
   
   # Load image and configuration with initial parameter estimates
   observation = obs.LimbObservation(
       image_filepath="horizon_image.jpg",
       fit_config="config/earth_iss_1.yaml"  # Contains camera specs and altitude
   )
   
   # Detect horizon in the image using AI segmentation (recommended)
   observation.detect_limb(method="segmentation")
   observation.smooth_limb(method="rolling-median", window_length=15)
   
   # Alternative detection methods:
   # observation.detect_limb(method="gradient-break")  # Legacy method
   # observation.detect_limb(method="manual")          # Interactive GUI annotation
   
   # Fit model to determine planetary radius
   observation.fit_limb()
   
   # Calculate radius with uncertainty
   radius_result = calculate_parameter_uncertainty(
       observation, 
       parameter="r", 
       scale_factor=1000,  # Convert to km
       uncertainty_type="std"
   )
   
   # Display results
   print(format_parameter_result(radius_result, "km"))
   # Output: r = 6371.2 ±15.8 km
   
   # For detailed uncertainty analysis
   print(f"Fitted radius: {radius_result['value']:.1f} km")
   print(f"Uncertainty (1σ): ±{radius_result['uncertainty']:.1f} km")
   print(f"Method: {radius_result['method']}")
   
   # Compare with known Earth radius
   known_earth_radius = 6371.0
   error = abs(radius_result['value'] - known_earth_radius)
   print(f"Error from true Earth radius: {error:.1f} km")

For Earth from ISS altitude (~400 km), you should get a radius close to 6371 km with uncertainties typically under ±50 km depending on image quality and camera specifications.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Documentation:

   installation
   tutorials
   examples
   api
   modules

.. toctree::
   :maxdepth: 1
   :caption: Development:

   testing
   benchmarks
   contributing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex` 
* :ref:`search`