This tutorial shows how to calculate Earth's radius using a horizon photograph with known altitude and camera parameters via the default interactive manual annotation method.

Prerequisites
~~~~~~~~~~~~

* Python 3.8+ with Planet Ruler installed (no additional dependencies needed)
* A horizon photograph (we'll use the demo Earth image)
* Basic knowledge of the observation altitude

.. note::
   **Manual vs Automatic Methods**: Planet Ruler's default **manual annotation** provides precise, user-controlled horizon detection with no additional dependencies. For automated processing, **gradient-field** and **AI segmentation** (requires PyTorch + Segment Anything) detection are also available.

Step 1: Setup and Imports
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import planet_ruler.observation as obs
   import planet_ruler.geometry as geom
   from planet_ruler.fit import calculate_parameter_uncertainty, format_parameter_result
   import matplotlib.pyplot as plt

Step 2: Load Configuration and Image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Planet Ruler uses YAML configuration files to specify camera parameters and initial estimates:

.. code-block:: python

   # Load Earth ISS observation
   observation = obs.LimbObservation(
       image_filepath="demo/images/ISS_Earth_horizon.jpg",
       fit_config="config/earth_iss_1.yaml"
   )
   
   # Display the loaded image
   observation.plot()

The configuration file contains:

* **Camera specifications**: focal length, detector width, field of view
* **Initial parameter estimates**: planet radius, observation altitude  
* **Optimization settings**: free parameters, parameter bounds

Step 3: Detect the Horizon (Default: Manual Annotation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Planet Ruler offers three detection methods, with manual annotation as the default:

.. code-block:: python

   # Method 1: Manual annotation (default - precise, no dependencies)
   observation.detect_limb(detection_method="manual")  # Opens interactive GUI
   
   # Method 2: Gradient-field (automated - good for clear horizons)
   # Here you get to skip detection! This method fits the planet radius to the image
   # directly so no need to identify the limb.
   
   # Method 3: AI segmentation (automated - requires PyTorch)
   # observation.detect_limb(detection_method="segmentation")
   # observation.smooth_limb(
   #    method="rolling-median",
   #    window_length=15,
   #    fill_nan=True
   # )
   
   # Plot the detected limb
   observation.plot()

.. tip::
   **Method Comparison**:
   
   * **Manual**: Precise user control, works with any image quality, no dependencies
   * **Gradient-field**: Automated, fast, works with clear horizons, no ML dependencies  
   * **Segmentation**: Most versatile, handles complex images, requires PyTorch + Segment Anything

Step 4: Fit Planetary Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now we optimize the planetary radius to match the observed horizon curvature:

.. code-block:: python

   # Perform the fit
   observation.fit_limb(
       minimizer="differential-evolution",
       seed=42  # For reproducible results
   )
   
   print("Fit completed successfully!")
   print(f"Fitted parameters: {observation.best_parameters}")

**Monitoring Progress with Dashboard:**

For long optimizations, enable the live progress dashboard:

.. code-block:: python

   # Enable dashboard for real-time monitoring
   observation.fit_limb(
       method="differential-evolution",
       dashboard=True  # Shows live progress
   )
   
   # Configure dashboard display
   observation.fit_limb(
       dashboard=True,
       dashboard_kwargs={
           'width': 80,         # Wider display
           'max_warnings': 5,   # More warning slots
           'max_hints': 4,      # More hint slots
       }
   )

The dashboard shows:
- Current parameter estimates
- Loss reduction progress
- Convergence status
- Warnings and optimization hints
- Adaptive refresh rate (fast during descent, slow at convergence)

Step 5: Calculate Uncertainty
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Planet Ruler provides multiple uncertainty estimation methods:

.. code-block:: python

   from planet_ruler.uncertainty import calculate_parameter_uncertainty
   
   # Auto-select best method (population for DE, hessian for others)
   radius_result = calculate_parameter_uncertainty(
       observation,
       parameter="r",
       scale_factor=1000,  # Convert to kilometers
       method='auto',
       confidence_level=0.68  # 1-sigma
   )
   
   print(f"Radius: {radius_result['uncertainty']:.1f} km")
   print(f"Method used: {radius_result['method']}")
   
   # Alternative methods:
   
   # 1. Population spread (fast, exact for differential-evolution)
   pop_result = calculate_parameter_uncertainty(
       observation, "r",
       scale_factor=1000,
       method='population',
       confidence_level=0.68
   )
   
   # 2. Hessian approximation (fast, works with all minimizers)
   hess_result = calculate_parameter_uncertainty(
       observation, "r",
       scale_factor=1000,
       method='hessian',
       confidence_level=0.68
   )
   
   # 3. Profile likelihood (slow, most accurate)
   profile_result = calculate_parameter_uncertainty(
       observation, "r",
       scale_factor=1000,
       method='profile',
       confidence_level=0.68,
       n_points=20
   )
   
   # Get confidence intervals at different levels
   for cl in [0.68, 0.95, 0.99]:
       result = calculate_parameter_uncertainty(
           observation, "r",
           scale_factor=1000,
           method='auto',
           confidence_level=cl
       )
       print(f"{int(cl*100)}% CI: ± {result['uncertainty']:.1f} km")

Step 6: Validate Results
~~~~~~~~~~~~~~~~~~~~~~~

Compare your results with the known Earth radius:

.. code-block:: python

   known_earth_radius = 6371.0  # km
   fitted_radius = radius_result["value"]
   uncertainty = radius_result["uncertainty"]
   
   error = abs(fitted_radius - known_earth_radius)
   error_in_sigma = error / uncertainty
   
   print(f"Known Earth radius: {known_earth_radius} km")
   print(f"Fitted radius: {fitted_radius:.1f} ± {uncertainty:.1f} km")
   print(f"Absolute error: {error:.1f} km")
   print(f"Error in standard deviations: {error_in_sigma:.1f}σ")
   
   if error_in_sigma < 2.0:
       print("✓ Result is within 2σ of known value!")
   else:
       print("⚠ Result differs significantly from known value")

**Expected Results**: For Earth from ISS altitude (~418 km) using manual annotation:
* Fitted radius: ~6,371 ± 15 km (high precision from careful point selection)
* Error: < 50 km from true radius