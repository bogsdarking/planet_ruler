Tutorials
=========

This section provides step-by-step tutorials for using Planet Ruler to determine planetary radius from horizon photographs using the default interactive manual annotation approach.

Tutorial 1: Basic Earth Radius Calculation
------------------------------------------

This tutorial shows how to calculate Earth's radius using a horizon photograph from a known altitude with the default interactive manual annotation method.

Prerequisites
~~~~~~~~~~~~

* Python 3.8+ with Planet Ruler installed (no additional dependencies needed)
* A horizon photograph (we'll use the demo Earth image)
* Basic knowledge of the observation altitude

.. note::
   **Manual vs Automatic Methods**: Planet Ruler's default **manual annotation** provides precise, user-controlled horizon detection with no additional dependencies. For automated processing, **AI segmentation** (requires PyTorch + Segment Anything) and **gradient-break** detection are also available.

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
   observation.plot(show=True)

The configuration file contains:

* **Camera specifications**: focal length, detector width, field of view
* **Initial parameter estimates**: planet radius, observation altitude  
* **Optimization settings**: free parameters, parameter bounds

Step 3: Detect the Horizon (Default: Manual Annotation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Planet Ruler uses interactive manual annotation for precise horizon detection:

.. code-block:: python

   # Detect horizon using manual annotation (default, no dependencies)
   observation.detect_limb(method="manual")  # Opens interactive GUI
   
   # Smooth the detected limb
   observation.smooth_limb(
       method="rolling-median",
       window_length=15,
       fill_nan=True
   )
   
   # Plot the detected limb
   observation.plot(gradient=False, show=True)

.. tip::
   **Manual annotation advantages**:
   
   * Precise user control over horizon selection
   * No additional dependencies or model downloads required
   * Works immediately after Planet Ruler installation
   * Interactive GUI with zoom, stretch, and save/load functionality
   * Handles any image type or quality level

**Alternative Methods:**

.. code-block:: python

   # Option 1: AI segmentation (requires PyTorch + Segment Anything)
   try:
       observation.detect_limb(
           method="segmentation",
           segmenter="segment-anything"
       )
   except ImportError:
       print("⚠ Segment Anything not available - install with: pip install segment-anything torch")
   
   # Option 2: Legacy gradient-break detection
   observation.detect_limb(
       method="gradient-break",
       window_length=21,
       threshold=0.1
   )

Step 4: Fit Planetary Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now we optimize the planetary radius to match the observed horizon curvature:

.. code-block:: python

   # Perform the fit
   observation.fit_limb(
       method="differential_evolution",
       maxiter=1000,
       popsize=15,
       seed=42  # For reproducible results
   )
   
   print("Fit completed successfully!")
   print(f"Fitted parameters: {observation.best_parameters}")

Step 5: Calculate Uncertainty
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the uncertainty calculation functions:

.. code-block:: python

   # Calculate radius uncertainty
   radius_result = calculate_parameter_uncertainty(
       observation,
       parameter="r",
       scale_factor=1000,  # Convert to kilometers
       uncertainty_type="std"
   )
   
   # Display formatted results
   print(format_parameter_result(radius_result, "km"))
   
   # Get confidence interval
   ci_result = calculate_parameter_uncertainty(
       observation,
       parameter="r", 
       scale_factor=1000,
       uncertainty_type="ci"
   )
   
   print(f"95% Confidence Interval: {ci_result['uncertainty']['lower']:.1f} - {ci_result['uncertainty']['upper']:.1f} km")

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

Tutorial 2: Advanced Manual Annotation Techniques
-------------------------------------------------

Interactive GUI Features
~~~~~~~~~~~~~~~~~~~~~~~

The manual annotation interface provides several advanced features:

.. code-block:: python

   from planet_ruler.annotate import TkLimbAnnotator
   
   # Load image for manual annotation
   observation = obs.LimbObservation("complex_horizon_image.jpg", "config.yaml")
   
   # Manual annotation opens interactive GUI with these features:
   # - Left click: Add limb points
   # - Right click: Remove nearby points
   # - Mouse wheel: Zoom in/out
   # - Arrow keys: Adjust image stretch/contrast
   # - 'g': Generate target array from points
   # - 's': Save points to JSON file
   # - 'l': Load points from JSON file
   # - ESC or 'q': Close window
   
   observation.detect_limb(method="manual")

Working with Difficult Images
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For challenging images with clouds, terrain, or atmospheric effects:

.. code-block:: python

   # Use manual annotation with custom stretch for better visibility
   observation = obs.LimbObservation("difficult_image.jpg", "config.yaml")
   
   # The GUI allows real-time contrast adjustment:
   # - Up arrow: Increase stretch (brighter)
   # - Down arrow: Decrease stretch (darker)
   # - Use zoom to focus on specific horizon sections
   
   observation.detect_limb(method="manual")

Saving and Loading Annotation Sessions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Save your work during annotation:
   # 1. Click points along the horizon
   # 2. Press 's' to save points to JSON file
   # 3. Continue later by pressing 'l' to load saved points
   
   # You can also save/load programmatically:
   from planet_ruler.annotate import TkLimbAnnotator
   
   annotator = TkLimbAnnotator("image.jpg", initial_stretch=1.0)
   # ... add points in GUI ...
   annotator.save_points("my_horizon_points.json")
   
   # Later session:
   annotator.load_points("my_horizon_points.json")

Tutorial 3: Multi-planetary Analysis (Manual Annotation)
-------------------------------------------------------

Comparing Earth, Pluto, and Saturn with Precise Manual Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   
   # Scenarios to analyze
   scenarios = [
       ("Earth ISS", "config/earth_iss_1.yaml", "demo/images/earth_iss.jpg"),
       ("Pluto New Horizons", "config/pluto-new-horizons.yaml", "demo/images/pluto_nh.jpg"),
       ("Saturn Cassini", "config/saturn-cassini-1.yaml", "demo/images/saturn_cassini.jpg")
   ]
   
   results = []
   
   for name, config_path, image_path in scenarios:
       print(f"\nProcessing {name}...")
       
       # Load and process observation with manual annotation
       obs_obj = obs.LimbObservation(image_path, config_path)
       
       print(f"  Opening manual annotation GUI for {name}...")
       print("  Instructions:")
       print("    - Click along the horizon to mark limb points")
       print("    - Use mouse wheel to zoom, arrows for contrast")
       print("    - Press 'g' to generate target, 's' to save, 'q' to close")
       
       # Use manual annotation (default, precise)
       obs_obj.detect_limb(method="manual")
       method_used = "Manual Annotation"
       
       obs_obj.smooth_limb()
       obs_obj.fit_limb()
       
       # Calculate uncertainties
       radius_result = calculate_parameter_uncertainty(
           obs_obj, "r", scale_factor=1000, uncertainty_type="std"
       )
       
       results.append({
           "Scenario": name,
           "Method": method_used,
           "Radius (km)": f"{radius_result['value']:.0f} ± {radius_result['uncertainty']:.0f}",
           "Uncertainty (km)": f"{radius_result['uncertainty']:.1f}",
           "Quality": "High (User-controlled precision)"
       })
   
   # Display results table
   df = pd.DataFrame(results)
   print("\n" + "="*70)
   print("MULTI-PLANETARY ANALYSIS RESULTS")
   print("="*70)
   print(df.to_string(index=False))

Tutorial 4: Detection Method Comparison
--------------------------------------

Comparing Manual vs Automatic Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import time
   
   # Load test image
   observation = obs.LimbObservation("test_image.jpg", "config/earth_iss_1.yaml")
   
   methods_to_test = [
       ("manual", {}),  # Interactive GUI - time depends on user
       ("gradient-break", {"window_length": 21, "threshold": 0.1})
   ]
   
   # Optional: test segmentation if available
   try:
       from segment_anything import sam_model_registry
       methods_to_test.append(("segmentation", {"segmenter": "segment-anything"}))
   except ImportError:
       print("⚠ Segmentation not available - install with: pip install segment-anything torch")
   
   results = {}
   
   for method_name, kwargs in methods_to_test:
       print(f"\nTesting {method_name}...")
       
       # Fresh observation for each test
       test_obs = obs.LimbObservation("test_image.jpg", "config/earth_iss_1.yaml")
       
       if method_name == "manual":
           print("  Manual annotation - time depends on user interaction")
           print("  Opening GUI... Click points along horizon, press 'g' to generate, 'q' to close")
           # Time manual interaction
           start_time = time.time()
           test_obs.detect_limb(method=method_name, **kwargs)
           detection_time = time.time() - start_time
       else:
           # Time automatic methods
           start_time = time.time()
           try:
               test_obs.detect_limb(method=method_name, **kwargs)
               detection_time = time.time() - start_time
           except Exception as e:
               results[method_name] = {
                   "error": str(e),
                   "success": False
               }
               continue
       
       test_obs.smooth_limb()
       test_obs.fit_limb()
       
       # Calculate uncertainty
       radius_result = calculate_parameter_uncertainty(
           test_obs, "r", scale_factor=1000
       )
       
       results[method_name] = {
           "time": detection_time,
           "radius": radius_result["value"],
           "uncertainty": radius_result["uncertainty"],
           "success": True
       }
   
   # Compare results
   print("\n" + "="*50)
   print("METHOD COMPARISON")
   print("="*50)
   
   for method, result in results.items():
       if result["success"]:
           print(f"{method.upper()}:")
           if method == "manual":
               print(f"  Time: {result['time']:.1f} seconds (user-dependent)")
               print(f"  Precision: User-controlled (typically highest)")
           else:
               print(f"  Time: {result['time']:.1f} seconds (automatic)")
           print(f"  Radius: {result['radius']:.1f} ± {result['uncertainty']:.1f} km")
           print(f"  Relative uncertainty: {100*result['uncertainty']/result['radius']:.1f}%")
       else:
           print(f"{method.upper()}: FAILED - {result['error']}")

Installation and Setup
----------------------

Basic Installation
~~~~~~~~~~~~~~~~~

Planet Ruler works immediately after installation with no additional dependencies:

.. code-block:: bash

   # Essential: Install Planet Ruler (manual annotation works immediately)
   pip install planet-ruler

Verification Test
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Test basic Planet Ruler functionality
   import planet_ruler.observation as obs
   import planet_ruler.geometry as geom
   
   # Test geometry functions
   horizon_dist = geom.horizon_distance(r=6371000, h=400000)
   print(f"✓ Planet Ruler installed - ISS horizon distance: {horizon_dist/1000:.1f} km")
   
   # Test manual annotation interface
   try:
       from planet_ruler.annotate import TkLimbAnnotator
       print("✓ Manual annotation GUI available")
   except ImportError as e:
       print(f"⚠ GUI not available: {e}")

Optional: Advanced Detection Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For automatic detection methods, install additional dependencies:

.. code-block:: bash

   # Optional: AI segmentation support (requires PyTorch + Segment Anything)
   pip install segment-anything torch torchvision
   
   # Optional: GPU support for faster AI processing
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

Testing Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Test AI segmentation installation (optional)
   try:
       from planet_ruler.image import ImageSegmentation
       from segment_anything import sam_model_registry
       print("✓ AI segmentation available")
   except ImportError as e:
       print(f"⚠ AI segmentation not available: {e}")
       print("Install with: pip install segment-anything torch")

Troubleshooting
~~~~~~~~~~~~~~

**Common issues and solutions:**

1. **Manual annotation GUI not opening**
   
   .. code-block:: bash
   
      # Ensure tkinter is installed (usually included with Python)
      python -m tkinter  # Should open a test window

2. **"No module named 'segment_anything'"** (for AI segmentation only)
   
   .. code-block:: bash
   
      pip install segment-anything torch

3. **Performance tips for manual annotation**
   
   .. code-block:: python
   
      # For large images, consider downsampling for easier annotation:
      from PIL import Image
      
      # Resize image before annotation if needed
      img = Image.open("large_image.jpg")
      img_resized = img.resize((img.width//2, img.height//2))
      img_resized.save("resized_for_annotation.jpg")

Next Steps
----------

* Review :doc:`installation` for detailed setup instructions
* Explore :doc:`examples` section for real mission data with manual annotation
* Check :doc:`api` documentation for all detection method parameters
* See :doc:`benchmarks` for performance analysis across detection methods