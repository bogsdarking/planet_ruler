Tutorials
=========

This section provides step-by-step tutorials for using Planet Ruler to determine planetary radius from horizon photographs using the recommended segmentation-based approach.

Tutorial 1: Basic Earth Radius Calculation
------------------------------------------

This tutorial shows how to calculate Earth's radius using a horizon photograph from a known altitude with the recommended segmentation method.

Prerequisites
~~~~~~~~~~~~

* Python 3.8+ with Planet Ruler installed
* **Segment Anything model** (recommended): ``pip install segment-anything torch``
* A horizon photograph (we'll use the demo Earth image)
* Basic knowledge of the observation altitude

.. note::
   **Segmentation vs Gradient Break**: While Planet Ruler supports gradient-break detection for legacy compatibility, **segmentation is strongly recommended** as it's much more reliable for horizon detection in real-world images.

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

Step 3: Detect the Horizon (Recommended: Segmentation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Planet Ruler uses AI-powered segmentation to reliably detect horizons:

.. code-block:: python

   # Detect horizon using segmentation (recommended)
   observation.detect_limb(
       method="segmentation",
       segmenter="segment-anything"  # Requires segment-anything installation
   )
   
   # Smooth the detected limb
   observation.smooth_limb(
       method="rolling-median",
       window_length=15,
       fill_nan=True
   )
   
   # Plot the detected limb
   observation.plot(gradient=False, show=True)

.. tip::
   **Segmentation advantages**:
   
   * Works reliably with complex backgrounds
   * Handles clouds, terrain features, and atmospheric effects
   * More robust than gradient-based methods
   * Automatically identifies the best horizon boundary

**Fallback Method (if Segment Anything unavailable):**

.. code-block:: python

   # Alternative: gradient-break method (less reliable)
   try:
       observation.detect_limb(method="segmentation")
   except ImportError:
       print("⚠ Segment Anything not available, falling back to gradient-break")
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

**Expected Results**: For Earth from ISS altitude (~418 km) using segmentation:
* Fitted radius: ~6,371 ± 10 km (much better than gradient-break!)
* Error: < 25 km from true radius

Tutorial 2: Advanced Segmentation Techniques
-------------------------------------------

Handling Complex Images with Multiple Objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For challenging images with clouds, terrain, or multiple planetary bodies:

.. code-block:: python

   from planet_ruler.image import ImageSegmentation
   
   # Load complex image
   observation = obs.PlanetObservation("complex_horizon_image.jpg")
   
   # Use advanced segmentation with custom settings
   observation.detect_limb(
       method="segmentation",
       segmenter="segment-anything",
       points_per_side=32,        # Higher resolution segmentation
       pred_iou_thresh=0.88,      # Higher quality threshold
       stability_score_thresh=0.95,  # More stable masks
       crop_n_layers=1,           # Multi-scale processing
       min_mask_region_area=1000  # Filter small regions
   )

Visualizing Segmentation Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Plot segmentation masks
   from planet_ruler.observation import plot_segmentation_masks
   
   plot_segmentation_masks(observation)
   
   # Show detected limb overlaid on original image
   observation.plot(show_limb=True, show=True)

Custom Segmentation Models
~~~~~~~~~~~~~~~~~~~~~~~~~

For specialized use cases, you can provide custom segmentation models:

.. code-block:: python

   # Future: Custom segmenter interface (requires refactoring)
   # This functionality is planned for future releases
   
   class CustomSegmenter:
       def __init__(self, model_path):
           # Load your custom model
           pass
           
       def segment(self, image):
           # Return segmentation masks
           pass
   
   # observation.detect_limb(method="segmentation", segmenter=CustomSegmenter("my_model.pt"))

Tutorial 3: Multi-planetary Analysis (Segmentation)
--------------------------------------------------

Comparing Earth, Pluto, and Saturn with Robust Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
       
       # Load and process observation with segmentation
       obs_obj = obs.LimbObservation(image_path, config_path)
       
       try:
           # Use segmentation (recommended)
           obs_obj.detect_limb(method="segmentation")
           method_used = "Segmentation"
       except ImportError:
           # Fallback to gradient-break if needed
           print(f"  ⚠ Using gradient-break fallback for {name}")
           obs_obj.detect_limb(method="gradient-break", window_length=21)
           method_used = "Gradient-break"
       
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
           "Quality": "High" if method_used == "Segmentation" else "Medium"
       })
   
   # Display results table
   df = pd.DataFrame(results)
   print("\n" + "="*70)
   print("MULTI-PLANETARY ANALYSIS RESULTS")
   print("="*70)
   print(df.to_string(index=False))

Tutorial 4: Performance and Reliability Comparison
-------------------------------------------------

Comparing Detection Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import time
   
   # Load test image
   observation = obs.LimbObservation("test_image.jpg", "config/earth_iss_1.yaml")
   
   methods_to_test = [
       ("segmentation", {"segmenter": "segment-anything"}),
       ("gradient-break", {"window_length": 21, "threshold": 0.1})
   ]
   
   results = {}
   
   for method_name, kwargs in methods_to_test:
       print(f"\nTesting {method_name}...")
       
       # Fresh observation for each test
       test_obs = obs.LimbObservation("test_image.jpg", "config/earth_iss_1.yaml")
       
       # Time the detection
       start_time = time.time()
       try:
           test_obs.detect_limb(method=method_name, **kwargs)
           test_obs.smooth_limb()
           test_obs.fit_limb()
           
           detection_time = time.time() - start_time
           
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
           
       except Exception as e:
           results[method_name] = {
               "error": str(e),
               "success": False
           }
   
   # Compare results
   print("\n" + "="*50)
   print("METHOD COMPARISON")
   print("="*50)
   
   for method, result in results.items():
       if result["success"]:
           print(f"{method.upper()}:")
           print(f"  Time: {result['time']:.1f} seconds")
           print(f"  Radius: {result['radius']:.1f} ± {result['uncertainty']:.1f} km")
           print(f"  Relative uncertainty: {100*result['uncertainty']/result['radius']:.1f}%")
       else:
           print(f"{method.upper()}: FAILED - {result['error']}")

Installation and Setup for Segmentation
---------------------------------------

Required Dependencies
~~~~~~~~~~~~~~~~~~~~

For the best Planet Ruler experience with segmentation:

.. code-block:: bash

   # Essential: Install Planet Ruler with segmentation support
   pip install planet-ruler
   
   # Required for segmentation: Segment Anything + PyTorch
   pip install segment-anything torch torchvision
   
   # Optional: GPU support for faster processing
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

Verification Test
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Test segmentation installation
   try:
       from planet_ruler.image import ImageSegmentation
       from segment_anything import sam_model_registry
       print("✓ Segmentation support available")
   except ImportError as e:
       print(f"⚠ Segmentation not available: {e}")
       print("Install with: pip install segment-anything torch")

Troubleshooting Segmentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Common issues and solutions:**

1. **"No module named 'segment_anything'"**
   
   .. code-block:: bash
   
      pip install segment-anything

2. **"CUDA out of memory"**
   
   .. code-block:: python
   
      # Use CPU instead of GPU
      observation.detect_limb(method="segmentation", device="cpu")

3. **"Model checkpoint not found"**
   
   .. code-block:: python
   
      # Manually download SAM model
      import torch
      from segment_anything import sam_model_registry
      
      # This will auto-download the model
      sam = sam_model_registry["vit_h"](checkpoint="path/to/sam_vit_h_4b8939.pth")

Performance Tips
~~~~~~~~~~~~~~~

.. code-block:: python

   # For faster segmentation on large images:
   
   # 1. Reduce image resolution
   observation.image_data = observation.image_data[::2, ::2]  # 2x downsampling
   
   # 2. Use fewer segmentation points
   observation.detect_limb(
       method="segmentation", 
       points_per_side=16  # Default: 32, lower = faster
   )
   
   # 3. Use CPU for small images, GPU for large ones
   device = "cpu" if observation.image_data.size < 1000000 else "cuda"
   observation.detect_limb(method="segmentation", device=device)

Next Steps
----------

* Review :doc:`installation` for detailed segmentation setup
* Explore :doc:`examples` section for real mission data with segmentation
* Check :doc:`api` documentation for segmentation parameters
* See :doc:`benchmarks` for segmentation performance analysis