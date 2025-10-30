Tutorials
=========

This section provides step-by-step tutorials for using Planet Ruler to determine planetary radius from horizon photographs using the default interactive manual annotation approach.

Tutorial 0: Zero-Configuration Quick Start
------------------------------------------

The fastest way to get started with Planet Ruler using automatic camera parameter detection from image EXIF data.

Prerequisites
~~~~~~~~~~~~

* Python 3.8+ with Planet Ruler installed
* A horizon photograph with EXIF data (from phone, DSLR, mirrorless camera)
* Known or estimated altitude when photo was taken

Step 1: Automatic Camera Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Planet Ruler can automatically extract camera parameters from image EXIF data:

.. code-block:: python

   from planet_ruler.camera import create_config_from_image
   
   # Automatically generate config from image EXIF
   auto_config = create_config_from_image(
       image_path="your_horizon_photo.jpg",
       altitude_km=10.0,  # Your altitude in kilometers
       planet="earth"
   )
   
   # View detected camera info
   print("Auto-detected camera:")
   camera = auto_config["camera"]
   print(f"  Make/Model: {camera.get('make', 'Unknown')} {camera.get('model', 'Unknown')}")
   print(f"  Focal length: {camera['focal_length_mm']:.1f} mm")
   print(f"  Sensor width: {camera['sensor_width_mm']:.1f} mm")
   print(f"  Field of view: {auto_config['observation']['field_of_view_deg']:.1f}°")

Step 2: Direct Analysis (No Config Files)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the auto-generated configuration directly:

.. code-block:: python

   import planet_ruler.observation as obs
   
   # Load observation using auto-generated config
   observation = obs.LimbObservation(
       image_filepath="your_horizon_photo.jpg",
       fit_config=auto_config  # Use dict instead of file path
   )
   
   # Standard workflow continues the same
   observation.detect_limb(method="manual")
   observation.smooth_limb()
   observation.fit_limb()

Step 3: CLI Usage (Even Simpler)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the simplest workflow, use the command line:

.. code-block:: bash

   # One command to measure planetary radius
   planet-ruler measure --auto-config --altitude 10 --planet earth your_photo.jpg
   
   # Override auto-detected field-of-view if needed
   planet-ruler measure --auto-config --altitude 10 --planet earth --field-of-view 60 your_photo.jpg

**Advantages of Zero-Config Approach:**

* **No manual camera configuration needed**
* **Works immediately with any EXIF-enabled image**
* **Automatic sensor size database lookup**
* **Parameter override capability when needed**

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

Planet Ruler offers three detection methods, with manual annotation as the default:

.. code-block:: python

   # Method 1: Manual annotation (default - precise, no dependencies)
   observation.detect_limb(method="manual")  # Opens interactive GUI
   
   # Method 2: Gradient-field (automated - good for clear horizons)
   # observation.detect_limb(method="gradient-field")
   
   # Method 3: AI segmentation (automated - requires PyTorch)
   # observation.detect_limb(method="segmentation")
   
   # Smooth the detected limb
   observation.smooth_limb(
       method="rolling-median",
       window_length=15,
       fill_nan=True
   )
   
   # Plot the detected limb
   observation.plot(gradient=False, show=True)

.. tip::
   **Method Comparison**:
   
   * **Manual**: Precise user control, works with any image quality, no dependencies
   * **Gradient-field**: Automated, fast, works with clear horizons, no ML dependencies  
   * **Segmentation**: Most versatile, handles complex images, requires PyTorch + Segment Anything

**Alternative Methods:**

.. code-block:: python

   # Option 1: Gradient-field (automated gradient analysis)
   observation.detect_limb(
       method="gradient-field",
       loss_function="gradient_field",
       gradient_sigma=1.0
   )
   
   # Option 2: AI segmentation (requires PyTorch + Segment Anything)
   try:
       observation.detect_limb(
           method="segmentation",
           segmenter="segment-anything"
       )
   except ImportError:
       print("⚠ Segment Anything not available - install with: pip install segment-anything torch")

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

Tutorial 2.5: Gradient-Field Automated Detection
-----------------------------------------------

This tutorial demonstrates the gradient-field detection method, which provides automated horizon detection without requiring ML dependencies or manual annotation.

When to Use Gradient-Field
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Best for:**

* Clear, well-defined horizons with strong gradients
* Atmospheric limbs without complex cloud structure
* Batch processing multiple images
* When PyTorch/ML dependencies are unavailable
* Situations where reproducibility is critical

**Not ideal for:**

* Horizons with multiple strong edges (use manual)
* Very noisy or low-contrast images (use manual)
* Complex cloud structures (use segmentation or manual)

Prerequisites
~~~~~~~~~~~~

* Python 3.8+ with Planet Ruler installed
* Clear horizon photograph
* Camera configuration file or auto-config

Step 1: Basic Gradient-Field Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import planet_ruler.observation as obs
   from planet_ruler.uncertainty import calculate_parameter_uncertainty
   
   # Load observation
   observation = obs.LimbObservation(
       image_filepath="demo/images/ISS_Earth_horizon.jpg",
       fit_config="config/earth_iss_1.yaml"
   )
   
   # Gradient-field detection (fully automated)
   print("Detecting horizon with gradient-field method...")
   observation.detect_limb(method="gradient-field")
   
   # Smooth and fit
   observation.smooth_limb()
   observation.fit_limb(
       minimizer='dual-annealing',
       resolution_stages='auto',  # Multi-resolution optimization
       maxiter=1000
   )
   
   # Calculate uncertainty
   result = calculate_parameter_uncertainty(
       observation, "r",
       scale_factor=1000,
       method='auto'
   )
   
   print(f"Radius: {result['uncertainty']:.1f} km")
   observation.plot()

Step 2: Understanding Gradient-Field Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The gradient-field method has several configurable parameters:

.. code-block:: python

   # Configure gradient-field detection
   observation.detect_limb(
       method="gradient-field",
       gradient_sigma=1.0,          # Gaussian smoothing (higher = smoother)
       directional_blur_distance=5,  # Distance for directional sampling
       directional_blur_decay=0.1    # Exponential decay rate
   )

**Parameter Effects:**

* **gradient_sigma**: Controls image smoothing before gradient calculation
  
  * Lower (0.5): Preserves fine details, more sensitive to noise
  * Higher (2.0): Smoother gradients, less noise sensitivity

* **directional_blur_distance**: How far to sample along gradient directions
  
  * Larger values: Enhance coherent features over larger scales
  * Smaller values: More localized feature detection

* **directional_blur_decay**: Rate of exponential decay in directional sampling
  
  * Larger values: Faster decay, emphasizes nearby pixels
  * Smaller values: Slower decay, includes more distant pixels

Step 3: Multi-Resolution Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Gradient-field detection benefits greatly from multi-resolution optimization:

.. code-block:: python

   # Automatic multi-resolution (recommended)
   observation.fit_limb(
       minimizer='dual-annealing',
       resolution_stages='auto',  # Automatically generates stages
       maxiter=1000
   )
   
   # Manual multi-resolution configuration
   observation.fit_limb(
       minimizer='differential-evolution',
       resolution_stages=[4, 2, 1],  # Downsample factors: 4x → 2x → 1x
       maxiter=1000
   )
   
   # Single resolution (faster but may miss global optimum)
   observation.fit_limb(
       minimizer='dual-annealing',
       resolution_stages=None,  # No multi-resolution
       maxiter=1000
   )

**Multi-Resolution Benefits:**

* Avoids local minima by starting coarse
* Progressively refines solution
* More robust convergence
* Slightly slower but much more reliable

Step 4: Choosing Minimizers for Gradient-Field
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Different minimizers have different characteristics:

.. code-block:: python

   # Differential evolution: Best for complex problems
   observation.fit_limb(
       minimizer='differential-evolution',
       resolution_stages='auto',
       popsize=15,
       maxiter=1000,
       seed=42
   )
   # Pros: Most robust, provides population for uncertainty
   # Cons: Slowest
   
   # Dual annealing: Good balance
   observation.fit_limb(
       minimizer='dual-annealing',
       resolution_stages='auto',
       maxiter=1000,
       seed=42
   )
   # Pros: Fast, good global optimization
   # Cons: No population (use Hessian for uncertainty)
   
   # Basinhopping: Fastest
   observation.fit_limb(
       minimizer='basinhopping',
       resolution_stages='auto',
       niter=100,
       seed=42
   )
   # Pros: Fastest optimization
   # Cons: May miss global optimum, use with multi-resolution

Step 5: Visualizing Gradient-Field Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np
   
   # Create comprehensive visualization
   fig, axes = plt.subplots(2, 2, figsize=(14, 10))
   
   # Original image
   axes[0, 0].imshow(observation.image_data)
   axes[0, 0].set_title("Original Image")
   axes[0, 0].axis('off')
   
   # Gradient field
   observation.plot(gradient=True, ax=axes[0, 1], show=False)
   axes[0, 1].set_title("Gradient Field")
   
   # Detected limb overlay
   axes[1, 0].imshow(observation.image_data)
   x = np.arange(len(observation.features["limb"]))
   axes[1, 0].plot(x, observation.features["limb"], 'r-', linewidth=2)
   axes[1, 0].set_title("Detected Limb")
   axes[1, 0].axis('off')
   
   # Fit quality
   x = np.arange(len(observation.features["limb"]))
   axes[1, 1].plot(x, observation.features["limb"], 'b-', 
                   linewidth=2, label="Detected")
   
   # Theoretical limb
   final_params = observation.init_parameter_values.copy()
   final_params.update(observation.best_parameters)
   theoretical = planet_ruler.geometry.limb_arc(
       n_pix_x=len(observation.features["limb"]),
       n_pix_y=observation.image_data.shape[0],
       **final_params
   )
   axes[1, 1].plot(x, theoretical, 'r--', 
                   linewidth=2, label="Fitted model")
   axes[1, 1].set_title("Fit Quality")
   axes[1, 1].set_xlabel("Pixel X")
   axes[1, 1].set_ylabel("Pixel Y")
   axes[1, 1].legend()
   axes[1, 1].grid(alpha=0.3)
   
   plt.tight_layout()
   plt.show()

Step 6: Batch Processing with Gradient-Field
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The gradient-field method is ideal for batch processing:

.. code-block:: python

   from pathlib import Path
   import pandas as pd
   
   # Process multiple images
   image_dir = Path("demo/images/")
   image_files = list(image_dir.glob("*_horizon_*.jpg"))
   
   results = []
   
   for image_file in image_files:
       print(f"Processing {image_file.name}...")
       
       try:
           # Load and process
           obs = obs.LimbObservation(
               str(image_file),
               "config/earth_iss_1.yaml"
           )
           
           # Automated detection
           obs.detect_limb(method="gradient-field")
           obs.smooth_limb()
           obs.fit_limb(
               minimizer='dual-annealing',
               resolution_stages='auto',
               maxiter=500  # Reduce for speed
           )
           
           # Calculate uncertainty
           result = calculate_parameter_uncertainty(
               obs, "r", scale_factor=1000, method='auto'
           )
           
           results.append({
               'file': image_file.name,
               'radius_km': result['uncertainty'],
               'status': 'success'
           })
           
       except Exception as e:
           results.append({
               'file': image_file.name,
               'radius_km': None,
               'status': f'failed: {str(e)}'
           })
   
   # Summary
   df = pd.DataFrame(results)
   print(df)
   
   # Save results
   df.to_csv("batch_results.csv", index=False)

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

Comparing Manual, Gradient-Field, and AI Segmentation Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import time
   
   # Load test image
   observation = obs.LimbObservation("test_image.jpg", "config/earth_iss_1.yaml")
   
   methods_to_test = [
       ("manual", {}),  # Interactive GUI - time depends on user
       ("gradient-field", {})  # Automated gradient analysis
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
       test_obs.fit_limb(
           minimizer='dual-annealing',
           resolution_stages='auto' if method_name == "gradient-field" else None,
           maxiter=500
       )
       
       # Calculate uncertainty
       from planet_ruler.uncertainty import calculate_parameter_uncertainty
       radius_result = calculate_parameter_uncertainty(
           test_obs, "r", scale_factor=1000, method='auto'
       )
       
       results[method_name] = {
           "time": detection_time,
           "radius": radius_result['uncertainty'],
           "method_info": radius_result['method'],
           "success": True
       }
   
   # Compare results
   print("\n" + "="*70)
   print("METHOD COMPARISON")
   print("="*70)
   
   for method, result in results.items():
       if result["success"]:
           print(f"\n{method.upper()}:")
           if method == "manual":
               print(f"  Time: {result['time']:.1f} seconds (user-dependent)")
               print(f"  Precision: User-controlled (typically highest)")
           elif method == "gradient-field":
               print(f"  Time: {result['time']:.1f} seconds (automated)")
               print(f"  Precision: Good for clear horizons")
               print(f"  Dependencies: None (uses SciPy only)")
           else:
               print(f"  Time: {result['time']:.1f} seconds (automated)")
               print(f"  Precision: Best for complex images")
               print(f"  Dependencies: PyTorch + Segment Anything (~2GB)")
           print(f"  Radius: {result['radius']:.1f} km")
           print(f"  Uncertainty method: {result['method_info']}")
       else:
           print(f"\n{method.upper()}: FAILED - {result['error']}")
   
   # Summary recommendation
   print("\n" + "="*70)
   print("RECOMMENDATIONS")
   print("="*70)
   print("""
   Choose detection method based on your needs:
   
   • MANUAL: Best when precision is critical, any image quality
     - User controls every aspect
     - Works immediately (no dependencies)
     - Interactive and educational
   
   • GRADIENT-FIELD: Best for batch processing clear horizons
     - Fully automated
     - No ML dependencies
     - Fast and reproducible
     - Requires clear, well-defined horizons
   
   • SEGMENTATION: Best for complex or challenging images
     - Most versatile
     - Handles clouds, haze, complex scenes
     - Requires PyTorch (~2GB) + Segment Anything
     - Slower due to model inference
   """)

Installation and Setup
----------------------

Basic Installation
~~~~~~~~~~~~~~~~~

Planet Ruler works immediately after installation with no additional dependencies:

.. code-block:: bash

   # Essential: Install Planet Ruler (manual annotation works immediately)
   python -m pip install planet-ruler

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
   python -m pip install segment-anything torch torchvision
   
   # Optional: GPU support for faster AI processing
   python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

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
   
      python -m pip install segment-anything torch

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