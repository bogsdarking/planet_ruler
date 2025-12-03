Tutorials
=========

This section provides step-by-step tutorials for using Planet Ruler to determine planetary radius from horizon photographs using the default interactive manual annotation approach.

Tutorial 0: Pre-Configured Earth Radius Measurement
------------------------------------------

.. include:: tutorial_preconfig.rst

Tutorial 1: Auto-Configured Earth Radius Measurement
------------------------------------------

.. include:: tutorial_autoconfig.rst

Tutorial 1.5: Measure Earth from an Airplane Window
--------------------------------------------------

.. include:: tutorial_airplane.rst

Tutorial 2: Advanced Manual Annotation Techniques
-------------------------------------------------

.. include:: tutorial_advanced_manual.rst

Tutorial 2.5: Gradient-Field Automated Detection
-----------------------------------------------

.. include:: tutorial_gradient_field.rst

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

.. include:: tutorial_method_selection.rst

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
       from planet_ruler.image import MaskSegmenter
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