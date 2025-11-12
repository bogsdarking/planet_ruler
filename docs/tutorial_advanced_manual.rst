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
   
   observation.detect_limb(detection_method="manual")

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