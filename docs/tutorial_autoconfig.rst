The fastest way to get started using your own images with Planet Ruler by using automatic camera parameter detection from EXIF data.

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
       altitude_m=10_000,  # Your altitude in kilometers
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
   
   # Standard workflow
   observation.detect_limb(detection_method="manual")
   observation.fit_limb()

Step 3: CLI Usage (Even Simpler)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the simplest workflow, use the command line:

.. code-block:: bash

   # One command to measure planetary radius
   planet-ruler measure --auto-config --altitude 10_000 --planet earth your_photo.jpg
   
   # Override auto-detected field-of-view if needed
   planet-ruler measure --auto-config --altitude 10_000 --planet earth --field-of-view 60 your_photo.jpg

**Advantages of Zero-Config Approach:**

* **No manual camera configuration needed**
* **Works immediately with any EXIF-enabled image**
* **Automatic sensor size database lookup**
* **Parameter override capability when needed**