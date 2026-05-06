The fastest way to get started using your own images with Planet Ruler by using automatic camera parameter detection from EXIF data.

Prerequisites
~~~~~~~~~~~~~

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
       altitude_m=10_000,  # Your altitude in meters
       planet="earth",
       limits_preset="balanced",  # "tight", "balanced" (default), or "loose"
   )

   # View detected camera info
   print("Auto-detected camera:")
   camera_info = auto_config["camera_info"]
   print(f"  Model: {camera_info.get('camera_model', 'Unknown')}")
   print(f"  Type:  {camera_info.get('camera_type', 'Unknown')}")
   # Focal length and sensor width are stored in meters in init_parameter_values
   f_mm = auto_config["init_parameter_values"]["f"] * 1000
   w_mm = auto_config["init_parameter_values"]["w"] * 1000
   print(f"  Focal length: {f_mm:.1f} mm")
   print(f"  Sensor width: {w_mm:.1f} mm")

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
   observation.fit_arc()

Step 3: CLI Usage (Even Simpler)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the simplest workflow, use the command line:

.. code-block:: bash

   # One command to measure planetary radius
   planet-ruler measure --auto-config --altitude 10000 --planet earth your_photo.jpg

   # Override auto-detected field-of-view if needed
   planet-ruler measure --auto-config --altitude 10000 --planet earth --field-of-view 60 your_photo.jpg

.. note::
   **Multi-camera phones**: Modern phones have multiple lenses (wide, main, tele).
   Planet Ruler reads the EXIF aperture tag to automatically infer which camera
   module was used and selects the correct sensor size — no manual selection needed.

**Advantages of Zero-Config Approach:**

* **No manual camera configuration needed**
* **Works immediately with any EXIF-enabled image**
* **Automatic sensor size database lookup**
* **Multi-camera phone support** — correct lens selected from EXIF aperture
* **Adjustable parameter limits** via ``limits_preset`` ("tight", "balanced", "loose")