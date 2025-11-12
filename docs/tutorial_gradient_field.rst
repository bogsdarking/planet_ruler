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

Step 1: Basic Gradient-Field Measurement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import planet_ruler.observation as obs
   from planet_ruler.uncertainty import calculate_parameter_uncertainty
   
   # Load observation
   observation = obs.LimbObservation(
       image_filepath="demo/images/ISS_Earth_horizon.jpg",
       fit_config="config/earth_iss_1.yaml"
   )
   
   # Gradient-field detection will do nothing (it isn't needed!)
   observation.detect_limb(detection_method="gradient-field")
   
   # Fitting is where the magic happens instead
   observation.fit_limb(
       loss_function="gradient_field", # Need to specify this!
       minimizer='dual-annealing',
       resolution_stages='auto',  # Multi-resolution optimization
       maxiter=3000
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
    observation.fit_limb(
        minimizer = "dual-annealing",
        loss_function="gradient_field",
        image_smoothing=2.0,
        kernel_smoothing=8.0,
        directional_smoothing=50,
        directional_decay_rate=0.10, # Directional smoothing fall-off.
        prefer_direction="up" # A trick that can help if you know that the horizon is darker along the top (up) or the bottom (down)
    )

**Parameter Effects:**

* **image_smoothing**: Controls image smoothing _before_ gradient calculation. Helps with artifacts and other local minima.
  
  * Lower (0.5): Preserves fine details, more sensitive to noise
  * Higher (2.0): Smoother gradients, less noise sensitivity

* **kernel_smoothing**: Controls image smoothing _after_ gradient calculation. Helps to homogenize noisy gradient directions.

  * Lower (1.0): Preserves fine details, more sensitive to noise
  * Higher (8.0): Smoother gradients, less noise sensitivity

* **directional_smoothing**: Directional smoothing distance (along gradients). Helps avoid vanishing gradient in loss function.
  
  * Larger values (50): Casts a wide net to keep optimization from getting stuck. Can distort results depending on strength and how far the limb is from the edge of frame.
  * Smaller values (5): Slight speedup to optimization if already converging, but too low and fit may not converge to the true minimum.

* **directional_decay_rate**: Rate of exponential decay in directional sampling
  
  * Larger values: Faster decay, emphasizes nearby pixels
  * Smaller values: Slower decay, includes more distant pixels

Step 3: Multi-Resolution Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Gradient-field detection benefits greatly from multi-resolution optimization:

.. code-block:: python

   # Automatic multi-resolution (recommended)
   observation.fit_limb(
       minimizer="dual-annealing",
       loss_function="gradient_field",
       resolution_stages='auto',  # Automatically generates stages
       maxiter=1000
   )
   
   # Manual multi-resolution configuration
   observation.fit_limb(
       minimizer="dual-annealing"
       loss_function="gradient_field",
       resolution_stages=[4, 2, 1],  # Downsample factors: 4x → 2x → 1x
       maxiter=1000
   )
   
   # Single resolution (faster but may miss global optimum)
   observation.fit_limb(
       minimizer="dual-annealing",
       loss_function="gradient_field",
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
           obs.fit_limb(
               minimizer='dual-annealing',
               loss_function="gradient_field",
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