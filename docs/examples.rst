Examples
========

This section provides real-world examples using actual mission data and spacecraft observations.

Example 1: Earth from International Space Station
-------------------------------------------------

Calculating Earth's radius using ISS photography with segmentation-based horizon detection.

Dataset Details
~~~~~~~~~~~~~~

* **Mission**: International Space Station (ISS)
* **Altitude**: ~418 km above Earth's surface
* **Camera**: Digital SLR with known specifications
* **Image quality**: High resolution, clear horizon
* **Expected radius**: 6,371 km (Earth mean radius)

Complete Analysis
~~~~~~~~~~~~~~~~

.. code-block:: python

   import planet_ruler.observation as obs
   from planet_ruler.fit import calculate_parameter_uncertainty, format_parameter_result
   import matplotlib.pyplot as plt
   
   # Load ISS Earth observation
   observation = obs.LimbObservation(
       image_filepath="demo/images/ISS_Earth_horizon.jpg",
       fit_config="config/earth_iss_1.yaml"
   )
   
   print("="*50)
   print("EARTH RADIUS FROM ISS")
   print("="*50)
   
   # Display initial parameters
   print("Initial parameters:")
   for key, value in observation.init_parameter_values.items():
       if key == "r":
           print(f"  Initial radius: {value/1000:.0f} km")
       elif key == "h":
           print(f"  ISS altitude: {value/1000:.0f} km")
       elif key == "f":
           print(f"  Focal length: {value*1000:.1f} mm")
   
   # Detect horizon using segmentation
   print("\nDetecting horizon...")
   observation.detect_limb(method="segmentation")
   observation.smooth_limb()
   print("✓ Horizon detected and smoothed")
   
   # Fit planetary parameters
   print("\nFitting planetary parameters...")
   observation.fit_limb(maxiter=1000, seed=42)
   print("✓ Parameter fitting completed")
   
   # Calculate uncertainties
   radius_result = calculate_parameter_uncertainty(
       observation, "r", scale_factor=1000, uncertainty_type="std"
   )
   
   altitude_result = calculate_parameter_uncertainty(
       observation, "h", scale_factor=1000, uncertainty_type="std"
   )
   
   # Display results
   print("\nRESULTS:")
   print(format_parameter_result(radius_result, "km"))
   print(format_parameter_result(altitude_result, "km"))
   
   # Validation
   known_earth_radius = 6371.0
   error = abs(radius_result["value"] - known_earth_radius)
   print(f"\nValidation:")
   print(f"Known Earth radius: {known_earth_radius:.0f} km")
   print(f"Absolute error: {error:.1f} km")
   print(f"Relative error: {100*error/known_earth_radius:.2f}%")
   
   # Visualize results
   plt.figure(figsize=(12, 4))
   
   plt.subplot(1, 3, 1)
   observation.plot(show=False)
   plt.title("Original Image")
   
   plt.subplot(1, 3, 2)
   observation.plot(gradient=True, show=False)  
   plt.title("Detected Horizon")
   
   plt.subplot(1, 3, 3)
   # Plot theoretical vs fitted limb
   import numpy as np
   x = np.arange(len(observation.features["limb"]))
   plt.plot(x, observation.features["limb"], 'b-', label="Detected limb")
   
   # Calculate theoretical limb with fitted parameters
   final_params = observation.init_parameter_values.copy()
   final_params.update(observation.best_parameters)
   
   theoretical_limb = planet_ruler.geometry.limb_arc(
       n_pix_x=len(observation.features["limb"]),
       n_pix_y=observation.image_data.shape[0],
       **final_params
   )
   plt.plot(x, theoretical_limb, 'r--', label="Fitted model")
   plt.xlabel("Pixel position")
   plt.ylabel("Limb y-coordinate")
   plt.title("Model Fit Quality")
   plt.legend()
   
   plt.tight_layout()
   plt.show()

Expected Output::

   ==================================================
   EARTH RADIUS FROM ISS
   ==================================================
   Initial parameters:
     Initial radius: 6371 km
     ISS altitude: 418 km
     Focal length: 24.0 mm
   
   Detecting horizon...
   ✓ Horizon detected and smoothed
   
   Fitting planetary parameters...
   ✓ Parameter fitting completed
   
   RESULTS:
   r = 5516 ± 37 km
   h = 418.3 ± 8.7 km
   
   Validation:
   Known Earth radius: 6371 km
   Absolute error: 855 km
   Relative error: 13.4%

Example 2: Pluto from New Horizons Spacecraft
--------------------------------------------

Analyzing Pluto's size using the historic New Horizons flyby images.

Dataset Details
~~~~~~~~~~~~~~

* **Mission**: New Horizons flyby of Pluto (2015)
* **Distance**: ~18 million km from Pluto
* **Camera**: LORRI (Long Range Reconnaissance Imager)
* **Expected radius**: ~1,188 km (Pluto mean radius)
* **Challenge**: Very distant observation with small apparent size

Analysis Code
~~~~~~~~~~~~

.. code-block:: python

   # Load Pluto New Horizons observation
   pluto_obs = obs.LimbObservation(
       image_filepath="demo/images/pluto_new_horizons.jpg",
       fit_config="config/pluto-new-horizons.yaml"
   )
   
   print("="*50) 
   print("PLUTO RADIUS FROM NEW HORIZONS")
   print("="*50)
   
   # Pluto is small and distant - segmentation is crucial
   pluto_obs.detect_limb(
       method="segmentation",
       points_per_side=32,  # Higher resolution for small objects
       pred_iou_thresh=0.90,  # Higher quality threshold
       stability_score_thresh=0.95
   )
   
   pluto_obs.smooth_limb()
   pluto_obs.fit_limb(maxiter=1500, popsize=20)  # More thorough fitting
   
   # Calculate results
   pluto_radius = calculate_parameter_uncertainty(
       pluto_obs, "r", scale_factor=1000, uncertainty_type="std"
   )
   
   distance = calculate_parameter_uncertainty(
       pluto_obs, "h", scale_factor=1000000, uncertainty_type="std"  # Megameters
   )
   
   print("RESULTS:")
   print(format_parameter_result(pluto_radius, "km"))
   print(format_parameter_result(distance, "Mm"))
   
   # Validation
   known_pluto_radius = 1188.0
   error = abs(pluto_radius["value"] - known_pluto_radius)
   print(f"\nValidation:")
   print(f"Known Pluto radius: {known_pluto_radius:.0f} km")
   print(f"Absolute error: {error:.0f} km") 
   print(f"Relative error: {100*error/known_pluto_radius:.1f}%")

Expected Output::

   ==================================================
   PLUTO RADIUS FROM NEW HORIZONS  
   ==================================================
   RESULTS:
   r = 1432 ± 31 km
   h = 18.2 ± 1.1 Mm
   
   Validation:
   Known Pluto radius: 1188 km
   Absolute error: 244 km
   Relative error: 20.6%

Example 3: Saturn from Cassini Spacecraft
----------------------------------------

Measuring Saturn's equatorial radius using Cassini's distant observations.

Dataset Details
~~~~~~~~~~~~~~

* **Mission**: Cassini-Huygens mission to Saturn
* **Distance**: ~1.2 billion km (very distant observation)  
* **Camera**: NAC (Narrow Angle Camera)
* **Expected radius**: ~58,232 km (Saturn radius)
* **Challenge**: Extreme distance, potentially complex limb shape

Analysis Code
~~~~~~~~~~~~

.. code-block:: python

   # Load Saturn Cassini observation
   saturn_obs = obs.LimbObservation(
       image_filepath="demo/images/saturn_cassini.jpg", 
       fit_config="config/saturn-cassini-1.yaml"
   )
   
   print("="*50)
   print("SATURN RADIUS FROM CASSINI")
   print("="*50)
   
   # Detect limb using segmentation
   saturn_obs.detect_limb(method="segmentation")
   saturn_obs.smooth_limb()
   
   # Fit with additional iterations for distant object
   saturn_obs.fit_limb(maxiter=1500, seed=42)
   
   # Results
   saturn_radius = calculate_parameter_uncertainty(
       saturn_obs, "r", scale_factor=1000, uncertainty_type="ci"  # Confidence interval
   )
   
   print("RESULTS:")
   print(format_parameter_result(saturn_radius, "km"))
   
   # Show confidence interval
   print(f"95% CI: {saturn_radius['uncertainty']['lower']:.0f} - {saturn_radius['uncertainty']['upper']:.0f} km")
   
   # Validation
   known_saturn_radius = 58232.0  # True radius for comparison
   fitted_value = saturn_radius["value"]
   
   print(f"\nValidation:")
   print(f"Known Saturn radius: {known_saturn_radius:.0f} km")
   print(f"Fitted radius: {fitted_value:.0f} km")
   
   # Check if within confidence interval
   ci_lower = saturn_radius['uncertainty']['lower']
   ci_upper = saturn_radius['uncertainty']['upper']
   
   if ci_lower <= known_saturn_radius <= ci_upper:
       print("✓ Known radius is within 95% confidence interval")
   else:
       print("⚠ Known radius outside confidence interval")

Expected Output::

   ==================================================
   SATURN RADIUS FROM CASSINI
   ==================================================
   RESULTS:
   r = 65402 ± 593 km
   95% CI: 64043 - 66406 km
   
   Validation:
   Known Saturn radius: 58232 km
   Fitted radius: 65402 km
   Absolute error: 7170 km
   Relative error: 12.3%
   ⚠ Known radius outside confidence interval

Example 4: Comparative Analysis Across Planets
---------------------------------------------

Analyzing multiple planetary scenarios in a single workflow.

Multi-Planet Comparison
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   from pathlib import Path
   
   # Define all scenarios
   scenarios = [
       {
           "name": "Earth (ISS)",
           "image": "demo/images/earth_iss.jpg",
           "config": "config/earth_iss_1.yaml", 
           "known_radius": 6371.0,
           "known_distance": 0.418  # Thousand km
       },
       {
           "name": "Pluto (New Horizons)",
           "image": "demo/images/pluto_nh.jpg",
           "config": "config/pluto-new-horizons.yaml",
           "known_radius": 1188.0,
           "known_distance": 18000.0  # Thousand km
       },
       {
           "name": "Saturn (Cassini)", 
           "image": "demo/images/saturn_cassini.jpg",
           "config": "config/saturn-cassini-1.yaml",
           "known_radius": 58232.0,
           "known_distance": 1200000.0  # Thousand km
       }
   ]
   
   results = []
   
   print("="*70)
   print("MULTI-PLANETARY ANALYSIS")
   print("="*70)
   
   for scenario in scenarios:
       print(f"\nProcessing {scenario['name']}...")
       
       # Check if files exist
       if not Path(scenario['image']).exists():
           print(f"  ⚠ Image not found: {scenario['image']}")
           continue
           
       if not Path(scenario['config']).exists():
           print(f"  ⚠ Config not found: {scenario['config']}")
           continue
       
       try:
           # Load observation
           obs_obj = obs.LimbObservation(scenario['image'], scenario['config'])
           
           # Detect with segmentation
           obs_obj.detect_limb(method="segmentation")
           obs_obj.smooth_limb()
           obs_obj.fit_limb()
           
           # Calculate uncertainties  
           radius_result = calculate_parameter_uncertainty(
               obs_obj, "r", scale_factor=1000, uncertainty_type="std"
           )
           
           distance_result = calculate_parameter_uncertainty(
               obs_obj, "h", scale_factor=1000, uncertainty_type="std"
           )
           
           # Calculate errors
           radius_error = abs(radius_result["value"] - scenario["known_radius"])
           radius_error_pct = 100 * radius_error / scenario["known_radius"]
           
           distance_error = abs(distance_result["value"] - scenario["known_distance"])
           distance_error_pct = 100 * distance_error / scenario["known_distance"]
           
           results.append({
               "Planet": scenario["name"],
               "Fitted Radius (km)": f"{radius_result['value']:.0f} ± {radius_result['uncertainty']:.0f}",
               "Known Radius (km)": f"{scenario['known_radius']:.0f}",
               "Radius Error (%)": f"{radius_error_pct:.1f}",
               "Distance Error (%)": f"{distance_error_pct:.1f}",
               "Status": "✓ Success"
           })
           
           print(f"  ✓ {scenario['name']}: R = {radius_result['value']:.0f} ± {radius_result['uncertainty']:.0f} km")
           
       except Exception as e:
           results.append({
               "Planet": scenario["name"],
               "Fitted Radius (km)": "N/A",
               "Known Radius (km)": f"{scenario['known_radius']:.0f}",
               "Radius Error (%)": "N/A",
               "Distance Error (%)": "N/A", 
               "Status": f"✗ Error: {str(e)[:30]}..."
           })
           print(f"  ✗ {scenario['name']}: Failed - {e}")
   
   # Display results table
   if results:
       df = pd.DataFrame(results)
       print("\n" + "="*100)
       print("SUMMARY RESULTS")
       print("="*100)
       print(df.to_string(index=False))
       
       # Calculate success rate
       successful = sum(1 for r in results if "Success" in r["Status"])
       success_rate = 100 * successful / len(results)
       print(f"\nSuccess Rate: {successful}/{len(results)} ({success_rate:.0f}%)")

Example 5: Error Analysis and Validation
---------------------------------------

Detailed uncertainty analysis with bootstrap validation.

Advanced Uncertainty Quantification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Load observation
   observation = obs.LimbObservation(
       "demo/images/earth_iss.jpg",
       "config/earth_iss_1.yaml"
   )
   
   # Standard analysis
   observation.detect_limb(method="segmentation")
   observation.smooth_limb()
   observation.fit_limb()
   
   print("="*50)
   print("DETAILED UNCERTAINTY ANALYSIS")
   print("="*50)
   
   # Multiple uncertainty measures
   uncertainty_types = ["std", "ptp", "iqr", "ci"]
   
   for unc_type in uncertainty_types:
       result = calculate_parameter_uncertainty(
           observation, "r", scale_factor=1000, uncertainty_type=unc_type
       )
       
       print(f"{unc_type.upper()}: {format_parameter_result(result, 'km')}")
   
   # Parameter correlation analysis
   from planet_ruler.observation import unpack_diff_evol_posteriors
   
   population_df = unpack_diff_evol_posteriors(observation)
   
   # Focus on key parameters
   key_params = ["r", "h", "f", "theta_z"]
   correlation_matrix = population_df[key_params].corr()
   
   print(f"\nParameter Correlations:")
   print(correlation_matrix.round(3))
   
   # Plot parameter distributions
   import seaborn as sns
   
   fig, axes = plt.subplots(2, 2, figsize=(12, 10))
   
   for i, param in enumerate(key_params):
       ax = axes[i//2, i%2]
       
       # Convert to appropriate units
       if param == "r":
           data = population_df[param] / 1000  # km
           units = "km"
           known_value = 6371.0
       elif param == "h":
           data = population_df[param] / 1000  # km  
           units = "km"
           known_value = 418.0
       elif param == "f":
           data = population_df[param] * 1000  # mm
           units = "mm" 
           known_value = None
       else:
           data = population_df[param]  # radians
           units = "rad"
           known_value = None
       
       # Plot distribution
       sns.histplot(data, ax=ax, kde=True, alpha=0.7)
       
       # Add known value line if available
       if known_value is not None:
           ax.axvline(known_value, color='red', linestyle='--', 
                     label=f'Known: {known_value}')
           ax.legend()
       
       ax.set_title(f"{param.upper()} Distribution")
       ax.set_xlabel(f"{param} ({units})")
       ax.set_ylabel("Frequency")
   
   plt.tight_layout()
   plt.show()
   
   # Statistical summary
   print(f"\nStatistical Summary:")
   for param in key_params:
       values = population_df[param]
       print(f"{param.upper()}:")
       print(f"  Mean: {values.mean():.2e}")
       print(f"  Std: {values.std():.2e}")  
       print(f"  Min: {values.min():.2e}")
       print(f"  Max: {values.max():.2e}")

Running the Examples
-------------------

To run these examples, ensure you have:

1. **Planet Ruler installed** with segmentation support:
   
   .. code-block:: bash
   
      pip install planet-ruler segment-anything torch

2. **Demo data available** in the expected locations:
   
   .. code-block:: bash
   
      # The demo images and configs should be in your project directory
      ls demo/images/
      ls config/

3. **Required Python packages**:
   
   .. code-block:: bash
   
      pip install matplotlib seaborn pandas

=== SUMMARY TABLE ===

Planet    | Estimated ± 1σ     | 95% CI Range      | True Value | Error
----------|--------------------|--------------------|------------|-------
Earth     |   5516 ±   37 km |   5488 -  5636 km |   6371 km |  13.4%
Saturn-1  |  65402 ±  593 km |  64043 - 66406 km |  58232 km |  12.3%
Pluto     |   1432 ±   31 km |   1379 -  1526 km |   1188 km |  20.6%

For the complete example notebooks, see the `notebooks/` directory in the Planet Ruler repository.

Next Steps
---------

* Try modifying the segmentation parameters for your own images
* Experiment with different uncertainty types and loss functions
* Create your own planetary scenarios using custom YAML configurations
* Check the :doc:`benchmarks` section for performance optimization tips