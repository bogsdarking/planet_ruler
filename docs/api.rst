API Reference
=============

This section provides detailed documentation for all Planet Ruler modules and functions.

Core Modules
-----------

Geometry Module
~~~~~~~~~~~~~~

.. automodule:: planet_ruler.geometry
   :members:
   :undoc-members:
   :show-inheritance:

The geometry module contains fundamental mathematical functions for planetary calculations:

* **Horizon calculations**: `horizon_distance`, `limb_camera_angle`
* **Camera optics**: `focal_length`, `detector_size`, `field_of_view`
* **Coordinate transforms**: `intrinsic_transform`, `extrinsic_transform`
* **Limb arc modeling**: `limb_arc`, `limb_arc_sample` - generates theoretical limb curves
* **Rotation matrices**: `get_rotation_matrix` - Euler angle to rotation matrix conversion

Image Processing Module
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: planet_ruler.image
   :members:
   :undoc-members:
   :show-inheritance:

The image module handles computer vision tasks:

* **Image loading**: `load_image` - loads and validates image files
* **Gradient-field detection**: `gradient_field` - automated limb detection using directional blur and flux analysis
* **Legacy detection**: `gradient_break` - simpler gradient-based detection
* **Image segmentation**: `MaskSegmenter` class with pluggable backends including Segment Anything integration (optional)
* **Limb processing**: `smooth_limb`, `fill_nans` - post-processing operations
* **Interpolation**: `bilinear_interpolate` - sub-pixel image sampling

Observation Module
~~~~~~~~~~~~~~~~~

.. automodule:: planet_ruler.observation
   :members:
   :undoc-members:
   :show-inheritance:

The observation module provides high-level interfaces:

* **PlanetObservation**: Base class for planetary observations
* **LimbObservation**: Complete workflow for limb-based radius fitting (default: manual annotation)
* **Results processing**: `unpack_diff_evol_posteriors`, `package_results`

Annotation Module
~~~~~~~~~~~~~~~~

.. automodule:: planet_ruler.annotate
   :members:
   :undoc-members:
   :show-inheritance:

The annotation module provides interactive manual limb detection:

* **TkLimbAnnotator**: Interactive GUI for manual horizon point selection
* **Point management**: Add, remove, and edit limb points with mouse interaction
* **Image controls**: Zoom, pan, and contrast adjustment for precise annotation
* **File I/O**: Save and load annotation sessions to/from JSON files
* **Target generation**: Convert point annotations to dense limb arrays

Camera Module
~~~~~~~~~~~~~

.. automodule:: planet_ruler.camera
   :members:
   :undoc-members:
   :show-inheritance:

The camera module provides automatic camera parameter extraction from image EXIF data:

* **EXIF processing**: `extract_exif`, `get_camera_model`, `get_focal_length_mm`
* **Camera database**: Comprehensive database of sensor dimensions for phones, DSLRs, mirrorless cameras
* **Parameter extraction**: `extract_camera_parameters` - automatic detection of focal length and sensor size
* **GPS integration**: `get_gps_altitude` - extract altitude from GPS EXIF data
* **Configuration generation**: `create_config_from_image` - complete auto-config from any camera image
* **Planet radii**: `get_initial_radius` - perturbed initial radius guesses for robust optimization

Fitting and Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: planet_ruler.fit
   :members:
   :undoc-members:
   :show-inheritance:

The fit module handles parameter optimization:

* **CostFunction**: Configurable cost function for parameter fitting
* **Parameter handling**: `pack_parameters`, `unpack_parameters`
* **Uncertainty analysis**: `calculate_parameter_uncertainty`, `format_parameter_result`

Uncertainty Module
~~~~~~~~~~~~~~~~~

.. automodule:: planet_ruler.uncertainty
   :members:
   :undoc-members:
   :show-inheritance:

The uncertainty module provides comprehensive parameter uncertainty estimation:

* **Multiple methods**: Population spread, Hessian approximation, profile likelihood, bootstrap
* **Auto-selection**: Automatically chooses best method based on minimizer used
* **Confidence intervals**: Configurable confidence levels (68%, 95%, 99%, etc.)
* **Method-specific details**: Additional diagnostics for each uncertainty estimation method

**Available Uncertainty Methods:**

* **population**: Uses parameter spread from differential-evolution population (fast, exact for DE)
* **hessian**: Inverse Hessian approximation at optimum (fast, approximate, works with all minimizers)
* **profile**: Profile likelihood re-optimization (slow, most accurate, works with all minimizers)
* **bootstrap**: Multiple fits with different seeds (slow, robust, partially implemented)
* **auto**: Automatically selects population for DE, hessian for others

Plotting and Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: planet_ruler.plot
   :members:
   :undoc-members:
   :show-inheritance:

The plot module provides visualization functions:

* **Image display**: `plot_image` - displays images with optional gradient overlay
* **Limb visualization**: `plot_limb` - plots detected horizon curves
* **3D solutions**: `plot_3d_solution` - 3D parameter space visualization
* **Topography**: `plot_topography` - terrain height visualization
* **Analysis plots**: `plot_diff_evol_posteriors` - differential evolution posterior distributions
* **Full limb plots**: `plot_full_limb` - complete limb visualization with uncertainty
* **Segmentation plots**: `plot_segmentation_masks` - image segmentation mask visualization

Validation Module
~~~~~~~~~~~~~~~~~

.. automodule:: planet_ruler.validation
   :members:
   :undoc-members:
   :show-inheritance:

The validation module provides configuration validation:

* **Config validation**: `validate_limb_config` - checks parameter limits, theta ranges, and consistency
* **Parameter bounds**: Validates initial values are within specified limits
* **Optimization warnings**: Identifies potential issues with tight constraints that could affect convergence

Command-Line Interface
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: planet_ruler.cli
   :members:
   :undoc-members:
   :show-inheritance:

The CLI module provides command-line access to planet_ruler:

* **Main interface**: `main` - primary CLI entry point
* **Measurement**: `measure_command` - automated radius measurement from images
* **Demo functionality**: `demo_command` - run example scenarios
* **Configuration**: `load_config`, `list_command` - manage configuration files

Dashboard Module
~~~~~~~~~~~~~~~~

.. automodule:: planet_ruler.dashboard
   :members:
   :undoc-members:
   :show-inheritance:

The dashboard module provides real-time optimization monitoring:

* **FitDashboard**: Live progress dashboard with adaptive refresh rate
* **OutputCapture**: Capture stdout/stderr for display in dashboard
* **Adaptive refresh**: Automatically adjusts update frequency (20Hz → 2Hz) based on optimization activity
* **Configurable display**: Adjust width, message slots, and display time

Demo and Configuration
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: planet_ruler.demo
   :members:
   :undoc-members:
   :show-inheritance:

The demo module manages example scenarios:

* **Parameter loading**: `load_demo_parameters` - loads predefined planetary scenarios
* **Interactive widgets**: `make_dropdown` - creates UI elements for Jupyter notebooks

Function Categories
------------------

Mathematical Functions
~~~~~~~~~~~~~~~~~~~~~

Core geometric calculations:

.. autosummary::
   :toctree: generated/

   planet_ruler.geometry.horizon_distance
   planet_ruler.geometry.limb_camera_angle
   planet_ruler.geometry.focal_length
   planet_ruler.geometry.detector_size
   planet_ruler.geometry.field_of_view
   planet_ruler.geometry.get_rotation_matrix

Image Processing Functions
~~~~~~~~~~~~~~~~~~~~~~~~~

Computer vision and image analysis:

.. autosummary::
   :toctree: generated/

   planet_ruler.image.load_image
   planet_ruler.image.gradient_break
   planet_ruler.image.smooth_limb
   planet_ruler.image.fill_nans
   planet_ruler.image.bilinear_interpolate
   planet_ruler.image.gradient_field

Manual Annotation Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Interactive limb detection and annotation:

.. autosummary::
   :toctree: generated/

   planet_ruler.annotate.TkLimbAnnotator
   planet_ruler.annotate.TkLimbAnnotator.run
   planet_ruler.annotate.TkLimbAnnotator.get_target
   planet_ruler.annotate.TkLimbAnnotator.generate_target
   planet_ruler.annotate.TkLimbAnnotator.save_points
   planet_ruler.annotate.TkLimbAnnotator.load_points

Camera Parameter Functions
~~~~~~~~~~~~~~~~~~~~~~~~~

Automatic camera parameter extraction:

.. autosummary::
   :toctree: generated/

   planet_ruler.camera.extract_exif
   planet_ruler.camera.get_camera_model
   planet_ruler.camera.get_focal_length_mm
   planet_ruler.camera.get_focal_length_35mm_equiv
   planet_ruler.camera.extract_camera_parameters
   planet_ruler.camera.get_gps_altitude
   planet_ruler.camera.create_config_from_image
   planet_ruler.camera.get_initial_radius

Coordinate Transform Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Camera geometry and projections:

.. autosummary::
   :toctree: generated/

   planet_ruler.geometry.intrinsic_transform
   planet_ruler.geometry.extrinsic_transform
   planet_ruler.geometry.limb_arc
   planet_ruler.geometry.limb_arc_sample

Optimization Functions
~~~~~~~~~~~~~~~~~~~~

Parameter fitting and uncertainty:

.. autosummary::
   :toctree: generated/

   planet_ruler.fit.pack_parameters
   planet_ruler.fit.unpack_parameters
   planet_ruler.fit.calculate_parameter_uncertainty
   planet_ruler.fit.format_parameter_result
   planet_ruler.uncertainty.calculate_parameter_uncertainty
   planet_ruler.uncertainty._uncertainty_from_population
   planet_ruler.uncertainty._uncertainty_from_hessian
   planet_ruler.uncertainty._uncertainty_from_profile
   planet_ruler.uncertainty._uncertainty_from_bootstrap

Validation Functions
~~~~~~~~~~~~~~~~~~~

Configuration and parameter validation:

.. autosummary::
   :toctree: generated/

   planet_ruler.validation.validate_limb_config

Command-Line Functions
~~~~~~~~~~~~~~~~~~~~~

CLI interface and commands:

.. autosummary::
   :toctree: generated/

   planet_ruler.cli.main
   planet_ruler.cli.measure_command
   planet_ruler.cli.demo_command
   planet_ruler.cli.list_command
   planet_ruler.cli.load_config

Visualization Functions
~~~~~~~~~~~~~~~~~~~~~

Plotting and display:

.. autosummary::
   :toctree: generated/

   planet_ruler.plot.plot_image
   planet_ruler.plot.plot_limb
   planet_ruler.plot.plot_3d_solution
   planet_ruler.plot.plot_topography
   planet_ruler.plot.plot_diff_evol_posteriors
   planet_ruler.plot.plot_full_limb
   planet_ruler.plot.plot_segmentation_masks

Classes
-------

CostFunction
~~~~~~~~~~~

.. autoclass:: planet_ruler.fit.CostFunction
   :members:
   :undoc-members:
   :show-inheritance:

Cost function wrapper for optimization problems. Supports multiple loss functions (L1, L2, log-L1) and flexible parameter handling.

PlanetObservation
~~~~~~~~~~~~~~~~

.. autoclass:: planet_ruler.observation.PlanetObservation
   :members:
   :undoc-members:
   :show-inheritance:

Base class for planetary observations. Handles image loading and basic plotting functionality.

LimbObservation
~~~~~~~~~~~~~~

.. autoclass:: planet_ruler.observation.LimbObservation
   :members:
   :undoc-members:
   :show-inheritance:

Complete workflow class for limb-based planetary radius determination. Default detection method is interactive manual annotation. Includes horizon detection, limb fitting with multi-resolution optimization, and comprehensive uncertainty analysis.

**Detection Methods Available:**

* **manual** (default): Interactive GUI for precise point selection
* **gradient-field**: Automated detection using gradient flow analysis with directional blur
* **segmentation**: AI-powered automatic detection (requires PyTorch + Segment Anything)

**Key Features:**

* Multi-resolution optimization with coarse-to-fine refinement
* Multiple uncertainty estimation methods (population spread, Hessian, profile likelihood)
* Flexible cost functions (gradient-field flux, L1, L2, log-L1)
* Support for multiple minimizers (differential-evolution, dual-annealing, basinhopping)

MaskSegmenter
~~~~~~~~~~~~

.. autoclass:: planet_ruler.image.MaskSegmenter
   :members:
   :undoc-members:
   :show-inheritance:

Advanced image segmentation with pluggable backends. Supports Segment Anything model (optional dependency) and custom segmentation functions. Provides automated mask generation, interactive classification, and limb extraction from complex images.

TkLimbAnnotator
~~~~~~~~~~~~~~

.. autoclass:: planet_ruler.annotate.TkLimbAnnotator
   :members:
   :undoc-members:
   :show-inheritance:

Interactive GUI for manual limb annotation. Provides precise user control over horizon detection with no additional dependencies required.

**Key Features:**

* **Interactive point selection**: Left click to add points, right click to remove
* **Zoom and pan**: Mouse wheel zoom, drag to pan for precise annotation
* **Contrast adjustment**: Arrow keys to adjust image stretch/brightness
* **Session management**: Save ('s') and load ('l') annotation sessions
* **Target generation**: Convert sparse points to dense limb arrays

Constants and Configuration
--------------------------

Camera Database
~~~~~~~~~~~~~~

The camera module includes a comprehensive database of sensor dimensions for automatic parameter extraction:

* **Smartphones**: iPhone models (iPhone 11-14 series), Samsung Galaxy, Google Pixel, etc.
* **Point-and-shoot**: Canon PowerShot series, Nikon Coolpix, Sony Cyber-shot
* **DSLRs**: Canon EOS series, Nikon D series, Sony Alpha
* **Mirrorless**: Sony ILCE series, Canon EOS R, Nikon Z series
* **Generic sensors**: Common sensor sizes (1/2.3", 1/1.7", APS-C, Full Frame)

Planet Radius Database
~~~~~~~~~~~~~~~~~~~~~

Built-in planetary radii for initial optimization guesses (automatically perturbed):

* **Earth**: 6,371,000 m
* **Mars**: 3,389,500 m
* **Jupiter**: 69,911,000 m
* **Saturn**: 58,232,000 m
* **Moon**: 1,737,400 m
* **Pluto**: 1,188,300 m
* And others...

Default Parameters
~~~~~~~~~~~~~~~~~

The following default values are used throughout Planet Ruler:

* **ISS altitude**: 418,000 m
* **Image processing window**: 21 pixels
* **Optimization tolerance**: 1e-6
* **Maximum iterations**: 1000
* **Perturbation factor**: 50% (for initial radius guessing)

File Formats
~~~~~~~~~~~

Supported file formats:

* **Images**: JPEG, PNG, TIFF, BMP (with EXIF support)
* **Configuration**: YAML (.yaml, .yml), JSON (.json)
* **Data output**: CSV, JSON, pickle
* **Annotation sessions**: JSON

Error Handling
-------------

Planet Ruler raises specific exceptions for different error conditions:

* **ValueError**: Invalid parameter values or configurations
* **FileNotFoundError**: Missing image or configuration files
* **ImportError**: Missing optional dependencies (e.g., PyTorch for Segment Anything)
* **RuntimeError**: Optimization convergence failures
* **AssertionError**: Configuration validation failures (when strict=True)

See individual function documentation for specific error conditions and handling recommendations.