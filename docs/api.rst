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
* **Limb arc modeling**: `limb_arc` - generates theoretical limb curves

Image Processing Module
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: planet_ruler.image
   :members:
   :undoc-members:
   :show-inheritance:

The image module handles computer vision tasks:

* **Image loading**: `load_image` - loads and validates image files
* **Horizon detection**: `gradient_break` - detects horizon using gradient analysis
* **Image segmentation**: `ImageSegmentation` class with Segment Anything integration (optional)
* **Limb processing**: `smooth_limb`, `fill_nans` - post-processing operations

Observation Module
~~~~~~~~~~~~~~~~~

.. automodule:: planet_ruler.observation
   :members:
   :undoc-members:
   :show-inheritance:

The observation module provides high-level interfaces:

* **PlanetObservation**: Base class for planetary observations
* **LimbObservation**: Complete workflow for limb-based radius fitting (default: manual annotation)
* **Visualization**: `plot_full_limb`, `plot_segmentation_masks`
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

Image Processing Functions
~~~~~~~~~~~~~~~~~~~~~~~~~

Computer vision and image analysis:

.. autosummary::
   :toctree: generated/

   planet_ruler.image.load_image
   planet_ruler.image.gradient_break
   planet_ruler.image.smooth_limb
   planet_ruler.image.fill_nans

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

Coordinate Transform Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Camera geometry and projections:

.. autosummary::
   :toctree: generated/

   planet_ruler.geometry.intrinsic_transform
   planet_ruler.geometry.extrinsic_transform
   planet_ruler.geometry.limb_arc

Optimization Functions
~~~~~~~~~~~~~~~~~~~~

Parameter fitting and uncertainty:

.. autosummary::
   :toctree: generated/

   planet_ruler.fit.pack_parameters
   planet_ruler.fit.unpack_parameters
   planet_ruler.fit.calculate_parameter_uncertainty
   planet_ruler.fit.format_parameter_result

Visualization Functions
~~~~~~~~~~~~~~~~~~~~~

Plotting and display:

.. autosummary::
   :toctree: generated/

   planet_ruler.plot.plot_image
   planet_ruler.plot.plot_limb
   planet_ruler.plot.plot_3d_solution
   planet_ruler.plot.plot_topography

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

Complete workflow class for limb-based planetary radius determination. Default detection method is interactive manual annotation. Includes horizon detection, limb fitting, and uncertainty analysis.

**Detection Methods Available:**

* **manual** (default): Interactive GUI for precise point selection
* **segmentation**: AI-powered automatic detection (requires PyTorch + Segment Anything)
* **gradient-break**: Legacy gradient-based detection

ImageSegmentation
~~~~~~~~~~~~~~~~

.. autoclass:: planet_ruler.image.ImageSegmentation
   :members:
   :undoc-members:
   :show-inheritance:

Advanced image segmentation using Segment Anything model (optional dependency). Provides automated mask generation and limb extraction from complex images.

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

Default Parameters
~~~~~~~~~~~~~~~~~

The following default values are used throughout Planet Ruler:

* **Earth radius**: 6,371,000 m
* **ISS altitude**: 418,000 m  
* **Image processing window**: 21 pixels
* **Optimization tolerance**: 1e-6
* **Maximum iterations**: 1000

File Formats
~~~~~~~~~~~

Supported file formats:

* **Images**: JPEG, PNG, TIFF, BMP
* **Configuration**: YAML (.yaml, .yml)
* **Data output**: CSV, JSON, pickle

Error Handling
-------------

Planet Ruler raises specific exceptions for different error conditions:

* **ValueError**: Invalid parameter values or configurations
* **FileNotFoundError**: Missing image or configuration files
* **ImportError**: Missing optional dependencies (e.g., PyTorch for Segment Anything)
* **RuntimeError**: Optimization convergence failures

See individual function documentation for specific error conditions and handling recommendations.