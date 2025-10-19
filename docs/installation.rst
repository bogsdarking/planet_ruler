Installation
============

System Requirements
------------------

Planet Ruler requires Python 3.8 or higher and supports the following platforms:

* **Linux**: Ubuntu 18.04+, CentOS 7+, other modern distributions
* **macOS**: 10.15+ (Catalina and newer)  
* **Windows**: 10+ with Python 3.8+

Python Dependencies
------------------

Core dependencies include:

* **NumPy** ≥1.20.0: Numerical computing
* **SciPy** ≥1.7.0: Scientific computing and optimization
* **Matplotlib** ≥3.5.0: Plotting and visualization
* **Pandas** ≥1.3.0: Data manipulation
* **Pillow** ≥8.0.0: Image loading and processing
* **PyYAML** ≥5.4.0: Configuration file parsing
* **tqdm** ≥4.60.0: Progress bars
* **Tkinter**: GUI toolkit for manual annotation (usually included with Python)

Optional dependencies for advanced features:

* **Segment Anything** ≥1.0 (Python 3.8+): AI-powered automatic segmentation (alternative to manual annotation)
* **PyTorch** ≥1.11.0: Required for Segment Anything
* **Seaborn** ≥0.11.0: Statistical plotting
* **IPython** ≥7.16.0: Interactive computing (for notebooks)

Installation Methods
-------------------

From PyPI (Recommended)
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python -m pip install planet-ruler
   
**Optional: For AI segmentation support:**

.. code-block:: bash

   python -m pip install planet-ruler segment-anything torch

From Source
~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/bogsdarking/planet-ruler.git
   cd planet-ruler
   python -m pip install -e .

For development with testing dependencies:

.. code-block:: bash

   git clone https://github.com/bogsdarking/planet-ruler.git
   cd planet-ruler
   python -m pip install -e ".[dev]"

Using Conda
~~~~~~~~~~~

.. code-block:: bash

   conda install -c conda-forge planet-ruler

Virtual Environment Setup
-------------------------

We recommend using a virtual environment:

**Using venv:**

.. code-block:: bash

   python -m venv planet-ruler-env
   source planet-ruler-env/bin/activate  # On Windows: planet-ruler-env\Scripts\activate
   python -m pip install planet-ruler

**Using conda:**

.. code-block:: bash

   conda create -n planet-ruler python=3.9
   conda activate planet-ruler
   python -m pip install planet-ruler

Verification
-----------

Test your installation:

.. code-block:: python

   import planet_ruler.geometry as geom
   import planet_ruler.observation as obs
   
   # Test basic geometry function
   horizon_dist = geom.horizon_distance(r=6371000, h=400000)
   print(f"ISS horizon distance: {horizon_dist/1000:.1f} km")
   
   # Should output: ISS horizon distance: 2290.7 km

Troubleshooting
--------------

Common Installation Issues
~~~~~~~~~~~~~~~~~~~~~~~~~

**ImportError: No module named 'torch'**

The Segment Anything model requires PyTorch. Install with:

.. code-block:: bash

   python -m pip install torch torchvision

**PIL/Pillow conflicts**

If you encounter PIL import errors:

.. code-block:: bash

   python -m pip uninstall PIL Pillow
   python -m pip install Pillow

**NumPy/SciPy build errors**

On some systems, you may need system-level dependencies:

**Ubuntu/Debian:**

.. code-block:: bash

   sudo apt-get install python3-dev libopenblas-dev

**macOS:**

.. code-block:: bash

   brew install openblas

**Windows:**

Install Microsoft Visual C++ Build Tools or use pre-compiled wheels via pip.

Performance Optimization
------------------------

For improved performance, especially with large images:

1. **Use conda-forge NumPy/SciPy** (often includes optimized BLAS):

   .. code-block:: bash

      conda install -c conda-forge numpy scipy

2. **Install OpenMP support** for multi-threading:

   .. code-block:: bash

      conda install -c conda-forge openmp

3. **Use SSD storage** for faster image I/O operations

GPU Support
-----------

While Planet Ruler primarily uses CPU computations, GPU acceleration is available for:

* **Segment Anything model**: Requires CUDA-capable GPU and PyTorch with CUDA support
* **Large-scale batch processing**: Use PyTorch DataLoader with GPU tensors

To enable GPU support:

.. code-block:: bash

   # Install PyTorch with CUDA support
   python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

Next Steps
----------

* Review the :doc:`tutorials` for guided examples
* Explore :doc:`examples` with real planetary data  
* Check the :doc:`api` reference for detailed function documentation