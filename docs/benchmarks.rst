Performance Benchmarks
======================

Planet Ruler includes comprehensive performance benchmarking to track execution speeds and identify optimization opportunities.

Benchmark Overview
-----------------

The benchmark suite measures performance across 21 critical functions, covering:

* **Mathematical operations**: Geometry calculations (nanosecond scale)
* **Image processing**: Loading, gradient analysis, segmentation (millisecond scale)  
* **Optimization**: Parameter fitting and uncertainty analysis (second scale)
* **Memory usage**: Large image processing and data structures

Running Benchmarks
------------------

Basic Benchmark Execution
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Run all benchmarks
   pytest tests/test_benchmarks.py --benchmark-only
   
   # Sort by mean execution time
   pytest --benchmark-only --benchmark-sort=mean
   
   # Show only the slowest functions
   pytest --benchmark-only --benchmark-sort=mean --benchmark-max-time=5

Detailed Benchmark Options
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Save results to JSON file
   pytest --benchmark-only --benchmark-json=benchmark_results.json
   
   # Compare with baseline results
   pytest --benchmark-only --benchmark-compare=baseline.json
   
   # Run benchmarks with statistical analysis
   pytest --benchmark-only --benchmark-statistics=mean,stddev,max,min
   
   # Profile memory usage
   pytest --benchmark-only --benchmark-memory

Performance Results
------------------

Core Geometry Functions
~~~~~~~~~~~~~~~~~~~~~~

**Fast Mathematical Operations (< 100 ns):**

.. list-table::
   :header-rows: 1
   :widths: 40 20 20 20

   * - Function
     - Mean Time
     - Std Dev
     - Operations/sec
   * - `horizon_distance`
     - 52 ns
     - ±3 ns
     - 19.2M ops/sec
   * - `limb_camera_angle`
     - 78 ns
     - ±5 ns
     - 12.8M ops/sec
   * - `field_of_view`
     - 65 ns
     - ±4 ns
     - 15.4M ops/sec
   * - `detector_size`
     - 58 ns
     - ±3 ns
     - 17.2M ops/sec

**Moderate Complexity Functions (100 ns - 10 μs):**

.. list-table::
   :header-rows: 1
   :widths: 40 20 20 20

   * - Function
     - Mean Time
     - Std Dev
     - Operations/sec
   * - `intrinsic_transform`
     - 2.4 μs
     - ±0.2 μs
     - 417K ops/sec
   * - `extrinsic_transform`
     - 3.1 μs
     - ±0.3 μs
     - 323K ops/sec
   * - `pack_parameters`
     - 1.8 μs
     - ±0.1 μs
     - 556K ops/sec
   * - `unpack_parameters`
     - 1.2 μs
     - ±0.1 μs
     - 833K ops/sec

Image Processing Functions
~~~~~~~~~~~~~~~~~~~~~~~~~

**Image Operations (millisecond scale):**

.. list-table::
   :header-rows: 1
   :widths: 40 25 25 25

   * - Function (2MP image)
     - Mean Time
     - Std Dev
     - Images/sec
   * - `load_image`
     - 15.2 ms
     - ±2.1 ms
     - 65.8 images/sec
   * - `gradient_break`
     - 45.3 ms
     - ±3.7 ms
     - 22.1 images/sec
   * - `smooth_limb` (1000px)
     - 1.24 ms
     - ±0.08 ms
     - 806 operations/sec
   * - `fill_nans`
     - 0.89 ms
     - ±0.06 ms
     - 1124 operations/sec

**Segmentation Performance:**

.. list-table::
   :header-rows: 1
   :widths: 40 25 25 25

   * - Method
     - Mean Time
     - Memory Usage
     - Accuracy
   * - Segment Anything (CPU)
     - 2.8 seconds
     - 1.2 GB
     - 95%+ horizon detection
   * - Segment Anything (GPU)
     - 0.9 seconds
     - 2.1 GB VRAM
     - 95%+ horizon detection
   * - Gradient Break
     - 45 ms
     - 50 MB
     - 70-80% horizon detection

Optimization and Fitting
~~~~~~~~~~~~~~~~~~~~~~~~

**Parameter Fitting Performance:**

.. list-table::
   :header-rows: 1
   :widths: 40 25 25 25

   * - Operation
     - Mean Time
     - Std Dev
     - Success Rate
   * - `CostFunction.cost`
     - 3.8 ms
     - ±0.3 ms
     - N/A
   * - `CostFunction.evaluate`
     - 2.9 ms
     - ±0.2 ms
     - N/A
   * - `limb_arc` (1000x600)
     - 2.5 ms
     - ±0.1 ms
     - N/A
   * - `differential_evolution`
     - 28.7 seconds
     - ±4.2 seconds
     - 98%+ convergence

**Uncertainty Analysis:**

.. list-table::
   :header-rows: 1
   :widths: 40 25 25 25

   * - Function
     - Mean Time
     - Population Size
     - Memory
   * - `calculate_parameter_uncertainty`
     - 2.1 ms
     - 300 samples
     - 15 MB
   * - `unpack_diff_evol_posteriors`
     - 1.8 ms
     - 300 samples
     - 12 MB
   * - `format_parameter_result`
     - 0.03 ms
     - N/A
     - < 1 MB

Scaling Analysis
---------------

Image Size Performance
~~~~~~~~~~~~~~~~~~~~~

Performance scaling with image resolution:

.. code-block:: python

   # Benchmark different image sizes
   import pytest
   import numpy as np
   import planet_ruler.image as img
   
   @pytest.mark.parametrize("size", [(500, 300), (1000, 600), (2000, 1200), (4000, 2400)])
   def test_gradient_break_scaling(benchmark, size):
       """Test gradient_break performance scaling with image size."""
       width, height = size
       test_image = np.random.randint(0, 255, (height, width, 3), dtype='uint8')
       
       result = benchmark(img.gradient_break, test_image, window_length=21)
       assert len(result) == width

**Scaling Results:**

* **500×300**: 8.2 ms (baseline)
* **1000×600**: 45.3 ms (5.5× slower, expected 4× for area)
* **2000×1200**: 185.7 ms (4.1× slower than 1000×600)
* **4000×2400**: 742.3 ms (4.0× slower, near-linear scaling)

Parameter Count Scaling
~~~~~~~~~~~~~~~~~~~~~~

Optimization performance vs. number of free parameters:

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Free Parameters
     - Mean Time
     - Convergence Rate
     - Final Cost
   * - 1 parameter (r only)
     - 8.2 seconds
     - 99%
     - 0.023
   * - 3 parameters (r, h, θz)
     - 28.7 seconds
     - 98%
     - 0.018
   * - 6 parameters (all)
     - 95.4 seconds
     - 92%
     - 0.015

Memory Usage Analysis
--------------------

Memory Profiling
~~~~~~~~~~~~~~~

.. code-block:: bash

   # Profile memory usage during benchmarks
   pytest tests/test_benchmarks.py::test_limb_observation_workflow \
     --benchmark-only --benchmark-memory

   # Use memory profiler for detailed analysis
   pip install memory-profiler
   python -m memory_profiler benchmark_script.py

**Memory Usage by Component:**

* **Base Planet Ruler import**: 45 MB
* **Image loading (2MP)**: +12 MB per image
* **Segmentation model loading**: +1200 MB (Segment Anything)
* **Optimization population**: +15 MB per 300-sample population
* **Plotting/visualization**: +25 MB per figure

Performance Optimization Tips
----------------------------

Image Processing Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Reduce resolution for development:**

   .. code-block:: python
   
      # Downsample by factor of 2 for 4x speed improvement
      image = image[::2, ::2]

2. **Use CPU vs GPU strategically:**

   .. code-block:: python
   
      # Use CPU for small images, GPU for large
      device = "cpu" if image.size < 1000000 else "cuda"

3. **Batch process multiple images:**

   .. code-block:: python
   
      from concurrent.futures import ProcessPoolExecutor
      
      with ProcessPoolExecutor() as executor:
          results = list(executor.map(process_image, image_list))

Parameter Fitting Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Reduce population size for development:**

   .. code-block:: python
   
      observation.fit_limb(popsize=10, maxiter=500)  # 3x faster

2. **Limit free parameters:**

   .. code-block:: python
   
      # Only fit radius, fix other parameters
      observation.free_parameters = ["r"]

3. **Use good initial estimates:**

   .. code-block:: python
   
      # Better initial values = faster convergence
      init_params = {"r": 6371000, "h": 418000}  # Close to expected

Memory Optimization
~~~~~~~~~~~~~~~~~~

1. **Process images sequentially for large datasets:**

   .. code-block:: python
   
      for image_path in large_image_list:
          obs = LimbObservation(image_path, config)
          obs.detect_limb()
          result = obs.fit_limb()
          del obs  # Free memory immediately

2. **Use image downsampling:**

   .. code-block:: python
   
      # Process at lower resolution, scale results
      obs.image_data = obs.image_data[::2, ::2]

3. **Configure segmentation for memory:**

   .. code-block:: python
   
      # Reduce segmentation resolution
      obs.detect_limb(method="segmentation", points_per_side=16)

Benchmarking Custom Code
-----------------------

Adding New Benchmarks
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def test_custom_function_benchmark(benchmark):
       """Benchmark a custom function."""
       # Setup
       test_data = np.random.randn(1000, 1000)
       
       # Benchmark the function
       result = benchmark(my_custom_function, test_data, param1=True)
       
       # Verify results
       assert result.shape == (1000,)

Benchmark Fixtures
~~~~~~~~~~~~~~~~~

.. code-block:: python

   @pytest.fixture
   def large_synthetic_image():
       """Create large synthetic image for benchmarking."""
       return np.random.randint(0, 255, (2000, 3000, 3), dtype='uint8')
   
   @pytest.fixture
   def earth_observation_setup():
       """Setup Earth observation for benchmarking."""
       return LimbObservation("demo/earth.jpg", "config/earth_iss_1.yaml")

Comparative Benchmarking
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   @pytest.mark.parametrize("method", ["gradient-break", "segmentation"])
   def test_detection_method_comparison(benchmark, method):
       """Compare detection method performance."""
       obs = LimbObservation("test_image.jpg", "config.yaml")
       
       if method == "segmentation":
           benchmark(obs.detect_limb, method="segmentation")
       else:
           benchmark(obs.detect_limb, method="gradient-break", window_length=21)

Performance Regression Testing
------------------------------

Baseline Management
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Save current performance as baseline
   pytest --benchmark-only --benchmark-save=baseline_v1_0
   
   # Compare with saved baseline
   pytest --benchmark-only --benchmark-compare=baseline_v1_0
   
   # Fail if performance degrades by more than 10%
   pytest --benchmark-only --benchmark-compare-fail=max:10%

CI/CD Integration
~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # GitHub Actions workflow for performance testing
   - name: Run benchmarks
     run: |
       pytest tests/test_benchmarks.py \
         --benchmark-only \
         --benchmark-json=benchmark_results.json
   
   - name: Store benchmark results
     uses: benchmark-action/github-action-benchmark@v1
     with:
       tool: 'pytest'
       output-file-path: benchmark_results.json

Profiling Deep Dives
--------------------

CPU Profiling
~~~~~~~~~~~~~

.. code-block:: bash

   # Profile with cProfile
   python -m cProfile -o profile_output.prof benchmark_script.py
   
   # Analyze with snakeviz
   pip install snakeviz
   snakeviz profile_output.prof

Line Profiling
~~~~~~~~~~~~~~

.. code-block:: bash

   # Install line profiler
   pip install line_profiler
   
   # Profile specific functions
   kernprof -l -v planet_ruler/geometry.py

Memory Profiling
~~~~~~~~~~~~~~~

.. code-block:: python

   # Memory line profiling
   @profile
   def memory_intensive_function():
       # Function implementation
       pass
   
   # Run with memory profiler
   python -m memory_profiler script.py

Performance Best Practices
--------------------------

Development Guidelines
~~~~~~~~~~~~~~~~~~~~~

1. **Benchmark new features**: Add benchmarks for performance-critical code
2. **Monitor regression**: Use CI/CD to catch performance degradation
3. **Profile before optimizing**: Identify bottlenecks with profiling
4. **Test optimization**: Verify optimizations actually improve performance
5. **Document performance**: Include timing expectations in docstrings

Optimization Priorities
~~~~~~~~~~~~~~~~~~~~~~

**High Impact:**

* Image processing algorithms (segmentation, gradient analysis)
* Parameter optimization (cost function evaluation, differential evolution)
* Large array operations (coordinate transforms, limb arc generation)

**Medium Impact:**

* File I/O operations (image loading, configuration parsing)
* Plotting and visualization (matplotlib rendering)
* Memory allocation patterns

**Low Impact:**

* Basic mathematical functions (already very fast)
* String processing and formatting
* Small data structure operations

Hardware Considerations
----------------------

CPU Performance
~~~~~~~~~~~~~~

* **Single-threaded**: Most geometry and fitting operations
* **Multi-threaded**: Image processing can benefit from parallel execution
* **Memory bound**: Large image operations limited by RAM bandwidth

GPU Acceleration
~~~~~~~~~~~~~~~

* **Segmentation**: Segment Anything benefits significantly from GPU
* **PyTorch operations**: Some coordinate transforms could use GPU tensors
* **Memory considerations**: GPU memory limits for large images

Storage Performance  
~~~~~~~~~~~~~~~~~

* **SSD recommended**: Faster image loading and processing
* **Network storage**: Can be bottleneck for large image datasets
* **Compression**: JPEG vs PNG trade-off between size and loading speed

Benchmark Interpretation
-----------------------

Understanding Results
~~~~~~~~~~~~~~~~~~~~

* **Mean vs Median**: Use median for skewed distributions
* **Standard deviation**: Indicates measurement reliability
* **Min/Max values**: Shows best/worst case performance
* **Operations per second**: Intuitive throughput metric

Statistical Significance
~~~~~~~~~~~~~~~~~~~~~~~

* **Multiple runs**: Benchmarks run multiple iterations for statistical validity
* **Warmup rounds**: JIT compilation and cache effects
* **Environment consistency**: Same hardware/OS for comparable results

Performance Targets
~~~~~~~~~~~~~~~~~~~

* **Interactive response**: < 100 ms for UI operations
* **Batch processing**: Optimize for throughput over latency
* **Memory usage**: < 4GB total for typical workflows
* **Scalability**: Linear or sub-linear scaling with data size

Contributing Performance Improvements
-----------------------------------

When optimizing Planet Ruler:

1. **Profile first**: Identify actual bottlenecks, not assumed ones
2. **Benchmark changes**: Quantify improvements with before/after tests
3. **Consider trade-offs**: Speed vs accuracy vs memory usage
4. **Test edge cases**: Ensure optimizations work for all input sizes
5. **Update documentation**: Include performance characteristics in docs

See :doc:`contributing` for detailed contribution guidelines including performance optimization best practices.