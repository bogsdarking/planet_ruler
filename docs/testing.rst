Testing
=======

Planet Ruler includes a comprehensive test suite with 208+ tests covering all functionality.

Test Categories
--------------

Unit Tests
~~~~~~~~~

* **Geometry functions**: Mathematical calculations and coordinate transforms
* **Image processing**: Loading, gradient detection, and limb smoothing
* **Cost functions**: Parameter optimization and loss function evaluation
* **Configuration management**: YAML loading and parameter validation

Integration Tests
~~~~~~~~~~~~~~~~

* **Real planetary data**: Earth ISS, Pluto New Horizons, Saturn Cassini scenarios
* **End-to-end workflows**: Complete observation processing pipelines
* **Configuration consistency**: Parameter bounds and initial value validation
* **Multi-planetary scenarios**: Cross-scenario parameter scaling validation

Property-Based Tests
~~~~~~~~~~~~~~~~~~~

Using Hypothesis framework for mathematical function validation:

* **Geometry invariants**: Physical constraints and mathematical relationships
* **Parameter bounds**: Valid ranges for planetary and camera parameters  
* **Coordinate transforms**: Invertibility and numerical stability
* **Edge case handling**: Extreme values and boundary conditions

Performance Benchmarks
~~~~~~~~~~~~~~~~~~~~~~

* **21 benchmark tests**: Function execution timing from nanoseconds to milliseconds
* **Memory usage**: Profiling large image processing operations
* **Optimization performance**: Convergence rates and iteration counts
* **Scaling analysis**: Performance vs. image size and parameter complexity

Running Tests
------------

Basic Test Execution
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Run all tests
   pytest
   
   # Run with coverage report
   pytest --cov=planet_ruler --cov-report=html
   
   # Run specific test categories
   pytest tests/test_geometry.py -v
   pytest -m integration tests/ -v
   pytest -m benchmark tests/ -v

Detailed Test Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Run with detailed output
   pytest -v --tb=short
   
   # Run performance benchmarks
   pytest --benchmark-only --benchmark-sort=mean
   
   # Run property-based tests with more examples
   pytest tests/test_property_based.py --hypothesis-max-examples=1000
   
   # Run integration tests only
   pytest tests/test_integration_real_data.py -v

Test Environment Setup
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Install test dependencies
   pip install -r requirements-test.txt
   
   # Or install with development extras
   pip install -e ".[dev]"
   
   # Verify test environment
   python -m pytest --version
   python -m pytest --markers

Test Coverage Analysis
---------------------

Current Coverage
~~~~~~~~~~~~~~~

Planet Ruler maintains >95% test coverage across all modules:

* **planet_ruler.geometry**: 100% line coverage, 98% branch coverage
* **planet_ruler.image**: 96% line coverage, 94% branch coverage  
* **planet_ruler.observation**: 98% line coverage, 95% branch coverage
* **planet_ruler.fit**: 100% line coverage, 97% branch coverage
* **planet_ruler.plot**: 92% line coverage, 89% branch coverage
* **planet_ruler.demo**: 95% line coverage, 93% branch coverage

Coverage Reports
~~~~~~~~~~~~~~~

.. code-block:: bash

   # Generate HTML coverage report
   pytest --cov=planet_ruler --cov-report=html
   open htmlcov/index.html  # View in browser
   
   # Terminal coverage report
   pytest --cov=planet_ruler --cov-report=term-missing
   
   # XML coverage for CI
   pytest --cov=planet_ruler --cov-report=xml

Integration Test Details
-----------------------

Real Mission Data Tests
~~~~~~~~~~~~~~~~~~~~~~

**Earth ISS Scenario:**

.. code-block:: python

   def test_earth_iss_horizon_calculation(earth_iss_config):
       params = earth_iss_config["init_parameter_values"]
       horizon_dist = geom.horizon_distance(r=params["r"], h=params["h"])
       # ISS horizon should be ~2600 km
       assert 2500000 < horizon_dist < 2700000

**Pluto New Horizons Scenario:**

.. code-block:: python

   def test_pluto_new_horizons_scenario(pluto_config):
       params = pluto_config["init_parameter_values"]
       limb_angle = geom.limb_camera_angle(r=params["r"], h=params["h"])
       # From 18M km, significant angle due to camera geometry
       assert 1.0 < limb_angle < 2.0

**Saturn Cassini Scenario:**

.. code-block:: python

   def test_saturn_cassini_scenario(saturn_config):
       params = saturn_config["init_parameter_values"]
       horizon_dist = geom.horizon_distance(r=params["r"], h=params["h"])
       # From 1.2B km distance, horizon should be substantial
       assert horizon_dist > 10000000

Configuration Validation Tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def test_parameter_limit_consistency():
       """Test that initial values are within specified limits."""
       for config_file in config_files:
           config = yaml.safe_load(config_file)
           init_vals = config["init_parameter_values"]
           limits = config["parameter_limits"]
           
           for param in config["free_parameters"]:
               init_val = init_vals[param]
               low, high = limits[param]
               assert low <= init_val <= high

Performance Benchmarks
---------------------

Benchmark Categories
~~~~~~~~~~~~~~~~~~~

**Mathematical Functions:**

* ``horizon_distance``: ~50 ns per call
* ``limb_camera_angle``: ~75 ns per call
* ``limb_arc``: ~2.5 ms for 1000x600 image

**Image Processing:**

* ``load_image``: ~15 ms for 2MP image
* ``gradient_break``: ~45 ms for 2MP image
* ``smooth_limb``: ~1.2 ms for 1000-pixel limb

**Optimization:**

* ``CostFunction.cost``: ~3.8 ms per evaluation
* ``differential_evolution``: ~30 seconds for typical fit
* ``parameter_uncertainty``: ~2.1 ms for posterior analysis

Running Benchmarks
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Run all benchmarks
   pytest tests/test_benchmarks.py --benchmark-only
   
   # Sort by execution time
   pytest --benchmark-only --benchmark-sort=mean
   
   # Save benchmark results
   pytest --benchmark-only --benchmark-json=benchmark_results.json
   
   # Compare with previous results
   pytest --benchmark-only --benchmark-compare=baseline.json

Custom Test Fixtures
-------------------

Planetary Configuration Fixtures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   @pytest.fixture
   def earth_iss_config():
       """Load Earth ISS configuration."""
       config_path = Path(__file__).parent.parent / "config/earth_iss_1.yaml"
       with open(config_path, 'r') as f:
           return yaml.safe_load(f)
   
   @pytest.fixture
   def synthetic_planet_image():
       """Create synthetic planet image for testing."""
       height, width = 400, 600
       image = np.zeros((height, width, 3), dtype='uint8')
       # Create horizon at y=200
       image[:200, :, :] = 30    # Space
       image[200:, :, :] = 180   # Planet
       return image

Mock Objects and Patches
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   @patch('PIL.Image.open')
   def test_image_processing_integration(mock_image_open):
       """Test image processing with mocked PIL."""
       mock_img = Mock()
       mock_image_open.return_value = mock_img
       
       with patch('numpy.array', return_value=synthetic_image):
           loaded_image = img.load_image("fake_planet.jpg")
           # Test processing...

Continuous Integration Testing
-----------------------------

GitHub Actions Workflow
~~~~~~~~~~~~~~~~~~~~~~

Planet Ruler uses GitHub Actions for automated testing across:

* **Python versions**: 3.8, 3.9, 3.10, 3.11
* **Operating systems**: Ubuntu, macOS, Windows  
* **Dependencies**: Core and optional (PyTorch, Segment Anything)

**CI Workflow Features:**

* Dependency installation and validation
* Full test suite execution with coverage
* Benchmark regression testing  
* Code quality checks (Black, flake8)
* Documentation building
* Artifact collection (coverage reports, benchmarks)

Test Workflow Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   - name: Run tests
     run: |
       python -m pytest tests/ \
         --cov=planet_ruler \
         --cov-report=xml \
         --benchmark-skip \
         -v
   
   - name: Run benchmarks
     run: |
       python -m pytest tests/test_benchmarks.py \
         --benchmark-only \
         --benchmark-json=benchmark_results.json

Writing New Tests
----------------

Test Organization
~~~~~~~~~~~~~~~~

Follow the existing test structure:

.. code-block::

   tests/
   ├── test_geometry.py           # Unit tests for geometry module
   ├── test_image.py             # Image processing tests
   ├── test_observation.py       # Observation class tests
   ├── test_fit.py              # Fitting and optimization tests
   ├── test_plot.py             # Plotting function tests
   ├── test_demo.py             # Demo configuration tests
   ├── test_integration_real_data.py  # Integration tests
   ├── test_property_based.py    # Hypothesis-based tests
   └── test_benchmarks.py        # Performance benchmarks

Test Function Naming
~~~~~~~~~~~~~~~~~~~

Use descriptive test names that indicate:

* What is being tested
* What conditions/inputs are used  
* What behavior is expected

.. code-block:: python

   def test_horizon_distance_earth_iss_altitude():
       """Test horizon distance calculation for Earth from ISS altitude."""
       
   def test_limb_arc_handles_invalid_parameters():
       """Test limb_arc raises appropriate errors for invalid inputs."""
       
   def test_cost_function_minimizes_with_perfect_parameters():
       """Test cost function returns low cost with perfect parameters."""

Property-Based Test Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from hypothesis import given, strategies as st
   
   @given(
       radius=st.floats(min_value=1e5, max_value=1e8),
       altitude=st.floats(min_value=1e3, max_value=1e9)
   )
   def test_horizon_distance_increases_with_altitude(radius, altitude):
       """Test horizon distance increases monotonically with altitude."""
       h1 = geom.horizon_distance(radius, altitude)
       h2 = geom.horizon_distance(radius, altitude * 1.1)
       assert h2 > h1

Adding Benchmarks
~~~~~~~~~~~~~~~~

.. code-block:: python

   def test_limb_arc_performance(benchmark):
       """Benchmark limb_arc function performance."""
       result = benchmark(
           geom.limb_arc,
           r=6371000, h=418000, n_pix_x=1000, n_pix_y=600,
           f=0.024, w=0.036, theta_x=0, theta_y=0, theta_z=0
       )
       assert len(result) == 1000  # Verify correct output

Best Practices
-------------

Test Design Principles
~~~~~~~~~~~~~~~~~~~~~

1. **Independence**: Tests should not depend on each other
2. **Determinism**: Use fixed seeds for reproducible results
3. **Clear assertions**: Test one concept per test function
4. **Comprehensive coverage**: Test normal cases, edge cases, and error conditions
5. **Performance awareness**: Use benchmarks for performance-critical functions

Mock Strategy
~~~~~~~~~~~~

* Mock external dependencies (PIL, PyTorch models)
* Use synthetic data for consistent test results
* Patch file I/O operations for isolated testing
* Mock expensive computations in unit tests

Error Testing
~~~~~~~~~~~~

.. code-block:: python

   def test_limb_arc_raises_on_negative_radius():
       """Test limb_arc raises ValueError for negative radius."""
       with pytest.raises(ValueError, match="radius must be positive"):
           geom.limb_arc(r=-1000, h=400000, n_pix_x=100, n_pix_y=100,
                        f=0.024, w=0.036, theta_x=0, theta_y=0, theta_z=0)

Running Tests Locally
--------------------

Development Workflow
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Install in development mode
   pip install -e ".[dev]"
   
   # Run tests during development
   pytest tests/test_geometry.py::test_horizon_distance_basic -v
   
   # Run tests with file watching
   pytest-watch tests/
   
   # Run specific test categories
   pytest -m "not benchmark" tests/ -v     # Skip benchmarks
   pytest -m integration tests/ -v         # Integration tests only

Test Debugging
~~~~~~~~~~~~~

.. code-block:: bash

   # Run with detailed output and stop on first failure
   pytest tests/ -vvv -x
   
   # Debug specific test
   pytest tests/test_fit.py::test_uncertainty_calculation -vvv --tb=long
   
   # Run with Python debugger
   pytest tests/test_observation.py::test_limb_observation_workflow --pdb

Test Quality Metrics
-------------------

The Planet Ruler test suite maintains high quality standards:

* **Test Coverage**: >95% line coverage across all modules
* **Execution Time**: Full test suite completes in <2 minutes
* **Reliability**: 100% pass rate in CI across all supported environments
* **Maintainability**: Clear test organization and comprehensive documentation
* **Performance Tracking**: Continuous benchmarking prevents performance regressions

Contributing to Tests
-------------------

When contributing new features or bug fixes:

1. **Add corresponding tests**: New functionality requires test coverage
2. **Update existing tests**: Ensure changes don't break existing functionality  
3. **Include benchmarks**: Performance-critical functions need benchmark coverage
4. **Test edge cases**: Consider boundary conditions and error scenarios
5. **Update documentation**: Keep test documentation current

See :doc:`contributing` for detailed contribution guidelines.