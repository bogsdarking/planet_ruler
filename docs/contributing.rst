Contributing
============

We welcome contributions to Planet Ruler! This guide will help you get started with developing, testing, and submitting improvements.

Getting Started
--------------

Development Setup
~~~~~~~~~~~~~~~~

1. **Fork and clone the repository:**

   .. code-block:: bash

      git clone https://github.com/yourusername/planet-ruler.git
      cd planet-ruler

2. **Create a virtual environment:**

   .. code-block:: bash

      python -m venv planet-ruler-dev
      source planet-ruler-dev/bin/activate  # On Windows: planet-ruler-dev\Scripts\activate

3. **Install in development mode:**

   .. code-block:: bash

      pip install -e ".[dev]"

4. **Install pre-commit hooks:**

   .. code-block:: bash

      pre-commit install

Development Dependencies
~~~~~~~~~~~~~~~~~~~~~~~

The development environment includes:

.. code-block:: bash

   pip install -e ".[dev]"

This installs:

* **Testing**: pytest, pytest-cov, pytest-benchmark, hypothesis
* **Code quality**: black, flake8, mypy, pre-commit
* **Documentation**: sphinx, sphinx-rtd-theme, myst-parser
* **Optional**: segment-anything, torch (for advanced features)

Code Standards
-------------

Code Style
~~~~~~~~~~

Planet Ruler follows PEP 8 with these specifications:

* **Line length**: 127 characters (configured in pyproject.toml)
* **Code formatter**: Black
* **Import sorting**: isort (integrated with Black)
* **Type hints**: Required for all public functions
* **Docstrings**: Google/NumPy style for all public APIs

**Format your code:**

.. code-block:: bash

   # Format all code
   black planet_ruler/ tests/
   
   # Check formatting
   black --check planet_ruler/ tests/

Type Checking
~~~~~~~~~~~~

Use type hints and run mypy for type checking:

.. code-block:: bash

   # Run type checking
   mypy planet_ruler/
   
   # Type check specific file
   mypy planet_ruler/geometry.py

**Type hint examples:**

.. code-block:: python

   from typing import Optional, Union, List, Dict
   import numpy as np
   
   def horizon_distance(r: float, h: float) -> float:
       """Calculate horizon distance with type hints."""
       return math.sqrt(2 * r * h + h * h)
   
   def load_config(config_path: str) -> Dict[str, Union[float, List[str]]]:
       """Load configuration with complex return type."""
       # Implementation

Docstring Style
~~~~~~~~~~~~~~

Use Google-style docstrings with mathematical notation support:

.. code-block:: python

   def limb_camera_angle(r: float, h: float) -> float:
       """
       Calculate limb angle as seen from camera.
       
       The limb angle is determined by the geometry of observation
       from altitude h above a spherical body of radius r.
       
       Args:
           r (float): Planetary radius in meters.
           h (float): Observation altitude above surface in meters.
       
       Returns:
           float: Limb angle in radians.
       
       Raises:
           ValueError: If r or h are negative.
       
       Note:
           The calculation assumes a spherical planet and uses the formula:
           
           .. math::
               \\theta = \\arcsin\\left(\\frac{r}{r + h}\\right)
       
       Example:
           Calculate Earth's limb angle from ISS altitude:
           
           >>> angle = limb_camera_angle(6371000, 418000)
           >>> print(f"Limb angle: {angle:.3f} rad")
           Limb angle: 1.354 rad
       """

Testing Guidelines
-----------------

Test Categories
~~~~~~~~~~~~~~

Write tests in the appropriate category:

1. **Unit tests**: Test individual functions in isolation
2. **Integration tests**: Test complete workflows with real data
3. **Property-based tests**: Test mathematical properties with Hypothesis
4. **Benchmarks**: Performance tests for critical functions

Test Writing
~~~~~~~~~~~

**File organization:**

.. code-block::

   tests/
   ├── test_geometry.py        # Unit tests for geometry module
   ├── test_integration_*.py   # Integration tests
   └── test_benchmarks.py      # Performance benchmarks

**Test function naming:**

.. code-block:: python

   def test_function_name_condition_expected_result():
       """Test that function behaves correctly under specific conditions."""

**Example unit test:**

.. code-block:: python

   import pytest
   import planet_ruler.geometry as geom
   
   def test_horizon_distance_earth_iss_returns_expected_value():
       """Test horizon distance calculation for Earth from ISS altitude."""
       result = geom.horizon_distance(r=6371000, h=418000)
       expected = 2290704.6  # Pre-calculated value
       assert abs(result - expected) < 1.0, f"Expected {expected}, got {result}"
   
   def test_horizon_distance_raises_on_negative_radius():
       """Test horizon distance raises ValueError for negative radius."""
       with pytest.raises(ValueError, match="radius must be positive"):
           geom.horizon_distance(r=-1000, h=418000)

**Integration test example:**

.. code-block:: python

   def test_complete_earth_workflow_produces_reasonable_radius():
       """Test complete Earth analysis workflow."""
       observation = obs.LimbObservation("demo/earth.jpg", "config/earth_iss_1.yaml")
       observation.detect_limb(method="segmentation")
       observation.fit_limb()
       
       radius_result = calculate_parameter_uncertainty(observation, "r", scale_factor=1000)
       assert 6300 < radius_result["value"] < 6400, "Earth radius should be ~6371 km"

**Property-based test example:**

.. code-block:: python

   from hypothesis import given, strategies as st
   
   @given(
       radius=st.floats(min_value=1e5, max_value=1e8),
       altitude=st.floats(min_value=1e3, max_value=1e9)
   )
   def test_horizon_distance_monotonic_in_altitude(radius, altitude):
       """Test that horizon distance increases with altitude."""
       h1 = geom.horizon_distance(radius, altitude)
       h2 = geom.horizon_distance(radius, altitude * 1.1)
       assert h2 > h1, "Horizon distance should increase with altitude"

Running Tests
~~~~~~~~~~~~

.. code-block:: bash

   # Run all tests
   pytest
   
   # Run with coverage
   pytest --cov=planet_ruler --cov-report=html
   
   # Run specific test categories  
   pytest tests/test_geometry.py -v
   pytest -m integration tests/ -v
   pytest --benchmark-only tests/

Performance Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~

When adding features that may impact performance:

1. **Add benchmarks**: Include performance tests for new functions
2. **Profile changes**: Use cProfile to identify bottlenecks
3. **Memory awareness**: Consider memory usage for large images
4. **Document performance**: Include timing expectations

.. code-block:: python

   def test_new_function_performance(benchmark):
       """Benchmark new function performance."""
       test_data = setup_test_data()
       result = benchmark(new_function, test_data)
       assert result is not None

Pull Request Process
-------------------

Before Submitting
~~~~~~~~~~~~~~~~

1. **Run the full test suite:**

   .. code-block:: bash

      pytest --cov=planet_ruler --cov-report=term-missing

2. **Check code formatting:**

   .. code-block:: bash

      black --check planet_ruler/ tests/
      flake8 planet_ruler/ tests/

3. **Run type checking:**

   .. code-block:: bash

      mypy planet_ruler/

4. **Update documentation if needed:**

   .. code-block:: bash

      cd docs/
      make html

Pull Request Guidelines
~~~~~~~~~~~~~~~~~~~~~~

**PR Title Format:**

* ``feat: add uncertainty calculation functions``
* ``fix: resolve numpy compatibility in fit.py``  
* ``docs: update installation instructions for segmentation``
* ``test: add integration tests for Saturn scenario``
* ``perf: optimize limb_arc calculation for large images``

**PR Description Template:**

.. code-block:: markdown

   ## Description
   Brief description of changes and motivation.
   
   ## Type of Change
   - [ ] Bug fix (non-breaking change that fixes an issue)
   - [ ] New feature (non-breaking change that adds functionality)
   - [ ] Breaking change (fix or feature that changes existing API)
   - [ ] Documentation update
   - [ ] Performance improvement
   
   ## Testing
   - [ ] New tests added for changed functionality
   - [ ] All existing tests pass
   - [ ] Integration tests pass with real data
   - [ ] Performance benchmarks added if applicable
   
   ## Checklist
   - [ ] Code follows project style guidelines
   - [ ] Self-review completed
   - [ ] Documentation updated
   - [ ] Type hints added for public APIs

Code Review Process
~~~~~~~~~~~~~~~~~~

All pull requests require:

1. **Automated checks passing**: CI/CD pipeline must be green
2. **Code review**: At least one maintainer approval
3. **Documentation**: Updates for user-facing changes
4. **Tests**: Adequate test coverage for new functionality
5. **Performance**: No significant performance regressions

Common Review Feedback
~~~~~~~~~~~~~~~~~~~~~

* **Add type hints**: All public functions need type annotations
* **Improve docstrings**: Include examples and parameter descriptions
* **Add error handling**: Validate inputs and provide clear error messages
* **Consider edge cases**: Test boundary conditions and error scenarios
* **Performance impact**: Profile and benchmark performance-critical changes

Feature Development
------------------

Adding New Functions
~~~~~~~~~~~~~~~~~~~

1. **Plan the API**: Consider function signature, parameters, return types
2. **Write tests first**: TDD approach with failing tests
3. **Implement functionality**: Focus on correctness before optimization
4. **Add documentation**: Docstrings, examples, and user guide updates
5. **Performance testing**: Add benchmarks for critical functions

**Example new function development:**

.. code-block:: python

   # 1. Write failing test
   def test_new_calculation_returns_expected_value():
       result = geom.new_calculation(param1=100, param2=200)
       assert abs(result - 141.42) < 0.01
   
   # 2. Implement function
   def new_calculation(param1: float, param2: float) -> float:
       """
       Calculate new geometric property.
       
       Args:
           param1: First parameter in meters.
           param2: Second parameter in meters.
       
       Returns:
           Calculated value in appropriate units.
       """
       return math.sqrt(param1**2 + param2**2)
   
   # 3. Add benchmark
   def test_new_calculation_performance(benchmark):
       result = benchmark(geom.new_calculation, 100.0, 200.0)
       assert result > 0

Adding New Modules
~~~~~~~~~~~~~~~~~

For substantial new functionality:

1. **Create module file**: ``planet_ruler/new_module.py``
2. **Add to __init__.py**: Import key functions/classes
3. **Create test file**: ``tests/test_new_module.py``
4. **Add documentation**: Module-level docstring and API docs
5. **Update dependencies**: Add to pyproject.toml if needed

Documentation
------------

Documentation Structure
~~~~~~~~~~~~~~~~~~~~~~

Planet Ruler uses Sphinx with reStructuredText:

.. code-block::

   docs/
   ├── conf.py              # Sphinx configuration
   ├── index.rst            # Main documentation page
   ├── installation.rst     # Installation guide
   ├── tutorials.rst        # Step-by-step tutorials
   ├── examples.rst         # Real-world examples
   ├── api.rst             # API reference
   ├── modules.rst         # Auto-generated module docs
   ├── testing.rst         # Testing documentation
   ├── benchmarks.rst      # Performance documentation  
   └── contributing.rst    # This file

Building Documentation
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Install documentation dependencies
   pip install -r docs/requirements.txt
   
   # Build HTML documentation
   cd docs/
   make html
   
   # Open in browser
   open _build/html/index.html

Documentation Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~

* **All public APIs**: Must have comprehensive docstrings
* **Examples**: Include usage examples in docstrings
* **Mathematical notation**: Use LaTeX for mathematical expressions
* **Cross-references**: Link between related functions and classes

.. code-block:: python

   def example_function(param: float) -> float:
       """
       Example function demonstrating documentation standards.
       
       This function calculates something important using the formula:
       
       .. math::
           result = \\sqrt{\\frac{param^2}{2}}
       
       Args:
           param: Input parameter in meters.
       
       Returns:
           float: Calculated result in meters.
       
       See Also:
           :func:`related_function`: For related calculations.
           :class:`SomeClass`: For object-oriented approach.
       
       Example:
           Basic usage:
           
           >>> result = example_function(10.0)
           >>> print(f"Result: {result:.2f}")
           Result: 7.07
       """

Release Process
--------------

Version Numbering
~~~~~~~~~~~~~~~~

Planet Ruler uses semantic versioning (SemVer):

* **Major** (1.0.0): Breaking API changes
* **Minor** (1.1.0): New functionality, backward compatible  
* **Patch** (1.0.1): Bug fixes, backward compatible

Release Checklist
~~~~~~~~~~~~~~~~~

Before releasing a new version:

1. **Update version number**: In pyproject.toml and __init__.py
2. **Update changelog**: Document all changes since last release
3. **Run full test suite**: Ensure all tests pass across Python versions
4. **Build documentation**: Verify docs build without warnings
5. **Performance check**: Ensure no significant regressions
6. **Tag release**: Create Git tag with version number

.. code-block:: bash

   # Create release tag
   git tag -a v1.1.0 -m "Release version 1.1.0"
   git push origin v1.1.0

Communication Guidelines
-----------------------

Issue Reporting
~~~~~~~~~~~~~~

When reporting bugs or requesting features:

1. **Search existing issues**: Avoid duplicates
2. **Use issue templates**: Provide required information
3. **Include minimal examples**: Reproducible test cases
4. **Provide context**: Operating system, Python version, dependencies

**Bug report template:**

.. code-block:: markdown

   **Bug Description**
   Clear description of the bug.
   
   **To Reproduce**
   Steps to reproduce the behavior:
   1. Load image '...'
   2. Call function '...'  
   3. See error
   
   **Expected Behavior**
   What you expected to happen.
   
   **Environment**
   - OS: [e.g. Ubuntu 20.04]
   - Python: [e.g. 3.9.7]
   - Planet Ruler: [e.g. 1.0.0]

Discussion Etiquette
~~~~~~~~~~~~~~~~~~~

* **Be respectful**: Constructive feedback and professional communication
* **Stay on topic**: Keep discussions focused on the issue at hand
* **Provide context**: Include relevant background information
* **Be patient**: Maintainers contribute in their spare time

Recognition
----------

Contributors are recognized in:

* **CONTRIBUTORS.md**: List of all contributors
* **Release notes**: Acknowledgment of major contributions  
* **Documentation**: Author attribution for significant additions

Types of Contributions
~~~~~~~~~~~~~~~~~~~~~

All contributions are valued:

* **Code**: Bug fixes, new features, optimizations
* **Documentation**: Tutorials, examples, API improvements
* **Testing**: Test cases, integration tests, benchmarks
* **Bug reports**: Well-documented issues help improve quality
* **Feature requests**: Ideas for improvements and new functionality

Getting Help
-----------

If you need help contributing:

* **GitHub Issues**: Ask questions about development
* **Documentation**: Review existing guides and examples
* **Code Review**: Learn from feedback on pull requests
* **Community**: Connect with other contributors

Development Tools
----------------

Recommended IDE Setup
~~~~~~~~~~~~~~~~~~~~

**VS Code configuration (.vscode/settings.json):**

.. code-block:: json

   {
       "python.linting.enabled": true,
       "python.linting.flake8Enabled": true,
       "python.formatting.provider": "black",
       "python.testing.pytestEnabled": true,
       "python.testing.pytestArgs": ["tests/"],
       "python.linting.mypyEnabled": true
   }

**PyCharm configuration:**

* Enable Black formatter in File → Settings → Tools → External Tools
* Configure pytest as test runner in Run/Debug Configurations
* Enable type checking with mypy plugin

Debugging Tips
~~~~~~~~~~~~~

.. code-block:: python

   # Add breakpoints for debugging
   import pdb; pdb.set_trace()
   
   # Debug tests
   pytest tests/test_geometry.py::test_specific_function --pdb
   
   # Profile performance
   python -m cProfile -o profile.stats script.py

Git Workflow
~~~~~~~~~~~

.. code-block:: bash

   # Create feature branch
   git checkout -b feature/new-uncertainty-function
   
   # Make changes and commit
   git add -A
   git commit -m "feat: add uncertainty calculation with multiple methods"
   
   # Push and create PR
   git push origin feature/new-uncertainty-function

Thank you for contributing to Planet Ruler! Your efforts help make planetary radius measurements accessible to researchers, educators, and curious minds worldwide.