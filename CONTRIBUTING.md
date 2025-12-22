# Contributing to Planet Ruler

Thank you for your interest in contributing to Planet Ruler! This document provides guidelines and information for contributors.

## Quick Start for Contributors

1. **Fork and Clone**
```bash
   git clone https://github.com/YOUR_USERNAME/planet_ruler.git
   cd planet_ruler
```

2. **Set up Development Environment**
```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install package in development mode with test dependencies
   pip install -e .
   pip install -r requirements-test.txt
```

   *Alternatively, if you just want to use the latest release:*
```bash
   pip install planet-ruler
```

3. **Run Tests**
```bash
   pytest tests/ -v
```

## Ways to Contribute

### Bug Reports
Found a bug? Please create an issue using our [bug report template](.github/ISSUE_TEMPLATE/bug_report.md).

**Before reporting:**
- Search existing issues to avoid duplicates
- Try the latest version from PyPI: `pip install --upgrade planet-ruler`
- Provide minimal reproducible example

### Feature Requests
Have an idea? Use our [feature request template](.github/ISSUE_TEMPLATE/feature_request.md).

**Good feature requests include:**
- Clear use case description
- Expected behavior
- Willingness to contribute implementation

### Documentation
Documentation improvements are always welcome!  See **Contribution Areas** below for ideas.

### Code Contributions

#### Good First Issues
Look for issues labeled `good first issue`:
- Documentation improvements
- Adding camera sensor presets to the database
- Test coverage improvements
- Code formatting/cleanup
- Tutorial examples

#### Development Workflow
1. **Create a branch** for your feature/fix
```bash
   git checkout -b feature/your-feature-name
```

2. **Write tests** for your changes
```bash
   # Add tests in tests/ directory
   # Run specific test
   pytest tests/test_your_feature.py -v
```

3. **Follow code style**
```bash
   # Format code (if you have black installed)
   black planet_ruler/ tests/
   
   # Check style (if you have flake8 installed)
   flake8 planet_ruler/ tests/
```

4. **Update documentation** if needed
```bash
   cd docs/
   make html
```

5. **Commit with clear messages**
```bash
   git commit -m "Add multi-resolution gradient optimization
   
   - Implement coarse-to-fine optimization strategy
   - Add parameter scaling between resolution levels
   - Include tests for convergence improvement
   - Update documentation with optimization guide
   
   Closes #123"
```

## Testing Guidelines

### Running Tests
```bash
# All tests
pytest tests/ -v

# Fast tests only (skip slow integration tests)
pytest tests/ -v -m "not slow and not benchmark"

# With coverage report
pytest tests/ --cov=planet_ruler --cov-report=html

# Specific test file
pytest tests/test_geometry.py -v

# Integration tests with real data
pytest tests/test_integration_real_data.py -v
```

### Writing Tests
- **Unit tests**: Test individual functions in isolation
- **Integration tests**: Test complete measurement workflows
- **Property tests**: Use Hypothesis for mathematical invariants
- **Benchmark tests**: Performance measurements (mark with `@pytest.mark.benchmark`)

Example test:
```python
import planet_ruler as pr
from planet_ruler.camera import create_config_from_image

def test_radius_measurement_workflow():
    """Test complete radius measurement from image."""
    config = create_config_from_image(
        "tests/data/test_image.jpg",
        altitude_m=10668,
        planet="earth"
    )
    
    obs = pr.LimbObservation("tests/data/test_image.jpg", config)
    obs.detect_limb(method="manual")
    obs.fit_limb()
    
    # Verify reasonable Earth radius
    assert 6200 < obs.radius_km < 6500
    assert obs.radius_uncertainty_km > 0
```

## Code Standards

### Python Style
- **Format**: Use `black` for code formatting (optional but appreciated)
- **Docstrings**: Follow NumPy docstring convention
- **Type hints**: Use where it improves clarity
- **Python version**: Support Python 3.8+

Example function:
```python
import numpy as np
from typing import Tuple

def fit_circle_to_points(
    points: np.ndarray, 
    initial_guess: Tuple[float, float, float]
) -> Tuple[float, float, float]:
    """Fit circle to horizon points using geometric optimization.
    
    Parameters
    ----------
    points : np.ndarray
        Array of (x, y) coordinates, shape (N, 2)
    initial_guess : tuple of float
        Initial (center_x, center_y, radius) estimate
        
    Returns
    -------
    center_x : float
        Fitted circle center x-coordinate in pixels
    center_y : float
        Fitted circle center y-coordinate in pixels
    radius : float
        Fitted circle radius in pixels
        
    Examples
    --------
    >>> points = np.array([[100, 200], [150, 180], [200, 200]])
    >>> cx, cy, r = fit_circle_to_points(points, (150, 190, 50))
    >>> print(f"Center: ({cx:.1f}, {cy:.1f}), Radius: {r:.1f}")
    """
    # Implementation here
    pass
```

### Project Structure
```
planet_ruler/            # Main package
├── __init__.py         
├── observation.py       # Core LimbObservation class
├── geometry.py          # Geometric calculations
├── camera.py            # Camera models and EXIF
├── fit.py               # Optimization routines
└── ...                  # Other modules

tests/
├── test_*.py            # Unit and integration tests
└── conftest.py          # Pytest fixtures

docs/
├── *.rst                # Documentation source files
├── images/              # Documentation images
└── conf.py              # Sphinx configuration

notebooks/
└── *.ipynb              # Interactive demos and tutorials

paper/
├── paper.md             # JOSS submission
└── paper.bib            # References

demo/
└── ...                  # Example images and narrations
```

## Contribution Areas

### High-Impact Contributions
1. **Algorithm improvements**
   - Better or faster horizon detection methods
   - Better handling of complex scenes (e.g., obscuring foreground)
   - Atmospheric corrections
   - More robust optimization
   - Advanced uncertainty quantification techniques

2. **New features**
   - Additional detection methods (beyond manual/gradient/ML)
   - Support for additional celestial bodies
   - Batch processing multiple images
   - Interactive web dashboard

3. **Performance optimization**
   - Faster gradient-field optimization
   - Memory-efficient image processing
   - Parallelization for batch workflows

4. **Camera support**
   - Expand camera sensor database
   - Support for fisheye/wide-angle lenses
   - Support for uncertainty from lens distortion or aberration

### Documentation Needs
1. **Mathematical explanations**
   - Detailed geometry derivations
   - Camera projection models
   - Uncertainty propagation theory

2. **Tutorials**
   - More airplane photography examples
   - Spacecraft imagery workflows (Mars, Moon, etc.)
   - High-altitude balloon measurements
   - Method comparison studies

3. **Examples**
   - Gallery of successful measurements
   - Edge cases and troubleshooting
   - Integration with other tools (e.g., exiftool)

### Testing Priorities
1. **Coverage expansion**
   - Edge cases (low altitudes, panorama images)
   - Error conditions and validation
   - Cross-platform compatibility

2. **Integration tests**
   - Real-world image validation
   - Method comparison accuracy
   - EXIF extraction robustness

## Recognition

Contributors are recognized in:
- GitHub contributors list
- README acknowledgments
- Release notes (CHANGELOG.md)
- Documentation credits

## Getting Help

**Questions about contributing?**
- Open a [discussion](https://github.com/bogsdarking/planet_ruler/discussions)
- Email the maintainer
- Create an issue with the `question` label

**Development help:**
- Check existing documentation at https://bogsdarking.github.io/planet_ruler/
- Search closed issues and PRs
- Ask in discussions before starting large changes

## Pull Request Checklist

Before submitting a PR, ensure:
- [ ] Tests pass locally (`pytest tests/ -v`)
- [ ] Code follows style guidelines (or at least is readable and documented)
- [ ] Documentation updated if needed (especially for new features)
- [ ] Commit messages are clear and descriptive
- [ ] PR description explains the change and motivation
- [ ] Related issue linked if applicable

## Release Process

For maintainers:
1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md` with new features/fixes
3. Create annotated tag: `git tag -a v1.x.x -m "Release v1.x.x"`
4. Push tag: `git push origin v1.x.x`
5. Create release on GitHub
6. CI will automatically build and upload to PyPI

## Code of Conduct

Be respectful and constructive in all interactions. We're all here to learn and improve the project together.

**In summary:**
- Be welcoming to newcomers
- Help others learn
- Focus on technical merits
- Assume good intentions
- Give constructive feedback
- Remember this is an educational tool—pedagogy matters!

## License

By contributing to Planet Ruler, you agree that your contributions will be licensed under the Apache License 2.0.

---

**Thank you for contributing to Planet Ruler!**

Every contribution, no matter how small, helps make planetary science more accessible to everyone.