# Contributing to Planet Ruler 🌍📏

Thank you for your interest in contributing to Planet Ruler! This document provides guidelines and information for contributors.

## 🚀 Quick Start for Contributors

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
   
   # Install in development mode
   pip install -e .
   pip install -r requirements-test.txt
   ```

3. **Run Tests**
   ```bash
   pytest tests/ -v
   ```

## 🎯 Ways to Contribute

### 🐛 Bug Reports
Found a bug? Please create an issue using our [bug report template](.github/ISSUE_TEMPLATE/bug_report.md).

**Before reporting:**
- Search existing issues to avoid duplicates
- Try the latest version
- Provide minimal reproducible example

### 💡 Feature Requests
Have an idea? Use our [feature request template](.github/ISSUE_TEMPLATE/feature_request.md).

**Good feature requests include:**
- Clear use case description
- Expected behavior
- Willingness to contribute implementation

### 📖 Documentation
Documentation improvements are always welcome!

**Areas that need help:**
- API documentation
- Tutorial improvements
- Example gallery
- Mathematical explanations
- Installation guides

### 🔧 Code Contributions

#### Good First Issues
Look for issues labeled `good first issue`:
- Documentation improvements
- Adding camera presets
- Test coverage improvements
- Code formatting/cleanup

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
   # Format code
   black planet_ruler/ tests/
   
   # Check style
   flake8 planet_ruler/ tests/
   ```

4. **Update documentation** if needed
   ```bash
   cd docs/
   make html
   ```

5. **Commit with clear messages**
   ```bash
   git commit -m "Add horizon detection for planetary images
   
   - Implement segment-anything integration
   - Add camera calibration support
   - Include tests and documentation
   
   Closes #123"
   ```

## 🧪 Testing Guidelines

### Running Tests
```bash
# All tests
pytest tests/ -v

# Fast tests only
pytest tests/ -v -m "not slow and not benchmark"

# With coverage
pytest tests/ --cov=planet_ruler --cov-report=html

# Specific test file
pytest tests/test_geometry.py -v
```

### Writing Tests
- **Unit tests**: Test individual functions
- **Integration tests**: Test complete workflows
- **Property tests**: Use Hypothesis for mathematical properties
- **Benchmark tests**: Performance measurements

Example test:
```python
def test_radius_calculation():
    """Test planetary radius calculation."""
    obs = pr.LimbObservation("test_image.jpg")
    obs.altitude_km = 400
    obs.focal_length_mm = 50
    
    # Your test logic here
    assert obs.radius_km > 0
```

## 📏 Code Standards

### Python Style
- **Format**: Use `black` for code formatting
- **Linting**: Use `flake8` for style checking
- **Docstrings**: Follow NumPy docstring convention
- **Type hints**: Use type hints where helpful

Example function:
```python
def calculate_radius(altitude: float, curvature: float) -> float:
    """Calculate planetary radius from altitude and curvature.
    
    Parameters
    ----------
    altitude : float
        Observer altitude in kilometers
    curvature : float
        Observed horizon curvature in radians
        
    Returns
    -------
    float
        Estimated planetary radius in kilometers
        
    Examples
    --------
    >>> radius = calculate_radius(400, 0.001)
    >>> print(f"Radius: {radius:.0f} km")
    """
    # Implementation here
    return radius
```

### Project Structure
```
planet_ruler/
├── __init__.py          # Main API exports
├── observation.py       # LimbObservation class
├── geometry.py         # Mathematical functions
├── image.py            # Image processing
├── fit.py              # Optimization routines
├── plot.py             # Visualization
├── demo.py             # Demo functionality
└── cli.py              # Command-line interface

tests/
├── test_geometry.py    # Geometry tests
├── test_image.py       # Image processing tests
├── test_fit.py         # Optimization tests
└── conftest.py         # Test configuration
```

## 🔬 Contribution Areas

### High-Impact Contributions
1. **Algorithm improvements**
   - Better horizon detection methods
   - More robust optimization
   - Uncertainty quantification

2. **New features**
   - Real-time processing
   - Multiple image support
   - Web interface

3. **Performance optimization**
   - Faster image processing
   - Parallelization
   - Memory usage reduction

### Documentation Needs
1. **Mathematical explanations**
   - Geometry derivations
   - Camera models
   - Error analysis

2. **Tutorials**
   - Step-by-step guides
   - Video walkthroughs
   - Academic use cases

3. **Examples**
   - More planetary bodies
   - Different camera types
   - Edge cases and troubleshooting

### Testing Priorities
1. **Coverage expansion**
   - Edge cases
   - Error conditions
   - Performance regression tests

2. **Integration tests**
   - End-to-end workflows
   - Real image validation
   - Cross-platform testing

## 🌟 Recognition

Contributors are recognized in:
- GitHub contributors list
- README acknowledgments
- Release notes
- Documentation credits

## 📞 Getting Help

**Questions about contributing?**
- 💬 Open a [discussion](https://github.com/bogsdarking/planet_ruler/discussions)
- 📧 Email the maintainer
- 🐛 Create an issue with the question label

**Development help:**
- 📖 Check existing documentation
- 🔍 Search closed issues and PRs
- 💡 Ask in discussions before starting large changes

## 📋 Pull Request Checklist

Before submitting a PR, ensure:
- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation updated if needed
- [ ] Commit messages are clear
- [ ] PR description explains the change
- [ ] Issue linked if applicable

## 🏷️ Release Process

For maintainers:
1. Update version in `pyproject.toml`
2. Update CHANGELOG
3. Create release on GitHub
4. CI will build and upload to PyPI

## 🤝 Code of Conduct

Be respectful and constructive in all interactions. We're all here to learn and improve the project together.

## License

By contributing to planet_ruler, you agree that your contributions will be licensed under the Apache License 2.0.

**In summary:**
- Be welcoming to newcomers
- Help others learn
- Focus on the technical merits
- Assume good intentions
- Give constructive feedback

---

**Thank you for contributing to Planet Ruler!** 🌍📏

Every contribution, no matter how small, helps make this project better for everyone.