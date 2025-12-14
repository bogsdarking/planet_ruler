# PLANET RULER

**Measure planetary radii with nothing but a camera and science!**

[![PyPI version](https://badge.fury.io/py/planet-ruler.svg)](https://badge.fury.io/py/planet-ruler)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://bogsdarking.github.io/planet_ruler/)
[![CI/CD Pipeline](https://github.com/bogsdarking/planet_ruler/actions/workflows/ci.yml/badge.svg)](https://github.com/bogsdarking/planet_ruler/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/bogsdarking/planet_ruler/branch/dev/graph/badge.svg)](https://codecov.io/gh/bogsdarking/planet_ruler)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Versions](https://img.shields.io/badge/python-3.8%20|%203.9%20|%203.10%20|%203.11%20|%203.12-blue)](https://github.com/bogsdarking/planet_ruler)

<div align="center">

**Got a horizon photo? Measure your planet in 3 lines of code!**

</div>

```python
import planet_ruler as pr
obs = pr.LimbObservation("horizon_photo.jpg", "config/camera.yaml")
obs.detect_limb().fit_limb()  # â†’ Planet radius: 6,234 km
```

<div align="center">

[Try Interactive Demo](notebooks/limb_demo.ipynb) • [Documentation](https://bogsdarking.github.io/planet_ruler/) • [Discussions](https://github.com/bogsdarking/planet_ruler/discussions)

</div>

---

<!-- ![Horizon analysis showcase](demo/images/cartoon_medley.png) -->

<div align="center">
<table>
<tr>
<td align="center" width="33%">
<img src="demo/images/Normandy_Pasture_-_HDR_(11988359173).jpg" width="300" alt="Commercial Aircraft View"/>
<br><b>Ground Level</b><br>
<em>~100 ft</em><br>
Horizon appears flat
</td>
<td align="center" width="33%">
<img src="demo/images/2013-08-05_22-42-14_Wikimania.jpg" width="300" alt="High Altitude Balloon"/>
<br><b>Commerical Aircraft</b><br>
<em>~35,000 ft</em><br>
Perceptible curvature
</td>
<td align="center" width="33%">
<img src="demo/images/An_aurora_streams_across_Earth's_horizon_(iss073e0293986).jpg" width="300" alt="Space Station View"/>
<br><b>International Space Station</b><br>
<em>~250 miles</em><br>
Dramatic spherical curvature
</td>
</tr>
</table>
</div>

<!-- <p style="text-align:right;">*From left to right: How horizon curvature changes with altitude, revealing the planetary radius beneath*</p> -->

<!-- *From left to right: How horizon curvature changes with altitude, revealing the planetary radius beneath* -->

## Quick Start

### Installation

**From PyPI (recommended):**
```bash
pip install planet-ruler
```

The package name is `planet-ruler` (with hyphen), but you import it with an underscore:
```python
import planet_ruler as pr  # Import uses underscore
```

**Optional dependencies:**

For ML segmentation:
```bash
pip install planet-ruler[ml]
```

For Jupyter notebooks:
```bash
pip install planet-ruler[jupyter]
```

For everything:
```bash
pip install planet-ruler[all]
```

**From source (development):**
```bash
git clone https://github.com/bogsdarking/planet_ruler.git
cd planet_ruler
pip install -e .
```

After installation, the command-line tool is available:
```bash
planet-ruler --help
```

### Python API
```python
import planet_ruler as pr

# Basic analysis
obs = pr.LimbObservation("photo.jpg", "camera_config.yaml")

# Choose detection method:
obs.detect_limb(method='manual')          # Interactive GUI (default)
# obs.detect_limb(method='gradient-break')  # Simple gradient-based detection
# obs.detect_limb(method='gradient-field')  # Gradient flow analysis
# obs.detect_limb(method='segmentation')    # AI-powered (requires PyTorch)

# OR skip detection and use gradient-field optimization directly:
# obs.fit_limb(loss_function='gradient_field')  # Direct gradient optimization

obs.fit_limb()     # Multi-resolution parameter optimization
obs.plot()         # Visualize results

# Access results with uncertainty
print(f"Radius: {obs.best_parameters['r']/1000:.0f} ± {obs.radius_uncertainty:.0f} km")
```

### Command Line
```bash
# Measure planetary radius using existing config file
planet-ruler measure photo.jpg --camera-config camera_config.yaml

# Auto-generate config from image EXIF data (requires altitude)
planet-ruler measure photo.jpg --auto-config --altitude 10668

# Auto-config with specific planet (affects initial radius guess)
planet-ruler measure photo.jpg --auto-config --altitude 10668 --planet mars

# Choose detection method (manual, gradient-break, gradient-field, or segmentation)
planet-ruler measure photo.jpg --auto-config --altitude 10668 --detection-method gradient-field

# Try built-in examples
planet-ruler demo --planet earth
```

## The Science Behind It

**The Problem**: How big is your planet?

**The Solution**: Depending on your altitude, planetary curvature becomes visible in the horizon. By measuring this curvature and accounting for your camera, we can reverse-engineer the planet's size.

<details>
<summary><strong>How It Works (Click to expand)</strong></summary>

1. **Capture**: Photograph showing horizon/limb from altitude
2. **Detect**: Choose your detection method:
   - **Manual**: Interactive GUI for precise point selection (default, no dependencies)
   - **Gradient-field**: Automated detection using gradient flow analysis
   - **Segmentation**: AI-powered detection (requires PyTorch + Segment Anything)
3. **Measure**: Extract curvature from the detected horizon
4. **Model**: Account for camera optics, altitude, and viewing geometry  
5. **Optimize**: Fit theoretical curves to observations using multi-resolution optimization
6. **Uncertainty**: Quantify measurement precision using population spread, Hessian approximation, or profile likelihood

**Mathematical Foundation**:
- Spherical geometry and horizon distance calculations
- Camera intrinsic/extrinsic parameter modeling
- Gradient-field analysis with directional blur and flux integration
- Multi-resolution optimization with coarse-to-fine refinement
- Non-linear optimization with robust error handling

</details>

## Real Results

**Validated on actual space mission data:**

| **Planet** | **Source** | **Estimated** | **True Value** | **Error** |
|------------|------------|---------------|----------------|-----------|
| **Earth** | ISS Photo | 5,516 km | 6,371 km | **13.4%** |
| **Saturn** | Cassini | 65,402 km | 58,232 km | **12.3%** |
| **Pluto** | New Horizons | 1,432 km | 1,188 km | **20.6%** |


## Key Features

<table>
<tr>
<td width="50%">

**Automatic Camera Detection**
- Auto-extract camera parameters from EXIF data
- Supports phones, DSLRs, mirrorless, point-and-shoot
- No manual camera configuration needed

**Flexible Detection Methods**
- **Manual**: Interactive GUI with precision controls (default)
- **Gradient-field**: Automated detection via directional blur and flux analysis
- **AI Segmentation**: Deep learning-powered (optional)

**Advanced Camera Models**
- Camera parameter optimization
- Multiple camera configurations supported
- Flexible parameter fitting framework

**Multi-Planetary Support**
- Earth, Saturn, Pluto examples included
- Extensible to any spherical body

</td>
<td width="50%">

**Scientific Rigor**
- Multi-resolution optimization with coarse-to-fine refinement
- Advanced uncertainty estimation (population spread, Hessian, profile likelihood)
- Mathematical validation with property tests

**Live Progress Dashboard**
- Real-time optimization monitoring
- Adaptive refresh rate (fast during descent, slow at convergence)
- Configurable warnings, hints, and output display

**Rich Visualizations**
- Interactive plots
- 3D planetary geometry views

**Multiple Interfaces**
- Python API for scripting
- Command-line tool for automation
- Jupyter notebooks for exploration

**Works with Any Camera**
- iPhones, Android phones, DSLRs, mirrorless
- Automatic sensor size detection
- Intelligent parameter estimation

</td>
</tr>
</table>

## Installation & Setup

### Requirements
- **Python 3.8+**
- **RAM**: 4GB+ recommended (for AI models)
- **Storage**: ~2GB for full installation with models

### Install Options

<details>
<summary><strong>Quick Start (Recommended)</strong></summary>

```bash
# Clone and install in one go
git clone https://github.com/bogsdarking/planet_ruler.git
cd planet_ruler
python -m pip install -e .

# Verify installation
planet-ruler --help
python -c "import planet_ruler; print('Ready to measure planets!')"
```
</details>

<details>
<summary><strong>Minimal Install (Core features only)</strong></summary>

```bash
git clone https://github.com/bogsdarking/planet_ruler.git
cd planet_ruler

# Install without heavy AI dependencies
python -m pip install -e . --no-deps
python -m pip install numpy scipy matplotlib pillow pyyaml pandas tqdm seaborn

# Note: Manual horizon detection required without segment-anything
```
</details>

<details>
<summary><strong>Development Install</strong></summary>

```bash
git clone https://github.com/bogsdarking/planet_ruler.git
cd planet_ruler

# Full development environment
python -m pip install -e .
python -m pip install -r requirements.txt
python -m pip install -r requirements-test.txt

# Run tests to verify
pytest tests/ -v
```
</details>

### Troubleshooting
- **Segment Anything issues?** See [installation guide](docs/installation.rst)
- **M1 Mac problems?** Use conda for better compatibility
- **Memory errors?** Try the minimal install option

## Try It Now

### Zero-Configuration Workflow
```python
# Just need your photo and altitude - planet_ruler handles the rest!
from planet_ruler.camera import create_config_from_image
import planet_ruler as pr

# Step 1: Auto-detect camera parameters
config = create_config_from_image("my_photo.jpg", altitude_m=10668)

# Step 2: Measure the planet
obs = pr.LimbObservation("my_photo.jpg", config)
obs.detect_limb().fit_limb()

print(f"Your planet's radius: {obs.best_parameters['r']/1000:.0f} km")
```

### Interactive Demo
```python
# Launch interactive widget with examples (in Jupyter notebook)
from planet_ruler.demo import make_dropdown, load_demo_parameters
demo = make_dropdown()  # Choose Earth, Saturn, or Pluto
params = load_demo_parameters(demo)
```

### Use Your Own Photo

#### Option 1: Auto-Config
```python
import planet_ruler as pr
from planet_ruler.camera import create_config_from_image

# Auto-generate config from image EXIF data
config = create_config_from_image("your_photo.jpg", altitude_m=10668, planet="earth")

# Use the auto-generated config
obs = pr.LimbObservation("your_photo.jpg", config)
obs.detect_limb()
obs.fit_limb()

print(f"Detected camera: {config['camera_info']['camera_model']}")
print(f"Fitted radius: {obs.best_parameters['r']/1000:.0f} km")
```

#### Option 2: Manual Config File
```python
import planet_ruler as pr

# Requires camera configuration file
obs = pr.LimbObservation(
    "your_photo.jpg",
    "config/your_camera.yaml"
)

# Choose detection method: 'manual', 'gradient-break', 'gradient-field', or 'segmentation'
obs.detect_limb(method='manual')  # Interactive GUI detection
# obs.detect_limb(method='gradient-break')  # Simple gradient detection
# obs.detect_limb(method='gradient-field')  # Gradient flow analysis

# OR use gradient-field optimization (skips traditional detection):
# obs.fit_limb(loss_function='gradient_field')  # Direct gradient optimization

obs.detect_limb()
obs.fit_limb()

print(f"Fitted radius: {obs.best_parameters['r']/1000:.0f} km")
```

### Camera Configuration Template
```yaml
# config/your_camera.yaml
description: "Your camera setup"
free_parameters:
  - r  # planetary radius
  - h  # altitude
  - f  # focal length
  - w  # sensor width

init_parameter_values:
  r: 6371000      # Earth radius in meters
  h: 400000       # Altitude in meters
  f: 0.05         # Focal length in meters
  w: 0.035        # Sensor width in meters

parameter_limits:
  r: [1000000, 20000000]
  h: [100000, 1000000]
  f: [0.01, 0.2]
  w: [0.01, 0.1]
```

## Usage Examples

### Example 1: Smartphone Photo with Auto-Config
```python
import planet_ruler as pr
from planet_ruler.camera import create_config_from_image

# Your iPhone/Android photo with horizon - just need altitude!
config = create_config_from_image(
    "airplane_window_photo.jpg",
    altitude_m=10668,  # 35,000 feet in meters
    planet="earth"
)

print(f"Detected: {config['camera_info']['camera_model']}")
# -> "iPhone 14 Pro" (or your camera model)

# Measure the planet
obs = pr.LimbObservation("airplane_window_photo.jpg", config)
obs.detect_limb()
obs.fit_limb()
obs.plot()

print(f"Earth radius: {obs.best_parameters['r']/1000:.0f} km")
```

### Example 2: Command Line with Auto-Config
```bash
# Simple one-liner with any camera!
planet-ruler measure my_photo.jpg --auto-config --altitude 10668

# With specific planet and detection method
planet-ruler measure mars_photo.jpg --auto-config --altitude 4500 --planet mars --detection-method gradient-field

# Override auto-detected parameters if needed
planet-ruler measure photo.jpg --auto-config --altitude 10668 --focal-length 50
```

### Example 3: Earth from ISS (Traditional Config)
```python
import planet_ruler as pr

# Load ISS Earth photo with manual configuration
obs = pr.LimbObservation(
    "demo/images/iss064e002941.jpg",
    "config/earth_iss_1.yaml"
)

# Full analysis pipeline
obs.detect_limb(method='gradient-break')  # Automated gradient-based detection
obs.fit_limb()                             # Multi-resolution parameter optimization

# OR use gradient-field optimization (more advanced):
# obs.fit_limb(loss_function='gradient_field', resolution_stages='auto')
obs.plot()                                 # Visualize results

# Results
print(f"Best fit parameters: {obs.best_parameters}")
print(f"Planetary radius (r): {obs.best_parameters['r']/1000:.0f} km")
```

### Example 4: Saturn from Cassini
```python
# Analyze Saturn's limb from Cassini spacecraft
obs = pr.LimbObservation(
    "demo/images/saturn_pia21341-1041.jpg",
    "config/saturn-cassini-1.yaml"
)

# Two-step analysis
obs.detect_limb(method='gradient-break')  # Automated detection
obs.fit_limb()                            # Multi-resolution fitting

# OR single-step gradient-field optimization:
# obs.fit_limb(loss_function='gradient_field', resolution_stages='auto')

# Rich visualization
from planet_ruler.plot import plot_3d_solution
plot_3d_solution(**obs.best_parameters)  # 3D planetary geometry view
```

## Documentation & Resources

### Learning Resources
| Resource | Description | Best For |
|----------|-------------|----------|
| [**Interactive Tutorial**](notebooks/limb_demo.ipynb) | Complete walkthrough with examples | **First-time users** |
| [**API Documentation**](https://bogsdarking.github.io/planet_ruler) | Detailed function reference | **Developers** |
| [**Camera Setup Guide**](config/) | Configuration examples | **Custom setups** |
| [**Example Gallery**](demo/) | Real space mission results | **Inspiration** |

### Quick References
```python
# Core classes and functions
pr.LimbObservation(image_path, fit_config)                      # Main analysis class
pr.geometry.horizon_distance(altitude, radius)                   # Theoretical calculations  
pr.fit.optimize_parameters(obs, method='differential_evolution') # Optimization
pr.uncertainty.calculate_parameter_uncertainty(obs, 'r')         # Uncertainty estimation
pr.plot.show_analysis(obs, style='comprehensive')                # Visualization

# Key methods
obs.detect_limb(method='manual')          # Interactive detection (default)
obs.fit_limb(resolution_stages='auto')    # Multi-resolution optimization
obs.plot()                                # Show results with uncertainty

# Advanced gradient-field optimization:
# obs.fit_limb(loss_function='gradient_field', resolution_stages='auto')
```

## Use Cases & Applications

- **Astronomy courses**: Demonstrate planetary geometry concepts
- **Computer vision**: Real-world optimization and AI applications  
- **Mathematics**: Applied geometry and curve-fitting examples
- **Physics**: Observational techniques and measurement uncertainty

## Limitations & Best Practices

### **Accuracy Expectations**
- **Typical accuracy**: ~20%
- **Best case**: ~15% with optimal conditions and camera calibration (so far!)
- **Factors affecting precision**: Image quality, horizon clarity, altitude, camera specs

### **Technical Limitations**
- **Optimization challenges**: Complex parameter space → potential local minima (mitigated by multi-resolution optimization)
- **Detection method trade-offs**: Manual (precise, time-intensive), gradient-field (automated, works for clear horizons), AI segmentation (most versatile, requires PyTorch)
- **Computational cost**: Multi-resolution optimization can be slow on older hardware

### **Best Practices**
1. **Image quality**: Sharp, high-resolution horizons work best
2. **Altitude**: Higher = more curvature = better measurements  
3. **Camera knowledge**: Focal length and sensor specs improve results
4. **Horizon clarity**: Mountains, clouds, or haze reduce accuracy
5. **Run multiple optimizations** and compare results for consistency

## Contributing

**We welcome contributions from astronomers, developers, educators, and enthusiasts!**

Planet Ruler is maintained by one developer in their spare time. Issue responses may take 3-7 days. Before opening an issue, please check the [documentation](https://bogsdarking.github.io/planet_ruler/) and [existing issues](https://github.com/bogsdarking/planet_ruler/issues).

### Quick Contribution Setup
```bash
# Fork the repo, then:
git clone https://github.com/YOUR_USERNAME/planet_ruler.git
cd planet_ruler
python -m pip install -e . && python -m pip install -r requirements.txt && python -m pip install -r requirements-test.txt
pytest tests/ -v  # Verify everything works
```

### Ways to Contribute
| Type | Examples | Good For |
|------|----------|----------|
| **Bug Reports** | Detection failures, optimization issues | **Everyone** |
| **Features** | New algorithms, UI improvements | **Developers** |
| **Documentation** | Tutorials, examples, API docs | **Educators** |
| **Examples** | New planetary bodies, camera setups | **Researchers** |
| **Testing** | Edge cases, performance tests | **QA enthusiasts** |

### Contribution Guidelines
- **Found a bug?** → [Create an issue](https://github.com/bogsdarking/planet_ruler/issues/new?template=bug_report.md)
- **Have an idea?** → [Start a discussion](https://github.com/bogsdarking/planet_ruler/discussions)
- **Ready to code?** → See our [CONTRIBUTING.md](CONTRIBUTING.md) guide

> **First-time contributors welcome!** Look for issues labeled [`good first issue`](https://github.com/bogsdarking/planet_ruler/labels/good%20first%20issue)

## Acknowledgments & References

### **Built With**
- [Segment Anything (Meta)](https://segment-anything.com/) - AI-powered horizon detection
- [SciPy](https://scipy.org/) - Scientific optimization algorithms  
- [NumPy](https://numpy.org/) - High-performance numerical computing
- [Matplotlib](https://matplotlib.org/) - Publication-quality visualizations

### **Scientific References**
- [Horizon geometry fundamentals](https://en.wikipedia.org/wiki/Horizon) - Basic theory
- [Camera calibration techniques](https://courses.cs.washington.edu/courses/cse455/09wi/Lects/lect5.pdf) - Optics modeling
- [Earth curvature visibility](https://earthscience.stackexchange.com/questions/7283/) - Observational considerations
- [Camera resectioning methods](https://en.wikipedia.org/wiki/Camera_resectioning) - Parameter estimation
- [Intrinsic camera parameters](https://ksimek.github.io/2013/08/13/intrinsic/) - Mathematical foundations

### **Inspiration**
If you've ever wondered about the size of your planet, you are not alone -- humanity has tried to measure this [throughout the ages](https://en.wikipedia.org/wiki/History_of_geodesy). Though Earth is large enough to defy the usual methods we have for measuring things, a creative mind can do it with surprisingly little. Eratosthenes, in ancient Greece, was able to do it to impressive accuracy using only a rod and the sun. How much better can we do today?

## License

Licensed under the Apache License, Version 2.0 - see [LICENSE](LICENSE) file for details.

---

<div align="center">

__μεταξὺ δὲ τοῦ πυρὸς καὶ τῶν δεσμωτῶν__<br />between the fire and the captives -- Plato

[⭐ Star this repo](https://github.com/bogsdarking/planet_ruler/stargazers) • [Report issues](https://github.com/bogsdarking/planet_ruler/issues) • [Join discussions](https://github.com/bogsdarking/planet_ruler/discussions)

*Made with ❤️ for curious minds exploring our cosmic neighborhood*

</div>