# Conda/Jupyter Installation Guide

This guide addresses common installation issues when using Planet Ruler in conda/Jupyter environments.

## Quick Install (Recommended)

```bash
# Create fresh conda environment
conda create -n planet-ruler python=3.9
conda activate planet-ruler

# Install core scientific packages via conda first
conda install numpy scipy matplotlib pandas pillow pyyaml tqdm seaborn

# Install planet-ruler with minimal dependencies
pip install -e . --no-deps
pip install ipython ipywidgets jupyter

# Optional: Install ML dependencies (may cause conflicts)
# pip install torch torchvision segment-anything
```

## Troubleshooting Common Issues

### 1. Segment Anything Installation Issues
```bash
# Skip ML dependencies if they cause conflicts
pip install -e .[minimal]

# Manual horizon detection will be required without segment-anything
```

### 2. Conda Environment Conflicts
```bash
# Use pip-only environment
conda create -n planet-ruler-pip python=3.9
conda activate planet-ruler-pip
pip install -e .
```

### 3. M1 Mac Issues
```bash
# Use conda-forge channel
conda install -c conda-forge numpy scipy matplotlib
pip install -e .[minimal]
```

## Installation Options

- `pip install -e .` - Full installation (may have conflicts)
- `pip install -e .[minimal]` - Core features only
- `pip install -e .[jupyter]` - With Jupyter support
- `pip install -e .[ml]` - ML features only
- `pip install -e .[all]` - All optional features

## Verification

```python
import planet_ruler as pr
print("Planet Ruler imported successfully!")

# Test CLI (requires successful installation)
planet-ruler --help