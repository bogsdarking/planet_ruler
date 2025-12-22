---
title: 'Planet Ruler: Measuring Planetary Curvature from Consumer Imagery'
tags:
  - Python
  - astronomy
  - computer vision
  - planetary science
  - education
  - photogrammetry
authors:
  - name: Brandon Anderson
    orcid: 0000-0003-0009-7976
    affiliation: 1
affiliations:
 - name: Independent Researcher
   index: 1
date: 21 December 2025
bibliography: paper.bib
---

# Summary

Planet Ruler is an open-source Python library for measuring planetary radii from photographs using computer vision and geometric optimization. The tool enables educators, students, researchers, and hobbyists to measure Earth's curvature from airplane windows, spacecraft imagery, or high-altitude balloon experiments using consumer cameras and smartphones. Users can choose from three detection methods: interactive manual annotation (default), automated gradient-field optimization, or machine learning segmentation, making method selection itself an educational opportunity. The software automatically extracts camera parameters from EXIF metadata, eliminating a primary barrier to entry for non-specialists.

Planet Ruler prioritizes educational transparency. The tool exposes geometric relationships and optimization processes. It uses customized annotation GUIs to involve the user directly in the data generation process, helping build intuition about planetary and measurement science. Comprehensive uncertainty quantification—including annotation errors, camera parameter uncertainties, and fit confidence—makes error analysis a central part of the learning experience. With 80%+ test coverage, detailed documentation including setting-specific tutorials, and continuous integration pipelines, the software is production-ready and promotes healthy coding practices. Planet Ruler is distributed under the Apache 2.0 license and available via PyPI (`pip install planet-ruler`), requiring no institutional affiliation or specialized equipment.

# Statement of Need

Measuring planetary curvature from photographs is a powerful educational demonstration that connects ancient methods (Eratosthenes) to modern computer vision [@Lynch2008], but existing tools create significant barriers to entry. Commercial photogrammetry software (e.g., Agisoft Metashape, Pix4D) is expensive and designed for general 3D reconstruction rather than horizon-specific measurements. Mission-specific research software developed for spacecraft imagery analysis are typically not publicly available and require institutional affiliation. Manual approaches using general image analysis tools (ImageJ, GIMP) combined with spreadsheet calculations are error-prone, lack proper optimization, and provide no inherent uncertainty quantification.

Planet Ruler fills this gap as the only open-source tool specifically designed for planetary radius measurements using consumer cameras. It serves four key audiences: (1) **educators** teaching planetary science concepts in K-12 and undergraduate courses, (2) **students** conducting science fair projects or hands-on learning activities, (3) **researchers** needing quick analysis of spacecraft or high-altitude balloon imagery, and (4) **hobbyists** participating in citizen science. The tool lowers barriers through automatic camera parameter extraction from smartphone EXIF data, eliminating the need for specialized equipment or calibration knowledge.

Educational tools that enable direct comparison of human annotation, physics-based algorithms, and machine learning represent a significant missed opportunity in STEM education [@NRC2000]. Students rarely encounter single-task contexts where they can examine the same problem through multiple methodological lenses—manual measurement, classical computer vision, and modern AI—with immediate feedback on accuracy and uncertainty. Planet Ruler also addresses this gap through its method-agnostic architecture where users can measure the same horizon using manual annotation, gradient-field optimization, or ML segmentation, then directly compare results. This comparative framework transforms method selection from a hidden implementation detail into an explicit pedagogical tool, helping students understand the tradeoffs between human judgment, physics-based models, and data-driven approaches.

# Key Features and Innovations

**Comparative Method Implementation**

Each detection method offers distinct pedagogical insights. Manual annotation uses an interactive GUI where users click horizon points, engaging directly with measurement uncertainty and the challenge of defining curved features in discrete pixels. Gradient-field optimization demonstrates physics-based computer vision by maximizing edge strength along hypothesized horizons, making the geometric modeling process explicit and tunable. ML segmentation integrates the Segment Anything Model [@Kirillov2023] with an interactive mask refinement interface, allowing users to examine and correct model predictions on this single-image, geometrically-constrained task. All three methods feed the same geometric optimization pipeline, enabling direct accuracy and uncertainty comparisons.

**Gradient-Field Optimization for Horizon Detection**

Planet Ruler introduces a novel physics-based approach to horizon detection using gradient flux maximization [@Canny1986]. Rather than detecting edges first and fitting geometry second, the method directly optimizes a parametric circle by maximizing the total gradient flux along the hypothesized horizon. Sub-pixel precision is achieved through Taylor expansion of the gradient field, enabling discrimination at fractional pixel locations. This model-based approach naturally handles partial occlusions and varying image quality without requiring edge detection thresholds or post-processing. A multi-resolution implementation [@Fieguth2011]—optimizing on progressively higher-resolution images—enables robust convergence while avoiding local minima, achieving 30-60% better success rates than single-resolution approaches while running 2-3× faster.

**Accessibility Features**

Smartphone EXIF metadata provides automatic extraction of camera parameters [@Monteiro2022] (focal length, sensor dimensions) using principles from photogrammetric camera calibration [@Hartley2004], eliminating calibration requirements. Multiple uncertainty quantification methods [@Eadie1971]—population spread for stochastic optimizers, Hessian approximation for gradient-based methods, profile likelihood for robust intervals—are automatically selected based on the minimizer used. A live dashboard displays optimization progress with real-time parameter updates and smart warnings for common pitfalls. The interface works in both terminal and Jupyter notebook environments with comprehensive tutorials for airplane photography, spacecraft imagery, and method comparison workflows.

# Comparison to Similar Tools

Planet Ruler occupies a unique position in planetary measurement tools. Commercial photogrammetry software (Agisoft Metashape, Pix4D) focuses on general 3D reconstruction with expensive licenses. Mission-specific spacecraft tools exist within research institutions but are rarely public or instrument-agnostic. Manual approaches using ImageJ or GIMP with spreadsheet calculations lack optimization and uncertainty analysis. General-purpose ML pipelines can segment horizons but provide no comparative framework for understanding methodological tradeoffs.

Planet Ruler is the only open-source tool purpose-built for planetary horizon measurements that treats method diversity as an educational feature rather than an implementation choice. The Apache 2.0 license and PyPI distribution (`pip install planet-ruler`) ensure accessibility without institutional requirements.

# Implementation and Availability

Planet Ruler is implemented in Python 3.8+ with core dependencies limited to NumPy [@Harris2020], SciPy [@Virtanen2020], Matplotlib [@Hunter2007], OpenCV [@Bradski2000], and tkinter. Optional PyTorch and Segment Anything Model support enables ML segmentation. The software achieves 80%+ test coverage with continuous integration via GitHub Actions ensuring cross-platform compatibility. Documentation includes comprehensive tutorials, API references, and interactive Jupyter notebooks.

A minimal example demonstrates the workflow:
```python
from planet_ruler.camera import create_config_from_image
import planet_ruler as pr

# Auto-config from EXIF data
config = create_config_from_image(
    "airplane_photo.jpg", 
    altitude_m=10668, 
    planet="earth"
)

# Measure with any method
obs = pr.LimbObservation("airplane_photo.jpg", config)
obs.detect_limb(method="manual")  # or "gradient" or "sam"
obs.fit_limb(dashboard=True)

print(f"Radius: {obs.radius_km:.0f} ± {obs.radius_uncertainty_km:.0f} km")
```

**Repository:** https://github.com/bogsdarking/planet_ruler  
**Documentation:** https://bogsdarking.github.io/planet_ruler/  
**License:** Apache 2.0  
**Version:** 1.7.0

# Acknowledgments

This work was developed independently without institutional funding. Development relied on open-source tools including NumPy, SciPy, Matplotlib, OpenCV, and Meta AI's Segment Anything Model. AI assistance (Claude, Anthropic) was used to expand the core project via code, documentation, and manuscript drafting.

# References