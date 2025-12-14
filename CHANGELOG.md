# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.7.0] - 2025-12-13

### Added

- **Initial PyPI release** - Planet Ruler is now pip-installable!
  - Package available as `planet-ruler` on PyPI
  - Import as `import planet_ruler` in Python code
  - Full documentation at https://bogsdarking.github.io/planet_ruler/

### Changed

- Package name changed from `planet_ruler` to `planet-ruler` for PyPI (import name unchanged)
- Better warnings for low-curvature observations
- Reduced resolution for demo images to keep things light

## [1.6.3] - 2025-12-12

### Fixed

- Documentation inconsistencies

## [1.6.2] - 2025-12-04

### Fixed

- Restored missing doc image
- Merged method tutorial with primary tutorial

## [1.6.1] - 2025-12-04

### Fixed

- Added missing requirement to build mermaid diagrams -- see new tutorials!

## [1.6.0] - 2025-11-30

### Added

- Expanded test benchmark suite including end-to-end detection and measurement methods.
- Added 'plot_residuals' function to zoom in on fit quality along with a 'plot_gradient_field_quiver' to directly visualize the field.
- Added 'plot_sam_masks' and generic 'plot_segmentation_masks' to visualize segmentation output.
- New tutorial 1.5 specifically for taking limb measurements from an airplane.
- New tutorial 4 on selecting a detection method.
- New manual annotation step available for ML segmentation -- user can tag masks to increase accuracy.

### Changed

- ImageSegmentation class replaced by the more method-agnostic MaskSegmenter

## [1.5.0] - 2025-11-11

### Added

- Fit Dashboard -- an easy-to-read interface that shows status, warnings, hints and recent output.
- New tutorial notebook for measuring your own photos: see notebooks/tutorials/measure_your_planet.ipynb .

### Changed

- Renamed gradient smoothing parameters to be more distinct.
- Reworked tutorials to move sequentially through demo, pre-configured, auto-configured, then advanced fits.

## [1.4.0] - 2025-11-03

### Added

- 'Gradient-field' fitting option that allows the minimizer to fit directly to the image without the intermediate step of detecting the horizon.
- Warm-start capability: you can now continue minimization from any previous solution.
- Multi-stage resolution fits to help navigate local minima when using gradient-field optimization method.

### Fixed

- Profile likelihood was not set up correctly.

## [1.3.0] - 2025-10-19

### Added

- Ability to automatically extract camera parameters from image metadata.

### Changed

- Ignoring 'main' actions when computing code coverage.

## [1.2.0] - 2025-10-11

### Added

- Manual annotation using custom GUI as primary limb detection method.
- CI/CD pipeline with >80% coverage and deployment to github-pages for full project documentation.

### Changed

- Now using Apache 2.0 license to align with educational usage.

### Removed

- String-drop limb detection method (fun simulation but too sensitive to configuration).
- Nested sampling for establishing focal length / detector width / field of view boundaries (we just fix one).

---

## Changelog Guidelines

When adding entries to this changelog:

### Categories
- **Added** for new features
- **Changed** for changes in existing functionality  
- **Deprecated** for soon-to-be removed features
- **Removed** for now removed features
- **Fixed** for any bug fixes
- **Security** for vulnerability fixes

### Version Format
- Use [Semantic Versioning](https://semver.org/)
- Format: `[MAJOR.MINOR.PATCH] - YYYY-MM-DD`
- Link versions to GitHub releases when available

### Entry Guidelines
- Write for users, not developers
- Include relevant issue/PR numbers when applicable
- Group related changes together
- Use present tense ("Add feature" not "Added feature")
- Be specific about what changed and why it matters to users

### Example Entry
```markdown
## [1.6.0] - 2024-12-01

### Added
- New Mars detection algorithms optimized for dusty atmospheres (#123)
- Export results to multiple formats (JSON, CSV, HDF5) (#145)

### Changed
- Improved gradient-field detection accuracy by 15% (#134)
- Updated documentation with mobile photography best practices (#142)

### Fixed
- Memory leak in multi-resolution optimization for large images (#138)
- EXIF parsing errors for certain camera models (#140)
```