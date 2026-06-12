# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Sensor dimensions calculated from the 35mm-equivalent focal length were ~4% too wide on
  4:3 sensors. `calculate_sensor_dimensions` split the equivalent focal length as a 3:2 width
  ratio (`36 / crop_factor`), but the equivalent focal length encodes the *diagonal* field of
  view. It now recovers the sensor diagonal and splits it at the image's true aspect ratio
  (taken from the decoded image dimensions, defaulting to the 3:2 full-frame reference).

### Changed

- `extract_camera_parameters` now prefers the 35mm-equivalent calculation over the camera
  database (the two strategies were reordered). The camera's own reported field of view is
  trusted ahead of a stored DB entry; the database is the fallback for images lacking the
  35mm-equivalent tag (common on DSLRs and older compacts).
- The calculated-sensor path is now reported as `high` confidence rather than `medium`,
  reflecting that it is the primary source.

### Added

- `camera_sanity_warnings()` flags conditions that make the EXIF-derived field of view
  unreliable: digital zoom, a non-native capture aspect ratio, and a decoded aspect that
  disagrees with the camera-recorded EXIF dimensions (cropped after capture). Surfaced via
  `params["warnings"]` from `extract_camera_parameters`.
- Camera database now keys Samsung devices by marketing name (e.g. "Galaxy A56",
  "Galaxy S22 Ultra") in addition to `SM-` model codes, since Samsung EXIF reports the
  marketing name on many devices. New entries for Galaxy S22 / S22 Ultra / S23 / A54 / A56,
  Pixel 8 / 8 Pro / 9 Pro, and iPhone 16 / 16 Pro (sensor dimensions for new models reuse a
  sibling's sensor class and are marked nominal in their notes).

## [2.0.1] - 2026-06-03

### Added

- Concavity penalty for the L2 annotation cost function. The optimizer can occasionally
  converge to a physically wrong, inverted arc — a failure mode on off-apex images or with
  loose parameter bounds. The penalty detects this by checking whether the predicted arc
  curves in the correct direction relative to its chord, and adds a large cost when it does
  not. Enabled by default; configurable via `concavity_penalty` (bool) and
  `concavity_penalty_scale` (float) kwargs in `fit_arc()`. The correct curvature direction
  is auto-detected from `theta_z`, so non-standard viewing geometries (e.g. camera pointing
  away from nadir) are handled correctly without any user configuration.

### Changed

- Sagitta pre-fitter (`fit_sagitta`) upgraded from a 2-D to a 3-D optimisation: the arc
  apex x-position is now a free parameter alongside curvature and camera angle. This removes
  a systematic radius bias on asymmetric arcs whose true apex falls outside the annotated
  region.
- Sagitta pre-fitter now seeds the differential-evolution warm-start with a geometrically
  computed camera angle (`arccos(r / (r + h))`) rather than the raw optimiser estimate,
  avoiding a near-degeneracy that could misguide the final fit.

## [2.0.0] - 2026-05-07

### Added

- Ability to handle phones with multiple cameras (infer which via EXIF aperture tag).
- Cropping tool (`planet_ruler.crop.TkImageCropper`) that allows subselection of images while preserving camera properties.
- Initial suite of test imagery for benchmarking and calibration. Stored in Git LFS -- retrieve via `git lfs install; git lfs pull`.
- Extended codebase for performing suites of benchmark testing. See new .py files under planet_ruler/benchmarks, example configs under planet_ruler/benchmarks/configs, and a basic notebook for evaluation, planet_ruler/benchmarks/visualize.ipynb.
- More cameras/parameters to the database.
- A new tutorial that derives the minimum viable altitude for detecting limb curvature as a function of camera resolution (see notebooks/minimum_altitude_demo.ipynb).
- Parameter limit presets (tight, balanced, loose) for create_config_from_image.
- New minimizer hooks for l-bfgs-b and shgo.
- An entirely new method for limb meaurement based on fitting the sagitta (pixel distance between the top and bottom of the horizon), rather than the arc itself.

### Changed

- The fit_limb() function has been refactored to accept a fit_stages argument that specifies a 
list of methods that chain into one another. For example one could run `[{"method": "sagitta"}, 
{"method": "arc"}]` and the code will find new initial conditions and bounds based on the sagitta method that are fed into the arc fitter for an overall faster and more reliable answer. Play with combinations to see what works best!
- The old fit_limb() methods for gradient-field and l2, etc. losses have been split up into individual functions that only require the kwargs for their respective methods: fit_arc(), fit_gradient(), and fit_sagitta().
- Individual method functions do not link independently to the display dashboard -- dashboard is now a fit_limb (staged) tool only.
- Parameter limits are now set initially via preset combinations of tolerance, rather than as a flat fraction.
- Minimizer presets now lead to different optimized parameters depending on which detection method is being employed.
- User can now control where manual annotations are saved.
- Python versions below 3.10 are no longer supported.

### Fixed

- Small speedup to gradient-field eval function by caching the mean rather than recomputing
- Restored a missing image referenced in the README
- Large speedup to gradient-field detection by removing an unused full-resolution field calculation
- Faster fitting on manual annotation by only calculating the limb at labeled points
- Parameter initialization is now randomized from within bounds rather than limited after init, which caused bunching.
- Many API fixes in the documentation

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