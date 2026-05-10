# Planet Ruler Benchmark Suite

Systematic infrastructure for running and analyzing optimizer performance
across parameter-space configurations, minimizer choices, and image types.

## Quick Start

```bash
# Run smoke tests
python -m planet_ruler.benchmarks.run_benchmarks \
    planet_ruler/benchmarks/configs/smoke_test.yaml

# Run the synthetic grid (overwrites any previous DB)
python -m planet_ruler.benchmarks.run_benchmarks \
    planet_ruler/benchmarks/configs/synth_grid.yaml \
    --overwrite --parallel

# Run the real-image manual grid
python -m planet_ruler.benchmarks.run_benchmarks \
    planet_ruler/benchmarks/configs/manual.yaml \
    --parallel

# Vet real images before including them in a benchmark
python -m planet_ruler.benchmarks.vet_images

# Annotate a new image (saves JSON to annotations/)
python -m planet_ruler.annotate path/to/image.jpg \
    --output-dir planet_ruler/benchmarks/annotations/

# Analyze results
jupyter notebook planet_ruler/benchmarks/visualize.ipynb
```

## Directory Structure

```text
benchmarks/
├── __init__.py
├── runner.py              # BenchmarkRunner — executes scenarios, writes DB
├── analyzer.py            # BenchmarkAnalyzer — queries, Pareto, reliability
├── run_benchmarks.py      # CLI wrapper for runner.py
├── analyze_benchmarks.py  # CLI wrapper for analyzer.py
├── vet_images.py          # Vetting tool (checks images recover plausible h)
├── augment_annotations.py # Noisy annotation variant generator
├── synthetic.py           # Synthetic image + annotation generator
├── visualize.ipynb        # Interactive optimizer-performance notebook
├── annotation_rms.ipynb   # Pixel-level residual analysis for annotations
├── vet_inspect.ipynb      # Inspect vetting results visually
├── README.md
├── configs/
│   ├── smoke_test.yaml        # Fast CI regression tests
│   ├── synth_grid.yaml        # Minimizer grid on synthetic images
│   ├── manual_synth_grid.yaml # Manual-annotation variant of synth grid
│   ├── manual.yaml            # Vetted real images, manual annotation
│   └── full_suite.yaml        # All vetted real images, all methods
├── images/                # Benchmark images (EXIF required)
├── annotations/           # *_limb_points.json annotation files
└── results/               # benchmark_results.db (gitignored)
```

## Configuration Schema

Scenarios can be listed explicitly or generated with `grid:`.

### Explicit scenarios

```yaml
scenarios:
  - name: "my_scenario"
    images:
      - synth_iphone_13_h10km_clean
    detection_method: manual
    annotation_file: synth_iphone_13_h10km_clean_limb_points.json
    planet_name: Earth
    expected_radius: 6371000      # meters
    uncertainty_radius: 10000

    free_parameters: [r, h, theta_x, theta_y, theta_z]

    init_parameter_values_override:
      theta_z: 3.14159

    parameter_limits_override:
      theta_z: [-3.1416, 3.1416]

    r_limits_km: [5734, 7009]     # override r bounds (km)
    h_limits_pct: 0.20            # ±20 % of GPS h (only when h is free)
    perturbation_factor: 0.25     # resample init from bounds (0 = no perturb)

    minimizer: basinhopping
    minimizer_preset: balanced

    fit_params:
      max_iter: 300

    augmentation:
      n_variants: 5
      noise_sigma: 2.0
      seed: 0
```

### Grid expansion

```yaml
grid:
  images:
    - synth_iphone_13_h10km_clean
    - synth_sm-s906e_h10km_clean

  annotation_file_pattern: "{image}_limb_points.json"

  fixed:
    detection_method: manual
    planet_name: Earth
    expected_radius: 6371000
    uncertainty_radius: 10000

  param_grid:
    - minimizer: basinhopping
      minimizer_preset: [fast, balanced, robust]
      free_parameters:
        - [r, h, theta_x, theta_y, theta_z]
        - [r, h, theta_x, theta_y, theta_z, f, w]
      r_limits_km:
        - [5734, 7009]
        - [3186, 9557]
      h_limits_pct: [null, 0.20]
      fit_params_max_iter: [100, 300]
```

Every list-valued key in `param_grid` is a sweep axis; scalar values are
fixed for that sub-grid.  Multiple dicts in `param_grid` are unioned.
`h_limits_pct` rows are automatically skipped when `h` is not free.

## Database Columns

`results/benchmark_results.db` → table `benchmark_results`:

| Column | Type | Notes |
| ------ | ---- | ----- |
| `scenario_name` | TEXT | Compact encoded name for grid runs |
| `image_name` | TEXT | Image stem (no extension) |
| `timestamp` | TEXT | ISO-8601 |
| `git_commit` | TEXT | Short hash |
| `detection_method` | TEXT | manual / gradient-field / segmentation |
| `fit_params` | JSON | `{"max_iter": N, ...}` |
| `free_parameters` | JSON | `["r", "h", ...]` |
| `init_parameter_values` | JSON | Actual init values used |
| `parameter_limits` | JSON | `{"r": [lo, hi], "h": [lo, hi], ...}` |
| `minimizer_config` | JSON | Minimizer method string |
| `minimizer_preset` | TEXT | fast / balanced / robust / scipy-default |
| `best_parameters` | JSON | `{"r": 6371000, "h": 10000, ...}` |
| `fitted_radius` | REAL | `best_parameters["r"]` |
| `convergence_status` | TEXT | success / error |
| `iterations` | INT | `nit` from optimizer result |
| `absolute_error` | REAL | meters |
| `relative_error` | REAL | fraction |
| `within_uncertainty` | INT | 1/0 |
| `total_time` | REAL | seconds |
| `optimization_time` | REAL | seconds |
| `error_message` | TEXT | set on convergence_status=error |
| `annotation_noise_sigma` | REAL | σ of Gaussian noise added to annotation (0 = clean) |

Use `BenchmarkAnalyzer.explode_parameters()` to unpack JSON columns into
flat columns (`r`, `h`, `r_lower`, `r_upper`, `h_free`, `minimizer`, …).

## Analysis API

```python
from planet_ruler.benchmarks.analyzer import BenchmarkAnalyzer

analyzer = BenchmarkAnalyzer("planet_ruler/benchmarks/results/benchmark_results.db")

# Load all results (JSON columns pre-parsed)
df = analyzer.get_results()

# Unpack nested columns into flat columns for groupby / correlation
df = analyzer.explode_parameters()
# → adds: r, h, r_lower, r_upper, r_free, max_iter, minimizer, …

# N-dimensional Pareto frontier
pareto = analyzer.get_pareto_frontier(
    [("total_time", "min"), ("relative_error", "min")]
)

# Weighted Pareto scoring (lower score = better)
scores = analyzer.score_pareto(
    pareto,
    [("total_time", "min", 0.4), ("relative_error", "min", 0.6)],
)
pareto["score"] = scores
pareto_ranked = pareto.sort_values("score")

# Reliability: fraction of runs with relative_error < threshold
rl = analyzer.reliability(df[df["minimizer_preset"] == "robust"], threshold=0.05)

# Per-group reliability table
rl_table = (
    df.groupby(["minimizer", "minimizer_preset"])
    .apply(lambda g: analyzer.reliability(g))
    .rename("reliability")
)
```

## CLI Reference

### run_benchmarks

```text
python -m planet_ruler.benchmarks.run_benchmarks <config> [options]

  --scenario NAME    Filter to specific scenario(s) (repeatable)
  --image NAME       Filter to specific image(s) (repeatable)
  --parallel         Run batches in parallel
  --workers N        Number of worker processes (default: CPU/2)
  --no-skip          Re-run scenarios already in DB
  --overwrite        Delete existing DB before running (fresh start)
  --db PATH          Override default DB path
```

### vet_images

```text
python -m planet_ruler.benchmarks.vet_images [options]

  --image STEM       Vet a specific image (repeatable)
  --fast             Quick preview: preset=balanced, max-iter=100
  --two-step         Bootstrap check: step 1 fixes r, recovers h; step 2 pins h, recovers r
  --update-exif      After --two-step, write fitted altitude back to EXIF GPS tag (PASS/WARN only)
  --final-scan       Pin h from EXIF (written by --update-exif), free r, verify r ≈ 6371 km
  --preset NAME      fast / balanced / robust (default: robust)
  --max-iter N       Override iteration budget (default: 1000)
  --log CSV_PATH     Save results to CSV
```

Default vetting (no flags): fixes r=6371 km, fits h free in [1.5–18 km] with the robust preset.
PASS: h in [3–14 km].  WARN: [1.5–3) or (14–18].  FAIL: outside WARN range or convergence error.

Recommended workflow for new images: `--two-step` to find h, `--update-exif` to persist it,
then `--final-scan` to confirm r recovery before adding the image to a config YAML.

## Adding Images

1. Place `image.jpg` in `benchmarks/images/` (EXIF with focal length required)
2. Annotate: `python -m planet_ruler.annotate image.jpg --output-dir benchmarks/annotations/`
3. Vet (two-step recommended):

   ```bash
   python -m planet_ruler.benchmarks.vet_images --image image_stem --two-step
   python -m planet_ruler.benchmarks.vet_images --image image_stem --two-step --update-exif
   python -m planet_ruler.benchmarks.vet_images --image image_stem --final-scan
   ```

4. If PASS/WARN on final-scan, add to a config YAML under `grid.images` or `scenarios[].images`
