# Planet Ruler: Quick Reference Guide

## Minimal Working Examples

### 1. Absolute Minimum (Auto-Config + Manual Annotation)
```python
from planet_ruler.observation import LimbObservation
from planet_ruler.camera import create_config_from_image

# Auto-generate config from your image
config = create_config_from_image(
    image_path="your_photo.jpg",
    altitude_m=10_000,
    planet="earth"
)

# Load observation
obs = LimbObservation("your_photo.jpg", config)

# Manual annotation (GUI opens)
obs.detect_limb()

# Fit and get results
obs.fit_limb(dashboard=True)
print(f"Radius: {obs.radius_km:.1f} ± {obs.radius_uncertainty:.1f} km")
```

### 2. Gradient-Field Method (Automatic, Lightweight)
```python
from planet_ruler.camera import create_config_from_image
from planet_ruler.observation import LimbObservation
from planet_ruler.dashboard import OutputCapture

config = create_config_from_image("your_photo.jpg", altitude_m=10_000, planet="earth")
obs = LimbObservation("your_photo.jpg", config)

# Direct gradient-field optimization (no detection needed)
# Ideal for batch processing - lightweight, no models
capture = OutputCapture(max_lines=20)
with capture:
    obs.fit_limb(
        minimizer="dual-annealing",
        loss_function="gradient_field",
        resolution_stages=[4, 2, 1],
        image_smoothing=2.0,
        kernel_smoothing=8.0,
        max_iter=3000,
        verbose=True,
        dashboard=True,
        dashboard_kwargs={'output_capture': capture}
    )

print(f"Radius: {obs.radius_km:.1f} ± {obs.radius_uncertainty:.1f} km")
```

### 3. ML Segmentation (Automatic but Heavyweight)
```python
from planet_ruler.camera import create_config_from_image
from planet_ruler.observation import LimbObservation

config = create_config_from_image("your_photo.jpg", altitude_m=10_000, planet="earth")
obs = LimbObservation("your_photo.jpg", config)

# ML segmentation (downloads ~2GB model on first use)
# Note: Gradient-field (Example 2) is lighter for batch processing
obs.detect_limb(detection_method="segmentation")
obs.smooth_limb(method="rolling-median", window_length=3)

# Fit with dashboard
obs.fit_limb(dashboard=True)
print(f"Radius: {obs.radius_km:.1f} ± {obs.radius_uncertainty:.1f} km")
```

### 4. With Parameter Extraction Visibility
```python
from planet_ruler.camera import extract_camera_parameters, create_config_from_image
from planet_ruler.observation import LimbObservation

# See what camera info was extracted
camera_info = extract_camera_parameters("your_photo.jpg")
print(f"Camera: {camera_info.get('camera_model', 'Unknown')}")
print(f"Focal length: {camera_info['focal_length_mm']:.2f} mm")
print(f"Sensor: {camera_info['sensor_width_mm']:.2f} mm")
print(f"Confidence: {camera_info['confidence']}")

# Check for GPS altitude
from planet_ruler.camera import get_gps_altitude
gps_alt = get_gps_altitude("your_photo.jpg")
if gps_alt:
    print(f"GPS altitude: {gps_alt:.1f} m")
    altitude_m = gps_alt
else:
    altitude_m = 10_000 # Your manual estimate

# Create config and proceed
config = create_config_from_image("your_photo.jpg", altitude_m, "earth")
obs = LimbObservation("your_photo.jpg", config)
obs.detect_limb()
obs.fit_limb(dashboard=True)
```

### 5. With Configuration Validation
```python
from planet_ruler.camera import create_config_from_image
from planet_ruler.validation import validate_limb_config
from planet_ruler.observation import LimbObservation

# Generate config
config = create_config_from_image("photo.jpg", altitude_m=10_000, planet="earth")

# Validate before using
try:
    validate_limb_config(config, strict=True)
    print("✓ Configuration validated")
except AssertionError as e:
    print(f"⚠️  Warning: {e}")

# Proceed with measurement
obs = LimbObservation("photo.jpg", config)
obs.detect_limb()
obs.fit_limb()
```

### 6. With Full Uncertainty Analysis
```python
from planet_ruler.uncertainty import calculate_parameter_uncertainty
from planet_ruler.fit import format_parameter_result
from planet_ruler.camera import create_config_from_image
from planet_ruler.observation import LimbObservation

config = create_config_from_image("photo.jpg", altitude_m=10_000, planet="earth")
obs = LimbObservation("photo.jpg", config)
obs.detect_limb()
obs.fit_limb()

# Get radius uncertainty with auto method selection
radius_result = calculate_parameter_uncertainty(
    obs, 
    parameter='r',
    scale_factor=1000,  # Convert m to km
    method='auto'
)

print(format_parameter_result(radius_result, unit='km'))
print(f"Method used: {radius_result['method']}")

# Get uncertainty for any parameter
altitude_result = calculate_parameter_uncertainty(obs, 'h', scale_factor=1000)
print(format_parameter_result(altitude_result, unit='km'))
```

### 6. Comparing All Three Methods
```python
from planet_ruler.camera import create_config_from_image
from planet_ruler.observation import LimbObservation

config = create_config_from_image("photo.jpg", altitude_m=10_000, planet="earth")

# Method 1: Manual annotation
obs_manual = LimbObservation("photo.jpg", config)
obs_manual.detect_limb()
obs_manual.fit_limb()
radius_manual = obs_manual.radius_km

# Method 2: Gradient-field (lightweight, ideal for batch)
obs_gradient = LimbObservation("photo.jpg", config)
obs_gradient.fit_limb(
    minimizer="dual-annealing",
    loss_function="gradient_field",
    resolution_stages=[4, 2, 1],
    image_smoothing=2.0,
    kernel_smoothing=8.0
)
radius_gradient = obs_gradient.radius_km

# Method 3: ML Segmentation (heavyweight)
obs_ml = LimbObservation("photo.jpg", config)
obs_ml.limb_detection = "segmentation"
obs_ml.detect_limb()
obs_ml.smooth_limb()
obs_ml.fit_limb(loss_function="l1")
radius_ml = obs_ml.radius_km

print(f"Manual:          {radius_manual:.1f} km")
print(f"Gradient-field:  {radius_gradient:.1f} km")
print(f"ML Segmentation: {radius_ml:.1f} km")
print(f"\nSpread: {max(radius_manual, radius_gradient, radius_ml) - min(radius_manual, radius_gradient, radius_ml):.1f} km")
```

### 7. Batch Processing Multiple Images
```python
from planet_ruler.camera import create_config_from_image
from planet_ruler.observation import LimbObservation
import glob

image_files = glob.glob("flight_photos/*.jpg")
results = []

# Use gradient-field for batch processing (lightweight, no models, efficient)
for img_path in image_files:
    try:
        config = create_config_from_image(img_path, altitude_km=10, planet="earth")
        obs = LimbObservation(img_path, config)
        
        # Gradient-field: automatic, lightweight, ideal for batch
        obs.fit_limb(
            loss_function="gradient_field",
            minimizer="dual-annealing",
            resolution_stages=[4, 2, 1],
            image_smoothing=2.0,
            kernel_smoothing=8.0
        )
        
        results.append({
            'file': img_path,
            'radius_km': obs.radius_km,
            'uncertainty_km': obs.radius_uncertainty
        })
        print(f"✓ {img_path}: {obs.radius_km:.1f} ± {obs.radius_uncertainty:.1f} km")
    except Exception as e:
        print(f"✗ {img_path}: {e}")

# Calculate average
import numpy as np
avg_radius = np.mean([r['radius_km'] for r in results])
print(f"\nAverage radius: {avg_radius:.1f} km")
```

## Method Selection Guide

### When to Use Manual Annotation
- ✓ First time using planet-ruler
- ✓ Complex atmospheric features (clouds, haze)
- ✓ Subtle or ambiguous horizons
- ✓ Educational demonstrations
- ✓ When you want full control
- ✓ Any difficult case

### When to Use Gradient-Field
- ✓ **Batch processing many images** (most efficient)
- ✓ Clean, sharp horizons
- ✓ Want fully automatic processing
- ✓ Limited compute resources (no GPU needed)
- ✓ Mobile-ready, lightweight solution
- ✗ NOT for cloudy/hazy images
- ✗ NOT for complex atmospheric layers

### When to Use ML Segmentation
- ✓ Clear, distinct planet/space boundaries
- ✓ Have GPU or don't mind CPU time
- ✓ Single/few images (not ideal for batch)
- ✓ Want to compare with other methods
- ✗ **NOT for batch processing** (heavy model, high memory)
- ✗ NOT for first-time users
- ✗ NOT for subtle horizons
- ✗ NOT if compute-limited

**Rule of thumb:** 
- First time? → Manual
- Batch processing? → Gradient-field
- Single image with GPU? → Try gradient-field first, ML if that fails

## Dashboard Configuration Options

```python
# Basic dashboard
obs.fit_limb(dashboard=True)

# Custom dashboard settings
from planet_ruler.dashboard import OutputCapture

capture = OutputCapture(max_lines=20, line_width=70)
with capture:
    obs.fit_limb(
        dashboard=True,
        target_planet="earth",  # For comparison display
        dashboard_kwargs={
            'output_capture': capture,    # Show print/log output
            'width': 80,                   # Dashboard width
            'max_warnings': 5,             # Warning slots
            'max_hints': 4,                # Hint slots
            'min_message_display_time': 5.0,  # Message duration
            'min_refresh_delay': 0.0,      # 0.0 = adaptive refresh
            'refresh_frequency': 1         # Refresh every N iterations
        }
    )
```

## Common Issues and Solutions

### Issue: "No EXIF data found"
**Solution:** Some images strip EXIF data. Manually specify camera parameters:
```python
config = create_config_from_image(
    "photo.jpg",
    altitude_km=10,
    planet="earth",
    focal_length_mm=24.0,      # Override if known
    sensor_width_mm=23.6,      # Override if known
    field_of_view_deg=60.0     # Or specify FOV directly
)
```

### Issue: "ML segmentation model download failing"
**Solution:** The Segment Anything Model is large (~2GB). Options:
```python
# 1. Try manual annotation instead
obs.detect_limb(detection_method="manual")

# 2. Or use gradient-field (no model needed)
obs.fit_limb(loss_function="gradient_field", minimizer="differential-evolution", resolution_stages=[4, 2, 1])

# 3. Check your internet connection and try again
# The model is cached after first download
```

### Issue: "ML segmentation detecting wrong boundary"
**Solution:** SAM may struggle with subtle horizons. Try:
```python
# 1. Use manual annotation for more control
obs.limb_detection = "manual"
obs.detect_limb()

# 2. Or use gradient-field optimization
obs.fit_limb(loss_function="gradient_field", resolution_stages='auto')

# 3. For batch processing, validate ML results before fitting
obs.detect_limb()
obs.plot()  # Visually inspect before fit_limb()
```

### Issue: "Optimization not converging"
**Solution:** Try gradient-field with more stages or adjust smoothing:
```python
obs.fit_limb(
    loss_function="gradient_field",
    resolution_stages=[8, 4, 2, 1],  # More stages
    image_smoothing=3.0,              # More smoothing
    kernel_smoothing=12.0,
    max_iter=5000                     # More iterations
)
```

### Issue: "Radius is way off"
**Solution:** Check altitude accuracy - it's the most common source of error:
```python
# Check for GPS altitude in EXIF
from planet_ruler.camera import get_gps_altitude
gps_alt = get_gps_altitude("photo.jpg")
if gps_alt:
    print(f"GPS says: {gps_alt/1000:.2f} km")
```

## Tips for Best Results

1. **Altitude is critical**: 10% error in altitude ≈ 10% error in radius
2. **Clear horizons work best**: Avoid heavy clouds or atmospheric layers
3. **Use GPS altitude**: EXIF GPS data is more accurate than estimates
4. **Higher is better**: Measurements from >10km altitude are more reliable
5. **Try multiple methods**: Manual, ML, and gradient-field can validate each other
6. **For ML segmentation**: Always inspect with `obs.plot()` after detection - validate before fitting
7. **Check the dashboard**: Watch for warnings about parameter drift or stalling
8. **Validate first**: Always run `validate_limb_config()` before optimizing
9. **First ML use**: Be patient with ~2GB model download (one-time only)

## Performance Notes

- **Manual annotation**: ~1 minute for clicking, ~2-5 minutes for optimization
- **ML segmentation**: ~10-30 seconds (GPU) or 1-5 minutes (CPU) for detection, ~2-5 minutes for optimization
  - First use: Add 2-5 minutes for ~2GB model download (one-time)
  - Subsequent uses: Model is cached locally
- **Gradient-field**: ~5-10 minutes for multi-resolution optimization
- **Dashboard overhead**: ~2-5% with adaptive refresh (negligible impact)
- **Multi-resolution**: Each stage takes ~30-40% of single-resolution time
- **Memory**: Full-resolution optimization uses ~2-4x image size in RAM
- **ML model**: Additional ~2GB RAM when using segmentation