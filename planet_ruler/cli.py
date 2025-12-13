# Copyright 2025 Brandon Anderson
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3
"""
Command-line interface for planet_ruler

This module provides a simple CLI for measuring planetary radii from horizon photographs.
"""

import argparse
import sys
import os
import subprocess
from pathlib import Path
import yaml
import json
from typing import Optional, Dict, Any

import planet_ruler as pr
from planet_ruler.camera import create_config_from_image


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    with open(config_file, "r") as f:
        if config_file.suffix.lower() in [".yaml", ".yml"]:
            return yaml.safe_load(f)
        elif config_file.suffix.lower() == ".json":
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_file.suffix}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="planet-ruler",
        description="Measure planetary radii from horizon photographs using computer vision",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic measurement with config file
  planet-ruler measure photo.jpg --camera-config config/earth_iss.yaml
  
  # Auto-generate config from image EXIF data
  planet-ruler measure photo.jpg --auto-config --altitude 10668
  
  # Advanced gradient field optimization with multi-resolution
  planet-ruler measure photo.jpg --auto-config --altitude 10668 \\
    --loss-function gradient_field --multi-resolution auto --verbose
  
  # Warm start optimization (refine previous results)
  planet-ruler measure photo.jpg --camera-config config/earth_iss.yaml \\
    --loss-function gradient_field --warm-start
  
  # Remove image artifacts and use robust optimization
  planet-ruler measure photo.jpg --auto-config --altitude 10668 \\
    --loss-function gradient_field --image-smoothing 2.0 --minimizer-preset robust
  
  # Custom multi-resolution stages
  planet-ruler measure photo.jpg --camera-config config/earth_iss.yaml \\
    --loss-function gradient_field --multi-resolution "8,4,2,1" --max-iterations 20000
  
  # Traditional demo
  planet-ruler demo --planet earth
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Measure command
    measure_parser = subparsers.add_parser(
        "measure", help="Measure planetary radius from image"
    )
    measure_parser.add_argument("image", help="Path to horizon/limb photograph")

    # Config options (mutually exclusive)
    config_group = measure_parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument(
        "--camera-config",
        "-c",
        type=str,
        help="Path to camera configuration YAML/JSON file",
    )
    config_group.add_argument(
        "--auto-config",
        action="store_true",
        help="Auto-generate camera config from image EXIF data (requires --altitude)",
    )
    measure_parser.add_argument(
        "--output", "-o", type=str, help="Output file for results (JSON format)"
    )
    measure_parser.add_argument(
        "--plot", action="store_true", help="Show visualization plots"
    )
    measure_parser.add_argument(
        "--save-plots", type=str, help="Directory to save visualization plots"
    )

    # Detection method
    measure_parser.add_argument(
        "--detection-method",
        "-d",
        choices=["manual", "gradient-break", "gradient-field", "segmentation"],
        default="manual",
        help="Limb detection method (default: manual)",
    )

    # Fitting options
    measure_parser.add_argument(
        "--loss-function",
        "-l",
        choices=["l1", "l2", "log-l1", "gradient_field"],
        default="l2",
        help="Loss function for fitting (default: l2)",
    )
    measure_parser.add_argument(
        "--warm-start",
        action="store_true",
        help="Use results from previous fit as starting point for optimization (requires previous fit)",
    )
    measure_parser.add_argument(
        "--multi-resolution",
        "--resolution-stages",
        type=str,
        help="Multi-resolution optimization stages. Use 'auto' for automatic, or comma-separated downsampling factors (e.g., '4,2,1'). Only works with gradient_field loss function.",
    )
    measure_parser.add_argument(
        "--max-iterations",
        type=int,
        default=15000,
        help="Maximum optimization iterations (default: 15000)",
    )
    measure_parser.add_argument(
        "--minimizer-preset",
        choices=["fast", "balanced", "robust", "planet-ruler", "scipy-default"],
        default="balanced",
        help="Optimization strategy preset (default: balanced)",
    )
    measure_parser.add_argument(
        "--image-smoothing",
        type=float,
        help="Gaussian blur sigma for image preprocessing (removes artifacts like crater rims). Only applies to gradient_field loss.",
    )
    measure_parser.add_argument(
        "--kernel-smoothing",
        type=float,
        default=5.0,
        help="Gaussian blur for gradient field estimation (default: 5.0). Only applies to gradient_field loss.",
    )
    measure_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed fitting progress",
    )

    # Dashboard options
    measure_parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Show live progress dashboard during optimization",
    )
    measure_parser.add_argument(
        "--dashboard-width",
        type=int,
        default=63,
        help="Dashboard width in characters (default: 63)",
    )
    measure_parser.add_argument(
        "--max-warnings",
        type=int,
        default=3,
        help="Number of warning message slots in dashboard (default: 3)",
    )
    measure_parser.add_argument(
        "--max-hints",
        type=int,
        default=3,
        help="Number of hint message slots in dashboard (default: 3)",
    )

    # Auto-config parameters
    measure_parser.add_argument(
        "--altitude",
        "-a",
        type=float,
        help="Altitude above surface in meters (required with --auto-config, optional override for config files)",
    )
    measure_parser.add_argument(
        "--planet",
        "-p",
        type=str,
        default="earth",
        help="Planet name for auto-config initial radius guess (default: earth)",
    )

    # Manual override parameters (for config files or auto-config)
    measure_parser.add_argument(
        "--focal-length", type=float, help="Camera focal length in mm (override)"
    )
    measure_parser.add_argument(
        "--sensor-width", type=float, help="Sensor width in mm (override)"
    )
    measure_parser.add_argument(
        "--field-of-view", type=float, help="Camera field of view in degrees (override)"
    )

    # Demo command
    demo_parser = subparsers.add_parser(
        "demo", help="Run demonstration with example data"
    )
    demo_parser.add_argument(
        "--planet",
        choices=["earth", "saturn", "pluto"],
        help="Planet to demonstrate with",
    )
    demo_parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch interactive Jupyter notebook demo",
    )

    # List examples command
    list_parser = subparsers.add_parser(
        "list", help="List available example configurations"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        if args.command == "measure":
            return measure_command(args)
        elif args.command == "demo":
            return demo_command(args)
        elif args.command == "list":
            return list_command(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def measure_command(args):
    """Handle the measure command."""
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}", file=sys.stderr)
        return 1

    print(f"Loading image: {args.image}")

    # Handle configuration - either load from file or auto-generate
    if args.auto_config:
        # Auto-generate config from image
        if args.altitude is None:
            print(
                "Error: --altitude is required when using --auto-config",
                file=sys.stderr,
            )
            return 1

        print("Auto-generating camera configuration from image EXIF data...")
        try:
            config = create_config_from_image(
                args.image, altitude_m=args.altitude, planet=args.planet
            )
            print(
                f"✓ Auto-generated config for {config.get('camera_info', {}).get('camera_model', 'Unknown')} camera"
            )
            print(
                f"  Confidence: {config.get('camera_info', {}).get('confidence', 'unknown')}"
            )

        except ValueError as e:
            print(f"Error generating config: {e}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"Unexpected error generating config: {e}", file=sys.stderr)
            return 1
    else:
        # Load config from file
        try:
            config = load_config(args.camera_config)
            print(f"Loaded camera configuration from: {args.camera_config}")
        except Exception as e:
            print(f"Error loading configuration: {e}", file=sys.stderr)
            return 1

    # Override with command-line parameters (both auto-config and file config)
    if args.altitude is not None and not args.auto_config:
        # Convert km to m for consistency (config files might use km, auto-config uses m)
        altitude_m = args.altitude * 1000 if args.altitude < 1000 else args.altitude
        if "init_parameter_values" in config:
            config["init_parameter_values"]["h"] = altitude_m
        else:
            config["altitude_m"] = altitude_m

    if args.focal_length is not None:
        focal_m = args.focal_length / 1000  # Convert mm to m
        if "init_parameter_values" in config:
            config["init_parameter_values"]["f"] = focal_m
        else:
            config["focal_length_mm"] = args.focal_length

    if args.sensor_width is not None:
        sensor_m = args.sensor_width / 1000  # Convert mm to m
        if "init_parameter_values" in config:
            config["init_parameter_values"]["w"] = sensor_m
        else:
            config["sensor_width_mm"] = args.sensor_width

    if args.field_of_view is not None:
        fov_rad = args.field_of_view * 3.14159 / 180  # Convert degrees to radians
        if "init_parameter_values" in config:
            config["init_parameter_values"]["fov"] = fov_rad
        else:
            config["field_of_view_deg"] = args.field_of_view

    try:
        # Create observation with detection method
        # Use config dict directly if auto-generated, otherwise use file path
        fit_config = config if args.auto_config else args.camera_config
        obs = pr.LimbObservation(
            args.image,
            fit_config=fit_config,
            limb_detection=args.detection_method,
        )

        print(f"Detecting horizon/limb using {args.detection_method} method...")
        obs.detect_limb()

        print("Fitting limb model...")

        # Parse multi-resolution stages if provided
        resolution_stages = None
        if args.multi_resolution:
            if args.multi_resolution.lower() == "auto":
                resolution_stages = "auto"
            else:
                try:
                    resolution_stages = [
                        int(x.strip()) for x in args.multi_resolution.split(",")
                    ]
                except ValueError:
                    print(
                        f"Error: Invalid multi-resolution format '{args.multi_resolution}'. Use 'auto' or comma-separated integers like '4,2,1'",
                        file=sys.stderr,
                    )
                    return 1

        # Build fit_limb kwargs
        fit_kwargs = {
            "loss_function": args.loss_function,
            "max_iter": args.max_iterations,
            "minimizer_preset": args.minimizer_preset,
            "warm_start": args.warm_start,
            "verbose": args.verbose,
            "kernel_smoothing": args.kernel_smoothing,
            "dashboard": args.dashboard,
        }

        # Dashboard configuration
        if args.dashboard:
            fit_kwargs["dashboard_kwargs"] = {
                "width": args.dashboard_width,
                "max_warnings": args.max_warnings,
                "max_hints": args.max_hints,
            }

        # Add optional parameters
        if resolution_stages is not None:
            fit_kwargs["resolution_stages"] = resolution_stages

        if args.image_smoothing is not None:
            fit_kwargs["image_smoothing"] = args.image_smoothing

        # Apply fit with options
        obs.fit_limb(**fit_kwargs)

        # Display results
        print(f"\nResults:")
        print(
            f"  Estimated planetary radius: {obs.radius_km:.0f} ± {getattr(obs, 'radius_uncertainty', 0):.0f} km"
        )
        if hasattr(obs, "altitude_km"):
            print(f"  Observer altitude: {obs.altitude_km:.1f} km")
        if hasattr(obs, "focal_length_mm"):
            print(f"  Camera focal length: {obs.focal_length_mm:.1f} mm")

        # Save results if requested
        if args.output:
            results = {
                "image_path": args.image,
                "radius_km": float(obs.radius_km),
                "radius_uncertainty_km": float(getattr(obs, "radius_uncertainty", 0)),
                "configuration": config,
            }

            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {args.output}")

        # Show plots if requested
        if args.plot:
            obs.plot()
            print("Close the plot window to continue...")

        # Save plots if requested
        if args.save_plots:
            import matplotlib.pyplot as plt

            os.makedirs(args.save_plots, exist_ok=True)
            plot_path = os.path.join(
                args.save_plots, f"{Path(args.image).stem}_analysis.png"
            )
            obs.plot(show=False)
            plt.savefig(plot_path)
            plt.close()
            print(f"Plot saved to: {plot_path}")

        return 0

    except Exception as e:
        print(f"Error during measurement: {e}", file=sys.stderr)
        return 1


def demo_command(args):
    """Handle the demo command."""
    if args.interactive:
        try:
            import jupyter_core.command

            notebook_path = (
                Path(__file__).parent.parent / "notebooks" / "limb_demo.ipynb"
            )
            subprocess.run(["jupyter", "notebook", str(notebook_path)], check=True)
        except ImportError:
            print(
                "Jupyter notebook not available. Install with: pip install jupyter",
                file=sys.stderr,
            )
            return 1
        except subprocess.CalledProcessError:
            print("Error launching Jupyter notebook", file=sys.stderr)
            return 1
    else:
        # Run preset demo
        try:
            from planet_ruler.demo import make_dropdown, load_demo_parameters

            if args.planet:
                print(f"Running {args.planet.title()} demonstration...")
                # TODO: Implement specific planet demo logic
                print(
                    f"Demo for {args.planet} completed. Use --interactive for full experience."
                )
            else:
                print("Available demonstrations:")
                print("  earth  - Earth from ISS")
                print("  saturn - Saturn from Cassini")
                print("  pluto  - Pluto from New Horizons")
                print("\nUse: planet-ruler demo --planet <name>")
                print("Or try: planet-ruler demo --interactive")

        except ImportError as e:
            print(f"Demo functionality not available: {e}", file=sys.stderr)
            return 1

    return 0


def list_command(args):
    """Handle the list command."""
    config_dir = Path(__file__).parent.parent / "config"

    print("Available example configurations:")

    if config_dir.exists():
        for config_file in config_dir.glob("*.yaml"):
            try:
                config = load_config(str(config_file))
                name = config_file.stem
                description = config.get("description", "No description available")
                print(f"  {name:<20} - {description}")
            except Exception:
                print(f"  {config_file.stem:<20} - Error loading configuration")
    else:
        print("  No configuration directory found")

    return 0


if __name__ == "__main__":
    sys.exit(main())
