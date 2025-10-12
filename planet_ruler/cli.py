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
from pathlib import Path
import yaml
import json
from typing import Optional, Dict, Any

import planet_ruler as pr


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
  planet-ruler measure photo.jpg --camera-config config/earth_iss.yaml
  planet-ruler measure --image photo.jpg --altitude 400 --focal-length 50
  planet-ruler demo --planet earth
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Measure command
    measure_parser = subparsers.add_parser(
        "measure", help="Measure planetary radius from image"
    )
    measure_parser.add_argument("image", help="Path to horizon/limb photograph")
    measure_parser.add_argument(
        "--camera-config",
        "-c",
        type=str,
        required=True,
        help="Path to camera configuration YAML/JSON file (required)",
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
        choices=["manual", "gradient-break", "segmentation"],
        default="manual",
        help="Limb detection method (default: manual)",
    )

    # Quick measurement parameters (alternative to config file)
    measure_parser.add_argument(
        "--altitude", type=float, help="Altitude above surface in km"
    )
    measure_parser.add_argument(
        "--focal-length", type=float, help="Camera focal length in mm"
    )
    measure_parser.add_argument("--sensor-width", type=float, help="Sensor width in mm")

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

    # Load configuration (now required)
    try:
        config = load_config(args.camera_config)
        print(f"Loaded camera configuration from: {args.camera_config}")
    except Exception as e:
        print(f"Error loading configuration: {e}", file=sys.stderr)
        return 1

    # Override with command-line parameters
    if args.altitude is not None:
        config["altitude_km"] = args.altitude
    if args.focal_length is not None:
        config["focal_length_mm"] = args.focal_length
    if args.sensor_width is not None:
        config["sensor_width_mm"] = args.sensor_width

    try:
        # Create observation with detection method
        obs = pr.LimbObservation(
            args.image,
            fit_config=args.camera_config,
            limb_detection=args.detection_method,
        )

        # Apply configuration
        if config:
            for key, value in config.items():
                if hasattr(obs, key):
                    setattr(obs, key, value)

        print(f"Detecting horizon/limb using {args.detection_method} method...")
        obs.detect_limb()

        print("Fitting limb model...")
        obs.fit_limb()

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
            import subprocess

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
