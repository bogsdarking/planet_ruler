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

"""
Basic test coverage for planet_ruler CLI module focusing on functionality that can be tested
without complex mocking of the observation classes.
"""

import os
import sys
import json
import yaml
import tempfile
import pytest
import subprocess
import numpy as np
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock, call
from io import StringIO

# Add the planet_ruler package to the path for testing
sys.path.insert(0, str(Path(__file__).parent.parent))

from planet_ruler import cli


class TestLoadConfig:
    """Test configuration file loading functionality."""

    def test_load_yaml_config(self):
        """Test loading YAML configuration file."""
        yaml_content = """
        altitude_km: 400
        focal_length_mm: 50
        sensor_width_mm: 36
        description: "Test ISS configuration"
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name

        try:
            config = cli.load_config(yaml_path)
            assert config["altitude_km"] == 400
            assert config["focal_length_mm"] == 50
            assert config["sensor_width_mm"] == 36
            assert config["description"] == "Test ISS configuration"
        finally:
            os.unlink(yaml_path)

    def test_load_yml_config(self):
        """Test loading .yml configuration file."""
        yml_content = """
        altitude_km: 800
        focal_length_mm: 85
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(yml_content)
            yml_path = f.name

        try:
            config = cli.load_config(yml_path)
            assert config["altitude_km"] == 800
            assert config["focal_length_mm"] == 85
        finally:
            os.unlink(yml_path)

    def test_load_json_config(self):
        """Test loading JSON configuration file."""
        json_config = {
            "altitude_km": 200,
            "focal_length_mm": 24,
            "sensor_width_mm": 24,
            "description": "Test drone configuration",
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(json_config, f)
            json_path = f.name

        try:
            config = cli.load_config(json_path)
            assert config == json_config
        finally:
            os.unlink(json_path)

    def test_load_config_file_not_found(self):
        """Test error handling when configuration file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            cli.load_config("/nonexistent/path/config.yaml")

    def test_load_config_unsupported_format(self):
        """Test error handling for unsupported file formats."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("some content")
            txt_path = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported config format"):
                cli.load_config(txt_path)
        finally:
            os.unlink(txt_path)

    def test_load_config_invalid_yaml(self):
        """Test error handling for invalid YAML content."""
        invalid_yaml = """
        altitude_km: 400
        focal_length_mm: [invalid: yaml
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(invalid_yaml)
            yaml_path = f.name

        try:
            with pytest.raises(yaml.YAMLError):
                cli.load_config(yaml_path)
        finally:
            os.unlink(yaml_path)

    def test_load_config_invalid_json(self):
        """Test error handling for invalid JSON content."""
        invalid_json = '{"altitude_km": 400, "invalid": json}'

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(invalid_json)
            json_path = f.name

        try:
            with pytest.raises(json.JSONDecodeError):
                cli.load_config(json_path)
        finally:
            os.unlink(json_path)


class TestMainFunction:
    """Test the main CLI entry point."""

    @patch("sys.argv", ["planet-ruler"])
    @patch("planet_ruler.cli.argparse.ArgumentParser.print_help")
    def test_main_no_command(self, mock_print_help):
        """Test main function with no command provided."""
        result = cli.main()
        assert result == 1
        mock_print_help.assert_called_once()

    @patch(
        "sys.argv",
        ["planet-ruler", "measure", "test.jpg", "--camera-config", "config.yaml"],
    )
    @patch("planet_ruler.cli.measure_command")
    def test_main_measure_command(self, mock_measure_command):
        """Test main function with measure command."""
        mock_measure_command.return_value = 0
        result = cli.main()
        assert result == 0
        mock_measure_command.assert_called_once()

    @patch("sys.argv", ["planet-ruler", "demo"])
    @patch("planet_ruler.cli.demo_command")
    def test_main_demo_command(self, mock_demo_command):
        """Test main function with demo command."""
        mock_demo_command.return_value = 0
        result = cli.main()
        assert result == 0
        mock_demo_command.assert_called_once()

    @patch("sys.argv", ["planet-ruler", "list"])
    @patch("planet_ruler.cli.list_command")
    def test_main_list_command(self, mock_list_command):
        """Test main function with list command."""
        mock_list_command.return_value = 0
        result = cli.main()
        assert result == 0
        mock_list_command.assert_called_once()

    @patch(
        "sys.argv",
        ["planet-ruler", "measure", "test.jpg", "--camera-config", "config.yaml"],
    )
    @patch("planet_ruler.cli.measure_command")
    def test_main_keyboard_interrupt(self, mock_measure_command):
        """Test main function handling KeyboardInterrupt."""
        mock_measure_command.side_effect = KeyboardInterrupt()

        with patch("sys.stderr", new=StringIO()) as fake_stderr:
            result = cli.main()
            assert result == 1
            assert "Operation cancelled by user" in fake_stderr.getvalue()

    @patch(
        "sys.argv",
        ["planet-ruler", "measure", "test.jpg", "--camera-config", "config.yaml"],
    )
    @patch("planet_ruler.cli.measure_command")
    def test_main_exception_handling(self, mock_measure_command):
        """Test main function handling general exceptions."""
        mock_measure_command.side_effect = Exception("Test error")

        with patch("sys.stderr", new=StringIO()) as fake_stderr:
            result = cli.main()
            assert result == 1
            assert "Error: Test error" in fake_stderr.getvalue()


class TestMeasureCommand:
    """Test the measure command functionality with basic mocking."""

    def test_measure_command_image_not_found(self, capsys):
        """Test measure command with non-existent image file."""
        args = MagicMock()
        args.image = "/nonexistent/image.jpg"
        args.camera_config = None
        args.altitude = None
        args.focal_length = None
        args.sensor_width = None
        args.output = None
        args.plot = False
        args.save_plots = None

        result = cli.measure_command(args)
        assert result == 1

        captured = capsys.readouterr()
        assert "Image file not found" in captured.err

    @patch("os.path.exists")
    @patch("planet_ruler.cli.load_config")
    def test_measure_command_config_loading_error(
        self, mock_load_config, mock_exists, capsys
    ):
        """Test measure command with configuration loading error."""
        mock_exists.return_value = True
        mock_load_config.side_effect = Exception("Config error")

        args = MagicMock()
        args.image = "test.jpg"
        args.camera_config = "config.yaml"
        args.altitude = None
        args.focal_length = None
        args.sensor_width = None
        args.output = None
        args.plot = False
        args.save_plots = None

        result = cli.measure_command(args)
        assert result == 1

        captured = capsys.readouterr()
        assert "Error loading configuration" in captured.err


class TestMeasureCommandManualDetection:
    """Test the measure command with manual detection method."""

    @patch("os.path.exists")
    @patch("planet_ruler.cli.load_config")
    @patch("planet_ruler.cli.pr.LimbObservation")
    def test_measure_command_manual_method_success(
        self, mock_limb_observation, mock_load_config, mock_exists, capsys
    ):
        """Test measure command with manual detection method success."""

        # Mock both image file and config file existence
        def mock_exists_side_effect(path):
            return path in ["test.jpg", "config.yaml"]

        mock_exists.side_effect = mock_exists_side_effect

        mock_load_config.return_value = {
            "altitude_km": 408,
            "focal_length_mm": 50,
            "sensor_width_mm": 36,
        }

        # Mock the observation object and its methods
        mock_obs = MagicMock()
        mock_obs.radius_km = 6371.0
        mock_obs.altitude_km = 408.0
        mock_obs.focal_length_mm = 50.0
        mock_obs.radius_uncertainty = 100.0  # Add uncertainty attribute
        mock_obs.detect_limb.return_value = None
        mock_obs.fit_limb.return_value = None

        # Configure hasattr to return True for the attributes we need
        def mock_hasattr(obj, attr):
            return attr in ["altitude_km", "focal_length_mm", "radius_uncertainty"]

        mock_limb_observation.return_value = mock_obs

        args = MagicMock()
        args.image = "test.jpg"
        args.camera_config = "config.yaml"
        args.detection_method = "manual"
        args.altitude = None
        args.focal_length = None
        args.sensor_width = None
        args.output = None
        args.plot = False
        args.save_plots = None

        result = cli.measure_command(args)
        assert result == 0

        # Verify LimbObservation was called with manual detection method
        mock_limb_observation.assert_called_once()
        call_args = mock_limb_observation.call_args
        assert call_args[1]["limb_detection"] == "manual"

        # Verify that the observation methods were called
        mock_obs.detect_limb.assert_called_once()
        mock_obs.fit_limb.assert_called_once()

    @patch("os.path.exists")
    @patch("planet_ruler.cli.load_config")
    @patch("planet_ruler.cli.pr.LimbObservation")
    def test_measure_command_all_detection_methods(
        self, mock_limb_observation, mock_load_config, mock_exists
    ):
        """Test measure command with all supported detection methods."""

        # Mock both image file and config file existence
        def mock_exists_side_effect(path):
            return path in ["test.jpg", "config.yaml"]

        mock_exists.side_effect = mock_exists_side_effect

        mock_load_config.return_value = {
            "altitude_km": 408,
            "focal_length_mm": 50,
            "sensor_width_mm": 36,
        }

        # Mock the observation object and its methods
        mock_obs = MagicMock()
        mock_obs.radius_km = 6371.0
        mock_obs.altitude_km = 408.0
        mock_obs.focal_length_mm = 50.0
        mock_obs.radius_uncertainty = 100.0  # Add uncertainty attribute
        mock_obs.detect_limb.return_value = None
        mock_obs.fit_limb.return_value = None
        mock_limb_observation.return_value = mock_obs

        # Test each detection method
        for method in ["gradient-break", "segmentation", "manual"]:
            mock_limb_observation.reset_mock()
            mock_obs.reset_mock()

            args = MagicMock()
            args.image = "test.jpg"
            args.camera_config = "config.yaml"
            args.detection_method = method
            args.altitude = None
            args.focal_length = None
            args.sensor_width = None
            args.output = None
            args.plot = False
            args.save_plots = None

            result = cli.measure_command(args)
            assert result == 0

            # Verify correct detection method was passed
            call_args = mock_limb_observation.call_args
            assert call_args[1]["limb_detection"] == method

            # Verify that the observation methods were called
            mock_obs.detect_limb.assert_called_once()
            mock_obs.fit_limb.assert_called_once()

    @patch("os.path.exists")
    def test_measure_command_missing_camera_config_with_manual(
        self, mock_exists, capsys
    ):
        """Test measure command with manual method but missing camera config."""

        # Mock image exists but config doesn't
        def mock_exists_side_effect(path):
            return path == "test.jpg"

        mock_exists.side_effect = mock_exists_side_effect

        args = MagicMock()
        args.image = "test.jpg"
        args.camera_config = None  # Missing required camera config
        args.detection_method = "manual"
        args.altitude = None
        args.focal_length = None
        args.sensor_width = None
        args.output = None
        args.plot = False
        args.save_plots = None

        result = cli.measure_command(args)
        assert result == 1

        captured = capsys.readouterr()
        # Check for the actual error message we get when loading config fails
        assert "Error loading configuration" in captured.err


class TestDemoCommand:
    """Test the demo command functionality."""

    @patch("subprocess.run")
    @patch("planet_ruler.cli.Path")
    def test_demo_command_interactive_success(self, mock_path, mock_subprocess, capsys):
        """Test interactive demo command success."""
        # Mock path for notebook
        mock_notebook_path = MagicMock()
        mock_path.return_value.parent.parent = mock_notebook_path
        mock_notebook_path.__truediv__.return_value.__truediv__.return_value = (
            "notebook.ipynb"
        )

        args = MagicMock()
        args.interactive = True
        args.planet = None

        result = cli.demo_command(args)
        assert result == 0

        # Verify subprocess was called
        mock_subprocess.assert_called_once()

    def test_demo_command_no_planet_specified(self, capsys):
        """Test demo command with no planet specified."""
        args = MagicMock()
        args.interactive = False
        args.planet = None

        result = cli.demo_command(args)
        assert result == 0

        captured = capsys.readouterr()
        assert "Available demonstrations:" in captured.out
        assert "earth  - Earth from ISS" in captured.out
        assert "saturn - Saturn from Cassini" in captured.out
        assert "pluto  - Pluto from New Horizons" in captured.out


class TestListCommand:
    """Test the list command functionality."""

    @patch("planet_ruler.cli.Path")
    def test_list_command_with_configs(self, mock_path, capsys):
        """Test list command with available configurations."""
        # Mock config directory and files
        mock_config_dir = MagicMock()
        mock_config_dir.exists.return_value = True

        # Create mock config files
        mock_config1 = MagicMock()
        mock_config1.stem = "earth_iss"
        mock_config1.suffix = ".yaml"

        mock_config2 = MagicMock()
        mock_config2.stem = "saturn_cassini"
        mock_config2.suffix = ".yaml"

        mock_config_dir.glob.return_value = [mock_config1, mock_config2]
        mock_path.return_value.parent.parent.__truediv__.return_value = mock_config_dir

        # Mock load_config for each file
        with patch("planet_ruler.cli.load_config") as mock_load_config:
            mock_load_config.side_effect = [
                {"description": "Earth from ISS configuration"},
                {"description": "Saturn from Cassini probe"},
            ]

            args = MagicMock()
            result = cli.list_command(args)
            assert result == 0

            captured = capsys.readouterr()
            assert "Available example configurations:" in captured.out
            assert "earth_iss" in captured.out
            assert "Earth from ISS configuration" in captured.out
            assert "saturn_cassini" in captured.out
            assert "Saturn from Cassini probe" in captured.out

    @patch("planet_ruler.cli.Path")
    def test_list_command_no_config_dir(self, mock_path, capsys):
        """Test list command when config directory doesn't exist."""
        # Mock config directory not existing
        mock_config_dir = MagicMock()
        mock_config_dir.exists.return_value = False
        mock_path.return_value.parent.parent.__truediv__.return_value = mock_config_dir

        args = MagicMock()
        result = cli.list_command(args)
        assert result == 0

        captured = capsys.readouterr()
        assert "No configuration directory found" in captured.out


class TestArgumentParsing:
    """Test command-line argument parsing."""

    def test_measure_command_args(self):
        """Test measure command argument parsing."""
        import argparse

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        measure_parser = subparsers.add_parser("measure")
        measure_parser.add_argument("image")
        measure_parser.add_argument("--camera-config", "-c")
        measure_parser.add_argument("--output", "-o")
        measure_parser.add_argument("--plot", action="store_true")
        measure_parser.add_argument("--save-plots")
        measure_parser.add_argument("--altitude", type=float)
        measure_parser.add_argument("--focal-length", type=float)
        measure_parser.add_argument("--sensor-width", type=float)

        # Test basic args
        args = parser.parse_args(["measure", "test.jpg"])
        assert args.command == "measure"
        assert args.image == "test.jpg"
        assert args.camera_config is None
        assert args.plot is False

        # Test with all options
        args = parser.parse_args(
            [
                "measure",
                "test.jpg",
                "--camera-config",
                "config.yaml",
                "--output",
                "results.json",
                "--plot",
                "--save-plots",
                "/plots",
                "--altitude",
                "400",
                "--focal-length",
                "50",
                "--sensor-width",
                "36",
            ]
        )

        assert args.command == "measure"
        assert args.image == "test.jpg"
        assert args.camera_config == "config.yaml"
        assert args.output == "results.json"
        assert args.plot is True
        assert args.save_plots == "/plots"
        assert args.altitude == 400.0
        assert args.focal_length == 50.0
        assert args.sensor_width == 36.0

    def test_measure_command_detection_method_args(self):
        """Test measure command with detection method argument parsing."""
        import argparse

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        measure_parser = subparsers.add_parser("measure")
        measure_parser.add_argument("image")
        measure_parser.add_argument("--camera-config", "-c", required=True)
        measure_parser.add_argument(
            "--detection-method",
            choices=["gradient-break", "segmentation", "manual"],
            default="gradient-break",
        )

        # Test default detection method
        args = parser.parse_args(
            ["measure", "test.jpg", "--camera-config", "config.yaml"]
        )
        assert args.detection_method == "gradient-break"

        # Test manual detection method
        args = parser.parse_args(
            [
                "measure",
                "test.jpg",
                "--camera-config",
                "config.yaml",
                "--detection-method",
                "manual",
            ]
        )
        assert args.detection_method == "manual"

        # Test segmentation detection method
        args = parser.parse_args(
            [
                "measure",
                "test.jpg",
                "--camera-config",
                "config.yaml",
                "--detection-method",
                "segmentation",
            ]
        )
        assert args.detection_method == "segmentation"

    def test_demo_command_args(self):
        """Test demo command argument parsing."""
        import argparse

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command")
        demo_parser = subparsers.add_parser("demo")
        demo_parser.add_argument("--planet", choices=["earth", "saturn", "pluto"])
        demo_parser.add_argument("--interactive", action="store_true")

        # Test basic demo
        args = parser.parse_args(["demo"])
        assert args.command == "demo"
        assert args.planet is None
        assert args.interactive is False

        # Test with options
        args = parser.parse_args(["demo", "--planet", "earth", "--interactive"])
        assert args.command == "demo"
        assert args.planet == "earth"
        assert args.interactive is True


class TestConfigIntegration:
    """Integration tests for configuration loading with actual file I/O."""

    def test_config_yaml_roundtrip(self):
        """Test complete YAML configuration loading workflow."""
        config_data = {
            "altitude_km": 408,
            "focal_length_mm": 50,
            "sensor_width_mm": 36,
            "description": "Earth from ISS",
            "camera_model": "Nikon D4",
            "parameters": {"exposure": "1/500s", "iso": 800, "aperture": "f/8"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f, default_flow_style=False)
            config_path = f.name

        try:
            loaded_config = cli.load_config(config_path)

            # Test all top-level keys
            assert loaded_config["altitude_km"] == 408
            assert loaded_config["focal_length_mm"] == 50
            assert loaded_config["sensor_width_mm"] == 36
            assert loaded_config["description"] == "Earth from ISS"
            assert loaded_config["camera_model"] == "Nikon D4"

            # Test nested parameters
            assert loaded_config["parameters"]["exposure"] == "1/500s"
            assert loaded_config["parameters"]["iso"] == 800
            assert loaded_config["parameters"]["aperture"] == "f/8"

        finally:
            os.unlink(config_path)

    def test_config_json_roundtrip(self):
        """Test complete JSON configuration loading workflow."""
        config_data = {
            "altitude_km": 100,
            "focal_length_mm": 24,
            "sensor_width_mm": 15.6,
            "description": "Low altitude drone survey",
            "flight_params": {"max_altitude": 120, "ground_speed": 15, "overlap": 80},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f, indent=2)
            config_path = f.name

        try:
            loaded_config = cli.load_config(config_path)

            # Test exact equality for JSON
            assert loaded_config == config_data

            # Test specific values
            assert loaded_config["altitude_km"] == 100
            assert loaded_config["flight_params"]["max_altitude"] == 120
            assert loaded_config["flight_params"]["overlap"] == 80

        finally:
            os.unlink(config_path)


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_empty_config_file(self):
        """Test handling of empty configuration files."""
        # Test empty YAML
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")  # Empty file
            yaml_path = f.name

        try:
            config = cli.load_config(yaml_path)
            assert config is None  # yaml.safe_load returns None for empty files
        finally:
            os.unlink(yaml_path)

        # Test empty JSON
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{}")  # Empty JSON object
            json_path = f.name

        try:
            config = cli.load_config(json_path)
            assert config == {}
        finally:
            os.unlink(json_path)

    def test_corrupted_files(self):
        """Test handling of corrupted or malformed files."""
        # Corrupted YAML
        corrupted_yaml = """
        altitude_km: 400
        focal_length_mm: [unclosed list
        sensor_width: "unclosed string
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(corrupted_yaml)
            yaml_path = f.name

        try:
            with pytest.raises(yaml.YAMLError):
                cli.load_config(yaml_path)
        finally:
            os.unlink(yaml_path)

        # Corrupted JSON
        corrupted_json = '{"altitude_km": 400, "focal_length": incomplete'

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(corrupted_json)
            json_path = f.name

        try:
            with pytest.raises(json.JSONDecodeError):
                cli.load_config(json_path)
        finally:
            os.unlink(json_path)

    def test_permission_denied(self):
        """Test handling of permission errors."""
        # Create a config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"test": "value"}, f)
            config_path = f.name

        try:
            # Change permissions to deny read access
            os.chmod(config_path, 0o000)

            # This should raise a PermissionError on most systems
            try:
                with pytest.raises(PermissionError):
                    cli.load_config(config_path)
            except AssertionError:
                # On some systems, the file might still be readable
                # In that case, just ensure it doesn't crash
                config = cli.load_config(config_path)
                assert isinstance(config, (dict, type(None)))
        finally:
            # Restore permissions and clean up
            os.chmod(config_path, 0o644)
            os.unlink(config_path)


if __name__ == "__main__":
    pytest.main([__file__])
