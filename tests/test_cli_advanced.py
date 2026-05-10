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
Advanced test coverage for planet_ruler CLI module.
These tests focus on CLI functionality that can be reliably tested without complex mocking.
"""

import os
import sys
import json
import yaml
import tempfile
import pytest
import subprocess
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock, call
from io import StringIO
from PIL import Image
import numpy as np

# Add the planet_ruler package to the path for testing
sys.path.insert(0, str(Path(__file__).parent.parent))

from planet_ruler import cli


class TestDemoCommandEdgeCases:
    """Test edge cases and error conditions for demo command."""

    def test_demo_command_planet_specific_without_import(self, capsys):
        """Test demo command with specific planet (without problematic imports)."""
        args = MagicMock()
        args.interactive = False
        args.planet = "earth"

        # This should work without trying to import demo functions
        result = cli.demo_command(args)
        assert result == 0

        captured = capsys.readouterr()
        assert "Running Earth demonstration" in captured.out
        assert "Demo for earth completed" in captured.out


class TestDemoCommandAdvanced:
    """Test the demo command functionality with advanced scenarios."""

    @patch("planet_ruler.cli.Path")
    @patch("subprocess.run")
    def test_demo_command_interactive_subprocess_error(
        self, mock_subprocess, mock_path, capsys
    ):
        """Test interactive demo with subprocess error."""
        # Mock path for notebook
        mock_notebook_path = MagicMock()
        mock_path.return_value.parent.parent = mock_notebook_path
        mock_notebook_path.__truediv__.return_value.__truediv__.return_value = (
            "notebook.ipynb"
        )

        # Create the subprocess error in a Python 3.8 compatible way
        subprocess_error = subprocess.CalledProcessError(1, "jupyter")
        mock_subprocess.side_effect = subprocess_error

        args = MagicMock()
        args.interactive = True
        args.planet = None

        result = cli.demo_command(args)
        assert result == 1

        captured = capsys.readouterr()
        assert "Error launching Jupyter notebook" in captured.err


class TestListCommandAdvanced:
    """Test the list command functionality with advanced scenarios."""

    @patch("planet_ruler.cli.Path")
    def test_list_command_config_loading_error(self, mock_path, capsys):
        """Test list command when config loading fails."""
        # Mock config directory and files
        mock_config_dir = MagicMock()
        mock_config_dir.exists.return_value = True

        mock_config1 = MagicMock()
        mock_config1.stem = "broken_config"
        mock_config_dir.glob.return_value = [mock_config1]
        mock_path.return_value.parent.parent.__truediv__.return_value = mock_config_dir

        # Mock load_config to raise exception
        with patch(
            "planet_ruler.cli.load_config", side_effect=Exception("Parse error")
        ):
            args = MagicMock()
            result = cli.list_command(args)
            assert result == 0

            captured = capsys.readouterr()
            assert "broken_config" in captured.out
            assert "Error loading configuration" in captured.out


class TestConfigurationWorkflow:
    """Test configuration-related workflows that don't require observation mocking."""

    def test_config_loading_workflow(self):
        """Test the complete configuration loading workflow."""
        # Create test configuration
        config_data = {
            "altitude_km": 400,
            "focal_length_mm": 50,
            "sensor_width_mm": 36,
            "description": "Test configuration for CLI workflow",
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as config_file:
            yaml.dump(config_data, config_file)
            config_path = config_file.name

        try:
            # Test configuration loading
            loaded_config = cli.load_config(config_path)
            assert loaded_config == config_data

            # Test that the config is properly formatted
            assert isinstance(loaded_config["altitude_km"], int)
            assert isinstance(loaded_config["focal_length_mm"], int)
            assert isinstance(loaded_config["sensor_width_mm"], int)
            assert isinstance(loaded_config["description"], str)

        finally:
            os.unlink(config_path)

    def test_json_config_workflow(self):
        """Test JSON configuration workflow."""
        config_data = {
            "altitude_km": 100,
            "focal_length_mm": 24,
            "sensor_width_mm": 15.6,
            "description": "JSON test configuration",
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as config_file:
            json.dump(config_data, config_file)
            config_path = config_file.name

        try:
            # Test configuration loading
            loaded_config = cli.load_config(config_path)
            assert loaded_config == config_data

            # Test numeric precision is preserved
            assert loaded_config["sensor_width_mm"] == 15.6

        finally:
            os.unlink(config_path)


class TestManualDetectionMethodIntegration:
    """Test manual detection method integration with advanced scenarios."""

    @patch("planet_ruler.cli.pr.LimbObservation")
    @patch("planet_ruler.cli.load_config")
    @patch("os.path.exists")
    def test_manual_detection_with_yaml_config(
        self, mock_exists, mock_load_config, mock_limb_observation, capsys
    ):
        """Test manual detection method with YAML configuration."""
        # Mock file existence for both image and config
        mock_exists.return_value = True

        mock_load_config.return_value = {
            "altitude_km": 408,
            "focal_length_mm": 50,
            "sensor_width_mm": 36,
            "description": "ISS Camera Configuration",
        }

        # Mock successful observation
        mock_obs = MagicMock()
        mock_obs.radius_km = 6371.0
        mock_obs.radius_uncertainty = 50.0
        mock_obs.altitude_km = 408.0
        mock_obs.focal_length_mm = 50.0
        mock_obs.detect_limb.return_value = mock_obs
        mock_obs.fit_limb.return_value = mock_obs
        mock_limb_observation.return_value = mock_obs

        args = MagicMock()
        args.image = "test_image.jpg"
        args.camera_config = "config.yaml"
        args.auto_config = False
        args.detection_method = "manual"
        args.altitude = None
        args.focal_length = None
        args.sensor_width = None
        args.output = None
        args.plot = False
        args.save_plots = None

        result = cli.measure_command(args)
        assert result == 0

        # Verify configuration was loaded
        mock_load_config.assert_called_once_with("config.yaml")

        # Verify LimbObservation was called with correct parameters
        mock_limb_observation.assert_called_once_with(
            "test_image.jpg", fit_config="config.yaml", limb_detection="manual"
        )

        # Verify methods were called
        mock_obs.detect_limb.assert_called_once()
        mock_obs.fit_limb.assert_called_once()

        captured = capsys.readouterr()
        assert "Loading image: test_image.jpg" in captured.out
        assert "Loaded camera configuration from: config.yaml" in captured.out
        assert "Detecting horizon/limb using manual method..." in captured.out
        assert "Estimated planetary radius: 6371" in captured.out

    @patch("planet_ruler.cli.pr.LimbObservation")
    @patch("planet_ruler.cli.load_config")
    @patch("os.path.exists")
    def test_manual_detection_with_json_config(
        self, mock_exists, mock_load_config, mock_limb_observation
    ):
        """Test manual detection method with JSON configuration."""
        # Mock file existence for both image and config
        mock_exists.return_value = True

        config_data = {
            "altitude_km": 100.5,
            "focal_length_mm": 24.0,
            "sensor_width_mm": 15.6,
            "description": "Drone survey configuration",
        }
        mock_load_config.return_value = config_data

        mock_obs = MagicMock()
        mock_obs.radius_km = 6371.0
        mock_obs.radius_uncertainty = 50.0
        mock_obs.altitude_km = 100.5
        mock_obs.focal_length_mm = 24.0
        mock_obs.detect_limb.return_value = mock_obs
        mock_obs.fit_limb.return_value = mock_obs
        mock_limb_observation.return_value = mock_obs

        args = MagicMock()
        args.image = "test_image.jpg"
        args.camera_config = "config.json"
        args.auto_config = False
        args.detection_method = "manual"
        args.altitude = None
        args.focal_length = None
        args.sensor_width = None
        args.output = None
        args.plot = False
        args.save_plots = None

        result = cli.measure_command(args)
        assert result == 0

        # Verify configuration was loaded and applied
        mock_load_config.assert_called_once_with("config.json")
        mock_limb_observation.assert_called_once_with(
            "test_image.jpg", fit_config="config.json", limb_detection="manual"
        )
        mock_obs.detect_limb.assert_called_once()
        mock_obs.fit_limb.assert_called_once()

    @patch("planet_ruler.cli.pr.LimbObservation")
    @patch("planet_ruler.cli.load_config")
    @patch("os.path.exists")
    def test_manual_detection_parameter_override(
        self, mock_exists, mock_load_config, mock_limb_observation
    ):
        """Test manual detection with command-line parameter overrides."""
        # Mock file existence for both image and config
        mock_exists.return_value = True

        config_data = {"altitude_km": 400, "focal_length_mm": 50, "sensor_width_mm": 36}
        mock_load_config.return_value = config_data

        mock_obs = MagicMock()
        mock_obs.radius_km = 6371.0
        mock_obs.radius_uncertainty = 50.0
        mock_obs.altitude_km = 500.0  # Should be overridden value
        mock_obs.focal_length_mm = 85.0  # Should be overridden value
        mock_obs.detect_limb.return_value = mock_obs
        mock_obs.fit_limb.return_value = mock_obs
        mock_limb_observation.return_value = mock_obs

        args = MagicMock()
        args.image = "test_image.jpg"
        args.camera_config = "config.yaml"
        args.auto_config = False
        args.detection_method = "manual"
        args.altitude = 500.0  # Override config value
        args.focal_length = 85.0  # Override config value
        args.sensor_width = 24.0  # Override config value
        args.output = None
        args.plot = False
        args.save_plots = None

        result = cli.measure_command(args)
        assert result == 0

        # Verify the config was loaded but values were overridden
        mock_load_config.assert_called_once_with("config.yaml")

        # We can't easily test the setattr calls, but we can verify the observation was created
        mock_limb_observation.assert_called_once_with(
            "test_image.jpg", fit_config="config.yaml", limb_detection="manual"
        )

        # Check that setattr was called to override configuration
        # (We can't easily test the exact setattr calls with our current mock setup)
        mock_obs.detect_limb.assert_called_once()
        mock_obs.fit_limb.assert_called_once()

    @patch("os.path.exists")
    @patch("planet_ruler.cli.load_config")
    def test_manual_detection_config_error_handling(
        self, mock_load_config, mock_exists, capsys
    ):
        """Test error handling when config loading fails with manual method."""
        mock_exists.return_value = True
        mock_load_config.side_effect = Exception("Invalid YAML syntax")

        args = MagicMock()
        args.image = "test.jpg"
        args.camera_config = "broken_config.yaml"
        args.auto_config = False
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
        assert "Error loading configuration" in captured.err

    @patch("planet_ruler.cli.pr.LimbObservation")
    @patch("planet_ruler.cli.load_config")
    @patch("os.path.exists")
    def test_manual_detection_observation_error(
        self, mock_exists, mock_load_config, mock_limb_observation, capsys
    ):
        """Test error handling when observation fails with manual method."""
        # Mock file existence for both image and config
        mock_exists.return_value = True

        config_data = {"altitude_km": 408, "focal_length_mm": 50, "sensor_width_mm": 36}
        mock_load_config.return_value = config_data

        # Mock observation to raise an exception
        mock_limb_observation.side_effect = Exception("Manual annotation failed")

        args = MagicMock()
        args.image = "test_image.jpg"
        args.camera_config = "config.yaml"
        args.auto_config = False
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
        assert "Manual annotation failed" in captured.err


class TestManualDetectionWorkflows:
    """Test complete workflows with manual detection method."""

    def test_manual_detection_config_roundtrip(self):
        """Test complete config loading workflow for manual detection."""
        # Create a test configuration that would be used with manual detection
        config_data = {
            "altitude_km": 408,
            "focal_length_mm": 50,
            "sensor_width_mm": 36,
            "description": "Manual annotation test configuration",
            "camera_model": "Canon EOS R5",
            "detection_settings": {
                "method": "manual",
                "annotation_zoom": 2.0,
                "default_stretch": 1.0,
            },
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as config_file:
            yaml.dump(config_data, config_file)
            config_path = config_file.name

        try:
            # Test configuration loading
            loaded_config = cli.load_config(config_path)

            # Verify all manual detection relevant fields are preserved
            assert loaded_config["altitude_km"] == 408
            assert loaded_config["focal_length_mm"] == 50
            assert loaded_config["sensor_width_mm"] == 36
            assert (
                loaded_config["description"] == "Manual annotation test configuration"
            )
            assert loaded_config["camera_model"] == "Canon EOS R5"
            assert loaded_config["detection_settings"]["method"] == "manual"
            assert loaded_config["detection_settings"]["annotation_zoom"] == 2.0
            assert loaded_config["detection_settings"]["default_stretch"] == 1.0

        finally:
            os.unlink(config_path)


# ---------------------------------------------------------------------------
# Auto-config path, parameter overrides, output / save-plots
# ---------------------------------------------------------------------------


def _mock_obs():
    obs = MagicMock()
    obs.radius_km = 6371.0
    obs.radius_uncertainty = 50.0
    obs.altitude_km = 10.5
    del obs.focal_length_mm  # not all obs have this attr; avoid AttributeError
    obs.best_parameters = {"r": 6371000.0, "h": 10500.0}
    obs.detect_limb.return_value = obs
    obs.fit_limb.return_value = obs
    return obs


def _valid_auto_config():
    return {
        "free_parameters": ["r", "h"],
        "init_parameter_values": {
            "r": 6371000,
            "h": 10000,
            "f": 0.05,
            "w": 0.036,
            "theta_x": 0.0,
            "theta_y": 0.0,
            "theta_z": 3.14,
        },
        "parameter_limits": {
            "r": [1e6, 1e8],
            "h": [5000, 20000],
            "f": [0.01, 0.1],
            "w": [0.01, 0.1],
            "theta_x": [-1, 1],
            "theta_y": [-1, 1],
            "theta_z": [0, 7],
        },
        "camera_info": {"camera_model": "iPhone 13", "confidence": "high"},
    }


def _base_args(tmp_path=None):
    args = MagicMock()
    args.auto_config = False
    args.camera_config = "config.yaml"
    args.image = "photo.jpg"
    args.altitude = None
    args.planet = "earth"
    args.detection_method = "manual"
    args.loss_function = "l2"
    args.max_iterations = 100
    args.minimizer_preset = "balanced"
    args.warm_start = False
    args.verbose = False
    args.kernel_smoothing = 5.0
    args.dashboard = False
    args.multi_resolution = None
    args.image_smoothing = None
    args.focal_length = None
    args.sensor_width = None
    args.field_of_view = None
    args.output = None
    args.plot = False
    args.save_plots = None
    return args


@patch("planet_ruler.cli.pr")
@patch("planet_ruler.cli.create_config_from_image")
@patch("os.path.exists", return_value=True)
class TestAutoConfigPath:
    def test_auto_config_success(self, _exists, mock_create, mock_pr, capsys):
        mock_create.return_value = _valid_auto_config()
        mock_pr.LimbObservation.return_value = _mock_obs()

        args = _base_args()
        args.auto_config = True
        args.altitude = 10000.0

        result = cli.measure_command(args)
        assert result == 0
        out = capsys.readouterr().out
        assert "planetary radius" in out

    def test_auto_config_no_altitude_returns_error(
        self, _exists, mock_create, mock_pr, capsys
    ):
        args = _base_args()
        args.auto_config = True
        args.altitude = None

        result = cli.measure_command(args)
        assert result == 1
        err = capsys.readouterr().err
        assert "--altitude" in err

    def test_auto_config_value_error_returns_error(
        self, _exists, mock_create, mock_pr, capsys
    ):
        mock_create.side_effect = ValueError("no EXIF")
        args = _base_args()
        args.auto_config = True
        args.altitude = 10000.0

        result = cli.measure_command(args)
        assert result == 1

    def test_auto_config_unexpected_error_returns_error(
        self, _exists, mock_create, mock_pr, capsys
    ):
        mock_create.side_effect = RuntimeError("boom")
        args = _base_args()
        args.auto_config = True
        args.altitude = 10000.0

        result = cli.measure_command(args)
        assert result == 1


@patch("planet_ruler.cli.pr")
@patch("planet_ruler.cli.load_config")
@patch("os.path.exists", return_value=True)
class TestParameterOverrides:
    def _run(self, mock_pr, config_override=None):
        obs = _mock_obs()
        mock_pr.LimbObservation.return_value = obs
        config = config_override or {"init_parameter_values": {}, "free_parameters": []}
        return obs, config

    def test_altitude_override_updates_h(self, _exists, mock_load, mock_pr, capsys):
        config = {"init_parameter_values": {"h": 5000.0}, "free_parameters": []}
        mock_load.return_value = config
        obs = _mock_obs()
        mock_pr.LimbObservation.return_value = obs

        args = _base_args()
        args.altitude = 10.0  # < 1000 → treated as km → * 1000

        cli.measure_command(args)
        assert config["init_parameter_values"]["h"] == pytest.approx(10000.0)

    def test_focal_length_override(self, _exists, mock_load, mock_pr, capsys):
        config = {"init_parameter_values": {"f": 0.05}, "free_parameters": []}
        mock_load.return_value = config
        obs = _mock_obs()
        mock_pr.LimbObservation.return_value = obs

        args = _base_args()
        args.focal_length = 50.0  # mm → /1000 = 0.05 m

        cli.measure_command(args)
        assert config["init_parameter_values"]["f"] == pytest.approx(0.05)

    def test_sensor_width_override(self, _exists, mock_load, mock_pr, capsys):
        config = {"init_parameter_values": {"w": 0.036}, "free_parameters": []}
        mock_load.return_value = config
        obs = _mock_obs()
        mock_pr.LimbObservation.return_value = obs

        args = _base_args()
        args.sensor_width = 36.0  # mm → /1000 = 0.036 m

        cli.measure_command(args)
        assert config["init_parameter_values"]["w"] == pytest.approx(0.036)

    def test_fov_override(self, _exists, mock_load, mock_pr, capsys):
        config = {"init_parameter_values": {}, "free_parameters": []}
        mock_load.return_value = config
        obs = _mock_obs()
        mock_pr.LimbObservation.return_value = obs

        args = _base_args()
        args.field_of_view = 60.0  # degrees

        cli.measure_command(args)
        expected = 60.0 * 3.14159 / 180
        assert config["init_parameter_values"]["fov"] == pytest.approx(
            expected, rel=1e-4
        )


@patch("planet_ruler.cli.pr")
@patch("planet_ruler.cli.load_config")
@patch("os.path.exists", return_value=True)
class TestOutputAndSavePlots:
    def test_output_json_saved(self, _exists, mock_load, mock_pr, tmp_path, capsys):
        mock_load.return_value = {"init_parameter_values": {}, "free_parameters": []}
        obs = _mock_obs()
        mock_pr.LimbObservation.return_value = obs

        out_file = tmp_path / "results.json"
        args = _base_args()
        args.output = str(out_file)

        result = cli.measure_command(args)
        assert result == 0
        assert out_file.exists()
        with open(out_file) as f:
            data = json.load(f)
        assert "radius_km" in data

    def test_save_plots_creates_file(
        self, _exists, mock_load, mock_pr, tmp_path, capsys
    ):
        mock_load.return_value = {"init_parameter_values": {}, "free_parameters": []}
        obs = _mock_obs()
        mock_pr.LimbObservation.return_value = obs

        plots_dir = str(tmp_path / "plots")
        args = _base_args()
        args.save_plots = plots_dir
        args.image = "photo.jpg"

        # plt is imported locally inside the if-branch; patch the global module
        with (
            patch("matplotlib.pyplot.savefig") as mock_save,
            patch("matplotlib.pyplot.close"),
        ):
            result = cli.measure_command(args)
        assert result == 0
        mock_save.assert_called_once()


@patch("planet_ruler.cli.pr")
@patch("planet_ruler.cli.load_config")
@patch("os.path.exists", return_value=True)
class TestMultiResolutionParsing:
    def test_auto_multi_resolution(self, _exists, mock_load, mock_pr, capsys):
        mock_load.return_value = {"init_parameter_values": {}, "free_parameters": []}
        obs = _mock_obs()
        mock_pr.LimbObservation.return_value = obs

        args = _base_args()
        args.multi_resolution = "auto"

        result = cli.measure_command(args)
        assert result == 0
        _, kwargs = obs.fit_limb.call_args
        assert kwargs.get("resolution_stages") == "auto"

    def test_numeric_multi_resolution(self, _exists, mock_load, mock_pr, capsys):
        mock_load.return_value = {"init_parameter_values": {}, "free_parameters": []}
        obs = _mock_obs()
        mock_pr.LimbObservation.return_value = obs

        args = _base_args()
        args.multi_resolution = "4,2,1"

        result = cli.measure_command(args)
        assert result == 0
        _, kwargs = obs.fit_limb.call_args
        assert kwargs.get("resolution_stages") == [4, 2, 1]

    def test_invalid_multi_resolution_returns_error(
        self, _exists, mock_load, mock_pr, capsys
    ):
        mock_load.return_value = {"init_parameter_values": {}, "free_parameters": []}
        obs = _mock_obs()
        mock_pr.LimbObservation.return_value = obs

        args = _base_args()
        args.multi_resolution = "bad,input"

        result = cli.measure_command(args)
        assert result == 1


if __name__ == "__main__":
    pytest.main([__file__])
