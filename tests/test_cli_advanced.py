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
    @patch("builtins.__import__")
    def test_demo_command_interactive_subprocess_error(
        self, mock_import, mock_subprocess, mock_path, capsys
    ):
        """Test interactive demo with subprocess error."""

        # Mock the import to succeed for jupyter_core.command
        def mock_import_func(name, *args, **kwargs):
            if name == "jupyter_core.command":
                return MagicMock()  # Mock successful import
            return __import__(name, *args, **kwargs)

        mock_import.side_effect = mock_import_func

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


if __name__ == "__main__":
    pytest.main([__file__])
