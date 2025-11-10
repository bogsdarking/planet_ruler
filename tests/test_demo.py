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
Comprehensive tests for planet_ruler.demo module.

Tests cover:
- make_dropdown: ipywidgets dropdown creation for demo selection
- load_demo_parameters: Parameter loading for different demos (Pluto, Saturn, Earth)
- display_text: Markdown file display using IPython

Uses extensive mocking of ipywidgets and IPython.display to avoid Jupyter dependencies.
Tests parameter validation, file loading, and widget configuration.
"""

import pytest
import json
from unittest.mock import Mock, patch, mock_open, MagicMock

from planet_ruler.demo import make_dropdown, load_demo_parameters, display_text


class TestMakeDropdown:
    """Test dropdown widget creation for demo selection."""

    @patch("planet_ruler.demo.widgets")
    def test_make_dropdown_basic(self, mock_widgets):
        """Test basic dropdown creation."""
        mock_dropdown = Mock()
        mock_widgets.Dropdown.return_value = mock_dropdown

        result = make_dropdown()

        # Verify dropdown creation
        mock_widgets.Dropdown.assert_called_once_with(
            options=[("Pluto", 1), ("Saturn-1", 2), ("Saturn-2", 3), ("Earth", 4)],
            value=1,
            description="Demo:",
        )
        assert result == mock_dropdown

    @patch("planet_ruler.demo.widgets")
    def test_make_dropdown_options(self, mock_widgets):
        """Test dropdown has correct options and default value."""
        mock_dropdown = Mock()
        mock_widgets.Dropdown.return_value = mock_dropdown

        make_dropdown()

        call_args = mock_widgets.Dropdown.call_args[1]

        # Verify options
        expected_options = [
            ("Pluto", 1),
            ("Saturn-1", 2),
            ("Saturn-2", 3),
            ("Earth", 4),
        ]
        assert call_args["options"] == expected_options

        # Verify default value
        assert call_args["value"] == 1

        # Verify description
        assert call_args["description"] == "Demo:"

    @patch("planet_ruler.demo.widgets")
    def test_make_dropdown_widget_type(self, mock_widgets):
        """Test correct widget type is used."""
        mock_dropdown = Mock()
        mock_dropdown.__class__.__name__ = "Dropdown"
        mock_widgets.Dropdown.return_value = mock_dropdown

        result = make_dropdown()

        # Verify correct widget constructor was called
        mock_widgets.Dropdown.assert_called_once()
        assert result == mock_dropdown


class TestLoadDemoParameters:
    """Test demo parameter loading for different targets."""

    def create_mock_demo(self, value):
        """Helper to create mock demo object with value attribute."""
        demo = Mock()
        demo.value = value
        return demo

    def test_load_demo_parameters_pluto(self):
        """Test loading Pluto demo parameters."""
        demo = self.create_mock_demo(1)

        result = load_demo_parameters(demo)

        # Verify parameter structure
        expected_limb_config = {
            "log": False,
            "y_min": 0,
            "y_max": -1,
            "window_length": 501,
            "polyorder": 1,
            "deriv": 0,
            "delta": 1,
            "segmenter": "segment-anything",
        }

        expected_params = {
            "target": "Pluto",
            "true_radius": 1188,
            "image_filepath": "../../demo/images/PIA19948.tif",
            "fit_config": "../../config/pluto-new-horizons.yaml",
            "limb_config": expected_limb_config,
            "limb_save": "pluto_limb.npy",
            "parameter_walkthrough": "../../demo/pluto_init.md",
            "preamble": "../../demo/pluto_preamble.md",
        }

        assert result == expected_params
        assert result["target"] == "Pluto"
        assert result["true_radius"] == 1188
        assert result["limb_config"] == expected_limb_config

    def test_load_demo_parameters_saturn_1(self):
        """Test loading Saturn-1 demo parameters."""
        demo = self.create_mock_demo(2)

        result = load_demo_parameters(demo)

        # Verify Saturn-1 specific parameters
        expected_limb_config = {
            "log": False,
            "y_min": 0,
            "y_max": -1,
            "window_length": 501,
            "polyorder": 1,
            "deriv": 0,
            "delta": 1,
            "segmenter": "segment-anything",
        }

        assert result["target"] == "Saturn"
        assert result["true_radius"] == 58232
        assert result["image_filepath"] == "../../demo/images/PIA21341.jpg"
        assert result["fit_config"] == "../../config/saturn-cassini-1.yaml"
        assert result["limb_config"] == expected_limb_config
        assert result["limb_save"] == "saturn_limb_1.npy"
        assert result["parameter_walkthrough"] == "../../demo/saturn_init_1.md"
        assert result["preamble"] == "../../demo/saturn_preamble_1.md"

    def test_load_demo_parameters_saturn_2(self):
        """Test loading Saturn-2 demo parameters."""
        demo = self.create_mock_demo(3)

        result = load_demo_parameters(demo)

        # Verify Saturn-2 specific parameters
        expected_limb_config = {
            "log": False,
            "y_min": 0,
            "y_max": -1,
            "window_length": 501,
            "polyorder": 1,
            "deriv": 0,
            "delta": 1,
            "segmenter": "segment-anything",
        }

        assert result["target"] == "Saturn"
        assert result["true_radius"] == 58232
        assert (
            result["image_filepath"] == "../../demo/images/saturn_ciclops_5769_13427_1.jpg"
        )
        assert result["fit_config"] == "../../config/saturn-cassini-2.yaml"
        assert result["limb_config"] == expected_limb_config
        assert result["limb_save"] == "saturn_limb_2.npy"
        assert result["parameter_walkthrough"] == "../../demo/saturn_init_2.md"
        assert result["preamble"] == "../../demo/saturn_preamble_2.md"

    def test_load_demo_parameters_earth(self):
        """Test loading Earth demo parameters."""
        demo = self.create_mock_demo(4)

        result = load_demo_parameters(demo)

        # Verify Earth specific parameters
        expected_limb_config = {
            "log": False,
            "y_min": 0,
            "y_max": -1,
            "window_length": 501,
            "polyorder": 1,
            "deriv": 0,
            "delta": 1,
            "segmenter": "segment-anything",
        }

        assert result["target"] == "Earth"
        assert result["true_radius"] == 6371
        assert result["image_filepath"] == "../../demo/images/50644513538_56228a2027_o.jpg"
        assert result["fit_config"] == "../../config/earth_iss_1.yaml"
        assert result["limb_config"] == expected_limb_config
        assert result["limb_save"] == "earth_limb_1.npy"
        assert result["parameter_walkthrough"] == "../../demo/earth_init_1.md"
        assert result["preamble"] == "../../demo/earth_preamble_1.md"

    def test_load_demo_parameters_invalid_value(self):
        """Test loading parameters with invalid demo value."""
        demo = self.create_mock_demo(99)  # Invalid value

        result = load_demo_parameters(demo)

        assert result is None

    def test_load_demo_parameters_zero_value(self):
        """Test loading parameters with zero demo value."""
        demo = self.create_mock_demo(0)

        result = load_demo_parameters(demo)

        assert result is None

    def test_load_demo_parameters_negative_value(self):
        """Test loading parameters with negative demo value."""
        demo = self.create_mock_demo(-1)

        result = load_demo_parameters(demo)

        assert result is None

    def test_load_demo_parameters_all_demos_have_required_keys(self):
        """Test that all demo configurations have required parameter keys."""
        required_keys = {
            "target",
            "true_radius",
            "image_filepath",
            "fit_config",
            "limb_config",
            "limb_save",
            "parameter_walkthrough",
            "preamble",
        }

        # Test all valid demo values
        for demo_value in [1, 2, 3, 4]:
            demo = self.create_mock_demo(demo_value)
            result = load_demo_parameters(demo)

            assert result is not None
            assert all(key in result for key in required_keys)
            assert len(result) == len(required_keys)  # No extra keys

    def test_load_demo_parameters_limb_save_path_consistency(self):
        """Test that limb save paths follow expected patterns."""
        # Test path patterns for each demo
        demo_configs = [
            (1, "pluto_limb.npy"),
            (2, "saturn_limb_1.npy"),
            (3, "saturn_limb_2.npy"),
            (4, "earth_limb_1.npy"),
        ]

        for demo_value, expected_save_path in demo_configs:
            demo = self.create_mock_demo(demo_value)
            result = load_demo_parameters(demo)

            assert result["limb_save"] == expected_save_path


class TestDisplayText:
    """Test markdown file display functionality."""

    @patch("planet_ruler.demo.display")
    @patch("planet_ruler.demo.Markdown")
    def test_display_text_basic(self, mock_markdown, mock_display):
        """Test basic text file display."""
        file_content = "# Demo Title\n\nThis is demo content.\n"
        mock_markdown_obj = Mock()
        mock_markdown.return_value = mock_markdown_obj

        with patch("builtins.open", mock_open(read_data=file_content)):
            display_text("/path/to/demo.md")

        # Verify file was read and processed
        mock_markdown.assert_called_once_with(file_content)
        mock_display.assert_called_once_with(mock_markdown_obj)

    @patch("planet_ruler.demo.display")
    @patch("planet_ruler.demo.Markdown")
    def test_display_text_different_content(self, mock_markdown, mock_display):
        """Test display with different markdown content."""
        file_content = """
## Parameter Configuration

- **Target**: Earth
- **Radius**: 6371 km
- **Method**: Gradient Break

### Analysis Steps
1. Load image
2. Detect limb
3. Calculate radius
        """.strip()

        mock_markdown_obj = Mock()
        mock_markdown.return_value = mock_markdown_obj

        with patch("builtins.open", mock_open(read_data=file_content)):
            display_text("/path/to/config.md")

        mock_markdown.assert_called_once_with(file_content)
        mock_display.assert_called_once_with(mock_markdown_obj)

    @patch("planet_ruler.demo.display")
    @patch("planet_ruler.demo.Markdown")
    def test_display_text_empty_file(self, mock_markdown, mock_display):
        """Test display with empty file."""
        file_content = ""
        mock_markdown_obj = Mock()
        mock_markdown.return_value = mock_markdown_obj

        with patch("builtins.open", mock_open(read_data=file_content)):
            display_text("/path/to/empty.md")

        mock_markdown.assert_called_once_with("")
        mock_display.assert_called_once_with(mock_markdown_obj)

    @patch("planet_ruler.demo.display")
    @patch("planet_ruler.demo.Markdown")
    def test_display_text_file_not_found(self, mock_markdown, mock_display):
        """Test display when file doesn't exist."""
        with patch("builtins.open", side_effect=FileNotFoundError("File not found")):
            with pytest.raises(FileNotFoundError):
                display_text("/path/to/nonexistent.md")

        # Should not call markdown or display if file doesn't exist
        mock_markdown.assert_not_called()
        mock_display.assert_not_called()

    @patch("planet_ruler.demo.display")
    @patch("planet_ruler.demo.Markdown")
    def test_display_text_read_permission_error(self, mock_markdown, mock_display):
        """Test display with file permission error."""
        with patch("builtins.open", side_effect=PermissionError("Permission denied")):
            with pytest.raises(PermissionError):
                display_text("/path/to/protected.md")

        mock_markdown.assert_not_called()
        mock_display.assert_not_called()

    @patch("planet_ruler.demo.display")
    @patch("planet_ruler.demo.Markdown")
    def test_display_text_file_encoding(self, mock_markdown, mock_display):
        """Test display handles text encoding correctly."""
        # Test with special characters
        file_content = "# Démonstration\n\nRadius: 6371 ± 0.1 km\nAngle: 45°\n"
        mock_markdown_obj = Mock()
        mock_markdown.return_value = mock_markdown_obj

        with patch("builtins.open", mock_open(read_data=file_content)):
            display_text("/path/to/unicode.md")

        mock_markdown.assert_called_once_with(file_content)
        mock_display.assert_called_once_with(mock_markdown_obj)


# Integration tests
class TestDemoIntegration:
    """Integration tests for demo workflow."""

    @patch("planet_ruler.demo.widgets")
    def test_dropdown_to_parameters_workflow(self, mock_widgets):
        """Test complete workflow from dropdown creation to parameter loading."""
        # Setup dropdown
        mock_dropdown = Mock()
        mock_dropdown.value = 1  # Pluto
        mock_widgets.Dropdown.return_value = mock_dropdown

        # Create dropdown and load parameters
        dropdown = make_dropdown()
        params = load_demo_parameters(dropdown)

        # Verify workflow
        expected_limb_config = {
            "log": False,
            "y_min": 0,
            "y_max": -1,
            "window_length": 501,
            "polyorder": 1,
            "deriv": 0,
            "delta": 1,
            "segmenter": "segment-anything",
        }

        assert params["target"] == "Pluto"
        assert params["limb_config"] == expected_limb_config
        assert dropdown == mock_dropdown

    def test_parameter_consistency_across_demos(self):
        """Test that parameter structures are consistent across all demos."""
        # Load all demo configurations
        all_params = []
        for value in [1, 2, 3, 4]:
            demo = Mock()
            demo.value = value
            params = load_demo_parameters(demo)
            all_params.append(params)

        # Verify all have same keys
        first_keys = set(all_params[0].keys())
        for params in all_params[1:]:
            assert set(params.keys()) == first_keys

        # Verify data types
        for params in all_params:
            assert isinstance(params["target"], str)
            assert isinstance(params["true_radius"], int)
            assert isinstance(params["image_filepath"], str)
            assert isinstance(params["fit_config"], str)
            assert isinstance(params["limb_config"], dict)
            assert isinstance(params["limb_save"], str)
            assert isinstance(params["parameter_walkthrough"], str)
            assert isinstance(params["preamble"], str)

        # Verify limb_config has consistent structure across all demos
        expected_limb_keys = {
            "log",
            "y_min",
            "y_max",
            "window_length",
            "polyorder",
            "deriv",
            "delta",
            "segmenter",
        }
        for params in all_params:
            assert set(params["limb_config"].keys()) == expected_limb_keys


# Error handling and edge cases
class TestDemoErrorHandling:
    """Test error handling and edge cases."""

    def test_load_demo_parameters_valid_limb_config_structure(self):
        """Test that limb_config has valid structure for all demos."""
        for demo_value in [1, 2, 3, 4]:
            demo = Mock()
            demo.value = demo_value
            result = load_demo_parameters(demo)

            limb_config = result["limb_config"]

            # Verify required limb_config keys and types
            assert isinstance(limb_config["log"], bool)
            assert isinstance(limb_config["y_min"], int)
            assert isinstance(limb_config["y_max"], int)
            assert isinstance(limb_config["window_length"], int)
            assert isinstance(limb_config["polyorder"], int)
            assert isinstance(limb_config["deriv"], int)
            assert isinstance(limb_config["delta"], int)
            assert isinstance(limb_config["segmenter"], str)

            # Verify reasonable values
            assert limb_config["window_length"] > 0
            assert limb_config["polyorder"] >= 0
            assert limb_config["segmenter"] == "segment-anything"

    def test_load_demo_parameters_limb_config_consistency(self):
        """Test that all demos have identical limb_config structures."""
        # Load all demo configurations
        all_limb_configs = []
        for value in [1, 2, 3, 4]:
            demo = Mock()
            demo.value = value
            params = load_demo_parameters(demo)
            all_limb_configs.append(params["limb_config"])

        # All limb configs should be identical
        first_config = all_limb_configs[0]
        for config in all_limb_configs[1:]:
            assert config == first_config

    def test_load_demo_parameters_none_demo(self):
        """Test handling of None demo object."""
        with pytest.raises(AttributeError):
            load_demo_parameters(None)

    def test_load_demo_parameters_no_value_attribute(self):
        """Test handling of demo object without value attribute."""
        demo = Mock(spec=[])  # Mock with no attributes

        with pytest.raises(AttributeError):
            load_demo_parameters(demo)


# Test data validation
class TestDemoDataValidation:
    """Test validation of demo data and configurations."""

    def test_radius_values_are_reasonable(self):
        """Test that radius values are within reasonable ranges."""
        expected_radii = {
            1: 1188,  # Pluto (km)
            2: 58232,  # Saturn (km)
            3: 58232,  # Saturn (km)
            4: 6371,  # Earth (km)
        }

        for demo_value, expected_radius in expected_radii.items():
            demo = Mock()
            demo.value = demo_value
            params = load_demo_parameters(demo)

            assert params["true_radius"] == expected_radius
            assert params["true_radius"] > 0  # Positive radius

    def test_file_paths_have_correct_extensions(self):
        """Test that file paths have expected extensions."""
        for demo_value in [1, 2, 3, 4]:
            demo = Mock()
            demo.value = demo_value
            params = load_demo_parameters(demo)

            # Check file extensions
            assert params["image_filepath"].endswith((".jpg", ".tif"))
            assert params["fit_config"].endswith(".yaml")
            assert params["limb_save"].endswith(".npy")
            assert params["parameter_walkthrough"].endswith(".md")
            assert params["preamble"].endswith(".md")
