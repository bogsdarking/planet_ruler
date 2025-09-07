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
        expected_options = [("Pluto", 1), ("Saturn-1", 2), ("Saturn-2", 3), ("Earth", 4)]
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

    @patch("planet_ruler.demo.json.load")
    @patch("planet_ruler.demo.open")
    def test_load_demo_parameters_pluto(self, mock_open_func, mock_json_load):
        """Test loading Pluto demo parameters."""
        demo = self.create_mock_demo(1)

        # Mock JSON config loading
        mock_limb_config = {"limb_method": "gradient-break", "window_length": 501}
        mock_json_load.return_value = mock_limb_config

        result = load_demo_parameters(demo)

        # Verify JSON file was loaded
        mock_open_func.assert_called_once_with("../config/pluto_limb_1.json", "r")
        mock_json_load.assert_called_once()

        # Verify parameter structure
        expected_params = {
            "target": "Pluto",
            "true_radius": 1188,
            "image_filepath": "../demo/images/PIA19948.tif",
            "fit_config": "../config/pluto-new-horizons.yaml",
            "limb_config": mock_limb_config,
            "limb_save": "pluto_limb.npy",
            "parameter_walkthrough": "../demo/pluto_init.md",
            "preamble": "../demo/pluto_preamble.md",
        }

        assert result == expected_params
        assert result["target"] == "Pluto"
        assert result["true_radius"] == 1188
        assert result["limb_config"] == mock_limb_config

    @patch("planet_ruler.demo.json.load")
    @patch("planet_ruler.demo.open")
    def test_load_demo_parameters_saturn_1(self, mock_open_func, mock_json_load):
        """Test loading Saturn-1 demo parameters."""
        demo = self.create_mock_demo(2)

        mock_limb_config = {"limb_method": "segmentation", "smooth_method": "savgol"}
        mock_json_load.return_value = mock_limb_config

        result = load_demo_parameters(demo)

        # Verify correct JSON file loaded
        mock_open_func.assert_called_once_with("../config/saturn_limb_1.json", "r")

        # Verify Saturn-1 specific parameters
        assert result["target"] == "Saturn"
        assert result["true_radius"] == 58232
        assert result["image_filepath"] == "../demo/images/PIA21341.jpg"
        assert result["fit_config"] == "../config/saturn-cassini-1.yaml"
        assert result["limb_save"] == "saturn_limb_1.npy"
        assert result["parameter_walkthrough"] == "../demo/saturn_init_1.md"
        assert result["preamble"] == "../demo/saturn_preamble_1.md"

    @patch("planet_ruler.demo.json.load")
    @patch("planet_ruler.demo.open")
    def test_load_demo_parameters_saturn_2(self, mock_open_func, mock_json_load):
        """Test loading Saturn-2 demo parameters."""
        demo = self.create_mock_demo(3)

        mock_limb_config = {"limb_method": "gradient-break", "log": True}
        mock_json_load.return_value = mock_limb_config

        result = load_demo_parameters(demo)

        # Verify correct JSON file loaded
        mock_open_func.assert_called_once_with("../config/saturn_limb_2.json", "r")

        # Verify Saturn-2 specific parameters
        assert result["target"] == "Saturn"
        assert result["true_radius"] == 58232
        assert result["image_filepath"] == "../demo/images/saturn_ciclops_5769_13427_1.jpg"
        assert result["fit_config"] == "../config/saturn-cassini-2.yaml"
        assert result["limb_save"] == "saturn_limb_2.npy"
        assert result["parameter_walkthrough"] == "../demo/saturn_init_2.md"
        assert result["preamble"] == "../demo/saturn_preamble_2.md"

    @patch("planet_ruler.demo.json.load")
    @patch("planet_ruler.demo.open")
    def test_load_demo_parameters_earth(self, mock_open_func, mock_json_load):
        """Test loading Earth demo parameters."""
        demo = self.create_mock_demo(4)

        mock_limb_config = {"limb_method": "gradient-break", "window_length": 301}
        mock_json_load.return_value = mock_limb_config

        result = load_demo_parameters(demo)

        # Note: Earth demo uses saturn_limb_2.json config (as per code)
        mock_open_func.assert_called_once_with("../config/saturn_limb_2.json", "r")

        # Verify Earth specific parameters
        assert result["target"] == "Earth"
        assert result["true_radius"] == 6371
        assert result["image_filepath"] == "../demo/images/50644513538_56228a2027_o.jpg"
        assert result["fit_config"] == "../config/earth_iss_1.yaml"
        assert result["limb_save"] == "earth_limb_1.npy"
        assert result["parameter_walkthrough"] == "../demo/earth_init_1.md"
        assert result["preamble"] == "../demo/earth_preamble_1.md"

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

    @patch("planet_ruler.demo.json.load")
    @patch("planet_ruler.demo.open")
    def test_load_demo_parameters_all_demos_have_required_keys(self, mock_open_func, mock_json_load):
        """Test that all demo configurations have required parameter keys."""
        mock_limb_config = {"test": "config"}
        mock_json_load.return_value = mock_limb_config

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

    @patch("planet_ruler.demo.json.load")
    @patch("planet_ruler.demo.open")
    def test_load_demo_parameters_file_path_consistency(self, mock_open_func, mock_json_load):
        """Test that file paths follow expected patterns."""
        mock_json_load.return_value = {}

        # Test path patterns for each demo
        demo_configs = [
            (1, "../config/pluto_limb_1.json", "pluto_limb.npy"),
            (2, "../config/saturn_limb_1.json", "saturn_limb_1.npy"),
            (3, "../config/saturn_limb_2.json", "saturn_limb_2.npy"),
            (4, "../config/saturn_limb_2.json", "earth_limb_1.npy"),  # Earth uses Saturn config
        ]

        for demo_value, expected_config_path, expected_save_path in demo_configs:
            mock_open_func.reset_mock()
            demo = self.create_mock_demo(demo_value)
            result = load_demo_parameters(demo)

            mock_open_func.assert_called_once_with(expected_config_path, "r")
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
    @patch("planet_ruler.demo.json.load")
    @patch("planet_ruler.demo.open")
    def test_dropdown_to_parameters_workflow(self, mock_open_func, mock_json_load, mock_widgets):
        """Test complete workflow from dropdown creation to parameter loading."""
        # Setup dropdown
        mock_dropdown = Mock()
        mock_dropdown.value = 1  # Pluto
        mock_widgets.Dropdown.return_value = mock_dropdown

        # Setup JSON loading
        mock_limb_config = {"limb_method": "gradient-break"}
        mock_json_load.return_value = mock_limb_config

        # Create dropdown and load parameters
        dropdown = make_dropdown()
        params = load_demo_parameters(dropdown)

        # Verify workflow
        assert params["target"] == "Pluto"
        assert params["limb_config"] == mock_limb_config
        assert dropdown == mock_dropdown

    def test_parameter_consistency_across_demos(self):
        """Test that parameter structures are consistent across all demos."""

        with patch("planet_ruler.demo.json.load") as mock_json_load, patch("planet_ruler.demo.open"):

            mock_json_load.return_value = {}

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
                assert isinstance(params["limb_save"], str)
                assert isinstance(params["parameter_walkthrough"], str)
                assert isinstance(params["preamble"], str)


# Error handling and edge cases
class TestDemoErrorHandling:
    """Test error handling and edge cases."""

    @patch("planet_ruler.demo.json.load")
    @patch("planet_ruler.demo.open")
    def test_load_demo_parameters_json_error(self, mock_open_func, mock_json_load):
        """Test handling of JSON loading errors."""
        demo = Mock()
        demo.value = 1

        # Simulate JSON decode error
        mock_json_load.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)

        with pytest.raises(json.JSONDecodeError):
            load_demo_parameters(demo)

    @patch("planet_ruler.demo.json.load")
    @patch("planet_ruler.demo.open")
    def test_load_demo_parameters_file_not_found(self, mock_open_func, mock_json_load):
        """Test handling when config file doesn't exist."""
        demo = Mock()
        demo.value = 1

        mock_open_func.side_effect = FileNotFoundError("Config file not found")

        with pytest.raises(FileNotFoundError):
            load_demo_parameters(demo)

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

    @patch("planet_ruler.demo.json.load")
    @patch("planet_ruler.demo.open")
    def test_radius_values_are_reasonable(self, mock_open_func, mock_json_load):
        """Test that radius values are within reasonable ranges."""
        mock_json_load.return_value = {}

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

    @patch("planet_ruler.demo.json.load")
    @patch("planet_ruler.demo.open")
    def test_file_paths_have_correct_extensions(self, mock_open_func, mock_json_load):
        """Test that file paths have expected extensions."""
        mock_json_load.return_value = {}

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
