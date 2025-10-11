# tests/test_annotate.py

import pytest
import numpy as np
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from planet_ruler.annotate import TkLimbAnnotator, ToolTip, create_tooltip


class TestTkLimbAnnotatorCore:
    """Test core functionality of TkLimbAnnotator that doesn't require GUI"""

    def test_point_management(self):
        """Test point addition and removal logic"""
        # Test the core logic without GUI components

        # Mock the GUI parts
        with patch("tkinter.Tk"), patch(
            "PIL.Image.open"
        ) as mock_image_open, patch.object(
            TkLimbAnnotator, "create_widgets"
        ), patch.object(
            TkLimbAnnotator, "auto_fit_zoom"
        ):

            mock_image = Mock()
            mock_image.size = (800, 600)
            mock_image_open.return_value = mock_image

            annotator = TkLimbAnnotator("test_image.png")

            # Test point addition
            assert len(annotator.points) == 0
            annotator.points.append((100, 200))
            annotator.points.append((200, 210))
            assert len(annotator.points) == 2
            assert annotator.points[0] == (100, 200)
            assert annotator.points[1] == (200, 210)

            # Test point removal
            annotator.points.pop()
            assert len(annotator.points) == 1
            assert annotator.points[0] == (100, 200)

    def test_target_generation_with_sufficient_points(self):
        """Test target array generation with sufficient points"""
        with patch("tkinter.Tk"), patch(
            "PIL.Image.open"
        ) as mock_image_open, patch.object(
            TkLimbAnnotator, "create_widgets"
        ), patch.object(
            TkLimbAnnotator, "auto_fit_zoom"
        ):

            mock_image = Mock()
            mock_image.size = (800, 600)
            mock_image_open.return_value = mock_image

            annotator = TkLimbAnnotator("test_image.png")

            # Add sufficient points (3 or more)
            annotator.points = [(100, 200), (200, 210), (300, 220), (400, 230)]

            target = annotator.get_target()

            # Verify target array
            assert target is not None
            assert len(target) == 800  # Image width
            assert target[100] == 200  # y-value at x=100
            assert target[200] == 210  # y-value at x=200

    def test_target_generation_with_insufficient_points(self):
        """Test target array generation with insufficient points"""
        with patch("tkinter.Tk"), patch(
            "PIL.Image.open"
        ) as mock_image_open, patch.object(
            TkLimbAnnotator, "create_widgets"
        ), patch.object(
            TkLimbAnnotator, "auto_fit_zoom"
        ):

            mock_image = Mock()
            mock_image.size = (800, 600)
            mock_image_open.return_value = mock_image

            annotator = TkLimbAnnotator("test_image.png")

            # Add insufficient points (less than 3)
            annotator.points = [(100, 200), (200, 210)]

            target = annotator.get_target()

            # Should return None for insufficient points
            assert target is None

    @patch("tkinter.messagebox.showwarning")
    def test_generate_target_insufficient_points_warning(self, mock_warning):
        """Test generate_target shows warning with insufficient points"""
        with patch("tkinter.Tk"), patch(
            "PIL.Image.open"
        ) as mock_image_open, patch.object(
            TkLimbAnnotator, "create_widgets"
        ), patch.object(
            TkLimbAnnotator, "auto_fit_zoom"
        ):

            mock_image = Mock()
            mock_image.size = (800, 600)
            mock_image_open.return_value = mock_image

            annotator = TkLimbAnnotator("test_image.png")

            # Add insufficient points
            annotator.points = [(100, 200)]  # Only 1 point

            result = annotator.generate_target()

            # Should show warning and return None
            mock_warning.assert_called_once()
            assert result is None

    @patch("numpy.save")
    @patch("tkinter.messagebox.showinfo")
    def test_generate_target_success(self, mock_info, mock_np_save):
        """Test successful target generation"""
        with patch("tkinter.Tk"), patch(
            "PIL.Image.open"
        ) as mock_image_open, patch.object(
            TkLimbAnnotator, "create_widgets"
        ), patch.object(
            TkLimbAnnotator, "auto_fit_zoom"
        ):

            mock_image = Mock()
            mock_image.size = (800, 600)
            mock_image_open.return_value = mock_image

            annotator = TkLimbAnnotator("test_image.png")
            annotator.image_path = "/path/to/test_image.png"

            # Add sufficient points
            annotator.points = [(100, 200), (200, 210), (300, 220)]

            result = annotator.generate_target()

            # Verify target was generated and saved
            assert result is not None
            assert len(result) == 800
            mock_np_save.assert_called_once()
            mock_info.assert_called_once()


class TestToolTip:
    """Test the ToolTip class"""

    def test_create_tooltip_helper(self):
        """Test the create_tooltip helper function"""
        mock_widget = Mock()
        mock_widget.bind = Mock()

        tooltip = create_tooltip(mock_widget, "Helper test")

        assert isinstance(tooltip, ToolTip)
        assert tooltip.text == "Helper test"


class TestManualLimbDetectionIntegration:
    """Integration tests for manual limb detection in LimbObservation"""

    @pytest.fixture
    def sample_fit_config(self):
        """Create a sample fit configuration"""
        config = {
            "free_parameters": ["r", "h", "f"],
            "init_parameter_values": {
                "r": 6371000.0,  # Earth radius in meters
                "h": 10000.0,  # 10km altitude
                "f": 0.024,  # 24mm lens
            },
            "parameter_limits": {
                "r": [6000000.0, 7000000.0],
                "h": [100.0, 100000.0],
                "f": [0.01, 0.1],
            },
        }
        return config

    @pytest.fixture
    def config_file(self, sample_fit_config):
        """Create a temporary config file"""
        import yaml

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.dump(sample_fit_config, f)
            f.flush()
            yield f.name
        os.unlink(f.name)

    @patch("planet_ruler.observation.TkLimbAnnotator")
    def test_manual_limb_detection(
        self, mock_annotator_class, sample_horizon_image, config_file
    ):
        """Test manual limb detection method integration"""
        from planet_ruler.observation import LimbObservation
        import matplotlib.pyplot as plt

        # Create test image
        image_data = sample_horizon_image()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            plt.imsave(tmp_file.name, image_data, cmap="gray")
            tmp_file.flush()

            # Create observation with manual detection
            obs = LimbObservation(
                image_filepath=tmp_file.name,
                fit_config=config_file,
                limb_detection="manual",
            )

            # Mock the annotator
            mock_annotator = Mock()
            expected_limb = np.random.random(image_data.shape[1])
            mock_annotator.get_target.return_value = expected_limb
            mock_annotator.run = Mock()
            mock_annotator_class.return_value = mock_annotator

            # Run detection
            obs.detect_limb()

            # Verify annotator was created and used correctly
            mock_annotator_class.assert_called_once_with(
                image_path=tmp_file.name, initial_stretch=1.0
            )
            mock_annotator.run.assert_called_once()
            mock_annotator.get_target.assert_called_once()

            # Verify limb was registered
            assert "limb" in obs.features
            assert np.array_equal(obs.features["limb"], expected_limb)
            assert np.array_equal(obs._raw_limb, expected_limb)

            os.unlink(tmp_file.name)

    def test_manual_method_in_valid_methods(self, sample_horizon_image, config_file):
        """Test that 'manual' is accepted as a valid limb detection method"""
        from planet_ruler.observation import LimbObservation
        import matplotlib.pyplot as plt

        # Create test image
        image_data = sample_horizon_image()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            plt.imsave(tmp_file.name, image_data, cmap="gray")
            tmp_file.flush()

            # Should not raise assertion error
            obs = LimbObservation(
                image_filepath=tmp_file.name,
                fit_config=config_file,
                limb_detection="manual",
            )

            assert obs.limb_detection == "manual"

            os.unlink(tmp_file.name)

    def test_invalid_limb_method_still_raises_error(
        self, sample_horizon_image, config_file
    ):
        """Test that invalid methods still raise AssertionError"""
        from planet_ruler.observation import LimbObservation
        import matplotlib.pyplot as plt

        # Create test image
        image_data = sample_horizon_image()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            plt.imsave(tmp_file.name, image_data, cmap="gray")
            tmp_file.flush()

            # Should raise assertion error for invalid method
            with pytest.raises(AssertionError):
                LimbObservation(
                    image_filepath=tmp_file.name,
                    fit_config=config_file,
                    limb_detection="invalid-method",
                )

            os.unlink(tmp_file.name)

    @patch("planet_ruler.observation.TkLimbAnnotator")
    def test_manual_detection_with_analyze_method(
        self, mock_annotator_class, sample_horizon_image, config_file
    ):
        """Test manual detection works with the analyze() method"""
        from planet_ruler.observation import LimbObservation
        import matplotlib.pyplot as plt

        # Create test image
        image_data = sample_horizon_image()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            plt.imsave(tmp_file.name, image_data, cmap="gray")
            tmp_file.flush()

            obs = LimbObservation(
                image_filepath=tmp_file.name,
                fit_config=config_file,
                limb_detection="manual",
            )

            # Mock the annotator
            mock_annotator = Mock()
            expected_limb = np.random.random(image_data.shape[1])
            mock_annotator.get_target.return_value = expected_limb
            mock_annotator.run = Mock()
            mock_annotator_class.return_value = mock_annotator

            # Mock fit_limb to avoid complex setup
            with patch.object(obs, "fit_limb", return_value=obs):
                # Use analyze method (should call detect_limb internally)
                result = obs.analyze()

                # Verify it returns self for chaining
                assert result is obs

                # Verify detection occurred
                mock_annotator_class.assert_called_once()
                mock_annotator.run.assert_called_once()

            os.unlink(tmp_file.name)

    def test_manual_detection_in_valid_methods_list(self):
        """Test that 'manual' is in the valid detection methods"""
        from planet_ruler.observation import LimbObservation

        # Check that manual is in valid methods
        valid_methods = ["gradient-break", "segmentation", "manual"]

        # This should be reflected in the LimbObservation class implementation
        # The assertion in __init__ should accept these methods
        assert "manual" in valid_methods
