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

# tests/test_observation.py

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import yaml
from unittest.mock import Mock, patch, MagicMock
import matplotlib.pyplot as plt

from planet_ruler.observation import (
    PlanetObservation,
    LimbObservation,
    package_results,
)
from planet_ruler.fit import unpack_diff_evol_posteriors
from planet_ruler.plot import (
    plot_diff_evol_posteriors,
    plot_full_limb,
    plot_segmentation_masks,
)


class TestPlanetObservation:
    """Test the base PlanetObservation class"""

    def test_initialization(self, sample_horizon_image):
        """Test PlanetObservation initialization"""
        # Generate actual image using the fixture function
        image_data = sample_horizon_image()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            # Save the sample image to a temporary file
            plt.imsave(tmp_file.name, image_data, cmap="gray")
            tmp_file.flush()

            obs = PlanetObservation(tmp_file.name)

            # Handle that loaded images may have channel dimension
            assert (
                obs.image.shape[:2] == image_data.shape[:2]
            )  # Compare height and width
            assert isinstance(obs.features, dict)
            assert len(obs.features) == 0
            assert isinstance(obs._plot_functions, dict)
            assert len(obs._plot_functions) == 0
            assert len(obs._cwheel) == 6  # Color wheel

            # Clean up
            os.unlink(tmp_file.name)

    @patch("matplotlib.pyplot.show")
    @patch("planet_ruler.observation.plot_image")
    def test_plot_empty_features(
        self, mock_plot_image, mock_show, sample_horizon_image
    ):
        """Test plotting with no features"""
        image_data = sample_horizon_image()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            plt.imsave(tmp_file.name, image_data, cmap="gray")
            tmp_file.flush()

            obs = PlanetObservation(tmp_file.name)
            obs.plot()

            mock_plot_image.assert_called_once_with(
                obs.image, gradient=False, show=False
            )
            mock_show.assert_called_once()

            os.unlink(tmp_file.name)

    @patch("matplotlib.pyplot.show")
    @patch("planet_ruler.observation.plot_image")
    def test_plot_with_features(self, mock_plot_image, mock_show, sample_horizon_image):
        """Test plotting with features"""
        image_data = sample_horizon_image()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            plt.imsave(tmp_file.name, image_data, cmap="gray")
            tmp_file.flush()

            obs = PlanetObservation(tmp_file.name)

            # Add a mock feature and its plot function
            obs.features["test_feature"] = np.array([1, 2, 3])
            mock_plot_func = Mock()
            obs._plot_functions["test_feature"] = mock_plot_func

            obs.plot(gradient=True, show=False)

            mock_plot_image.assert_called_once_with(
                obs.image, gradient=True, show=False
            )
            mock_plot_func.assert_called_once()
            mock_show.assert_not_called()  # show=False

            os.unlink(tmp_file.name)


class TestLimbObservation:
    """Test the LimbObservation class"""

    @pytest.fixture
    def sample_fit_config(self):
        """Create a sample fit configuration"""
        config = {
            "free_parameters": ["planet_radius", "altitude"],
            "init_parameter_values": {
                "planet_radius": 6371000.0,  # Earth radius in meters
                "altitude": 10000.0,  # 10km altitude
                "focal_length": 0.024,  # 24mm lens
                "detector_width": 0.036,  # Full frame sensor
                "detector_height": 0.024,
            },
            "parameter_limits": {
                "planet_radius": [6000000.0, 7000000.0],
                "altitude": [100.0, 100000.0],
                "focal_length": [0.01, 0.1],
                "detector_width": [0.01, 0.1],
                "detector_height": [0.01, 0.1],
            },
        }
        return config

    @pytest.fixture
    def config_file(self, sample_fit_config):
        """Create a temporary config file"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.dump(sample_fit_config, f)
            f.flush()
            yield f.name
        os.unlink(f.name)

    def test_initialization(self, sample_horizon_image, config_file):
        """Test LimbObservation initialization"""
        image_data = sample_horizon_image()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            plt.imsave(tmp_file.name, image_data, cmap="gray")
            tmp_file.flush()

            obs = LimbObservation(
                image_filepath=tmp_file.name,
                fit_config=config_file,
                limb_detection="segmentation",
                minimizer="differential-evolution",
            )

            # Handle that loaded images may have channel dimension
            assert (
                obs.image.shape[:2] == image_data.shape[:2]
            )  # Compare height and width
            assert obs.limb_detection == "segmentation"
            assert obs.minimizer == "differential-evolution"
            assert obs._segmenter is None
            assert obs.free_parameters == ["planet_radius", "altitude"]
            assert obs.init_parameter_values["planet_radius"] == 6371000.0

            os.unlink(tmp_file.name)

    def test_initialization_invalid_limb_detection(
        self, sample_horizon_image, config_file
    ):
        """Test initialization with invalid limb detection method"""
        image_data = sample_horizon_image()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            plt.imsave(tmp_file.name, image_data, cmap="gray")
            tmp_file.flush()

            with pytest.raises(AssertionError):
                LimbObservation(
                    image_filepath=tmp_file.name,
                    fit_config=config_file,
                    limb_detection="invalid-method",
                )

            os.unlink(tmp_file.name)

    def test_initialization_invalid_minimizer(self, sample_horizon_image, config_file):
        """Test initialization with invalid minimizer"""
        image_data = sample_horizon_image()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            plt.imsave(tmp_file.name, image_data, cmap="gray")
            tmp_file.flush()

            with pytest.raises(AssertionError):
                LimbObservation(
                    image_filepath=tmp_file.name,
                    fit_config=config_file,
                    limb_detection="segmentation",
                    minimizer="invalid-minimizer",
                )

            os.unlink(tmp_file.name)

    def test_load_fit_config_invalid_initial_values(
        self, sample_horizon_image, sample_fit_config
    ):
        """Test load_fit_config with invalid initial values"""
        # Create config with initial value outside bounds
        invalid_config = sample_fit_config.copy()
        invalid_config["init_parameter_values"][
            "planet_radius"
        ] = 5000000.0  # Below lower bound

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.dump(invalid_config, f)
            f.flush()

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                sample_image = np.random.random((100, 100))
                plt.imsave(tmp_file.name, sample_image, cmap="gray")
                tmp_file.flush()

                with pytest.raises(AssertionError, match="violates stated lower limit"):
                    LimbObservation(image_filepath=tmp_file.name, fit_config=f.name)

                os.unlink(tmp_file.name)

            os.unlink(f.name)

    @patch("planet_ruler.observation.gradient_break")
    def test_detect_limb_gradient_break(
        self, mock_gradient_break, sample_horizon_image, config_file
    ):
        """Test limb detection using gradient-break method"""
        image_data = sample_horizon_image()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            plt.imsave(tmp_file.name, image_data, cmap="gray")
            tmp_file.flush()

            obs = LimbObservation(
                image_filepath=tmp_file.name,
                fit_config=config_file,
                limb_detection="gradient-break",
            )

            # Mock return value
            expected_limb = np.random.random(image_data.shape[1])
            mock_gradient_break.return_value = expected_limb

            obs.detect_limb(log=True, y_min=10, y_max=90)

            mock_gradient_break.assert_called_once_with(
                obs.image,
                log=True,
                y_min=10,
                y_max=90,
                window_length=501,
                polyorder=1,
                deriv=0,
                delta=1,
            )

            assert np.array_equal(obs.features["limb"], expected_limb)
            assert np.array_equal(obs._raw_limb, expected_limb)

            os.unlink(tmp_file.name)

    @patch("planet_ruler.observation.ImageSegmentation")
    def test_detect_limb_segmentation(
        self, mock_segmentation_class, sample_horizon_image, config_file
    ):
        """Test limb detection using segmentation method"""
        image_data = sample_horizon_image()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            plt.imsave(tmp_file.name, image_data, cmap="gray")
            tmp_file.flush()

            obs = LimbObservation(
                image_filepath=tmp_file.name,
                fit_config=config_file,
                limb_detection="segmentation",
            )

            # Mock segmenter and its methods
            mock_segmenter = Mock()
            expected_limb = np.random.random(image_data.shape[1])
            mock_segmenter.segment.return_value = expected_limb
            mock_segmentation_class.return_value = mock_segmenter

            obs.detect_limb(segmenter="segment-anything")

            mock_segmentation_class.assert_called_once_with(
                obs.image, segmenter="segment-anything"
            )
            mock_segmenter.segment.assert_called_once()

            assert np.array_equal(obs.features["limb"], expected_limb)
            assert np.array_equal(obs._raw_limb, expected_limb)
            assert obs._segmenter is mock_segmenter

            os.unlink(tmp_file.name)

    @patch("planet_ruler.observation.TkLimbAnnotator")
    def test_detect_limb_manual(
        self, mock_annotator_class, sample_horizon_image, config_file
    ):
        """Test limb detection using manual method"""
        image_data = sample_horizon_image()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            plt.imsave(tmp_file.name, image_data, cmap="gray")
            tmp_file.flush()

            obs = LimbObservation(
                image_filepath=tmp_file.name,
                fit_config=config_file,
                limb_detection="manual",
            )

            # Mock annotator and its methods
            mock_annotator = Mock()
            expected_limb = np.random.random(image_data.shape[1])
            mock_annotator.get_target.return_value = expected_limb
            mock_annotator.run = Mock()
            mock_annotator_class.return_value = mock_annotator

            obs.detect_limb()

            # Verify annotator was created with correct parameters
            mock_annotator_class.assert_called_once_with(
                image_path=tmp_file.name, initial_stretch=1.0
            )
            mock_annotator.run.assert_called_once()
            mock_annotator.get_target.assert_called_once()

            # Verify limb was registered
            assert np.array_equal(obs.features["limb"], expected_limb)
            assert np.array_equal(obs._raw_limb, expected_limb)

            os.unlink(tmp_file.name)

    @patch("planet_ruler.observation.TkLimbAnnotator")
    def test_detect_limb_manual_with_default_stretch(
        self, mock_annotator_class, sample_horizon_image, config_file
    ):
        """Test manual limb detection uses default stretch value"""
        image_data = sample_horizon_image()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            plt.imsave(tmp_file.name, image_data, cmap="gray")
            tmp_file.flush()

            obs = LimbObservation(
                image_filepath=tmp_file.name,
                fit_config=config_file,
                limb_detection="manual",
            )

            # Mock annotator
            mock_annotator = Mock()
            expected_limb = np.random.random(image_data.shape[1])
            mock_annotator.get_target.return_value = expected_limb
            mock_annotator.run = Mock()
            mock_annotator_class.return_value = mock_annotator

            # Manual method uses fixed initial_stretch of 1.0
            obs.detect_limb()

            # Verify annotator was created with default stretch
            mock_annotator_class.assert_called_once_with(
                image_path=tmp_file.name, initial_stretch=1.0
            )

            os.unlink(tmp_file.name)

    @patch("planet_ruler.observation.TkLimbAnnotator")
    def test_detect_limb_manual_no_target_returned(
        self, mock_annotator_class, sample_horizon_image, config_file
    ):
        """Test manual limb detection when annotator returns None"""
        image_data = sample_horizon_image()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            plt.imsave(tmp_file.name, image_data, cmap="gray")
            tmp_file.flush()

            obs = LimbObservation(
                image_filepath=tmp_file.name,
                fit_config=config_file,
                limb_detection="manual",
            )

            # Mock annotator to return None (insufficient points)
            mock_annotator = Mock()
            mock_annotator.get_target.return_value = None
            mock_annotator.run = Mock()
            mock_annotator_class.return_value = mock_annotator

            # Should handle None gracefully (no limb registered)
            obs.detect_limb()

            # Verify annotator was called but no limb registered
            mock_annotator_class.assert_called_once()
            mock_annotator.run.assert_called_once()
            mock_annotator.get_target.assert_called_once()

            # limb should not be in features since None was returned
            assert "limb" not in obs.features

            os.unlink(tmp_file.name)

    def test_manual_method_in_valid_detection_methods(
        self, sample_horizon_image, config_file
    ):
        """Test that 'manual' is accepted as a valid limb detection method"""
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

    @patch("planet_ruler.observation.smooth_limb")
    @patch("planet_ruler.observation.fill_nans")
    def test_smooth_limb(
        self, mock_fill_nans, mock_smooth_limb, sample_horizon_image, config_file
    ):
        """Test limb smoothing functionality"""
        image_data = sample_horizon_image()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            plt.imsave(tmp_file.name, image_data, cmap="gray")
            tmp_file.flush()

            obs = LimbObservation(image_filepath=tmp_file.name, fit_config=config_file)

            # Set up mock raw limb data
            raw_limb = np.random.random(100)
            obs._raw_limb = raw_limb

            smoothed_limb = np.random.random(100)
            filled_limb = np.random.random(100)

            mock_smooth_limb.return_value = smoothed_limb
            mock_fill_nans.return_value = filled_limb

            obs.smooth_limb(fill_nan=True, window_size=5)

            mock_smooth_limb.assert_called_once_with(raw_limb, window_size=5)
            mock_fill_nans.assert_called_once_with(smoothed_limb)

            assert np.array_equal(obs.features["limb"], filled_limb)

            os.unlink(tmp_file.name)

    @patch("planet_ruler.observation.differential_evolution")
    @patch("planet_ruler.observation.CostFunction")
    def test_fit_limb(
        self,
        mock_cost_function_class,
        mock_diff_evolution,
        sample_horizon_image,
        config_file,
    ):
        """Test limb fitting functionality"""
        image_data = sample_horizon_image()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            plt.imsave(tmp_file.name, image_data, cmap="gray")
            tmp_file.flush()

            obs = LimbObservation(image_filepath=tmp_file.name, fit_config=config_file)

            # Set up limb data
            limb_data = np.random.random(image_data.shape[1])
            obs.features["limb"] = limb_data

            # Mock cost function
            mock_cost_function = Mock()
            mock_cost_function_class.return_value = mock_cost_function

            # Mock differential evolution results
            mock_result = Mock()
            mock_result.x = np.array([6500000.0, 15000.0])  # Best fit parameters
            mock_diff_evolution.return_value = mock_result

            # Mock cost function evaluation
            fitted_limb = np.random.random(image_data.shape[1])
            mock_cost_function.evaluate.return_value = fitted_limb

            obs.fit_limb(loss_function="l2", max_iter=100, n_jobs=1, seed=42)

            # Verify cost function creation
            mock_cost_function_class.assert_called_once()
            call_args = mock_cost_function_class.call_args
            assert np.array_equal(call_args[1]["target"], limb_data)
            assert call_args[1]["free_parameters"] == ["planet_radius", "altitude"]
            assert call_args[1]["loss_function"] == "l2"

            # Verify differential evolution call
            mock_diff_evolution.assert_called_once()
            de_args = mock_diff_evolution.call_args
            assert de_args[1]["maxiter"] == 100
            assert de_args[1]["seed"] == 42

            # Verify results
            assert obs.fit_results == mock_result
            assert np.array_equal(obs.features["fitted_limb"], fitted_limb)
            assert obs.best_parameters is not None
            assert obs.best_parameters["planet_radius"] is not None
            assert obs.best_parameters["altitude"] is not None

            os.unlink(tmp_file.name)

    def test_save_and_load_limb(self, sample_horizon_image, config_file):
        """Test saving and loading limb data"""
        image_data = sample_horizon_image()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_image:
            plt.imsave(tmp_image.name, image_data, cmap="gray")
            tmp_image.flush()

            obs = LimbObservation(image_filepath=tmp_image.name, fit_config=config_file)

            # Create sample limb data
            limb_data = np.random.random(100)
            obs.features["limb"] = limb_data

            # Save limb data
            with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp_limb:
                obs.save_limb(tmp_limb.name)

                # Create new observation and load limb data
                obs2 = LimbObservation(
                    image_filepath=tmp_image.name, fit_config=config_file
                )

                with patch("planet_ruler.observation.fill_nans") as mock_fill_nans:
                    filled_limb = limb_data + 0.1  # Slightly different
                    mock_fill_nans.return_value = filled_limb

                    obs2.load_limb(tmp_limb.name)

                    mock_fill_nans.assert_called_once()
                    assert np.array_equal(obs2.features["limb"], filled_limb)
                    assert np.array_equal(obs2._raw_limb, filled_limb)

                os.unlink(tmp_limb.name)

            os.unlink(tmp_image.name)

    def test_warm_start_parameter_protection(self, sample_horizon_image, config_file):
        """Test warm start parameter protection and restoration functionality"""
        image_data = sample_horizon_image()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            plt.imsave(tmp_file.name, image_data, cmap="gray")
            tmp_file.flush()

            obs = LimbObservation(
                image_filepath=tmp_file.name,
                fit_config=config_file,
                limb_detection="manual",
            )

            # Test 1: Verify original values are properly stored
            original_values = obs.init_parameter_values.copy()
            stored_original_values = obs._original_init_parameter_values.copy()
            
            assert original_values == stored_original_values
            assert hasattr(obs, '_original_init_parameter_values')
            assert obs._original_init_parameter_values is not None

            # Test 2: Simulate a previous fit by setting best_parameters
            obs.best_parameters = {
                'planet_radius': 6500000.0,  # Different from original 6371000.0
                'altitude': 15000.0,         # Different from original 10000.0
            }

            # Test 3: Test warm_start=True behavior (manual simulation of fit_limb logic)
            # This simulates what happens in fit_limb when warm_start=True
            if hasattr(obs, 'best_parameters') and obs.best_parameters is not None:
                for param in obs.free_parameters:
                    if param in obs.best_parameters:
                        obs.init_parameter_values[param] = obs.best_parameters[param]

            # Verify warm start worked
            for param in obs.free_parameters:
                expected = obs.best_parameters[param]
                actual = obs.init_parameter_values[param]
                assert actual == expected, f"Warm start failed for {param}: {actual} != {expected}"

            # Test 4: Test warm_start=False behavior (restoration)
            # This simulates what happens in fit_limb when warm_start=False
            if hasattr(obs, '_original_init_parameter_values') and obs._original_init_parameter_values is not None:
                obs.init_parameter_values = obs._original_init_parameter_values.copy()

            # Verify restoration worked
            for param in obs.free_parameters:
                expected = original_values[param]
                actual = obs.init_parameter_values[param]
                assert actual == expected, f"Restoration failed for {param}: {actual} != {expected}"

            os.unlink(tmp_file.name)

    def test_warm_start_multiple_cycles(self, sample_horizon_image, config_file):
        """Test warm start behavior across multiple fit cycles"""
        image_data = sample_horizon_image()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            plt.imsave(tmp_file.name, image_data, cmap="gray")
            tmp_file.flush()

            obs = LimbObservation(
                image_filepath=tmp_file.name,
                fit_config=config_file,
                limb_detection="manual",
            )

            original_values = obs.init_parameter_values.copy()

            # First fit cycle
            obs.best_parameters = {
                'planet_radius': 6500000.0,
                'altitude': 15000.0,
            }

            # Apply warm start
            for param in obs.free_parameters:
                if param in obs.best_parameters:
                    obs.init_parameter_values[param] = obs.best_parameters[param]

            # Second fit cycle with different results
            obs.best_parameters = {
                'planet_radius': 6200000.0,
                'altitude': 8000.0,
            }

            # Apply second warm start
            for param in obs.free_parameters:
                if param in obs.best_parameters:
                    obs.init_parameter_values[param] = obs.best_parameters[param]

            # Verify we can still restore originals after multiple cycles
            obs.init_parameter_values = obs._original_init_parameter_values.copy()

            for param in obs.free_parameters:
                expected = original_values[param]
                actual = obs.init_parameter_values[param]
                assert actual == expected, f"Multiple cycle restoration failed for {param}: {actual} != {expected}"

            os.unlink(tmp_file.name)

    def test_warm_start_without_previous_fit(self, sample_horizon_image, config_file):
        """Test warm start behavior when no previous fit exists"""
        image_data = sample_horizon_image()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            plt.imsave(tmp_file.name, image_data, cmap="gray")
            tmp_file.flush()

            obs = LimbObservation(
                image_filepath=tmp_file.name,
                fit_config=config_file,
                limb_detection="manual",
            )

            original_values = obs.init_parameter_values.copy()

            # Simulate warm_start=True when no best_parameters exist
            # This should not change anything (fallback to original behavior)
            if not (hasattr(obs, 'best_parameters') and obs.best_parameters is not None):
                # Parameters should remain unchanged
                pass
            else:
                for param in obs.free_parameters:
                    if param in obs.best_parameters:
                        obs.init_parameter_values[param] = obs.best_parameters[param]

            # Verify parameters are unchanged when no previous fit exists
            for param in obs.free_parameters:
                expected = original_values[param]
                actual = obs.init_parameter_values[param]
                assert actual == expected, f"Parameters changed without previous fit for {param}: {actual} != {expected}"

            os.unlink(tmp_file.name)

    def test_original_values_immutable(self, sample_horizon_image, config_file):
        """Test that _original_init_parameter_values remains immutable"""
        image_data = sample_horizon_image()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            plt.imsave(tmp_file.name, image_data, cmap="gray")
            tmp_file.flush()

            obs = LimbObservation(
                image_filepath=tmp_file.name,
                fit_config=config_file,
                limb_detection="manual",
            )

            original_stored = obs._original_init_parameter_values.copy()

            # Modify init_parameter_values
            obs.init_parameter_values['planet_radius'] = 9999999.0

            # Verify _original_init_parameter_values is unchanged
            for param in obs.free_parameters:
                expected = original_stored[param]
                actual = obs._original_init_parameter_values[param]
                assert actual == expected, f"Original values were modified for {param}: {actual} != {expected}"

            os.unlink(tmp_file.name)


class TestUtilityFunctions:
    """Test utility functions for observation analysis"""

    def test_unpack_diff_evol_posteriors(self):
        """Test unpacking differential evolution posteriors"""
        # Create mock observation
        obs = Mock()
        obs.free_parameters = ["param1", "param2"]
        obs.init_parameter_values = {"param1": 1.0, "param2": 2.0, "fixed_param": 3.0}

        # Mock fit results
        obs.fit_results = {
            "population": [
                np.array([1.1, 2.1]),
                np.array([1.2, 2.2]),
                np.array([1.3, 2.3]),
            ],
            "population_energies": [0.1, 0.2, 0.3],
        }

        with patch("planet_ruler.fit.unpack_parameters") as mock_unpack:
            # Mock unpack_parameters to return parameter dicts
            mock_unpack.side_effect = [
                {"param1": 1.1, "param2": 2.1},
                {"param1": 1.2, "param2": 2.2},
                {"param1": 1.3, "param2": 2.3},
            ]

            result = unpack_diff_evol_posteriors(obs)

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 3  # Three population members
            assert "param1" in result.columns
            assert "param2" in result.columns
            assert "fixed_param" in result.columns
            assert "mse" in result.columns

            # Check values
            assert np.allclose(result["mse"].values, [0.1, 0.2, 0.3])
            assert np.allclose(result["param1"].values, [1.1, 1.2, 1.3])
            assert np.allclose(result["param2"].values, [2.1, 2.2, 2.3])

    @patch("matplotlib.pyplot.show")
    @patch("seaborn.kdeplot")
    @patch("matplotlib.pyplot.axvline")
    def test_plot_diff_evol_posteriors(self, mock_axvline, mock_kdeplot, mock_show):
        """Test plotting differential evolution posteriors"""
        # Create mock observation
        obs = Mock()
        obs.free_parameters = ["param1"]
        obs.parameter_limits = {"param1": [0.5, 1.5]}
        obs.init_parameter_values = {"param1": 1.0}

        # Mock fit_results with proper structure for unpack_diff_evol_posteriors
        obs.fit_results = {
            "population_energies": [0.1, 0.2, 0.3],
            "population": [[1.1], [1.2], [1.3]],  # Single parameter
        }

        plot_diff_evol_posteriors(obs, show_points=False, log=True)

        mock_kdeplot.assert_called_once()
        assert mock_axvline.call_count >= 2  # At least bounds lines
        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.imshow")
    @patch("matplotlib.pyplot.scatter")
    @patch("planet_ruler.observation.limb_arc")
    def test_plot_full_limb(self, mock_limb_arc, mock_scatter, mock_imshow, mock_show):
        """Test plotting full limb arc"""
        # Create mock observation
        obs = Mock()
        obs.image = np.random.random((100, 200))
        obs.best_parameters = {
            "r": 6371000.0,
            "h": 400000.0,
            "f": 0.024,  # 24mm focal length
            "w": 0.0236,  # APS-C sensor width (~24mm)
            "param1": 1.0,
            "param2": 2.0,
        }

        # Mock limb_arc returns
        full_limb_pixels = np.column_stack(
            [np.arange(200), np.random.random(200) * 100]
        )  # x coordinates  # y coordinates
        visible_limb = np.random.random(200)

        mock_limb_arc.side_effect = [full_limb_pixels, visible_limb]

        plot_full_limb(obs, x_min=0, x_max=200, y_min=0, y_max=100)

        mock_imshow.assert_called_once_with(obs.image)
        assert mock_scatter.call_count == 2  # Full arc and visible portion
        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.imshow")
    @patch("matplotlib.pyplot.title")
    def test_plot_segmentation_masks(self, mock_title, mock_imshow, mock_show):
        """Test plotting segmentation masks"""
        # Create mock observation
        obs = Mock()
        mock_masks = [
            {"segmentation": np.random.random((100, 100))},
            {"segmentation": np.random.random((100, 100))},
        ]
        obs._segmenter._masks = mock_masks

        plot_segmentation_masks(obs)

        assert mock_imshow.call_count == 2  # Two masks
        assert mock_title.call_count == 2  # Two titles
        assert mock_show.call_count == 2  # Show each mask

    def test_package_results(self):
        """Test packaging fit results"""
        # Create mock observation
        obs = Mock()
        obs.free_parameters = ["param1", "param2"]
        obs.init_parameter_values = {"param1": 1.0, "param2": 2.0}

        # Mock fit results
        obs.fit_results.x = np.array([1.5, 2.5])

        with patch("planet_ruler.fit.unpack_parameters") as mock_unpack:
            mock_unpack.return_value = {"param1": 1.5, "param2": 2.5}

            result = package_results(obs)

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 2  # Two parameters
            assert "fit value" in result.columns
            assert "initial value" in result.columns
            assert "parameter" in result.index.names

            # Check values by extracting scalar values
            assert result.loc["param1", "fit value"] == 1.5
            assert result.loc["param1", "initial value"] == 1.0
            assert result.loc["param2", "fit value"] == 2.5
            assert result.loc["param2", "initial value"] == 2.0


# Integration tests using real fixtures
class TestObservationIntegration:
    """Integration tests using realistic data"""

    @pytest.mark.integration
    def test_full_observation_workflow(self, sample_horizon_image):
        """Test a complete observation workflow with mocked components"""
        # Create inline sample fit config
        sample_fit_config = {
            "free_parameters": ["planet_radius", "altitude"],
            "init_parameter_values": {
                "planet_radius": 6371000.0,
                "altitude": 10000.0,
                "focal_length": 0.024,
                "detector_width": 0.036,
                "detector_height": 0.024,
            },
            "parameter_limits": {
                "planet_radius": [6000000.0, 7000000.0],
                "altitude": [100.0, 100000.0],
                "focal_length": [0.01, 0.1],
                "detector_width": [0.01, 0.1],
                "detector_height": [0.01, 0.1],
            },
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yml", delete=False
        ) as config_file:
            yaml.dump(sample_fit_config, config_file)
            config_file.flush()

            image_data = sample_horizon_image()
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as image_file:
                plt.imsave(image_file.name, image_data, cmap="gray")
                image_file.flush()

                # Create observation
                obs = LimbObservation(
                    image_filepath=image_file.name,
                    fit_config=config_file.name,
                    limb_detection="gradient-break",
                )

                # Mock limb detection
                with patch(
                    "planet_ruler.observation.gradient_break"
                ) as mock_gradient_break:
                    expected_limb = np.random.random(image_data.shape[1])
                    mock_gradient_break.return_value = expected_limb

                    obs.detect_limb()

                    assert "limb" in obs.features
                    assert obs._raw_limb is not None

                # Mock smoothing
                with patch("planet_ruler.observation.smooth_limb") as mock_smooth:
                    with patch("planet_ruler.observation.fill_nans") as mock_fill:
                        smoothed = expected_limb + 0.1
                        filled = smoothed + 0.1
                        mock_smooth.return_value = smoothed
                        mock_fill.return_value = filled

                        obs.smooth_limb()

                        assert np.array_equal(obs.features["limb"], filled)

                os.unlink(image_file.name)

            os.unlink(config_file.name)

    @pytest.mark.parametrize("limb_method", ["gradient-break", "segmentation"])
    def test_different_limb_detection_methods(self, sample_horizon_image, limb_method):
        """Test different limb detection methods"""
        # Create a sample fit config inline
        sample_fit_config = {
            "free_parameters": ["planet_radius", "altitude"],
            "init_parameter_values": {
                "planet_radius": 6371000.0,
                "altitude": 10000.0,
                "focal_length": 0.024,
                "detector_width": 0.036,
                "detector_height": 0.024,
            },
            "parameter_limits": {
                "planet_radius": [6000000.0, 7000000.0],
                "altitude": [100.0, 100000.0],
                "focal_length": [0.01, 0.1],
                "detector_width": [0.01, 0.1],
                "detector_height": [0.01, 0.1],
            },
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yml", delete=False
        ) as config_file:
            yaml.dump(sample_fit_config, config_file)
            config_file.flush()

            image_data = sample_horizon_image()
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as image_file:
                plt.imsave(image_file.name, image_data, cmap="gray")
                image_file.flush()

                obs = LimbObservation(
                    image_filepath=image_file.name,
                    fit_config=config_file.name,
                    limb_detection=limb_method,
                )

                assert obs.limb_detection == limb_method

                # Test that the appropriate method attributes are initialized
                if limb_method == "segmentation":
                    assert obs._segmenter is None

                os.unlink(image_file.name)

            os.unlink(config_file.name)
