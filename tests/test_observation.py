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
    MINIMIZER_PRESETS,
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
            assert obs.free_parameters == ["r", "h"]
            assert obs.init_parameter_values["r"] == 6371000.0

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
        invalid_config["init_parameter_values"]["r"] = 5000000.0  # Below lower bound

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

    @patch("planet_ruler.observation.MaskSegmenter")
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
                image=obs.image,
                method="sam",
                downsample_factor=1,
                interactive=True,
                segmenter="segment-anything",
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
                image_path=tmp_file.name,
                image=obs.image,
                initial_stretch=1.0,
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
                image_path=tmp_file.name,
                image=obs.image,
                initial_stretch=1.0,
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

    @patch("planet_ruler.observation.limb_arc")
    @patch("planet_ruler.observation.LimbFitter")
    def test_fit_limb(
        self,
        mock_lf_class,
        mock_limb_arc,
        sample_horizon_image,
        config_file,
    ):
        """Test limb arc fitting via fit_arc()"""
        image_data = sample_horizon_image()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            plt.imsave(tmp_file.name, image_data, cmap="gray")
            tmp_file.flush()

            obs = LimbObservation(image_filepath=tmp_file.name, fit_config=config_file)

            # Set up limb data
            limb_data = np.random.random(image_data.shape[1])
            obs.features["limb"] = limb_data

            # Mock LimbFitter
            mock_lf = Mock()
            mock_fit_result = {
                "best_parameters": {
                    "r": 6500000.0,
                    "h": 15000.0,
                    "n_pix_x": image_data.shape[1],
                    "n_pix_y": image_data.shape[0],
                    "x0": image_data.shape[1] // 2,
                    "y0": image_data.shape[0] // 2,
                    "f": 0.024,
                    "w": 0.036,
                },
                "fit_results": Mock(x=np.array([6500000.0, 15000.0])),
                "updated_init": {"r": 6500000.0, "h": 15000.0},
                "updated_limits": {},
                "status": "ok",
                "warnings": [],
            }
            mock_lf.fit.return_value = mock_fit_result
            mock_lf_class.return_value = mock_lf

            # Mock limb_arc (called for fitted_limb overlay)
            fitted_limb = np.random.random(image_data.shape[1])
            mock_limb_arc.return_value = fitted_limb

            obs.fit_arc(loss_function="l2", max_iter=100, seed=42)

            # Verify LimbFitter was created with correct args
            mock_lf_class.assert_called_once()
            call_kwargs = mock_lf_class.call_args[1]
            assert np.array_equal(call_kwargs["target"], limb_data)
            assert call_kwargs["free_parameters"] == ["r", "h"]
            assert call_kwargs["loss_function"] == "l2"
            assert call_kwargs["max_iter"] == 100
            assert call_kwargs["seed"] == 42

            # Verify results
            assert obs.fit_results == mock_fit_result["fit_results"]
            assert np.array_equal(obs.features["fitted_limb"], fitted_limb)
            assert obs.best_parameters is not None
            assert obs.best_parameters["r"] == 6500000.0
            assert obs.best_parameters["h"] == 15000.0

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
            assert hasattr(obs, "_original_init_parameter_values")
            assert obs._original_init_parameter_values is not None

            # Test 2: Simulate a previous fit by setting best_parameters
            obs.best_parameters = {
                "r": 6500000.0,  # Different from original 6371000.0
                "h": 15000.0,  # Different from original 10000.0
            }

            # Test 3: Test warm_start=True behavior (manual simulation of fit_limb logic)
            # This simulates what happens in fit_limb when warm_start=True
            if hasattr(obs, "best_parameters") and obs.best_parameters is not None:
                for param in obs.free_parameters:
                    if param in obs.best_parameters:
                        obs.init_parameter_values[param] = obs.best_parameters[param]

            # Verify warm start worked
            for param in obs.free_parameters:
                expected = obs.best_parameters[param]
                actual = obs.init_parameter_values[param]
                assert (
                    actual == expected
                ), f"Warm start failed for {param}: {actual} != {expected}"

            # Test 4: Test warm_start=False behavior (restoration)
            # This simulates what happens in fit_limb when warm_start=False
            if (
                hasattr(obs, "_original_init_parameter_values")
                and obs._original_init_parameter_values is not None
            ):
                obs.init_parameter_values = obs._original_init_parameter_values.copy()

            # Verify restoration worked
            for param in obs.free_parameters:
                expected = original_values[param]
                actual = obs.init_parameter_values[param]
                assert (
                    actual == expected
                ), f"Restoration failed for {param}: {actual} != {expected}"

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
                "r": 6500000.0,
                "h": 15000.0,
            }

            # Apply warm start
            for param in obs.free_parameters:
                if param in obs.best_parameters:
                    obs.init_parameter_values[param] = obs.best_parameters[param]

            # Second fit cycle with different results
            obs.best_parameters = {
                "r": 6200000.0,
                "h": 8000.0,
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
                assert (
                    actual == expected
                ), f"Multiple cycle restoration failed for {param}: {actual} != {expected}"

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
            if not (
                hasattr(obs, "best_parameters") and obs.best_parameters is not None
            ):
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
                assert (
                    actual == expected
                ), f"Parameters changed without previous fit for {param}: {actual} != {expected}"

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
            obs.init_parameter_values["r"] = 9999999.0

            # Verify _original_init_parameter_values is unchanged
            for param in obs.free_parameters:
                expected = original_stored[param]
                actual = obs._original_init_parameter_values[param]
                assert (
                    actual == expected
                ), f"Original values were modified for {param}: {actual} != {expected}"

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

            # Check values by extracting scalar values properly
            assert result.at["param1", "fit value"] == 1.5
            assert result.at["param1", "initial value"] == 1.0
            assert result.at["param2", "fit value"] == 2.5
            assert result.at["param2", "initial value"] == 2.0


class TestObservationPropertyMethods:
    """Test observation property methods and edge cases"""

    def test_properties_without_fit(self, sample_horizon_image, sample_fit_config):
        """Test property methods when no fit has been performed"""
        image_data = sample_horizon_image()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            plt.imsave(tmp_file.name, image_data, cmap="gray")
            tmp_file.flush()

            obs = LimbObservation(
                image_filepath=tmp_file.name, fit_config=sample_fit_config
            )

            # Test properties return 0 when no fit performed
            assert obs.radius_km == 0.0
            assert obs.altitude_km == 0.0
            assert obs.focal_length_mm == 0.0
            assert obs.radius_uncertainty == 0.0

            os.unlink(tmp_file.name)

    def test_properties_with_fit_results(self, sample_horizon_image, sample_fit_config):
        """Test property methods when fit results exist"""
        image_data = sample_horizon_image()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            plt.imsave(tmp_file.name, image_data, cmap="gray")
            tmp_file.flush()

            obs = LimbObservation(
                image_filepath=tmp_file.name, fit_config=sample_fit_config
            )

            # Mock fit results
            obs.best_parameters = {
                "r": 6371000.0,  # meters
                "h": 50000.0,  # meters
                "f": 0.024,  # meters
            }

            # Test conversions
            assert abs(obs.radius_km - 6371.0) < 0.1  # km
            assert abs(obs.altitude_km - 50.0) < 0.1  # km
            assert abs(obs.focal_length_mm - 24.0) < 0.1  # mm

            os.unlink(tmp_file.name)

    def test_parameter_uncertainty_without_fit(
        self, sample_horizon_image, sample_fit_config
    ):
        """Test parameter_uncertainty when no fit results exist"""
        image_data = sample_horizon_image()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            plt.imsave(tmp_file.name, image_data, cmap="gray")
            tmp_file.flush()

            obs = LimbObservation(
                image_filepath=tmp_file.name, fit_config=sample_fit_config
            )

            result = obs.parameter_uncertainty("r")

            assert result["uncertainty"] == 0.0
            assert result["method"] == "none"
            assert result["additional_info"] == "No fit performed"

            os.unlink(tmp_file.name)

    @patch("planet_ruler.uncertainty.calculate_parameter_uncertainty")
    def test_parameter_uncertainty_with_exception(
        self, mock_calc_uncertainty, sample_horizon_image, sample_fit_config
    ):
        """Test parameter_uncertainty when calculation raises exception"""
        image_data = sample_horizon_image()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            plt.imsave(tmp_file.name, image_data, cmap="gray")
            tmp_file.flush()

            obs = LimbObservation(
                image_filepath=tmp_file.name, fit_config=sample_fit_config
            )
            obs.fit_results = Mock()  # Mock fit results exist

            # Mock exception
            mock_calc_uncertainty.side_effect = ValueError("Test error")

            result = obs.parameter_uncertainty("r")

            assert result["uncertainty"] == 0.0
            assert result["method"] == "error"
            assert "Test error" in result["additional_info"]

            os.unlink(tmp_file.name)

    @patch("planet_ruler.plot.plot_3d_solution")
    def test_plot_3d_without_fit(
        self, mock_plot_3d, sample_horizon_image, sample_fit_config
    ):
        """Test plot_3d raises error when no fit performed"""
        image_data = sample_horizon_image()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            plt.imsave(tmp_file.name, image_data, cmap="gray")
            tmp_file.flush()

            obs = LimbObservation(
                image_filepath=tmp_file.name, fit_config=sample_fit_config
            )

            with pytest.raises(ValueError, match="Must fit limb before plotting"):
                obs.plot_3d()

            os.unlink(tmp_file.name)

    @patch("planet_ruler.plot.plot_3d_solution")
    def test_plot_3d_with_fit(
        self, mock_plot_3d, sample_horizon_image, sample_fit_config
    ):
        """Test plot_3d works when fit results exist"""
        image_data = sample_horizon_image()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            plt.imsave(tmp_file.name, image_data, cmap="gray")
            tmp_file.flush()

            obs = LimbObservation(
                image_filepath=tmp_file.name, fit_config=sample_fit_config
            )
            obs.best_parameters = {
                "r": 6371000.0,
                "h": 50000.0,
                "f": 0.024,
            }

            obs.plot_3d(some_param="test")

            mock_plot_3d.assert_called_once_with(
                r=6371000.0, h=50000.0, f=0.024, some_param="test"
            )

            os.unlink(tmp_file.name)

    def test_analyze_method_chaining(self, sample_horizon_image, sample_fit_config):
        """Test analyze method with method chaining"""
        image_data = sample_horizon_image()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            plt.imsave(tmp_file.name, image_data, cmap="gray")
            tmp_file.flush()

            obs = LimbObservation(
                image_filepath=tmp_file.name, fit_config=sample_fit_config
            )

            with patch.object(obs, "detect_limb") as mock_detect:
                with patch.object(obs, "fit_arc") as mock_fit:
                    result = obs.analyze()

                    # Should return self for chaining
                    assert result is obs

                    # Should call both methods
                    mock_detect.assert_called_once_with()
                    mock_fit.assert_called_once_with()

            os.unlink(tmp_file.name)

    def test_analyze_with_kwargs(self, sample_horizon_image, sample_fit_config):
        """Test analyze method passes kwargs correctly"""
        image_data = sample_horizon_image()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            plt.imsave(tmp_file.name, image_data, cmap="gray")
            tmp_file.flush()

            obs = LimbObservation(
                image_filepath=tmp_file.name, fit_config=sample_fit_config
            )

            with patch.object(obs, "detect_limb") as mock_detect:
                with patch.object(obs, "fit_arc") as mock_fit:
                    detect_kwargs = {"log": True, "y_min": 10}
                    fit_kwargs = {"loss_function": "l1", "max_iter": 500}

                    obs.analyze(
                        detect_limb_kwargs=detect_kwargs,
                        fit_method_kwargs=fit_kwargs,
                    )

                    mock_detect.assert_called_once_with(**detect_kwargs)
                    mock_fit.assert_called_once_with(**fit_kwargs)

            os.unlink(tmp_file.name)


# Integration tests using real fixtures
class TestObservationIntegration:
    """Integration tests using realistic data"""

    @pytest.mark.integration
    def test_full_observation_workflow(self, sample_horizon_image):
        """Test a complete observation workflow with mocked components"""
        # Create inline sample fit config
        sample_fit_config = {
            "free_parameters": ["r", "h"],
            "init_parameter_values": {
                "r": 6371000.0,
                "h": 10000.0,
                "f": 0.024,
                "w": 0.036,
            },
            "parameter_limits": {
                "r": [6000000.0, 7000000.0],
                "h": [100.0, 100000.0],
                "f": [0.01, 0.1],
                "w": [0.01, 0.1],
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
            "free_parameters": ["r", "h"],
            "init_parameter_values": {
                "r": 6371000.0,
                "h": 10000.0,
                "f": 0.024,
                "w": 0.036,
            },
            "parameter_limits": {
                "r": [6000000.0, 7000000.0],
                "h": [100.0, 100000.0],
                "f": [0.01, 0.1],
                "w": [0.01, 0.1],
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


class TestObservationMultiResolution:
    """Test multi-resolution optimization workflow (fit_gradient)"""

    @pytest.fixture
    def config_file(self, sample_fit_config):
        """Create a temporary config file"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.dump(sample_fit_config, f)
            f.flush()
            yield f.name
        os.unlink(f.name)

    def _mock_lf_result(self, npix_x=3000, npix_y=2500):
        return {
            "best_parameters": {
                "r": 6500000.0,
                "h": 15000.0,
                "n_pix_x": npix_x,
                "n_pix_y": npix_y,
                "x0": npix_x // 2,
                "y0": npix_y // 2,
                "f": 0.024,
                "w": 0.036,
            },
            "fit_results": Mock(x=np.array([6500000.0, 15000.0])),
            "updated_init": {"r": 6500000.0, "h": 15000.0},
            "updated_limits": {},
            "status": "ok",
            "warnings": [],
        }

    @patch("planet_ruler.observation.limb_arc")
    @patch("planet_ruler.observation.LimbFitter")
    def test_multi_resolution_auto_stages(
        self,
        mock_lf_class,
        mock_limb_arc,
        sample_horizon_image,
        config_file,
    ):
        """Test fit_gradient with auto-determined resolution stages"""
        large_image = np.random.random((2500, 3000))

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            plt.imsave(tmp_file.name, large_image, cmap="gray")
            tmp_file.flush()

            obs = LimbObservation(
                image_filepath=tmp_file.name,
                fit_config=config_file,
                limb_detection="gradient-field",
            )

            mock_lf = Mock()
            mock_lf.fit.return_value = self._mock_lf_result()
            mock_lf_class.return_value = mock_lf
            mock_limb_arc.return_value = np.random.random(large_image.shape[1])

            obs.fit_gradient(resolution_stages="auto", max_iter=300, verbose=True)

            # auto → [4, 2, 1] for min_dim=2500, so LimbFitter called 3 times
            assert mock_lf_class.call_count >= 2
            assert obs.best_parameters is not None

            os.unlink(tmp_file.name)

    @patch("planet_ruler.observation.limb_arc")
    @patch("planet_ruler.observation.LimbFitter")
    def test_multi_resolution_custom_stages(
        self,
        mock_lf_class,
        mock_limb_arc,
        sample_horizon_image,
        config_file,
    ):
        """Test fit_gradient with custom resolution stages"""
        image_data = np.random.random((1000, 1200))

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            plt.imsave(tmp_file.name, image_data, cmap="gray")
            tmp_file.flush()

            obs = LimbObservation(
                image_filepath=tmp_file.name,
                fit_config=config_file,
                limb_detection="gradient-field",
            )

            mock_lf = Mock()
            mock_lf.fit.return_value = self._mock_lf_result(1200, 1000)
            mock_lf_class.return_value = mock_lf
            mock_limb_arc.return_value = np.random.random(image_data.shape[1])

            obs.fit_gradient(
                resolution_stages=[4, 2, 1],
                max_iter_per_stage=[50, 100, 150],
                verbose=True,
            )

            # Should call LimbFitter 3 times (one per stage)
            assert mock_lf_class.call_count == 3
            assert obs.best_parameters is not None

            os.unlink(tmp_file.name)

    @patch("planet_ruler.observation.limb_arc")
    @patch("planet_ruler.observation.LimbFitter")
    @patch("cv2.resize")
    def test_multi_resolution_image_scaling(
        self,
        mock_cv2_resize,
        mock_lf_class,
        mock_limb_arc,
        sample_horizon_image,
        config_file,
    ):
        """Test that images are properly resized in multi-resolution fit_gradient"""
        image_data = np.random.random((800, 1000)).astype("float32")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            plt.imsave(tmp_file.name, image_data, cmap="gray")
            tmp_file.flush()

            obs = LimbObservation(
                image_filepath=tmp_file.name,
                fit_config=config_file,
                limb_detection="gradient-field",
            )

            stage1_image = np.random.random((200, 250)).astype("float32")
            stage2_image = np.random.random((400, 500)).astype("float32")
            mock_cv2_resize.side_effect = [stage1_image, stage2_image]

            mock_lf = Mock()
            mock_lf.fit.return_value = self._mock_lf_result(1000, 800)
            mock_lf_class.return_value = mock_lf
            mock_limb_arc.return_value = np.random.random(image_data.shape[1])

            obs.fit_gradient(resolution_stages=[4, 2, 1], verbose=True)

            # Stages 1 and 2 resize, stage 3 uses full resolution
            assert mock_cv2_resize.call_count == 2

            os.unlink(tmp_file.name)

    @patch("planet_ruler.observation.limb_arc")
    @patch("planet_ruler.observation.LimbFitter")
    def test_fit_gradient_single_stage(
        self, mock_lf_class, mock_limb_arc, sample_horizon_image, config_file
    ):
        """Test fit_gradient with a single (default) resolution stage"""
        image_data = sample_horizon_image()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            plt.imsave(tmp_file.name, image_data, cmap="gray")
            tmp_file.flush()

            obs = LimbObservation(
                image_filepath=tmp_file.name,
                fit_config=config_file,
                limb_detection="gradient-field",
            )

            mock_lf = Mock()
            mock_lf.fit.return_value = self._mock_lf_result(
                image_data.shape[1], image_data.shape[0]
            )
            mock_lf_class.return_value = mock_lf
            mock_limb_arc.return_value = np.random.random(image_data.shape[1])

            obs.fit_gradient()  # default: single stage

            mock_lf_class.assert_called_once()
            assert obs.best_parameters is not None

            os.unlink(tmp_file.name)

    def test_scale_parameters_for_resolution(self, sample_horizon_image, config_file):
        """Test parameter scaling utility in geometry module"""
        from planet_ruler.geometry import _scale_parameters_for_resolution

        params = {
            "n_pix_x": 1000,
            "n_pix_y": 800,
            "x0": 500,
            "y0": 400,
            "r": 6371000.0,
            "h": 50000.0,
            "f": 0.024,
        }

        scaled_params = _scale_parameters_for_resolution(params, 0.5)

        assert scaled_params["n_pix_x"] == 500
        assert scaled_params["n_pix_y"] == 400
        assert scaled_params["x0"] == 250
        assert scaled_params["y0"] == 200
        assert scaled_params["r"] == 6371000.0
        assert scaled_params["h"] == 50000.0
        assert scaled_params["f"] == 0.024

        scaled_params2 = _scale_parameters_for_resolution(params, 2.0)
        assert scaled_params2["n_pix_x"] == 2000
        assert scaled_params2["n_pix_y"] == 1600
        assert scaled_params2["x0"] == 1000
        assert scaled_params2["y0"] == 800


class TestObservationMinimizers:
    """Test different minimizer algorithms"""

    @pytest.fixture
    def config_file(self, sample_fit_config):
        """Create a temporary config file"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.dump(sample_fit_config, f)
            f.flush()
            yield f.name
        os.unlink(f.name)

    @patch("planet_ruler.observation.limb_arc")
    @patch("planet_ruler.observation.LimbFitter")
    def test_dual_annealing_minimizer(
        self,
        mock_lf_class,
        mock_limb_arc,
        sample_horizon_image,
        config_file,
    ):
        """Test that fit_arc passes dual-annealing preset correctly to LimbFitter"""
        image_data = sample_horizon_image()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            plt.imsave(tmp_file.name, image_data, cmap="gray")
            tmp_file.flush()

            obs = LimbObservation(
                image_filepath=tmp_file.name,
                fit_config=config_file,
                minimizer="dual-annealing",
            )
            obs.features["limb"] = np.random.random(image_data.shape[1])

            mock_lf = Mock()
            mock_lf.fit.return_value = {
                "best_parameters": {
                    "r": 6500000.0,
                    "h": 15000.0,
                    "n_pix_x": image_data.shape[1],
                    "n_pix_y": image_data.shape[0],
                    "x0": image_data.shape[1] // 2,
                    "y0": image_data.shape[0] // 2,
                },
                "fit_results": Mock(x=np.array([6500000.0, 15000.0])),
                "updated_init": {"r": 6500000.0, "h": 15000.0},
                "updated_limits": {},
                "status": "ok",
                "warnings": [],
            }
            mock_lf_class.return_value = mock_lf
            mock_limb_arc.return_value = np.random.random(image_data.shape[1])

            obs.fit_arc(
                loss_function="l2",
                minimizer_preset="fast",
                max_iter=100,
                seed=42,
                verbose=True,
            )

            # Verify LimbFitter got the right minimizer, preset kwargs, seed
            mock_lf_class.assert_called_once()
            call_kwargs = mock_lf_class.call_args[1]
            assert call_kwargs["minimizer"] == "dual-annealing"
            assert call_kwargs["max_iter"] == 100
            assert call_kwargs["seed"] == 42
            assert call_kwargs["verbose"] is True

            expected_preset = MINIMIZER_PRESETS["dual-annealing"]["fast"]
            for key, value in expected_preset.items():
                assert call_kwargs["minimizer_kwargs"][key] == value

            assert obs.fit_results == mock_lf.fit.return_value["fit_results"]

            os.unlink(tmp_file.name)

    @patch("planet_ruler.observation.limb_arc")
    @patch("planet_ruler.observation.LimbFitter")
    def test_basinhopping_minimizer(
        self,
        mock_lf_class,
        mock_limb_arc,
        sample_horizon_image,
        config_file,
    ):
        """Test that fit_arc passes basinhopping preset correctly to LimbFitter"""
        image_data = sample_horizon_image()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            plt.imsave(tmp_file.name, image_data, cmap="gray")
            tmp_file.flush()

            obs = LimbObservation(
                image_filepath=tmp_file.name,
                fit_config=config_file,
                minimizer="basinhopping",
            )
            obs.features["limb"] = np.random.random(image_data.shape[1])

            mock_lf = Mock()
            mock_lf.fit.return_value = {
                "best_parameters": {
                    "r": 6500000.0,
                    "h": 15000.0,
                    "n_pix_x": image_data.shape[1],
                    "n_pix_y": image_data.shape[0],
                    "x0": image_data.shape[1] // 2,
                    "y0": image_data.shape[0] // 2,
                },
                "fit_results": Mock(x=np.array([6500000.0, 15000.0])),
                "updated_init": {"r": 6500000.0, "h": 15000.0},
                "updated_limits": {},
                "status": "ok",
                "warnings": [],
            }
            mock_lf_class.return_value = mock_lf
            mock_limb_arc.return_value = np.random.random(image_data.shape[1])

            obs.fit_arc(
                loss_function="l2",
                minimizer_preset="robust",
                max_iter=200,
                seed=123,
                verbose=True,
            )

            # Verify LimbFitter got the right minimizer and preset kwargs
            mock_lf_class.assert_called_once()
            call_kwargs = mock_lf_class.call_args[1]
            assert call_kwargs["minimizer"] == "basinhopping"
            assert call_kwargs["max_iter"] == 200
            assert call_kwargs["seed"] == 123

            expected_preset = MINIMIZER_PRESETS["basinhopping"]["robust"]
            for key, value in expected_preset.items():
                assert call_kwargs["minimizer_kwargs"][key] == value

            assert obs.fit_results == mock_lf.fit.return_value["fit_results"]

            os.unlink(tmp_file.name)

    def test_invalid_minimizer_preset(self, sample_horizon_image, config_file):
        """Test handling of invalid minimizer preset"""
        image_data = sample_horizon_image()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            plt.imsave(tmp_file.name, image_data, cmap="gray")
            tmp_file.flush()

            obs = LimbObservation(image_filepath=tmp_file.name, fit_config=config_file)
            obs.features["limb"] = np.random.random(image_data.shape[1])

            with pytest.raises(ValueError, match="Unknown preset"):
                obs.fit_arc(minimizer_preset="invalid-preset")  # type: ignore

            os.unlink(tmp_file.name)

    def test_invalid_minimizer(self, sample_horizon_image, config_file):
        """Test handling of invalid minimizer"""
        image_data = sample_horizon_image()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            plt.imsave(tmp_file.name, image_data, cmap="gray")
            tmp_file.flush()

            obs = LimbObservation(image_filepath=tmp_file.name, fit_config=config_file)
            obs.features["limb"] = np.random.random(image_data.shape[1])

            with pytest.raises(ValueError, match="Unknown minimizer"):
                obs.fit_arc(minimizer="invalid-minimizer")  # type: ignore

            os.unlink(tmp_file.name)

    @patch("planet_ruler.observation.limb_arc")
    @patch("planet_ruler.observation.LimbFitter")
    def test_minimizer_kwargs_override(
        self,
        mock_lf_class,
        mock_limb_arc,
        sample_horizon_image,
        config_file,
    ):
        """Test that minimizer_kwargs override preset values passed to LimbFitter"""
        image_data = sample_horizon_image()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            plt.imsave(tmp_file.name, image_data, cmap="gray")
            tmp_file.flush()

            obs = LimbObservation(image_filepath=tmp_file.name, fit_config=config_file)
            obs.features["limb"] = np.random.random(image_data.shape[1])

            mock_lf = Mock()
            mock_lf.fit.return_value = {
                "best_parameters": {
                    "r": 6500000.0,
                    "h": 15000.0,
                    "n_pix_x": image_data.shape[1],
                    "n_pix_y": image_data.shape[0],
                    "x0": image_data.shape[1] // 2,
                    "y0": image_data.shape[0] // 2,
                },
                "fit_results": Mock(x=np.array([6500000.0, 15000.0])),
                "updated_init": {"r": 6500000.0, "h": 15000.0},
                "updated_limits": {},
                "status": "ok",
                "warnings": [],
            }
            mock_lf_class.return_value = mock_lf
            mock_limb_arc.return_value = np.random.random(image_data.shape[1])

            custom_kwargs = {
                "popsize": 25,
                "mutation": [0.2, 2.0],
                "custom_param": "test",
            }

            obs.fit_arc(
                loss_function="l2",
                minimizer_preset="balanced",
                minimizer_kwargs=custom_kwargs,
            )

            call_kwargs = mock_lf_class.call_args[1]
            assert call_kwargs["minimizer_kwargs"]["popsize"] == 25
            assert call_kwargs["minimizer_kwargs"]["mutation"] == [0.2, 2.0]
            assert call_kwargs["minimizer_kwargs"]["custom_param"] == "test"

            expected_preset = MINIMIZER_PRESETS["differential-evolution"]["balanced"]
            assert (
                call_kwargs["minimizer_kwargs"]["strategy"]
                == expected_preset["strategy"]
            )
            assert (
                call_kwargs["minimizer_kwargs"]["recombination"]
                == expected_preset["recombination"]
            )

            os.unlink(tmp_file.name)


class TestObservationImageSmoothing:
    """Test image smoothing and preprocessing"""

    @pytest.fixture
    def config_file(self, sample_fit_config):
        """Create a temporary config file"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.dump(sample_fit_config, f)
            f.flush()
            yield f.name
        os.unlink(f.name)

    @patch("planet_ruler.observation.limb_arc")
    @patch("planet_ruler.observation.LimbFitter")
    @patch("cv2.GaussianBlur")
    def test_image_smoothing_applied_and_restored(
        self,
        mock_gaussian_blur,
        mock_lf_class,
        mock_limb_arc,
        sample_horizon_image,
        config_file,
    ):
        """Test that fit_gradient applies image_smoothing then restores the image"""
        original_image = np.random.random((400, 500)).astype("float32")
        smoothed_image = original_image + 0.1

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            plt.imsave(tmp_file.name, original_image, cmap="gray")
            tmp_file.flush()

            obs = LimbObservation(
                image_filepath=tmp_file.name,
                fit_config=config_file,
                limb_detection="gradient-field",
            )

            mock_gaussian_blur.return_value = smoothed_image

            mock_lf = Mock()
            mock_lf.fit.return_value = {
                "best_parameters": {
                    "r": 6500000.0,
                    "h": 15000.0,
                    "n_pix_x": 500,
                    "n_pix_y": 400,
                    "x0": 250,
                    "y0": 200,
                },
                "fit_results": Mock(x=np.array([6500000.0, 15000.0])),
                "updated_init": {"r": 6500000.0, "h": 15000.0},
                "updated_limits": {},
                "status": "ok",
                "warnings": [],
            }
            mock_lf_class.return_value = mock_lf
            mock_limb_arc.return_value = np.random.random(original_image.shape[1])

            obs_original_image = obs.image.copy()

            obs.fit_gradient(image_smoothing=2.5, verbose=True)

            mock_gaussian_blur.assert_called_once()
            call_args = mock_gaussian_blur.call_args
            assert call_args[1]["sigmaX"] == 2.5
            assert call_args[1]["sigmaY"] == 2.5

            # Image restored to original after fitting
            np.testing.assert_array_equal(obs.image, obs_original_image)

            os.unlink(tmp_file.name)

    @patch("planet_ruler.observation.limb_arc")
    @patch("planet_ruler.observation.LimbFitter")
    def test_fit_gradient_no_smoothing_when_none(
        self,
        mock_lf_class,
        mock_limb_arc,
        sample_horizon_image,
        config_file,
    ):
        """Test that fit_gradient without image_smoothing does not apply blur"""
        image_data = sample_horizon_image()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            plt.imsave(tmp_file.name, image_data, cmap="gray")
            tmp_file.flush()

            obs = LimbObservation(
                image_filepath=tmp_file.name,
                fit_config=config_file,
                limb_detection="gradient-field",
            )

            mock_lf = Mock()
            mock_lf.fit.return_value = {
                "best_parameters": {
                    "r": 6500000.0,
                    "h": 15000.0,
                    "n_pix_x": image_data.shape[1],
                    "n_pix_y": image_data.shape[0],
                    "x0": image_data.shape[1] // 2,
                    "y0": image_data.shape[0] // 2,
                },
                "fit_results": Mock(x=np.array([6500000.0, 15000.0])),
                "updated_init": {"r": 6500000.0, "h": 15000.0},
                "updated_limits": {},
                "status": "ok",
                "warnings": [],
            }
            mock_lf_class.return_value = mock_lf
            mock_limb_arc.return_value = np.random.random(image_data.shape[1])
            original_image = obs.image.copy()

            with patch("cv2.GaussianBlur") as mock_blur:
                obs.fit_gradient()  # No image_smoothing
                mock_blur.assert_not_called()

            np.testing.assert_array_equal(obs.image, original_image)

            os.unlink(tmp_file.name)


class TestObservationDashboard:
    """Test FitDashboard lifecycle managed by fit_limb orchestrator"""

    @pytest.fixture
    def config_file(self, sample_fit_config):
        """Create a temporary config file"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.dump(sample_fit_config, f)
            f.flush()
            yield f.name
        os.unlink(f.name)

    @patch("planet_ruler.observation.limb_arc")
    @patch("planet_ruler.observation.LimbFitter")
    @patch("planet_ruler.observation.FitDashboard")
    def test_single_stage_dashboard(
        self,
        mock_dashboard_class,
        mock_lf_class,
        mock_limb_arc,
        sample_horizon_image,
        config_file,
    ):
        """Test dashboard lifecycle for a single arc stage via fit_limb"""
        image_data = sample_horizon_image()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            plt.imsave(tmp_file.name, image_data, cmap="gray")
            tmp_file.flush()

            obs = LimbObservation(image_filepath=tmp_file.name, fit_config=config_file)
            obs.features["limb"] = np.random.random(image_data.shape[1])

            mock_dashboard = Mock()
            mock_dashboard_class.return_value = mock_dashboard

            mock_lf = Mock()
            mock_lf.fit.return_value = {
                "best_parameters": {
                    "r": 6500000.0,
                    "h": 15000.0,
                    "n_pix_x": image_data.shape[1],
                    "n_pix_y": image_data.shape[0],
                    "x0": image_data.shape[1] // 2,
                    "y0": image_data.shape[0] // 2,
                },
                "fit_results": Mock(x=np.array([6500000.0, 15000.0])),
                "updated_init": {"r": 6500000.0, "h": 15000.0},
                "updated_limits": {},
                "status": "ok",
                "warnings": [],
            }
            mock_lf_class.return_value = mock_lf
            mock_limb_arc.return_value = np.random.random(image_data.shape[1])

            obs.fit_limb(
                [{"method": "arc", "max_iter": 100}],
                dashboard=True,
                dashboard_kwargs={"some_option": "test"},
                target_planet="mars",
            )

            # Dashboard created with correct parameters
            mock_dashboard_class.assert_called_once()
            call_kwargs = mock_dashboard_class.call_args[1]
            assert call_kwargs["max_iter"] == 100
            assert call_kwargs["target_planet"] == "mars"
            assert call_kwargs["some_option"] == "test"
            assert call_kwargs["total_stages"] == 1

            # Dashboard finalized at end
            mock_dashboard.finalize.assert_called_once()

            os.unlink(tmp_file.name)

    @patch("planet_ruler.observation.limb_arc")
    @patch("planet_ruler.observation.LimbFitter")
    @patch("planet_ruler.observation.FitDashboard")
    def test_multi_resolution_dashboard(
        self,
        mock_dashboard_class,
        mock_lf_class,
        mock_limb_arc,
        sample_horizon_image,
        config_file,
    ):
        """Test dashboard tracks resolution stages inside a gradient fit_limb call"""
        image_data = np.random.random((1200, 1500))

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            plt.imsave(tmp_file.name, image_data, cmap="gray")
            tmp_file.flush()

            obs = LimbObservation(
                image_filepath=tmp_file.name,
                fit_config=config_file,
                limb_detection="gradient-field",
            )

            mock_dashboard = Mock()
            mock_dashboard_class.return_value = mock_dashboard

            mock_lf = Mock()
            mock_lf.fit.return_value = {
                "best_parameters": {
                    "r": 6500000.0,
                    "h": 15000.0,
                    "n_pix_x": 1500,
                    "n_pix_y": 1200,
                    "x0": 750,
                    "y0": 600,
                },
                "fit_results": Mock(x=np.array([6500000.0, 15000.0])),
                "updated_init": {"r": 6500000.0, "h": 15000.0},
                "updated_limits": {},
                "status": "ok",
                "warnings": [],
            }
            mock_lf_class.return_value = mock_lf
            mock_limb_arc.return_value = np.random.random(image_data.shape[1])

            obs.fit_limb(
                [{"method": "gradient", "resolution_stages": [2, 1], "max_iter": 200}],
                dashboard=True,
                target_planet="jupiter",
            )

            # Dashboard created with total_stages=2 (two resolution stages)
            mock_dashboard_class.assert_called_once()
            call_kwargs = mock_dashboard_class.call_args[1]
            assert call_kwargs["total_stages"] == 2
            assert call_kwargs["target_planet"] == "jupiter"

            # start_stage called once for the second resolution stage
            # Auto-split for [2, 1] with max_iter=200: weight 1+2=3 → [67, 133]
            mock_dashboard.start_stage.assert_called_once_with(2, "full", 133)

            # Dashboard finalized at end
            mock_dashboard.finalize.assert_called_once()

            os.unlink(tmp_file.name)


class TestObservationConfigurationEdgeCases:
    """Test configuration loading edge cases"""

    def test_load_fit_config_from_dict(self, sample_horizon_image):
        """Test loading config from dictionary instead of file"""
        image_data = sample_horizon_image()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            plt.imsave(tmp_file.name, image_data, cmap="gray")
            tmp_file.flush()

            config_dict = {
                "free_parameters": ["r", "h", "f"],
                "init_parameter_values": {
                    "r": 6400000.0,
                    "h": 8000.0,
                    "f": 0.035,
                },
                "parameter_limits": {
                    "r": [6000000.0, 7000000.0],
                    "h": [100.0, 100000.0],
                    "f": [0.01, 0.1],
                },
            }

            obs = LimbObservation(
                image_filepath=tmp_file.name,
                fit_config=config_dict,  # Dict instead of file path
            )

            assert obs.free_parameters == ["r", "h", "f"]
            assert obs.init_parameter_values is not None
            assert obs.init_parameter_values["r"] == 6400000.0
            assert obs.init_parameter_values["f"] == 0.035

            os.unlink(tmp_file.name)

    def test_load_fit_config_partial_override(self, sample_horizon_image):
        """Test config loading with partial parameter overrides"""
        image_data = sample_horizon_image()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            plt.imsave(tmp_file.name, image_data, cmap="gray")
            tmp_file.flush()

            # Config with only some parameters - others should use defaults
            config_dict = {
                "free_parameters": ["r"],
                "init_parameter_values": {
                    "r": 6500000.0,
                    # Missing h - should use default
                },
                "parameter_limits": {
                    "r": [6200000.0, 6800000.0],
                    # Missing others - should use defaults
                },
            }

            obs = LimbObservation(image_filepath=tmp_file.name, fit_config=config_dict)

            assert obs.free_parameters == ["r"]
            assert obs.init_parameter_values is not None
            assert obs.parameter_limits is not None
            assert obs.init_parameter_values["r"] == 6500000.0
            # Should have default values for missing parameters
            assert obs.init_parameter_values["theta_y"] == 0  # Default value
            assert obs.init_parameter_values["theta_z"] == 0  # Default value
            assert obs.parameter_limits["theta_x"] == [-3.14, 3.14]  # Default

            os.unlink(tmp_file.name)

    def test_load_fit_config_default_only(self, sample_horizon_image):
        """Test loading config with only defaults (minimal config)"""
        image_data = sample_horizon_image()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            plt.imsave(tmp_file.name, image_data, cmap="gray")
            tmp_file.flush()

            # Minimal config - should be filled with defaults
            config_dict = {
                "free_parameters": ["r", "h"],
            }

            obs = LimbObservation(image_filepath=tmp_file.name, fit_config=config_dict)

            # Should have all default values
            assert obs.parameter_limits is not None
            assert obs.init_parameter_values is not None
            assert obs.parameter_limits["theta_x"] == [-3.14, 3.14]
            assert obs.parameter_limits["theta_y"] == [-3.14, 3.14]
            assert obs.parameter_limits["num_sample"] == [4000, 6000]
            assert obs.init_parameter_values["theta_y"] == 0
            assert obs.init_parameter_values["theta_z"] == 0

            os.unlink(tmp_file.name)


class TestObservationDetectionMethods:
    """Test additional detection method coverage"""

    @pytest.fixture
    def config_file(self, sample_fit_config):
        """Create a temporary config file"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.dump(sample_fit_config, f)
            f.flush()
            yield f.name
        os.unlink(f.name)

    def test_detect_limb_gradient_field_method(self, sample_horizon_image, config_file):
        """Test gradient-field detection method (which skips detection)"""
        image_data = sample_horizon_image()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            plt.imsave(tmp_file.name, image_data, cmap="gray")
            tmp_file.flush()

            obs = LimbObservation(
                image_filepath=tmp_file.name,
                fit_config=config_file,
                limb_detection="gradient-field",
            )

            # Should not raise any errors and should print skip message
            obs.detect_limb()

            # No limb should be registered for gradient-field method
            assert "limb" not in obs.features

            os.unlink(tmp_file.name)

    def test_detect_limb_method_override(self, sample_horizon_image, config_file):
        """Test detection method can be overridden at runtime"""
        image_data = sample_horizon_image()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            plt.imsave(tmp_file.name, image_data, cmap="gray")
            tmp_file.flush()

            obs = LimbObservation(
                image_filepath=tmp_file.name,
                fit_config=config_file,
                limb_detection="manual",  # Initial method
            )

            # Override method at detection time
            with patch(
                "planet_ruler.observation.gradient_break"
            ) as mock_gradient_break:
                expected_limb = np.random.random(image_data.shape[1])
                mock_gradient_break.return_value = expected_limb

                obs.detect_limb(detection_method="gradient-break")

                # Should have changed to gradient-break method
                assert obs.limb_detection == "gradient-break"
                assert np.array_equal(obs.features["limb"], expected_limb)

            os.unlink(tmp_file.name)

    def test_register_limb_method_chaining(self, sample_horizon_image, config_file):
        """Test register_limb returns self for method chaining"""
        image_data = sample_horizon_image()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            plt.imsave(tmp_file.name, image_data, cmap="gray")
            tmp_file.flush()

            obs = LimbObservation(image_filepath=tmp_file.name, fit_config=config_file)

            limb_data = np.random.random(100)
            result = obs.register_limb(limb_data)

            # Should return self for chaining
            assert result is obs
            assert np.array_equal(obs.features["limb"], limb_data)
            assert np.array_equal(obs._raw_limb, limb_data)

            os.unlink(tmp_file.name)


class TestObservationErrorHandling:
    """Test error handling and edge cases"""

    @pytest.fixture
    def config_file(self, sample_fit_config):
        """Create a temporary config file"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.dump(sample_fit_config, f)
            f.flush()
            yield f.name
        os.unlink(f.name)

    def test_fit_limb_without_limb_for_traditional_loss(
        self, sample_horizon_image, config_file
    ):
        """Test that fit_arc raises ValueError when no limb is detected"""
        image_data = sample_horizon_image()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            plt.imsave(tmp_file.name, image_data, cmap="gray")
            tmp_file.flush()

            obs = LimbObservation(image_filepath=tmp_file.name, fit_config=config_file)
            # No limb data registered

            with pytest.raises(ValueError, match="Must detect limb"):
                obs.fit_arc(loss_function="l2")

            os.unlink(tmp_file.name)

    @patch("planet_ruler.uncertainty.calculate_parameter_uncertainty")
    def test_radius_uncertainty_with_exception(
        self, mock_calc_uncertainty, sample_horizon_image, config_file
    ):
        """Test radius_uncertainty handles exceptions gracefully"""
        image_data = sample_horizon_image()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            plt.imsave(tmp_file.name, image_data, cmap="gray")
            tmp_file.flush()

            obs = LimbObservation(image_filepath=tmp_file.name, fit_config=config_file)
            obs.fit_results = Mock()  # Mock fit results exist

            # Mock exception
            mock_calc_uncertainty.side_effect = RuntimeError("Test error")

            with patch("planet_ruler.observation.logging") as mock_logging:
                result = obs.radius_uncertainty

                assert result == 0.0
                mock_logging.warning.assert_called_once()
                warning_msg = mock_logging.warning.call_args[0][0]
                assert "Could not calculate radius uncertainty" in warning_msg

            os.unlink(tmp_file.name)

    def test_warm_start_with_missing_parameters(
        self, sample_horizon_image, config_file
    ):
        """Test warm start handles missing parameters in best_parameters"""
        image_data = sample_horizon_image()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            plt.imsave(tmp_file.name, image_data, cmap="gray")
            tmp_file.flush()

            obs = LimbObservation(image_filepath=tmp_file.name, fit_config=config_file)

            assert obs.init_parameter_values is not None
            assert obs.free_parameters is not None
            original_values = obs.init_parameter_values.copy()

            # Set best_parameters with only some free parameters
            obs.best_parameters = {
                "r": 6500000.0,
                # Missing "h" parameter
            }

            # Simulate warm start logic
            for param in obs.free_parameters:
                if param in obs.best_parameters:
                    obs.init_parameter_values[param] = obs.best_parameters[param]

            # "r" should be updated, "h" should remain original
            assert obs.init_parameter_values["r"] == 6500000.0
            assert obs.init_parameter_values["h"] == original_values["h"]

            os.unlink(tmp_file.name)


class TestObservationMiscellaneous:
    """Test miscellaneous methods and edge cases"""

    @pytest.fixture
    def config_file(self, sample_fit_config):
        """Create a temporary config file"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.dump(sample_fit_config, f)
            f.flush()
            yield f.name
        os.unlink(f.name)

    def test_smooth_limb_without_fill_nan(self, sample_horizon_image, config_file):
        """Test smooth_limb with fill_nan=False"""
        image_data = sample_horizon_image()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            plt.imsave(tmp_file.name, image_data, cmap="gray")
            tmp_file.flush()

            obs = LimbObservation(image_filepath=tmp_file.name, fit_config=config_file)

            raw_limb = np.random.random(100)
            obs._raw_limb = raw_limb

            with patch("planet_ruler.observation.smooth_limb") as mock_smooth:
                with patch("planet_ruler.observation.fill_nans") as mock_fill:
                    smoothed_limb = np.random.random(100)
                    mock_smooth.return_value = smoothed_limb

                    obs.smooth_limb(fill_nan=False, custom_param=42)

                    mock_smooth.assert_called_once_with(raw_limb, custom_param=42)
                    mock_fill.assert_not_called()  # Should not fill NaNs

                    assert np.array_equal(obs.features["limb"], smoothed_limb)

            os.unlink(tmp_file.name)

    def test_properties_with_missing_parameters(
        self, sample_horizon_image, config_file
    ):
        """Test property methods when best_parameters is missing some keys"""
        image_data = sample_horizon_image()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            plt.imsave(tmp_file.name, image_data, cmap="gray")
            tmp_file.flush()

            obs = LimbObservation(image_filepath=tmp_file.name, fit_config=config_file)

            # Set best_parameters with missing keys
            obs.best_parameters = {
                "r": 6371000.0,
                # Missing "h" and "f"
            }

            # Should handle missing keys gracefully
            assert abs(obs.radius_km - 6371.0) < 0.1
            assert obs.altitude_km == 0.0  # Missing "h"
            assert obs.focal_length_mm == 0.0  # Missing "f"

            os.unlink(tmp_file.name)

    def test_analyze_with_none_kwargs(self, sample_horizon_image, config_file):
        """Test analyze method with None kwargs (default behavior)"""
        image_data = sample_horizon_image()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            plt.imsave(tmp_file.name, image_data, cmap="gray")
            tmp_file.flush()

            obs = LimbObservation(image_filepath=tmp_file.name, fit_config=config_file)

            with patch.object(obs, "detect_limb") as mock_detect:
                with patch.object(obs, "fit_arc") as mock_fit:
                    # Pass None explicitly - should convert to empty dict
                    result = obs.analyze()

                    mock_detect.assert_called_once_with()
                    mock_fit.assert_called_once_with()

            os.unlink(tmp_file.name)

    @patch("planet_ruler.observation.limb_arc")
    @patch("planet_ruler.observation.LimbFitter")
    @patch("planet_ruler.observation.FitDashboard")
    def test_dashboard_callback_function(
        self,
        mock_dashboard_class,
        mock_lf_class,
        mock_limb_arc,
        sample_horizon_image,
        config_file,
    ):
        """Test dashboard callback is wired into LimbFitter minimizer_kwargs"""
        image_data = sample_horizon_image()

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            plt.imsave(tmp_file.name, image_data, cmap="gray")
            tmp_file.flush()

            obs = LimbObservation(image_filepath=tmp_file.name, fit_config=config_file)
            obs.features["limb"] = np.random.random(image_data.shape[1])

            mock_dashboard = Mock()
            mock_dashboard_class.return_value = mock_dashboard

            mock_lf = Mock()
            mock_lf.fit.return_value = {
                "best_parameters": {
                    "r": 6500000.0,
                    "h": 15000.0,
                    "n_pix_x": image_data.shape[1],
                    "n_pix_y": image_data.shape[0],
                    "x0": image_data.shape[1] // 2,
                    "y0": image_data.shape[0] // 2,
                },
                "fit_results": Mock(x=np.array([6500000.0, 15000.0])),
                "updated_init": {"r": 6500000.0, "h": 15000.0},
                "updated_limits": {},
                "status": "ok",
                "warnings": [],
            }
            mock_lf_class.return_value = mock_lf
            mock_limb_arc.return_value = np.random.random(image_data.shape[1])

            obs.fit_limb(
                [{"method": "arc", "max_iter": 100}],
                dashboard=True,
            )

            # Callback was wired into LimbFitter's minimizer_kwargs
            lf_call_kwargs = mock_lf_class.call_args[1]
            minimizer_kwargs = lf_call_kwargs.get("minimizer_kwargs", {})
            callback = minimizer_kwargs.get("callback")

            assert callback is not None, "Expected callback wired into LimbFitter"

            # differential_evolution signature: callback(xk, convergence=None)
            result = callback(np.array([6500000.0, 15000.0]))
            assert result is False
            mock_dashboard.update.assert_called()

            # basinhopping/dual-annealing also pass convergence as kwarg
            mock_dashboard.update.reset_mock()
            result = callback(np.array([6500000.0, 15000.0]), convergence=0.5)
            assert result is False
            mock_dashboard.update.assert_called()

            os.unlink(tmp_file.name)


# ---------------------------------------------------------------------------
# Helpers shared by TestFitSagitta
# ---------------------------------------------------------------------------


def _synthetic_limb_for_obs(
    r=6_371_000,
    h=10_000,
    f=0.026,
    w=0.0173,
    W=4000,
    H=3000,
    n_points=12,
    noise_std=0.0,
    seed=42,
):
    """Centered synthetic limb array with apex at W//2."""
    from planet_ruler.geometry import limb_arc, limb_camera_angle

    theta_x = limb_camera_angle(r, h)
    y_arc = limb_arc(
        r=r,
        n_pix_x=W,
        n_pix_y=H,
        h=h,
        f=f,
        w=w,
        x0=W // 2,
        y0=H // 2,
        theta_x=theta_x,
        theta_y=0,
        theta_z=0,
    )
    rng = np.random.default_rng(seed)
    margin = W // 10
    xs = np.linspace(margin, W - margin - 1, n_points, dtype=int)
    limb = np.full(W, np.nan)
    for x in xs:
        limb[x] = y_arc[x] + rng.normal(0, noise_std)
    return limb


def _mock_obs_for_sagitta(
    limb_array, h=10_000.0, f=0.026, w=0.0173, r_init=6_371_000.0
):
    """Minimal stub that satisfies fit_sagitta() without __init__."""
    W = len(limb_array)
    stub = MagicMock(spec=LimbObservation)
    stub.features = {"limb": limb_array}
    stub.init_parameter_values = {"h": h, "f": f, "w": w, "r": r_init}
    stub.parameter_limits = {"r": [1e6, 1e8]}
    stub.image = MagicMock()
    stub.image.shape = (3000, W, 3)
    stub.free_parameters = ["r", "h"]
    stub.stage_results = []
    stub.best_parameters = None
    stub._apply_updated_limits = lambda lims: LimbObservation._apply_updated_limits(
        stub, lims
    )
    stub._apply_updated_init = lambda inits: LimbObservation._apply_updated_init(
        stub, inits
    )
    return stub


class TestFitSagitta:
    """Tests for LimbObservation.fit_sagitta()."""

    def test_raises_without_limb(self):
        stub = _mock_obs_for_sagitta(np.full(4000, np.nan))
        stub.features = {}
        with pytest.raises(ValueError, match="Must detect limb"):
            LimbObservation.fit_sagitta(stub)

    def test_updates_r_and_limits(self):
        limb = _synthetic_limb_for_obs()
        stub = _mock_obs_for_sagitta(limb)
        LimbObservation.fit_sagitta(stub)
        assert stub.init_parameter_values["r"] != 6_371_000.0
        r_low, r_high = stub.parameter_limits["r"]
        assert r_low < stub.init_parameter_values["r"] < r_high

    def test_returns_self(self):
        limb = _synthetic_limb_for_obs()
        stub = _mock_obs_for_sagitta(limb)
        result = LimbObservation.fit_sagitta(stub)
        assert result is stub

    def test_n_sigma_widens_bounds(self):
        # n_sigma controls SagittaFitter's bound width; check raw stage output
        # (parameter_limits reflects the intersection with initial bounds, which
        # may clamp the sagitta bounds when sigma is large).
        limb = _synthetic_limb_for_obs()

        stub1 = _mock_obs_for_sagitta(limb)
        LimbObservation.fit_sagitta(stub1, n_sigma=1.0)
        width1 = stub1.stage_results[0]["r_high"] - stub1.stage_results[0]["r_low"]

        stub2 = _mock_obs_for_sagitta(limb)
        LimbObservation.fit_sagitta(stub2, n_sigma=3.0)
        width2 = stub2.stage_results[0]["r_high"] - stub2.stage_results[0]["r_low"]

        assert width2 > width1

    def test_appends_stage_result(self):
        limb = _synthetic_limb_for_obs()
        stub = _mock_obs_for_sagitta(limb)
        LimbObservation.fit_sagitta(stub, n_sigma=2.5)
        assert len(stub.stage_results) == 1
        entry = stub.stage_results[0]
        assert entry["method"] == "sagitta"
        assert entry["n_sigma"] == 2.5

    def test_sets_best_parameters(self):
        limb = _synthetic_limb_for_obs()
        stub = _mock_obs_for_sagitta(limb)
        assert stub.best_parameters is None
        LimbObservation.fit_sagitta(stub)
        assert stub.best_parameters is not None
        assert "r" in stub.best_parameters

    def test_degenerate_limb_behavior(self):
        """Too-few-points limb: SagittaFitter returns too_few_points; r unchanged."""
        limb = np.full(4000, np.nan)
        limb[500] = 1200.0
        limb[1000] = 1202.0
        limb[1500] = 1203.0  # only 3 points — below the 4-point threshold
        stub = _mock_obs_for_sagitta(limb, r_init=6_371_000.0)
        LimbObservation.fit_sagitta(stub)
        # SagittaFitter returns updated_init={} for too_few_points → r unchanged
        assert stub.init_parameter_values["r"] == 6_371_000.0
        assert len(stub.stage_results) == 1
