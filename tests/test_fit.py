# tests/test_fit.py
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, Mock
from planet_ruler.fit import (
    unpack_parameters,
    pack_parameters,
    CostFunction,
    calculate_parameter_uncertainty,
    format_parameter_result,
)


class TestParameterHandling:
    """Test parameter packing and unpacking functions"""

    def test_unpack_parameters_basic(self):
        """Test basic parameter unpacking"""
        params = [1.0, 2.0, 3.0]
        template = ["a", "b", "c"]

        result = unpack_parameters(params, template)

        expected = {"a": 1.0, "b": 2.0, "c": 3.0}
        assert result == expected

    def test_unpack_parameters_empty(self):
        """Test unpacking empty parameters"""
        params = []
        template = []

        result = unpack_parameters(params, template)

        assert result == {}

    def test_unpack_parameters_single(self):
        """Test unpacking single parameter"""
        params = [42.0]
        template = ["radius"]

        result = unpack_parameters(params, template)

        assert result == {"radius": 42.0}

    def test_pack_parameters_basic(self):
        """Test basic parameter packing"""
        params = {"a": 1.0, "b": 2.0}
        template = {"a": 0.0, "b": 0.0, "c": 3.0}

        result = pack_parameters(params, template)

        # Should use params values for 'a', 'b' and template default for 'c'
        expected = [1.0, 2.0, 3.0]
        assert result == expected

    def test_pack_parameters_partial_override(self):
        """Test packing with partial parameter override"""
        params = {"radius": 100.0}
        template = {"radius": 1.0, "height": 10.0, "focal_length": 0.05}

        result = pack_parameters(params, template)

        expected = [100.0, 10.0, 0.05]
        assert result == expected

    def test_pack_parameters_empty_params(self):
        """Test packing with empty params (use all defaults)"""
        params = {}
        template = {"a": 1.0, "b": 2.0}

        result = pack_parameters(params, template)

        expected = [1.0, 2.0]
        assert result == expected

    def test_pack_unpack_roundtrip(self):
        """Test that pack and unpack are inverse operations"""
        original_dict = {"r": 6371000, "h": 10000, "f": 0.05}
        template = {"r": 1000000, "h": 1000, "f": 0.02}

        # Pack to list
        packed = pack_parameters(original_dict, template)
        # Unpack back to dict
        unpacked = unpack_parameters(packed, list(template.keys()))

        assert unpacked == original_dict


class TestCostFunction:
    """Test the CostFunction optimization wrapper class"""

    @pytest.fixture
    def simple_target(self):
        """Simple linear target for testing"""
        return np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    @pytest.fixture
    def simple_function(self):
        """Simple linear function for testing: f(x) = m*x + b"""

        def linear_func(m=1.0, b=0.0, **kwargs):
            x = np.arange(5)
            return m * x + b

        return linear_func

    @pytest.fixture
    def cost_func_basic(self, simple_target, simple_function):
        """Basic CostFunction instance for testing"""
        return CostFunction(
            target=simple_target,
            function=simple_function,
            free_parameters=["m", "b"],
            init_parameter_values={"m": 1.0, "b": 1.0},
        )

    def test_cost_function_initialization(self, simple_target, simple_function):
        """Test CostFunction initialization"""
        cost_func = CostFunction(
            target=simple_target,
            function=simple_function,
            free_parameters=["m", "b"],
            init_parameter_values={"m": 1.0, "b": 1.0},
            loss_function="l2",
        )

        assert np.array_equal(cost_func.target, simple_target)
        assert cost_func.function == simple_function
        assert cost_func.free_parameters == ["m", "b"]
        assert cost_func.init_parameter_values == {"m": 1.0, "b": 1.0}
        assert cost_func.loss_function == "l2"
        assert len(cost_func.x) == len(simple_target)

    def test_evaluate_with_dict_params(self, cost_func_basic):
        """Test evaluate method with dictionary parameters"""
        params = {"m": 1.0, "b": 1.0}

        result = cost_func_basic.evaluate(params)

        # f(x) = 1.0*x + 1.0 for x=[0,1,2,3,4] should give [1,2,3,4,5]
        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert np.array_equal(result, expected)

    def test_evaluate_with_array_params(self, cost_func_basic):
        """Test evaluate method with numpy array parameters"""
        params = np.array([1.0, 1.0])  # m=1.0, b=1.0

        result = cost_func_basic.evaluate(params)

        expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert np.array_equal(result, expected)

    def test_evaluate_partial_override(self, cost_func_basic):
        """Test evaluate with partial parameter override"""
        # Only override 'm', should use init value for 'b'
        params = {"m": 2.0}

        result = cost_func_basic.evaluate(params)

        # f(x) = 2.0*x + 1.0 for x=[0,1,2,3,4] should give [1,3,5,7,9]
        expected = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
        assert np.array_equal(result, expected)

    def test_cost_l2_perfect_fit(self, cost_func_basic):
        """Test L2 cost function with perfect fit (cost should be 0)"""
        # Perfect parameters: target = [1,2,3,4,5], so m=1, b=1
        params = {"m": 1.0, "b": 1.0}

        cost = cost_func_basic.cost(params)

        assert np.isclose(cost, 0.0, atol=1e-10)

    def test_cost_l2_imperfect_fit(self, cost_func_basic):
        """Test L2 cost function with imperfect fit"""
        # Wrong parameters should give non-zero cost
        params = {"m": 2.0, "b": 0.0}

        cost = cost_func_basic.cost(params)

        # Prediction: [0,2,4,6,8], Target: [1,2,3,4,5]
        # Residuals: [-1,0,1,2,3], Squared: [1,0,1,4,9], Mean: 3.0
        expected_cost = 3.0
        assert np.isclose(cost, expected_cost)

    def test_cost_l1_loss(self, simple_target, simple_function):
        """Test L1 (absolute) loss function"""
        cost_func = CostFunction(
            target=simple_target,
            function=simple_function,
            free_parameters=["m", "b"],
            init_parameter_values={"m": 1.0, "b": 1.0},
            loss_function="l1",
        )

        params = {"m": 2.0, "b": 0.0}
        cost = cost_func.cost(params)

        # Prediction: [0,2,4,6,8], Target: [1,2,3,4,5]
        # Abs residuals: [1,0,1,2,3], Mean: 1.4
        expected_cost = 1.4
        assert np.isclose(cost, expected_cost)

    def test_cost_log_l1_loss(self, simple_target, simple_function):
        """Test log-L1 loss function"""
        cost_func = CostFunction(
            target=simple_target,
            function=simple_function,
            free_parameters=["m", "b"],
            init_parameter_values={"m": 1.0, "b": 1.0},
            loss_function="log-l1",
        )

        params = {"m": 1.0, "b": 1.0}  # Perfect fit
        cost = cost_func.cost(params)

        # Perfect fit should give log(0+1) = log(1) = 0
        assert np.isclose(cost, 0.0, atol=1e-10)

    def test_cost_invalid_loss_function(self, simple_target, simple_function):
        """Test that invalid loss function raises ValueError"""
        cost_func = CostFunction(
            target=simple_target,
            function=simple_function,
            free_parameters=["m", "b"],
            init_parameter_values={"m": 1.0, "b": 1.0},
            loss_function="invalid_loss",
        )

        with pytest.raises(ValueError, match="Unrecognized loss function"):
            cost_func.cost({"m": 1.0, "b": 1.0})

    def test_cost_with_nans(self, simple_function):
        """Test cost function handling of NaN values"""
        target_with_nan = np.array([1.0, np.nan, 3.0, 4.0, 5.0])

        cost_func = CostFunction(
            target=target_with_nan,
            function=simple_function,
            free_parameters=["m", "b"],
            init_parameter_values={"m": 1.0, "b": 1.0},
        )

        cost = cost_func.cost({"m": 1.0, "b": 1.0})

        # Should handle NaNs gracefully with nanmean
        # Valid comparisons: [0, _, 0, 0, 0] -> mean of [0,0,0,0] = 0
        assert np.isclose(cost, 0.0, atol=1e-10)


class TestCostFunctionIntegration:
    """Integration tests with more complex scenarios"""

    def test_polynomial_fitting(self):
        """Test fitting a polynomial function"""
        # Target: quadratic function y = x^2 + 2x + 1
        x_values = np.linspace(0, 4, 5)
        target = x_values**2 + 2 * x_values + 1

        def quadratic_func(a=1.0, b=1.0, c=1.0, **kwargs):
            return a * x_values**2 + b * x_values + c

        cost_func = CostFunction(
            target=target,
            function=quadratic_func,
            free_parameters=["a", "b", "c"],
            init_parameter_values={"a": 1.0, "b": 1.0, "c": 1.0},
        )

        # Perfect parameters should give zero cost
        perfect_params = {"a": 1.0, "b": 2.0, "c": 1.0}
        cost = cost_func.cost(perfect_params)

        assert np.isclose(cost, 0.0, atol=1e-10)

    def test_geometry_function_integration(self):
        """Test with a geometry-like function"""

        # Simulate horizon distance calculation
        def horizon_func(r=6371000, h=10000, **kwargs):
            return np.sqrt(h**2 + 2 * h * r)

        # Target distances for different heights
        heights = np.array([1000, 5000, 10000, 20000, 50000])
        targets = np.array([113000, 252000, 357000, 505000, 798000])  # approximate

        def multi_horizon_func(r=6371000, **kwargs):
            return np.array([np.sqrt(h**2 + 2 * h * r) for h in heights])

        cost_func = CostFunction(
            target=targets,
            function=multi_horizon_func,
            free_parameters=["r"],
            init_parameter_values={"r": 6000000},  # Wrong initial guess
        )

        # Test that evaluation works
        prediction = cost_func.evaluate({"r": 6371000})
        assert len(prediction) == len(targets)

        # Test that cost is reasonable
        cost = cost_func.cost({"r": 6371000})
        assert cost >= 0  # Cost should always be non-negative

    def test_parameter_bounds_simulation(self):
        """Test parameter handling with realistic parameter counts"""
        # Simulate a fit with many parameters like in limb fitting
        param_names = ["r", "h", "f", "w", "x0", "y0", "theta_x", "theta_y"]
        init_values = {
            "r": 6371000,
            "h": 10000,
            "f": 0.05,
            "w": 0.036,
            "x0": 960,
            "y0": 540,
            "theta_x": 0,
            "theta_y": 0,
        }

        def complex_func(**kwargs):
            # Simple combination of parameters for testing
            return np.array([kwargs["r"] / 1e6, kwargs["h"] / 1e3, kwargs["f"] * 1000])

        target = np.array([6.4, 10.5, 51.0])  # Approximate expected values

        cost_func = CostFunction(
            target=target,
            function=complex_func,
            free_parameters=param_names[:3],  # Only fit first 3 parameters
            init_parameter_values=init_values,
        )

        # Test with array parameters
        params_array = np.array([6400000, 10500, 0.051])
        result = cost_func.evaluate(params_array)

        assert len(result) == len(target)
        assert all(np.isfinite(result))


class TestCalculateParameterUncertainty:
    """Test calculate_parameter_uncertainty function"""

    @pytest.fixture
    def mock_observation_basic(self):
        """Create a mock observation with basic fit results"""
        obs = Mock()
        obs.init_parameter_values = {"r": 6371000, "h": 10000, "f": 0.05}
        obs.best_parameters = {"r": 6371500, "h": 9800}

        # Mock fit_results with population data
        obs.fit_results = Mock()
        obs.fit_results.population = np.array(
            [
                [6371000, 9900, 0.051],
                [6371200, 9850, 0.049],
                [6371800, 9750, 0.052],
                [6372000, 9700, 0.048],
                [6371400, 9950, 0.050],
            ]
        )

        return obs

    @pytest.fixture
    def mock_population_df(self):
        """Create a mock population DataFrame"""
        return pd.DataFrame(
            {
                "r": [6371000, 6371200, 6371800, 6372000, 6371400],
                "h": [9900, 9850, 9750, 9700, 9950],
                "f": [0.051, 0.049, 0.052, 0.048, 0.050],
            }
        )

    def test_missing_best_parameters(self):
        """Test error when observation lacks best_parameters"""
        obs = Mock()
        obs.init_parameter_values = {"r": 6371000}
        # Create mock without best_parameters
        del obs.best_parameters

        with pytest.raises(
            AttributeError, match="must have completed fit with best_parameters"
        ):
            calculate_parameter_uncertainty(obs)

    def test_parameter_not_found(self, mock_observation_basic):
        """Test error when requested parameter not in fitted results"""
        with pytest.raises(ValueError, match="Parameter 'invalid_param' not found"):
            calculate_parameter_uncertainty(
                mock_observation_basic, parameter="invalid_param"
            )

    def test_auto_method_detection_success(self, mock_observation_basic, monkeypatch):
        """Test automatic method detection with differential evolution data"""
        # Mock the unpack function to return our test data
        mock_unpack = Mock(
            return_value=pd.DataFrame(
                {
                    "r": [6371000, 6371200, 6371800, 6372000, 6371400],
                }
            )
        )
        monkeypatch.setattr(
            "planet_ruler.observation.unpack_diff_evol_posteriors", mock_unpack
        )

        result = calculate_parameter_uncertainty(mock_observation_basic, method="auto")

        assert result["method"] == "differential_evolution"
        assert result["parameter"] == "r"
        assert "value" in result
        assert "uncertainty" in result

    def test_auto_method_detection_failure(self):
        """Test auto method detection failure when no supported data available"""
        obs = Mock()
        obs.init_parameter_values = {"r": 6371000}
        obs.best_parameters = {"r": 6371500}
        # Remove fit_results to simulate missing data
        del obs.fit_results

        with pytest.raises(
            ValueError, match="No supported uncertainty method detected"
        ):
            calculate_parameter_uncertainty(obs, method="auto")

    def test_differential_evolution_missing_data(self, mock_observation_basic):
        """Test error when DE method requested but population data missing"""
        # Remove the population attribute
        delattr(mock_observation_basic.fit_results, "population")

        with pytest.raises(
            AttributeError, match="Differential evolution posteriors not available"
        ):
            calculate_parameter_uncertainty(
                mock_observation_basic, method="differential_evolution"
            )

    def test_differential_evolution_std_uncertainty(
        self, mock_observation_basic, monkeypatch
    ):
        """Test DE method with standard deviation uncertainty"""
        mock_df = pd.DataFrame({"r": [6371000, 6371200, 6371800, 6372000, 6371400]})
        mock_unpack = Mock(return_value=mock_df)
        monkeypatch.setattr(
            "planet_ruler.observation.unpack_diff_evol_posteriors", mock_unpack
        )

        result = calculate_parameter_uncertainty(
            mock_observation_basic,
            method="differential_evolution",
            uncertainty_type="std",
        )

        expected_std = mock_df["r"].std()
        assert result["method"] == "differential_evolution"
        assert result["type"] == "std"
        assert np.isclose(result["uncertainty"], expected_std)
        assert result["parameter"] == "r"

    def test_differential_evolution_ptp_uncertainty(
        self, mock_observation_basic, monkeypatch
    ):
        """Test DE method with peak-to-peak uncertainty"""
        mock_df = pd.DataFrame({"r": [6371000, 6371200, 6371800, 6372000, 6371400]})
        mock_unpack = Mock(return_value=mock_df)
        monkeypatch.setattr(
            "planet_ruler.observation.unpack_diff_evol_posteriors", mock_unpack
        )

        result = calculate_parameter_uncertainty(
            mock_observation_basic,
            method="differential_evolution",
            uncertainty_type="ptp",
        )

        expected_ptp = mock_df["r"].max() - mock_df["r"].min()
        assert result["type"] == "ptp"
        assert np.isclose(result["uncertainty"], expected_ptp)

    def test_differential_evolution_iqr_uncertainty(
        self, mock_observation_basic, monkeypatch
    ):
        """Test DE method with interquartile range uncertainty"""
        mock_df = pd.DataFrame({"r": [6371000, 6371200, 6371800, 6372000, 6371400]})
        mock_unpack = Mock(return_value=mock_df)
        monkeypatch.setattr(
            "planet_ruler.observation.unpack_diff_evol_posteriors", mock_unpack
        )

        result = calculate_parameter_uncertainty(
            mock_observation_basic,
            method="differential_evolution",
            uncertainty_type="iqr",
        )

        expected_iqr = mock_df["r"].quantile(0.75) - mock_df["r"].quantile(0.25)
        assert result["type"] == "iqr"
        assert np.isclose(result["uncertainty"], expected_iqr)

    def test_differential_evolution_ci_uncertainty(
        self, mock_observation_basic, monkeypatch
    ):
        """Test DE method with confidence interval uncertainty"""
        mock_df = pd.DataFrame({"r": [6371000, 6371200, 6371800, 6372000, 6371400]})
        mock_unpack = Mock(return_value=mock_df)
        monkeypatch.setattr(
            "planet_ruler.observation.unpack_diff_evol_posteriors", mock_unpack
        )

        result = calculate_parameter_uncertainty(
            mock_observation_basic,
            method="differential_evolution",
            uncertainty_type="ci",
        )

        expected_lower = mock_df["r"].quantile(0.025)
        expected_upper = mock_df["r"].quantile(0.975)

        assert result["type"] == "ci"
        assert isinstance(result["uncertainty"], dict)
        assert "lower" in result["uncertainty"]
        assert "upper" in result["uncertainty"]
        assert "width" in result["uncertainty"]
        assert np.isclose(result["uncertainty"]["lower"], expected_lower)
        assert np.isclose(result["uncertainty"]["upper"], expected_upper)

    def test_scale_factor_application(self, mock_observation_basic, monkeypatch):
        """Test that scale factor is correctly applied to results"""
        mock_df = pd.DataFrame({"r": [6371000, 6371200, 6371800, 6372000, 6371400]})
        mock_unpack = Mock(return_value=mock_df)
        monkeypatch.setattr(
            "planet_ruler.observation.unpack_diff_evol_posteriors", mock_unpack
        )

        scale_factor = 1000.0  # Convert to km
        result = calculate_parameter_uncertainty(
            mock_observation_basic, scale_factor=scale_factor
        )

        # Check that values are scaled correctly
        # The fitted value comes from combining init_parameter_values and best_parameters
        final_params = mock_observation_basic.init_parameter_values.copy()
        final_params.update(mock_observation_basic.best_parameters)
        expected_value = final_params["r"] / scale_factor
        expected_uncertainty = mock_df["r"].std() / scale_factor

        assert result["scale_factor"] == scale_factor
        assert np.isclose(result["value"], expected_value)
        assert np.isclose(result["uncertainty"], expected_uncertainty)

    def test_different_parameter(self, mock_observation_basic, monkeypatch):
        """Test uncertainty calculation for different parameter"""
        mock_df = pd.DataFrame(
            {
                "r": [6371000, 6371200, 6371800, 6372000, 6371400],
                "h": [9900, 9850, 9750, 9700, 9950],
            }
        )
        mock_unpack = Mock(return_value=mock_df)
        monkeypatch.setattr(
            "planet_ruler.observation.unpack_diff_evol_posteriors", mock_unpack
        )

        result = calculate_parameter_uncertainty(mock_observation_basic, parameter="h")

        assert result["parameter"] == "h"
        expected_uncertainty = mock_df["h"].std()
        assert np.isclose(result["uncertainty"], expected_uncertainty)

    def test_parameter_not_in_population(self, mock_observation_basic, monkeypatch):
        """Test error when parameter not found in population data"""
        mock_df = pd.DataFrame({"r": [6371000, 6371200]})  # Missing 'h' parameter
        mock_unpack = Mock(return_value=mock_df)
        monkeypatch.setattr(
            "planet_ruler.observation.unpack_diff_evol_posteriors", mock_unpack
        )

        with pytest.raises(
            ValueError, match="Parameter 'h' not found in population posteriors"
        ):
            calculate_parameter_uncertainty(mock_observation_basic, parameter="h")

    def test_unsupported_uncertainty_type(self, mock_observation_basic, monkeypatch):
        """Test error for unsupported uncertainty type"""
        mock_df = pd.DataFrame({"r": [6371000, 6371200, 6371800]})
        mock_unpack = Mock(return_value=mock_df)
        monkeypatch.setattr(
            "planet_ruler.observation.unpack_diff_evol_posteriors", mock_unpack
        )

        with pytest.raises(
            ValueError, match="Unsupported uncertainty type: invalid_type"
        ):
            calculate_parameter_uncertainty(
                mock_observation_basic, uncertainty_type="invalid_type"
            )

    def test_unsupported_method(self, mock_observation_basic):
        """Test error for unsupported uncertainty method"""
        with pytest.raises(
            ValueError, match="Unsupported uncertainty calculation method"
        ):
            calculate_parameter_uncertainty(
                mock_observation_basic, method="invalid_method"
            )

    def test_bootstrap_not_implemented(self, mock_observation_basic):
        """Test that bootstrap method raises NotImplementedError"""
        with pytest.raises(
            NotImplementedError,
            match="Bootstrap uncertainty calculation not yet implemented",
        ):
            calculate_parameter_uncertainty(mock_observation_basic, method="bootstrap")

    def test_hessian_not_implemented(self, mock_observation_basic):
        """Test that hessian method raises NotImplementedError"""
        with pytest.raises(
            NotImplementedError,
            match="Hessian-based uncertainty calculation not yet implemented",
        ):
            calculate_parameter_uncertainty(mock_observation_basic, method="hessian")

    def test_import_error_handling(self, mock_observation_basic, monkeypatch):
        """Test handling of import error for unpack function"""

        # Mock import to raise ImportError
        def mock_import_error(*args, **kwargs):
            raise ImportError("Mocked import error")

        monkeypatch.setattr(
            "planet_ruler.observation.unpack_diff_evol_posteriors", mock_import_error
        )

        with pytest.raises(
            ImportError, match="Could not import unpack_diff_evol_posteriors function"
        ):
            calculate_parameter_uncertainty(
                mock_observation_basic, method="differential_evolution"
            )

    def test_raw_data_inclusion(self, mock_observation_basic, monkeypatch):
        """Test that raw data is included in results"""
        mock_df = pd.DataFrame({"r": [6371000, 6371200, 6371800, 6372000, 6371400]})
        mock_unpack = Mock(return_value=mock_df)
        monkeypatch.setattr(
            "planet_ruler.observation.unpack_diff_evol_posteriors", mock_unpack
        )

        result = calculate_parameter_uncertainty(mock_observation_basic)

        assert result["raw_data"] is not None
        assert isinstance(result["raw_data"], np.ndarray)
        assert len(result["raw_data"]) == len(mock_df)


class TestFormatParameterResult:
    """Test format_parameter_result function"""

    def test_format_std_uncertainty(self):
        """Test formatting standard deviation uncertainty"""
        uncertainty_result = {
            "value": 6371.5,
            "uncertainty": 2.3,
            "parameter": "r",
            "type": "std",
        }

        result = format_parameter_result(uncertainty_result, units="km")
        expected = "r = 6371.5 ±2.3 km"
        assert result == expected

    def test_format_ptp_uncertainty(self):
        """Test formatting peak-to-peak uncertainty"""
        uncertainty_result = {
            "value": 10.2,
            "uncertainty": 1.8,
            "parameter": "h",
            "type": "ptp",
        }

        result = format_parameter_result(uncertainty_result, units="km")
        expected = "h = 10.2 range ±1.8 km"
        assert result == expected

    def test_format_iqr_uncertainty(self):
        """Test formatting interquartile range uncertainty"""
        uncertainty_result = {
            "value": 0.051,
            "uncertainty": 0.003,
            "parameter": "f",
            "type": "iqr",
        }

        result = format_parameter_result(uncertainty_result, units="m")
        expected = "f = 0.1 IQR ±0.0 m"
        assert result == expected

    def test_format_ci_uncertainty(self):
        """Test formatting confidence interval uncertainty"""
        uncertainty_result = {
            "value": 6371.5,
            "uncertainty": {"lower": 6369.2, "upper": 6373.8, "width": 4.6},
            "parameter": "r",
            "type": "ci",
        }

        result = format_parameter_result(uncertainty_result, units="km")
        expected = "r = 6371.5 km (95% CI: 6369.2-6373.8 km)"
        assert result == expected

    def test_format_unknown_uncertainty_type(self):
        """Test formatting unknown uncertainty type defaults to ±"""
        uncertainty_result = {
            "value": 6371.5,
            "uncertainty": 2.3,
            "parameter": "r",
            "type": "unknown_type",
        }

        result = format_parameter_result(uncertainty_result, units="km")
        expected = "r = 6371.5 ±2.3 km"
        assert result == expected

    def test_format_no_units(self):
        """Test formatting without units"""
        uncertainty_result = {
            "value": 6371.5,
            "uncertainty": 2.3,
            "parameter": "r",
            "type": "std",
        }

        result = format_parameter_result(uncertainty_result)
        expected = "r = 6371.5 ±2.3 "
        assert result == expected

    def test_format_precision(self):
        """Test formatting precision is consistent at 1 decimal place"""
        uncertainty_result = {
            "value": 6371.555555,
            "uncertainty": 2.333333,
            "parameter": "r",
            "type": "std",
        }

        result = format_parameter_result(uncertainty_result, units="km")
        expected = "r = 6371.6 ±2.3 km"
        assert result == expected

    def test_format_ci_precision(self):
        """Test CI formatting precision"""
        uncertainty_result = {
            "value": 6371.555555,
            "uncertainty": {
                "lower": 6369.222222,
                "upper": 6373.888888,
                "width": 4.666666,
            },
            "parameter": "r",
            "type": "ci",
        }

        result = format_parameter_result(uncertainty_result, units="km")
        expected = "r = 6371.6 km (95% CI: 6369.2-6373.9 km)"
        assert result == expected

    def test_format_different_parameters(self):
        """Test formatting for different parameter names"""
        test_cases = [
            ("radius", "radius = 6371.5 ±2.3 km"),
            ("height", "height = 10.2 ±1.8 km"),
            ("focal_length", "focal_length = 0.1 ±0.0 m"),
            ("theta_x", "theta_x = 0.5 ±0.1 degrees"),
        ]

        for param_name, expected in test_cases:
            uncertainty_result = {
                "value": (
                    6371.5
                    if param_name == "radius"
                    else (
                        10.2
                        if param_name == "height"
                        else (0.051 if param_name == "focal_length" else 0.52)
                    )
                ),
                "uncertainty": (
                    2.3
                    if param_name == "radius"
                    else (
                        1.8
                        if param_name == "height"
                        else (0.003 if param_name == "focal_length" else 0.08)
                    )
                ),
                "parameter": param_name,
                "type": "std",
            }

            units = (
                "km"
                if param_name in ["radius", "height"]
                else ("m" if param_name == "focal_length" else "degrees")
            )
            result = format_parameter_result(uncertainty_result, units=units)
            assert result == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
