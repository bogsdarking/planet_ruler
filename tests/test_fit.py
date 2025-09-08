# tests/test_fit.py
import pytest
import numpy as np
from planet_ruler.fit import unpack_parameters, pack_parameters, CostFunction


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
