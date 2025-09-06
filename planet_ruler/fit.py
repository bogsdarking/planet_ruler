import numpy as np
from typing import Callable


def unpack_parameters(params: list,
                      template: list) -> dict:
    """
    Turn a list of parameters back into a dict.

    Args:
        params (list): Values of dictionary elements in a list.
        template (list): Ordered list of target keys.

    Returns:
        param_dict (dict): Parameter dictionary.
    """
    return {key: params[i] for i, key in enumerate(template)}


def pack_parameters(params: dict, template: dict) -> list:
    """
    Turn a dict of parameters (or defaults) into a list.

    Args:
        params (dict): Parameter dictionary (subset or full keys of template).
        template (dict): Template (full) parameter dictionary.

    Returns:
        param_list (list): List of parameter values.
    """
    return [params[key] if key in params else template[key] for key in template]


class CostFunction:
    """
    Wrapper to simplify interface with the minimization at hand.

    Args:
        target (np.ndarray): True value(s), e.g., the actual limb position.
        function (Callable): Function mapping parameters to target of interest.
        free_parameters (list): List of free parameter names.
        init_parameter_values (dict): Initial values for named parameters.
        loss_function (str): Type of loss function, must be one of ['l2', 'l1', 'log-l1'].

    Returns:
        param_list (list): List of parameter values.
    """
    def __init__(
            self,
            target: np.ndarray,
            function: Callable,
            free_parameters: list,
            init_parameter_values,
            loss_function='l2'):

        self.function = function
        self.free_parameters = free_parameters
        self.init_parameter_values = init_parameter_values
        self.x = np.arange(len(target))
        self.target = target
        self.loss_function = loss_function

    def cost(
            self,
            params: np.ndarray | dict) -> float:
        """
        Compute prediction and use desired metric to reduce difference
        from truth to a cost. AKA loss function.

        Args:
            params (np.ndarray | dict): Parameter values, either packed
                into array or as dict.

        Returns:
            cost (float): Cost given parameters.
        """
        y = self.evaluate(params)

        if self.loss_function == 'l2':
            cost = np.nanmean(pow(y - self.target, 2))
        elif self.loss_function == 'l1':
            abs_diff = abs(y - self.target)
            cost = np.nanmean(abs_diff)
        elif self.loss_function == 'log-l1':
            abs_diff = abs(y - self.target)
            # cost = np.sum(abs_diff) + np.sum(pow(abs_diff+1, -0.5))
            cost = np.nanmean(np.log(abs_diff + 1))
        else:
            raise ValueError("Unrecognized loss function.")

        return cost

    def evaluate(
            self,
            params: np.ndarray | dict) -> np.ndarray:
        """
        Compute prediction given parameters.

        Args:
            params (np.ndarray | dict): Parameter values, either packed
                into array or as dict.

        Returns:
            prediction (np.ndarray): Prediction value(s).
        """
        kwargs = self.init_parameter_values.copy()
        if type(params) == np.ndarray:
            kwargs.update(unpack_parameters(list(params), self.free_parameters))
        else:
            kwargs.update(params)

        y = self.function(**kwargs)

        return y
