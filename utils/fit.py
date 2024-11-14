import numpy as np


def unpack_parameters(params, template):
    return {key: params[i] for i, key in enumerate(template)}


def pack_parameters(params, template):
    return [params[key] if key in params else template[key] for key in template]


class CostFunction:
    def __init__(self, target, function, free_parameters, init_parameter_values, l2=True):
        self.function = function
        self.free_parameters = free_parameters
        self.init_parameter_values = init_parameter_values
        self.x = np.arange(len(target))
        self.target = target
        self.l2 = l2

    def cost(self, params):

        if type(params) == np.ndarray:
            kwargs = self.init_parameter_values.copy()
            kwargs.update(unpack_parameters(list(params), self.free_parameters))
        else:
            kwargs = params

        y = self.function(self.x, **kwargs)

        if self.l2:
            cost = np.sum(pow(y - self.target, 2))
        else:
            absdiff = abs(y - self.target)
            # cost = np.sum(absdiff) + np.sum(pow(absdiff+1, -0.5))
            cost = np.sum(np.log(absdiff + 1))


        return cost

    def evaluate(self, params):

        kwargs = self.init_parameter_values.copy()
        if type(params) == np.ndarray:
            kwargs.update(unpack_parameters(list(params), self.free_parameters))
        else:
            kwargs.update(params)

        y = self.function(self.x, **kwargs)

        return y
