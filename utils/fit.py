import numpy as np
import pandas as pd
import scipy.optimize
from scipy.optimize import minimize
from abcpy.probabilisticmodels import ProbabilisticModel, Continuous, InputConnector
from abcpy.continuousmodels import Uniform, Normal
from abcpy.statistics import Identity
from abcpy.distances import Euclidean
from abcpy.inferences import RejectionABC


def unpack_parameters(params, template):
    return {key: params[i] for i, key in enumerate(template)}


def pack_parameters(params, template):
    return [params[key] if key in params else template[key] for key in template]


class CostFunction:
    def __init__(self, target, function, free_parameters, init_parameter_values):
        self.function = function
        self.free_parameters = free_parameters
        self.init_parameter_values = init_parameter_values
        self.x = np.arange(len(target))
        self.target = target

    def cost(self, params):

        if type(params) == np.ndarray:
            kwargs = self.init_parameter_values.copy()
            kwargs.update(unpack_parameters(list(params), self.free_parameters))
        else:
            kwargs = params

        y = self.function(self.x, **kwargs)

        cost = np.sum(pow(y - self.target, 2))

        return cost

    def evaluate(self, params):

        kwargs = self.init_parameter_values.copy()
        if type(params) == np.ndarray:
            kwargs.update(unpack_parameters(list(params), self.free_parameters))
        else:
            kwargs.update(params)

        y = self.function(self.x, **kwargs)

        return y


# class LimbArc(ProbabilisticModel, Continuous):
class LimbArc:

    def __init__(self, target, function, free_parameters, init_parameter_values): #, name='LimbArc'):
        # if not isinstance(parameters, list):
        #     raise TypeError('Input of Normal model is of type list')

        self.target = target
        self.function = function
        self.cost_function = CostFunction(target, function, free_parameters, init_parameter_values)

        self.free_parameters = free_parameters
        self.init_parameter_values = init_parameter_values

        # self.n_fit_params = len(free_parameters)
        # if len(parameters) != self.n_fit_params:
        #     raise RuntimeError(f'Input list must be of length {self.n_fit_params},'
        #                        f'containing {free_parameters}.')

        # input_connector = InputConnector.from_list(parameters)
        # super().__init__(input_connector, name)

    def forward_simulate(self, input_values, k, rng=np.random.RandomState()):
        obs = self.cost_function.evaluate(unpack_parameters(input_values, self.free_parameters))
        # Format the output to obey API
        obs = obs.reshape(1, len(obs))
        # Hack just repeat samples instead of draw
        result = [obs for _ in range(k)]

        return result

    # def _check_input(self, input_values):
    #     # Check whether input has correct type or format
    #     if len(input_values) != self.n_fit_params:
    #         raise ValueError(f'Number of parameters of LimbArc model must be {self.n_fit_params}.')
    #
    #     # Check whether input is from correct domain
    #     # todo incorporate limit template
    #     params = unpack_parameters(input_values, self.free_parameters)
    #     if params['r'] < 0 or params['h'] < 0 or params['f'] < 0 or params['py'] <= 0:
    #         return False
    #
    #     return True
    #
    # def _check_output(self, values):
    #     # todo figure this out
    #     if not isinstance(values, np.array):
    #         raise ValueError('This returns a bivariate array')
    #
    #     #         if values.shape[0] != 2:
    #     #             raise RuntimeError('The size of the output has to be 2.')
    #
    #     return True
    #
    # def get_output_dimension(self):
    #     # todo figure this out
    #     #         return self.num_x_pixels
    #     return 2
    #
    # def forward_simulate(self, input_values, k, rng=np.random.RandomState()):
    #     obs = self.cost_function.evaluate(unpack_parameters(input_values, self.free_parameters))
    #
    #     # Format the output to obey API
    #     obs = obs.reshape(1, len(obs))
    #     # Hack just repeat samples instead of draw
    #     result = [obs for _ in range(k)]
    #     return result


# class LimbArcABC(LimbArc):
class LimbArcABC(LimbArc, ProbabilisticModel, Continuous):
    def __init__(self, target, function, free_parameters, init_parameter_values,
                 parameter_limits, parameter_priors, seed=1): #, n_cpu=4):

        self.parameter_limits = parameter_limits
        pdfs = []
        for key in free_parameters:
            if parameter_priors[key]['type'] == 'Normal':
                base = {'mean': init_parameter_values[key],
                        'std': abs(init_parameter_values[key] * 0.1)
                        }
                base.update(parameter_priors[key])
                pdfs += [Normal([[base['mean']],
                                 [base['std']]],
                                name=key)]
            elif parameter_priors[key]['type'] == 'Uniform':
                base = {'lower': parameter_limits[key][0],
                        'upper': parameter_limits[key][1]
                        }
                base.update(parameter_priors[key])
                pdfs += [Uniform([[base['lower']],
                                  [base['upper']]],
                                 name=key)]
        self.pdfs = pdfs

        input_connector = InputConnector.from_list(self.pdfs)
        # target, function, free_parameters, init_parameter_values
        # super().__init__(target=target, function=function, free_parameters=free_parameters,
        #                  init_parameter_values=init_parameter_values, name='LimbArc')
        LimbArc.__init__(self, target, function, free_parameters, init_parameter_values)

        ProbabilisticModel.__init__(self, input_connector, 'LimbArc')
        self.statistics_calculator = Identity()
        self.distance_calculator = Euclidean(self.statistics_calculator)

        self.obs = target.reshape(1, len(target))
        self.seed = seed
        self.journal = None
        self.accepted = []
        self.best_fit = np.inf
        self.best_parameters = None

    def _check_input(self, input_values):
        # Check whether input has correct type or format
        if len(input_values) != self.n_fit_params:
            raise ValueError(f'Number of parameters of LimbArc model must be {self.n_fit_params}.')

        # Check whether input is from correct domain
        # todo incorporate limit template
        params = unpack_parameters(input_values, self.free_parameters)
        if params['r'] < 0 or params['h'] < 0 or params['f'] < 0 or params['py'] <= 0:
            return False

        return True

    def _check_output(self, values):
        # todo figure this out
        if not isinstance(values, np.array):
            raise ValueError('This returns a bivariate array')

        #         if values.shape[0] != 2:
        #             raise RuntimeError('The size of the output has to be 2.')

        return True

    def get_output_dimension(self):
        # todo figure this out
        #         return self.num_x_pixels
        return 2

    # def forward_simulate(self, input_values, k, rng=np.random.RandomState()):
    #     obs = self.cost_function.evaluate(unpack_parameters(input_values, self.free_parameters))
    #
    #     # Format the output to obey API
    #     obs = obs.reshape(1, len(obs))
    #     # Hack just repeat samples instead of draw
    #     result = [obs for _ in range(k)]
    #     return result

    def generate_samples(self, backend, n_samples, n_samples_per_param=1, epsilon=100000):

        sampler = RejectionABC([self], [self.distance_calculator], backend, seed=self.seed)
        self.journal = sampler.sample([[self.obs]],
                                       n_samples_per_param=n_samples_per_param,
                                       n_samples=n_samples,
                                       epsilon=epsilon)

        posterior_samples = np.array(self.journal.get_accepted_parameters()).squeeze()
        for sample in posterior_samples:
            self.accepted += [unpack_parameters(sample, self.free_parameters)]

    def get_accepted(self):
        return pd.DataFrame.from_records(self.accepted)

    def fit_accepted(self, epsilon=100000, method='Nelder-Mead'):
        accepted = self.get_accepted()
        for i in accepted.index:
            d = self.distance_calculator.distance([self.obs],
                                                  self.forward_simulate(list(accepted.loc[i].values), 1))
            if d > epsilon:
                continue

            # results = scipy.optimize.differential_evolution(self.cost_function.cost,
            #                                                 [self.parameter_limits[key] for key in self.free_parameters],
            #                                                 workers=4, maxiter=100000, polish=True,
            #                                                 init='halton', updating='deferred')
            results = minimize(self.cost_function.cost,
                               accepted.loc[i].values,
                               method=method,
                               bounds=[self.parameter_limits[key] for key in self.free_parameters],
                               options={'maxiter':500000})

            print(results.success, results.message)
            if results.fun < self.best_fit:
                self.best_fit = results.fun
                print('new best', results.fun)
                # todo add function to preserve all optimize params, not just best
                self.best_parameters = unpack_parameters(results.x, self.free_parameters)
