import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
import seaborn as sns
import matplotlib.pyplot as plt


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
            abs_diff = abs(y - self.target)
            # cost = np.sum(abs_diff) + np.sum(pow(abs_diff+1, -0.5))
            cost = np.sum(np.log(abs_diff + 1))


        return cost

    def evaluate(self, params):

        kwargs = self.init_parameter_values.copy()
        if type(params) == np.ndarray:
            kwargs.update(unpack_parameters(list(params), self.free_parameters))
        else:
            kwargs.update(params)

        y = self.function(self.x, **kwargs)

        return y


def unpack_diff_evol_posteriors(observation):
    pop = []
    en = observation.fit_results['population_energies']
    for i, sol in enumerate(observation.fit_results['population']):
        mse = en[i]
        updated = observation.init_parameter_values.copy()
        updated.update(unpack_parameters(sol, observation.free_parameters))
        updated['mse'] = mse
        pop.append(updated)
    pop = pd.DataFrame.from_records(pop)

    return pop


def plot_diff_evol_posteriors(observation, show_points=False):
    pop = unpack_diff_evol_posteriors(observation)

    for col in pop.columns:
        if col == 'mse':
            continue
        if show_points:
            plt.scatter(pop[col], pop['mse'])
        sns.kdeplot(x=pop[col], y=pop['mse'], color='blue', warn_singular=False, label='posterior')
        plt.axvline(observation.parameter_limits[col][0], ls='--', c='k', alpha=0.5, label='bounds')
        plt.axvline(observation.parameter_limits[col][1], ls='--', c='k', alpha=0.5)
        plt.title(col)
        plt.grid(which='both', ls='--', alpha=0.2)
        ax = plt.gca()
        ax.set_yscale('log')

        handles, labels = ax.get_legend_handles_labels()
        h_plus, l_plus = [Line2D([0], [0], color='blue', lw=2)], ['posterior']
        plt.legend(handles + h_plus, labels + l_plus)

        plt.show()


def package_results(observation):
    full_fit_params = unpack_parameters(observation.fit_results.x, observation.free_parameters)

    results = []
    for key in observation.free_parameters:
        result = {'fit value': full_fit_params[key],
                  'initial value': observation.init_parameter_values[key],
                  'parameter': key}
        results.append(result)
    results = pd.DataFrame.from_records(results)
    results = results.set_index(['parameter'])
    return results
