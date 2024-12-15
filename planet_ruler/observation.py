from scipy.optimize import differential_evolution, shgo, minimize
import yaml
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from planet_ruler.plot import plot_image, plot_limb
from planet_ruler.image import load_image, gradient_break, StringDrop, smooth_limb, fill_nans
from planet_ruler.fit import CostFunction, unpack_parameters
from planet_ruler.geometry import limb_arc


class PlanetObservation:
    """
    Base class for planet observations.

    Args:
        image_filepath (str): Path to image file.
    """
    def __init__(
            self,
            image_filepath: str):
        self.raw_image = load_image(image_filepath)
        self.image = self.raw_image.copy()
        self.features = {}
        self._plot_functions = {}
        self._cwheel = ['y', 'b', 'r', 'orange', 'pink', 'black']

    def plot(
            self,
            gradient: bool = False,
            show: bool = True) -> None:
        """
        Display the observation and all current features.

        Args:
            gradient (bool): Show the image gradient instead of the raw image.
            show (bool): Show -- useful as False if intending to add more to the plot before showing.
        """
        plot_image(self.image, gradient=gradient, show=False)
        for i, feature in enumerate(self.features):
            self._plot_functions[feature](self.features[feature], show=False, c=self._cwheel[i])
        if show:
            plt.show()

    def restrict_image(
            self,
            xmin: int = 0,
            ymin: int = 0,
            xmax: int = -1,
            ymax: int = -1) -> None:
        """
        Limit the observation by pixels. Note it does not change the loaded image and can always
        be updated.

        Args:
            xmin (int): Left pixel bound.
            ymin (int): Bottom pixel bound.
            xmax (int): Right pixel bound.
            ymax (int): Top pixel bound.
        """
        self.image = self.raw_image[ymin:ymax, xmin:xmax]


class LimbObservation(PlanetObservation):
    """
    Observation of a planet's limb (horizon).

    Args:
        image_filepath (str): Path to image file.
        fit_config (str): Path to fit config file.
        limb_detection (str): Method to locate the limb in the image.
        minimizer (str): Choice of minimizer. Currently only supports
            'differential-evolution'.
    """
    def __init__(
            self,
            image_filepath,
            fit_config,
            limb_detection='string-drop',
            minimizer='differential-evolution'):
        super().__init__(image_filepath)

        self.free_parameters = None
        self.init_parameter_values = None
        self.parameter_limits = None
        self.load_fit_config(fit_config)
        assert limb_detection in ['gradient-break', 'string-drop']
        self.limb_detection = limb_detection
        self._string_drop = None
        assert minimizer in ['differential-evolution']
        self.minimizer = minimizer

        self._raw_limb = None
        self.cost_function = None
        self.fit = None
        self.best_parameters = None
        self.fit_results = None

    def load_fit_config(
            self,
            fit_config: str) -> None:
        """
        Load the fit configuration from file, setting all parameters
        to their initial values.

        Args:
            fit_config (str): Path to configuration file.
        """
        with open(fit_config, 'r') as f:
            base_config = yaml.safe_load(f)

        for p, v in base_config['init_parameter_values'].items():
            assert v >= base_config['parameter_limits'][p][0],\
                f"Initial value for parameter {p} violates stated lower limit."
            assert v <= base_config['parameter_limits'][p][1], \
                f"Initial value for parameter {p} violates stated upper limit."

        self.free_parameters = base_config['free_parameters']
        self.init_parameter_values = base_config['init_parameter_values']
        self.parameter_limits = base_config['parameter_limits']

    def detect_limb(self, **kwargs) -> None:
        """
        Use the instance-defined method to find the limb in our observation.
        Kwargs are passed to the method.
        """
        if self.limb_detection == 'string-drop':
            if self._string_drop is None:
                self._string_drop = StringDrop(self.image)
            print('computing gradient force map...')
            self._string_drop.compute_force_map(tilt=0.05, smoothing_window=50)
            print('dropping horizon string...')
            self.features['limb'] = self._string_drop.drop_string(**kwargs)
        elif self.limb_detection == 'gradient-break':
            self.features['limb'] = gradient_break(self.image, **kwargs)

        self._raw_limb = self.features['limb'].copy()
        self._plot_functions['limb'] = plot_limb

    def smooth_limb(
            self,
            fill_nan=True,
            **kwargs) -> None:
        """
        Apply the smooth_limb function to current observation.

        Args:
            fill_nan (bool): Fill any NaNs in the limb.
        """
        self.features['limb'] = smooth_limb(self._raw_limb, **kwargs)
        if fill_nan:
            logging.info("Filling NaNs in fitted limb.")
            self.features['limb'] = fill_nans(self.features['limb'])

    def fit_limb(self,
                 init_parameters: dict = None,
                 loss_function: str = 'l2',
                 max_iter: int = 1000,
                 n_jobs: int = 1,
                 seed: int = 0) -> None:
        """
        Fit the current limb using minimizer of choice.

        Args:
            init_parameters (dict): Initial values for named parameters.
            loss_function (str): Type of loss function, must be one of ['l2', 'l1', 'log-l1'].
            max_iter (int): Maximum steps used to converge.
            n_jobs (int): Number of cores to engage.
            seed (int): Random seed for minimizer.
        """
        if init_parameters is None:
            init_parameters = [self.init_parameter_values[key] for key in self.free_parameters]

        self.cost_function = CostFunction(self.features['limb'], limb_arc,
                                          self.free_parameters,
                                          self.init_parameter_values,
                                          loss_function=loss_function)
        if self.minimizer == 'differential-evolution':
            # 'rand1exp' and  Best/2 supposed to be good?  best2bin best2exp?  rand1bin?
            strategy = 'best2bin'
            # strategy = 'rand1exp'
            if n_jobs > 1:
                updating = 'deferred'
            else:
                updating = 'immediate'
            self.fit_results = differential_evolution(
                self.cost_function.cost,
                [self.parameter_limits[key] for key in self.free_parameters],
                workers=n_jobs, maxiter=max_iter, strategy=strategy,
                polish=True,
                init='sobol', # halton is another reasonable option
                mutation=[0.1, 1.9],
                updating=updating, disp=True, x0=init_parameters, seed=seed)
        self.best_parameters = unpack_parameters(self.fit_results.x, self.free_parameters)

        self.features['fitted_limb'] = self.cost_function.evaluate(self.best_parameters)
        self._plot_functions['fitted_limb'] = plot_limb

    def save_limb(self,
                  filepath: str) -> None:
        """
        Save the detected limb position as a numpy array.

        Args:
            filepath (str): Path to save file.
        """
        np.save(filepath, self.features['limb'])

    def load_limb(self,
                  filepath: str) -> None:
        """
        Load the detected limb position from a numpy array.

        Args:
            filepath (str): Path to save file.
        """
        self.features['limb'] = np.load(filepath)
        self.features['limb'] = fill_nans(self.features['limb'])
        self._plot_functions['limb'] = plot_limb


def unpack_diff_evol_posteriors(
        observation: LimbObservation) -> pd.DataFrame:
    """
    Extract the final state population of a differential evolution
    minimization and organize as a DataFrame.

    Args:
        observation (object): Instance of LimbObservation (must have
            used differential evolution minimizer).

    Returns:
        population (pd.DataFrame): Population (rows) and properties (columns).
    """
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


def plot_diff_evol_posteriors(
        observation: LimbObservation,
        show_points: bool = False):
    """
    Extract and display the final state population of a differential evolution
    minimization.

    Args:
        observation (object): Instance of LimbObservation (must have
            used differential evolution minimizer).
        show_points (bool): Show the individual population members in
            addition to the contour.

    Returns:
        None
    """
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


def package_results(
        observation: LimbObservation) -> pd.DataFrame:
    """
    Consolidate the results of a fit to see final vs. initial values.

    Args:
        observation (object): Instance of LimbObservation (must have
            used differential evolution minimizer).

    Returns:
        results (pd.DataFrame): DataFrame of results including
            - fit value
            - initial value
            - parameter
    """
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
