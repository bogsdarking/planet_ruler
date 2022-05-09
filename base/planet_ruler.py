from scipy.optimize import differential_evolution, minimize
import yaml
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'utils')
from plot import plot_image, plot_limb
from image import load_image, detect_limb, smooth_limb
from fit import CostFunction, unpack_parameters, pack_parameters, LimbArcABC, LimbArc
from geometry import limb_arc


class PlanetObservation:
    def __init__(self, image_filepath):
        self.raw_image = load_image(image_filepath)
        self.image = self.raw_image.copy()
        self.features = {}
        self._plot_functions = {}
        self._cwheel = ['y', 'b', 'r', 'orange', 'pink', 'black']

    def plot(self, gradient=True, show=True):
        plot_image(self.image, gradient=gradient, show=False)
        for i, feature in enumerate(self.features):
            self._plot_functions[feature](self.features[feature], show=False, c=self._cwheel[i])
        if show:
            plt.show()

    def restrict_image(self, xmin=0, ymin=0, xmax=-1, ymax=-1):
        self.image = self.raw_image[ymin:ymax, xmin:xmax]


class LimbObservation(PlanetObservation):
    def __init__(self, image_filepath,
                 fit_config='config/basic_resection.yaml',
                 model=limb_arc,
                 minimizer=differential_evolution):
        super().__init__(image_filepath)

        with open(fit_config, 'r') as f:
            base_config = yaml.safe_load(f)
        self.free_parameters = base_config['free_parameters']
        self.init_parameter_values = base_config['init_parameter_values']
        # self.free_parameters = {key: self.init_parameter_values[key] for key in base_config['free_parameters']}
        self.parameter_limits = base_config['parameter_limits']
        self.parameter_priors = base_config['parameter_priors']
        self.minimizer = minimizer  # todo awkward to have partially implemented choice of minimizer
        self.model = model

        self._raw_limb = None
        self.cost_function = None
        self.fit = None
        self.abc_multifit = None
        self.best_parameters = None

    def detect_limb(self, **kwargs):
        self.features['limb'] = detect_limb(self.image, **kwargs)
        self._raw_limb = self.features['limb'].copy()
        self._plot_functions['limb'] = plot_limb

    def smooth_limb(self, **kwargs):
        self.features['limb'] = smooth_limb(self._raw_limb, **kwargs)

    def fit_limb(self, method='oneshot', init_parameters=None,
                 backend=None, seed=1, n_samples=40, epsilon=100000):
        assert method in ['oneshot', 'abc-multifit']

        if init_parameters is None:
            init_parameters = self.init_parameter_values

        if method == 'oneshot':
            # target, function, parameters, free_parameters, init_parameter_values, name='LimbArc'
            self.fit = LimbArc(self.features['limb'], self.model,
                               self.free_parameters, self.init_parameter_values)
            # self.cost_function = CostFunction(self.features['limb'], self.model,
            #                                   self.free_parameters,
            #                                   self.init_parameter_values)
            print('fitting oneshot...')
            # self.fit = self.minimize(self.cost_function.cost,
            #                          np.array([init_parameters[key] for key in self.free_parameters]),
            #                          method='Nelder-Mead',
            #                          bounds=[self.parameter_limits[key] for key in self.free_parameters],
            #                          options={'maxiter': 100000})

            results = self.minimizer(self.fit.cost_function.cost,
                                     [self.parameter_limits[key] for key in self.free_parameters],
                                     workers=4, maxiter=100000, polish=True,
                                     init='halton', updating='deferred')

            self.best_parameters = unpack_parameters(results.x, self.free_parameters)

        elif method == 'abc-multifit':
            self.fit = LimbArcABC(self.features['limb'], self.model,
                                  self.free_parameters, self.init_parameter_values,
                                  self.parameter_limits, self.parameter_priors,
                                  seed=seed)
            print('generating samples...')
            self.fit.generate_samples(backend, n_samples, n_samples_per_param=1, epsilon=epsilon)
            print('fitting accepted parameters...')
            self.fit.fit_accepted(epsilon=epsilon)
            self.best_parameters = self.fit.best_parameters

        # self.features['fitted_limb'] = self.cost_function.evaluate(self.best_parameters)
        y = self.fit.forward_simulate(pack_parameters(self.best_parameters, self.free_parameters), 1)[0][0]
        self.features['fitted_limb'] = y
        self._plot_functions['fitted_limb'] = plot_limb
