from scipy.optimize import differential_evolution, shgo, minimize
import yaml
import numpy as np
import logging
import matplotlib.pyplot as plt
from planet_ruler.plot import plot_image, plot_limb
from planet_ruler.image import load_image, gradient_break, StringDrop, smooth_limb, fill_nans
from planet_ruler.fit import CostFunction, unpack_parameters, pack_parameters
from planet_ruler.geometry import limb_arc


class PlanetObservation:
    def __init__(self, image_filepath):
        self.raw_image = load_image(image_filepath)
        self.image = self.raw_image.copy()
        self.features = {}
        self._plot_functions = {}
        self._cwheel = ['y', 'b', 'r', 'orange', 'pink', 'black']

    def plot(self, gradient=False, show=True):
        plot_image(self.image, gradient=gradient, show=False)
        for i, feature in enumerate(self.features):
            self._plot_functions[feature](self.features[feature], show=False, c=self._cwheel[i])
        if show:
            plt.show()

    def restrict_image(self, xmin=0, ymin=0, xmax=-1, ymax=-1):
        self.image = self.raw_image[ymin:ymax, xmin:xmax]


class LimbObservation(PlanetObservation):
    def __init__(self, image_filepath,
                 fit_config,
                 model=limb_arc,
                 limb_detection='string-drop',
                 minimizer=differential_evolution):
        super().__init__(image_filepath)

        self.free_parameters = None
        self.init_parameter_values = None
        self.parameter_limits = None
        self.load_fit_config(fit_config)
        assert limb_detection in ['gradient-break', 'string-drop']
        self.limb_detection = limb_detection
        self._string_drop = None
        self.minimizer = minimizer  # todo awkward to have partially implemented choice of minimizer
        self.model = model

        self._raw_limb = None
        self.cost_function = None
        self.fit = None
        self.best_parameters = None
        self.fit_results = None

    def load_fit_config(self, fit_config):
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

    def detect_limb(self, **kwargs):
        if self.limb_detection == 'string-drop':
            if self._string_drop is None:
                self._string_drop = StringDrop(self.image)
            print('computing gradient force map')
            self._string_drop.compute_force_map(tilt=0.05, smoothing_window=50)
            print('dropping horizon string')
            self.features['limb'] = self._string_drop.drop_string(**kwargs)
        elif self.limb_detection == 'gradient-break':
            self.features['limb'] = gradient_break(self.image, **kwargs)

        self._raw_limb = self.features['limb'].copy()
        self._plot_functions['limb'] = plot_limb

    def smooth_limb(self, fill_nan=True, **kwargs):
        self.features['limb'] = smooth_limb(self._raw_limb, **kwargs)
        if fill_nan:
            logging.info("Filling NaNs in fitted limb.")
            self.features['limb'] = fill_nans(self.features['limb'])

    def fit_limb(self,
                 init_parameters: dict = None,
                 l2: bool = True,
                 max_iter: int = 1000):
        if init_parameters is None:
            init_parameters = [self.init_parameter_values[key] for key in self.free_parameters]

        self.cost_function = CostFunction(self.features['limb'], self.model,
                                          self.free_parameters,
                                          self.init_parameter_values,
                                          l2=l2)
        # 'rand1exp' and  Best/2 supposed to be good?  best2bin best2exp?  rand1bin?
        strategy = 'best2bin'
        # strategy = 'rand1exp'
        results = self.minimizer(self.cost_function.cost,
                                 [self.parameter_limits[key] for key in self.free_parameters],
                                 workers=4, maxiter=max_iter, strategy=strategy,
                                 polish=True,
                                 init='sobol', # halton is another reasonable option
                                 mutation=[0.1, 1.9],
                                 updating='deferred', disp=True, x0=init_parameters)
        self.fit_results = results
        self.best_parameters = unpack_parameters(results.x, self.free_parameters)

        self.features['fitted_limb'] = self.cost_function.evaluate(self.best_parameters)
        self._plot_functions['fitted_limb'] = plot_limb

    def save_limb(self,
                  filepath: str):
        np.save(filepath, self.features['limb'])

    def load_limb(self,
                  filepath: str):

        self.features['limb'] = np.load(filepath)
        self.features['limb'] = fill_nans(self.features['limb'])
        self._plot_functions['limb'] = plot_limb
