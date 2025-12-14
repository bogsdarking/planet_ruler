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

import json
import ipywidgets as widgets
from IPython.display import display, Markdown


def make_dropdown():
    demo = widgets.Dropdown(
        options=[("Pluto", 1), ("Saturn-1", 2), ("Saturn-2", 3), ("Earth", 4)],
        value=1,
        description="Demo:",
    )
    return demo


def load_demo_parameters(demo):
    if demo.value == 1:
        demo_parameters = {
            "target": "Pluto",
            "true_radius": 1188,
            "image_filepath": "../../demo/images/PIA19948.jpg",
            "fit_config": "../../config/pluto-new-horizons.yaml",
            "limb_config": {
                "log": False,
                "y_min": 0,
                "y_max": -1,
                "window_length": 501,
                "polyorder": 1,
                "deriv": 0,
                "delta": 1,
                "segmenter": "segment-anything",
            },
            "limb_save": "pluto_limb.npy",
            "parameter_walkthrough": "../../demo/pluto_init.md",
            "preamble": "../../demo/pluto_preamble.md",
        }
    elif demo.value == 2:
        demo_parameters = {
            "target": "Saturn",
            "true_radius": 58232,
            "image_filepath": "../../demo/images/PIA21341.jpg",
            "fit_config": "../../config/saturn-cassini-1.yaml",
            "limb_config": {
                "log": False,
                "y_min": 0,
                "y_max": -1,
                "window_length": 501,
                "polyorder": 1,
                "deriv": 0,
                "delta": 1,
                "segmenter": "segment-anything",
            },
            "limb_save": "saturn_limb_1.npy",
            "parameter_walkthrough": "../../demo/saturn_init_1.md",
            "preamble": "../../demo/saturn_preamble_1.md",
        }
    elif demo.value == 3:
        demo_parameters = {
            "target": "Saturn",
            "true_radius": 58232,
            "image_filepath": "../../demo/images/saturn_ciclops_5769_13427_1.jpg",
            "fit_config": "../../config/saturn-cassini-2.yaml",
            "limb_config": {
                "log": False,
                "y_min": 0,
                "y_max": -1,
                "window_length": 501,
                "polyorder": 1,
                "deriv": 0,
                "delta": 1,
                "segmenter": "segment-anything",
            },
            "limb_save": "saturn_limb_2.npy",
            "parameter_walkthrough": "../../demo/saturn_init_2.md",
            "preamble": "../../demo/saturn_preamble_2.md",
        }
    elif demo.value == 4:
        demo_parameters = {
            "target": "Earth",
            "true_radius": 6371,
            "image_filepath": "../../demo/images/50644513538_56228a2027_o.jpg",
            "fit_config": "../../config/earth_iss_1.yaml",
            "limb_config": {
                "log": False,
                "y_min": 0,
                "y_max": -1,
                "window_length": 501,
                "polyorder": 1,
                "deriv": 0,
                "delta": 1,
                "segmenter": "segment-anything",
            },
            "limb_save": "earth_limb_1.npy",
            "parameter_walkthrough": "../../demo/earth_init_1.md",
            "preamble": "../../demo/earth_preamble_1.md",
        }
    else:
        demo_parameters = None
    return demo_parameters


def display_text(filepath):
    with open(filepath, "r") as f:
        display(Markdown(f.read()))
