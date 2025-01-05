import json
import ipywidgets as widgets
from IPython.display import display, Markdown, Latex


def make_dropdown():
    demo = widgets.Dropdown(
        options=[('Pluto', 1), ('Saturn-1', 2), ('Saturn-2', 3), ('Earth', 4)],
        value=1,
        description='Demo:',
    )
    return demo


def load_demo_parameters(demo):
    if demo.value == 1:
        demo_parameters = {
            'target': 'Pluto',
            'true_radius': 1188,
            'image_filepath': '../demo/images/PIA19948.tif',
            'fit_config': '../config/pluto-new-horizons.yaml',
            'limb_config': json.load(open('../config/pluto_limb_1.json', 'r')),
            'limb_save': 'pluto_limb.npy',
            'parameter_walkthrough': '../demo/pluto_init.md',
            'preamble': '../demo/pluto_preamble.md'
        }
    elif demo.value == 2:
        demo_parameters = {
            'target': 'Saturn',
            'true_radius': 58232,
            'image_filepath': '../demo/images/PIA21341.jpg',
            'fit_config': '../config/saturn-cassini-1.yaml',
            'limb_config': json.load(open('../config/saturn_limb_1.json', 'r')),
            'limb_save': 'saturn_limb_1.npy',
            'parameter_walkthrough': '../demo/saturn_init_1.md',
            'preamble': '../demo/saturn_preamble_1.md'
        }
    elif demo.value == 3:
        demo_parameters = {
            'target': 'Saturn',
            'true_radius': 58232,
            'image_filepath': '../demo/images/saturn_ciclops_5769_13427_1.jpg',
            'fit_config': '../config/saturn-cassini-2.yaml',
            'limb_config': json.load(open('../config/saturn_limb_2.json', 'r')),
            'limb_save': 'saturn_limb_2.npy',
            'parameter_walkthrough': '../demo/saturn_init_2.md',
            'preamble': '../demo/saturn_preamble_2.md'
        }
    else:
        demo_parameters = None
    return demo_parameters

def display_text(filepath):
    with open(filepath, 'r') as f:
        display(Markdown(f.read()))


