"""
Planet Ruler: A package for measuring planetary radii from limb-fitting observations.
"""

from . import demo
from . import fit
from . import geometry
from . import image
from . import observation
from . import plot
from . import annotate

# Main classes for user-facing API
from .observation import LimbObservation, PlanetObservation
from .annotate import TkLimbAnnotator

__all__ = [
    # Modules
    "demo",
    "fit",
    "geometry",
    "image",
    "observation",
    "plot",
    "annotate",
    # Main classes
    "LimbObservation",
    "PlanetObservation",
    "TkLimbAnnotator",
]

# Version information
__version__ = "1.2.0"
