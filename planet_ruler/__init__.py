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
from . import camera
from . import validation
from . import uncertainty
from . import dashboard

# Main classes for user-facing API
from .observation import LimbObservation, PlanetObservation
from .annotate import TkLimbAnnotator
from .dashboard import OutputCapture

__all__ = [
    # Modules
    "demo",
    "fit",
    "geometry",
    "image",
    "observation",
    "plot",
    "annotate",
    "camera",
    "validation",
    "uncertainty",
    "dashboard",
    # Main classes
    "LimbObservation",
    "PlanetObservation",
    "TkLimbAnnotator",
    "OutputCapture",
]

# Version information
__version__ = "1.7.0"
