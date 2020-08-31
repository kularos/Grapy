# This Module extends matplotlib's scalar-to-color mapping functionality into arbitrarily high dimensional spaces

from . import Animation, Color, Mapping

from .Animation import *
from .Color import *
from .Mapping import *

Mapping.directory_path = __name__ + '/dir/'
