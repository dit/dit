"""
Module for basic inference tools.

"""
from .binning import binned
from .estimators import entropy_0, entropy_1, entropy_2
try:
    from .pycounts import *
except ImportError:
    pass
