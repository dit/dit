"""
Module for basic inference tools.

"""
from .binning import binned
try:
    from .pycounts import *
except ImportError:
    pass
