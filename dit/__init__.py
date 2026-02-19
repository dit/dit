"""
dit is a Python package for information theory on discrete random variables.

d = discrete
i = information
t = theory

However, the more precise statement (at this point) is that `dit` is a
Python package for sigma-algebras defined on finite sets.

"""

__version__ = "1.5"

# Order is important!
from loguru import logger as _logger

from .algorithms import expanded_samplespace, pruned_samplespace
from .bgm import *
from .cdisthelpers import joint_from_factors
from .distconst import *
from .helpers import copypmf
from .npdist import Distribution
from .npscalardist import ScalarDistribution
from .params import ditParams

# Order does not matter for these
from .samplespace import CartesianProduct, SampleSpace, ScalarSampleSpace

_logger.disable("dit")

import dit.algorithms
import dit.divergences
import dit.example_dists
import dit.inference
import dit.multivariate
import dit.other
import dit.pid
import dit.profiles
import dit.shannon
