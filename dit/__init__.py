"""
dit is a Python package for information theory on discrete random variables.

d = discrete
i = information
t = theory

However, the more precise statement (at this point) is that `dit` is a
Python package for sigma-algebras defined on finite sets.

"""

__version__ = "1.5"

from loguru import logger as _logger

from .algorithms import expanded_samplespace, pruned_samplespace
from .bgm import *
from .cdisthelpers import joint_from_factors
from .distconst import *
from .distribution import Distribution
from .helpers import copypmf
from .params import ditParams
from .samplespace import CartesianProduct, SampleSpace, ScalarSampleSpace

_logger.disable("dit")

import dit.algorithms  # noqa: E402
import dit.channelorder  # noqa: E402
import dit.divergences  # noqa: E402
import dit.example_dists  # noqa: E402
import dit.inference  # noqa: E402
import dit.multivariate  # noqa: E402
import dit.other  # noqa: E402
import dit.pid  # noqa: E402
import dit.profiles  # noqa: E402
import dit.shannon  # noqa: E402
