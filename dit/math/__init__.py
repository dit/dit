# -*- coding: utf-8 -*-

"""
Mathematical tools for dit.
"""

# Global random number generator
import numpy as np
prng = np.random.RandomState()
# Set the error level to ignore...for example: log2(0).
np.seterr(all='ignore')
del np

from .equal import close, allclose
from .sampling import sample, _sample, _samples, ball, norm, sample_simplex
from .ops import get_ops, LinearOperations, LogOperations
from .fraction import approximate_fraction
from .misc import *
from .sigmaalgebra import sigma_algebra, is_sigma_algebra, atom_set

from . import pmfops
from .pmfops import perturb_support as perturb_support_pmf

from . import aitchison
from . import combinatorics
