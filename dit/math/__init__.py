"""
Mathematical tools for dit.
"""

# Global random number generator
import numpy as np

from . import aitchison, combinatorics, pmfops
from .equal import allclose, close
from .fraction import approximate_fraction
from .misc import *
from .ops import LinearOperations, LogOperations, get_ops
from .pmfops import perturb_support as perturb_support_pmf
from .sampling import _sample, _samples, ball, norm, sample, sample_simplex
from .sigmaalgebra import atom_set, is_sigma_algebra, sigma_algebra

prng = np.random.RandomState()
# Set the error level to ignore...for example: log2(0).
np.seterr(all='ignore')
del np
