"""
Mathematical tools for dit.
"""

# Global random number generator
import numpy as np

from . import aitchison, combinatorics, pmfops
from .fraction import approximate_fraction
from .misc import *
from .ops import LinearOperations, LogOperations, get_ops
from .pmfops import perturb_support as perturb_support_pmf
from .sampling import _sample, _samples, ball, norm, sample, sample_simplex
from .sigmaalgebra import atom_set, is_sigma_algebra, sigma_algebra

prng = np.random.RandomState()


def as_random_state(rng=None):
    """
    Coerce `rng` into a ``numpy.random.RandomState``.

    Parameters
    ----------
    rng : np.random.Generator, np.random.RandomState, int, None
        The source of randomness. None means the global ``dit.math.prng``; a
        Generator is wrapped so that both share the same bit generator.

    Returns
    -------
    rng : np.random.RandomState
        A RandomState drawing from `rng`.
    """
    import numpy as np

    if rng is None:
        return prng
    if isinstance(rng, np.random.RandomState):
        return rng
    if isinstance(rng, np.random.Generator):
        return np.random.RandomState(rng.bit_generator)
    return np.random.RandomState(rng)


# Set the error level to ignore...for example: log2(0).
np.seterr(all="ignore")
del np
