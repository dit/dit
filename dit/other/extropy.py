"""
The extropy
"""

from ..helpers import RV_MODES
from ..math.ops import get_ops

import numpy as np

def extropy(dist, rvs=None, rv_mode=None):
    """
    Returns the extropy J[X] over the random variables in `rvs`.

    If the distribution represents linear probabilities, then the extropy
    is calculated with units of 'bits' (base-2).

    Parameters
    ----------
    dist : Distribution or float
        The distribution from which the extropy is calculated. If a float,
        then we calculate the binary extropy.
    rvs : list, None
        The indexes of the random variable used to calculate the extropy.
        If None, then the extropy is calculated over all random variables.
        This should remain `None` for ScalarDistributions.
    rv_mode : str, None
        Specifies how to interpret the elements of `rvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `rvs` are interpreted as random variable indices. If equal to 'names',
        the the elements are interpreted as random variable names. If `None`,
        then the value of `dist._rv_mode` is consulted.

    Returns
    -------
    J : float
        The extropy of the distribution.

    """
    try:
        # Handle binary extropy.
        float(dist)
    except TypeError:
        pass
    else:
        # Assume linear probability for binary extropy.
        import dit
        dist = dit.ScalarDistribution([dist, 1-dist])
        rvs = None
        rv_mode = RV_MODES.INDICES

    if dist.is_joint():
        if rvs is None:
            # Set to entropy of entire distribution
            rvs = list(range(dist.outcome_length()))
            rv_mode = RV_MODES.INDICES

        d = dist.marginal(rvs, rv_mode=rv_mode)
    else:
        d = dist

    pmf = d.pmf
    if d.is_log():
        base = d.get_base(numerical=True)
        npmf = d.ops.log(1-d.ops.exp(pmf))
        terms = -base**npmf * npmf
    else:
        # Calculate entropy in bits.
        log = get_ops(2).log
        npmf = 1 - pmf
        terms = -npmf * log(npmf)

    J = np.nansum(terms)
    return J
