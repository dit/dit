"""
The extropy
"""

from ..math import LogOperations

import numpy as np

def extropy(dist, rvs=None, rv_names=None):
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
    rv_names : bool
        If `True`, then the elements of `rvs` are treated as random variable
        names. If `False`, then the elements of `rvs` are treated as random
        variable indexes.  If `None`, then the value `True` is used if the
        distribution has specified names for its random variables.

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
        rv_names = False

    if dist.is_joint():
        if rvs is None:
            # Set to entropy of entire distribution
            rvs = list(range(dist.outcome_length()))
            rv_names = False

        d = dist.marginal(rvs, rv_names=rv_names)
    else:
        d = dist

    pmf = d.pmf
    if d.is_log():
        base = d.get_base(numerical=True)
        npmf = d.ops.log(1-d.ops.exp(pmf))
        terms = -base**npmf * npmf
    else:
        # Calculate entropy in bits.
        log = LogOperations(2).log
        npmf = 1 - pmf
        terms = -npmf * log(npmf)

    J = np.nansum(terms)
    return J
