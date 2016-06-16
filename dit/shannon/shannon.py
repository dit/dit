"""
Some basic Shannon information quantities.

"""

from ..math import LogOperations
from ..helpers import RV_MODES

import numpy as np

def entropy_pmf(pmf):
    """
    Returns the entropy of the probability mass function.

    Assumption: Linearly distributed probabilities.

    Parameters
    ----------
    pmf : NumPy array, shape (k,) or (n,k)
        Returns the entropy over the last index.

    """
    pmf = np.asarray(pmf)
    return np.nansum(-pmf * np.log2(pmf), axis=-1)

def entropy(dist, rvs=None, rv_mode=None):
    """
    Returns the entropy H[X] over the random variables in `rvs`.

    If the distribution represents linear probabilities, then the entropy
    is calculated with units of 'bits' (base-2). Otherwise, the entropy is
    calculated in whatever base that matches the distribution's pmf.

    Parameters
    ----------
    dist : Distribution or float
        The distribution from which the entropy is calculated. If a float,
        then we calculate the binary entropy.
    rvs : list, None
        The indexes of the random variable used to calculate the entropy.
        If None, then the entropy is calculated over all random variables.
        This should remain `None` for ScalarDistributions.
    rv_mode : str, None
        Specifies how to interpret the elements of `rvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `rvs` are interpreted as random variable indices. If equal to 'names',
        the the elements are interpreted as random variable names. If `None`,
        then the value of `dist._rv_mode` is consulted.

    Returns
    -------
    H : float
        The entropy of the distribution.

    """
    try:
        # Handle binary entropy.
        float(dist)
    except TypeError:
        pass
    else:
        # Assume linear probability for binary entropy.
        import dit
        dist = dit.ScalarDistribution([dist, 1-dist])

    if dist.is_joint():
        if rvs is None:
            # Set to entropy of entire distribution
            rvs = range(dist.outcome_length()) # pylint: disable=no-member
            rv_mode = RV_MODES.INDICES

        d = dist.marginal(rvs, rv_mode=rv_mode) # pylint: disable=no-member
    else:
        d = dist

    pmf = d.pmf
    if d.is_log():
        base = d.get_base(numerical=True)
        terms = -base**pmf * pmf
    else:
        # Calculate entropy in bits.
        log = LogOperations(2).log
        terms = -pmf * log(pmf)

    H = np.nansum(terms)
    return H

def conditional_entropy(dist, rvs_X, rvs_Y, rv_mode=None):
    """
    Returns the conditional entropy of H[X|Y].

    If the distribution represents linear probabilities, then the entropy
    is calculated with units of 'bits' (base-2).

    Parameters
    ----------
    dist : Distribution
        The distribution from which the conditional entropy is calculated.
    rvs_X : list, None
        The indexes of the random variables defining X.
    rvs_Y : list, None
        The indexes of the random variables defining Y.
    rv_mode : str, None
        Specifies how to interpret the elements of `rvs_X` and `rvs_Y`. Valid
        options are: {'indices', 'names'}. If equal to 'indices', then the
        elements of `rvs_X` and `rvs_Y` are interpreted as random variable
        indices. If equal to 'names', the the elements are interpreted as
        random variable names. If `None`, then the value of `dist._rv_mode`
        is consulted.

    Returns
    -------
    H_XgY : float
        The conditional entropy H[X|Y].

    """
    if set(rvs_X).issubset(rvs_Y):
        # This is not necessary, but it makes the answer *exactly* zero,
        # instead of 1e-12 or something smaller.
        return 0.0

    MI_XY = mutual_information(dist, rvs_X, rvs_Y, rv_mode=rv_mode)
    H_X = entropy(dist, rvs_X, rv_mode=rv_mode)
    H_XgY = H_X - MI_XY
    return H_XgY

def mutual_information(dist, rvs_X, rvs_Y, rv_mode=None):
    """
    Returns the mutual information I[X:Y].

    If the distribution represents linear probabilities, then the entropy
    is calculated with units of 'bits' (base-2).

    Parameters
    ----------
    dist : Distribution
        The distribution from which the mutual information is calculated.
    rvs_X : list, None
        The indexes of the random variables defining X.
    rvs_Y : list, None
        The indexes of the random variables defining Y.
    rv_mode : str, None
        Specifies how to interpret the elements of `rvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `rvs` are interpreted as random variable indices. If equal to 'names',
        the the elements are interpreted as random variable names. If `None`,
        then the value of `dist._rv_mode` is consulted.

    Returns
    -------
    I : float
        The mutual information I[X:Y].

    """
    H_X = entropy(dist, rvs_X, rv_mode=rv_mode)
    H_Y = entropy(dist, rvs_Y, rv_mode=rv_mode)
    # Make sure to union the indexes. This handles the case when X and Y
    # do not partition the set of all indexes.
    H_XY = entropy(dist, set(rvs_X) | set(rvs_Y), rv_mode=rv_mode)
    I = H_X + H_Y - H_XY
    return I
