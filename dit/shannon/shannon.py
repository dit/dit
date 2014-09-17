"""
Some basic Shannon information quantities.

This will be replaced with something better later.

"""

from ..math import LogOperations

import numpy as np

def entropy_pmf(pmf):
    pmf = np.asarray(pmf)
    return np.nansum(-pmf * np.log2(pmf))

def entropy(dist, rvs=None, rv_names=None):
    """
    Returns the entropy H[X] over the random variables in `rvs`.

    If the distribution represents linear probabilities, then the entropy
    is calculated with units of 'bits' (base-2).

    Parameters
    ----------
    dist : Distribution or float
        The distribution from which the entropy is calculated. If a float,
        then we calculate the binary entropy.
    rvs : list, None
        The indexes of the random variable used to calculate the entropy.
        If None, then the entropy is calculated over all random variables.
        This should remain `None` for ScalarDistributions.
    rv_names : bool
        If `True`, then the elements of `rvs` are treated as random variable
        names. If `False`, then the elements of `rvs` are treated as random
        variable indexes.  If `None`, then the value `True` is used if the
        distribution has specified names for its random variables.

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
        rvs = None
        rv_names = False

    if dist.is_joint():
        if rvs is None:
            # Set to entropy of entire distribution
            rvs = range(dist.outcome_length())
            rv_names = False

        d = dist.marginal(rvs, rv_names=rv_names)
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

def conditional_entropy(dist, rvs_X, rvs_Y, rv_names=None):
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
    rv_names : bool
        If `True`, then the elements of `rvs_X` and `rvs_Y` are treated as
        random variable names. If `False`, then their elements are treated as
        random variable indexes.  If `None`, then the value `True` is used if
        the distribution has specified names for its random variables.

    Returns
    -------
    H_XgY : float
        The conditional entropy H[X|Y].

    """
    if set(rvs_X).issubset(rvs_Y):
        # This is not necessary, but it makes the answer *exactly* zero,
        # instead of 1e-12 or something smaller.
        return 0.0

    MI_XY = mutual_information(dist, rvs_Y, rvs_X, rv_names)
    H_X = entropy(dist, rvs_X, rv_names)
    H_XgY = H_X - MI_XY
    return H_XgY

def mutual_information(dist, rvs_X, rvs_Y, rv_names=None):
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
    rv_names : bool
        If `True`, then the elements of `rvs_X` and `rvs_Y` are treated as
        random variable names. If `False`, then their elements are treated as
        random variable indexes.  If `None`, then the value `True` is used if
        the distribution has specified names for its random variables.

    Returns
    -------
    I : float
        The mutual information I[X:Y].

    """
    H_X = entropy(dist, rvs_X, rv_names)
    H_Y = entropy(dist, rvs_Y, rv_names)
    # Make sure to union the indexes. This handles the case when X and Y
    # do not partition the set of all indexes.
    H_XY = entropy(dist, set(rvs_X) | set(rvs_Y), rv_names)
    I = H_X + H_Y - H_XY
    return I
