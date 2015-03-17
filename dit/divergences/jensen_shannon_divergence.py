"""
The Jensen-Shannon Diverence.

This is a reasonable measure of distinguishablity between distribution.
"""

from __future__ import division

import numpy as np
from six.moves import zip # pylint: disable=redefined-builtin,import-error

import dit
from ..distconst import mixture_distribution
from ..shannon import entropy as H, entropy_pmf as H_pmf

__all__ = ('jensen_shannon_divergence',
           'jensen_shannon_divergence_pmf',
          )

def jensen_shannon_divergence_pmf(pmfs, weights=None):
    """
    The Jensen-Shannon Divergence: H(sum(w_i*P_i)) - sum(w_i*H(P_i)).

    The square root of the Jensen-Shannon divergence is a distance metric.

    Assumption: Linearly distributed probabilities.

    Parameters
    ----------
    pmfs : NumPy array, shape (n,k)
        The `n` distributions, each of length `k` that will be mixed.
    weights : NumPy array, shape (n,)
        The weights applied to each pmf. This array will be normalized
        automatically. If None, each pmf is weighted equally.

    Returns
    -------
    jsd: float
        The Jensen-Shannon Divergence

    """
    pmfs = np.atleast_2d(pmfs)
    if weights is None:
        weights = np.ones(pmfs.shape[0], dtype=float) / pmfs.shape[0]
    else:
        if len(weights) != len(pmfs):
            msg = "number of weights != number of pmfs"
            raise dit.exceptions.ditException(msg)
        weights = np.asarray(weights, dtype=float)
        weights /= weights.sum()

    mixture = dit.math.pmfops.convex_combination(pmfs, weights)
    one = H_pmf(mixture)
    entropies = np.apply_along_axis(H_pmf, 1, pmfs)
    two = (entropies * weights).sum()
    return one - two

def jensen_shannon_divergence(dists, weights=None):
    """
    The Jensen-Shannon Divergence: H(sum(w_i*P_i)) - sum(w_i*H(P_i)).

    The square root of the Jensen-Shannon divergence is a distance metric.

    Parameters
    ----------
    dists: [Distribution]
        The distributions, P_i, to take the Jensen-Shannon Divergence of.

    weights: [float], None
        The weights, w_i, to give the distributions. If None, the weights are
        assumed to be uniform.

    Returns
    -------
    jsd: float
        The Jensen-Shannon Divergence

    Raises
    ------
    ditException
        Raised if there `dists` and `weights` have unequal lengths.
    InvalidNormalization
        Raised if the weights do not sum to unity.
    InvalidProbability
        Raised if the weights are not valid probabilities.
    """
    if weights is None:
        weights = np.array([1/len(dists)] * len(dists))
    else:
        if hasattr(weights, 'pmf'):
            m = 'Likely user error. Second argument to JSD should be weights.'
            raise dit.exceptions.ditException(m)

    # validation of `weights` is done in mixture_distribution,
    # so we don't need to worry about it for the second part.
    mixture = mixture_distribution(dists, weights, merge=True)
    one = H(mixture)
    two = sum(w*H(d) for w, d in zip(weights, dists))
    jsd = one - two
    return jsd
