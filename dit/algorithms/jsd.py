"""
The Jensen-Shannon Diverence.

This is a reasonable measure of distinguishablity between distribution.
"""

from __future__ import division

import numpy as np
from six.moves import zip # pylint: disable=redefined-builtin

from ..distconst import mixture_distribution
from .shannon import entropy as H

def jensen_shannon_divergence(dists, weights=None):
    """
    The Jensen-Shannon Divergence: H(sum(w_i*P_i)) - sum(w_i*H(P_i)).

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
        weights = np.array([ 1/len(dists) ] * len(dists))

    # validation of `weights` is done in mixture_distribution,
    # so we don't need to worry about it for the second part.
    one = H(mixture_distribution(dists, weights, merge=True))
    two = sum(w*H(d) for w, d in zip(weights, dists))
    jsd = one - two
    return jsd
