"""
The Jensen-Shannon Diverence.

This is a reasonable measure of distinguishablity between distribution.
"""

from __future__ import division

import numpy as np

from ..exceptions import ditException
from ..distconst import mixture_distribution
from .shannon import entropy as H

def jensen_shannon_divergence(dists, weights=None):
    """
    The Jensen-Shannon Divergence: H( sum(w_i*P_i) ) - sum(w_i*H(P_i)).

    Parameters
    ----------
    dists: [Distribution]
        The distributions, P_i, to take the Jensen-Shannon Divergence of.

    weights: [float], None
        The weights, w_i, to give the distributions. If None, the weights are assumed to be uniform.

    Returns
    -------
    jsd: float
        The Jensen-Shannon Divergence

    Raises
    ------
    DitException
        Raised if there `dists` and `weights` have unequal lengths.
    InvalidNormalization
        Raised if the weights do not sum to unity.
    InvalidProbability
        Raised if the weights are not valid probabilities.
    """
    if weights is None:
        weights = [ 1/len(dists) ] * len(dists)

    weights = np.asarray(weights)
    if len(dists) != len(weights):
        msg = "Length of `dists` and `weights` must be equal."
        raise ditException(msg)

    return H(mixture_distribution(dists, weights, merge=True)) - sum(w*H(d) for w, d in zip(weights, dists))