"""
The Jensen-Shannon Diverence.

This is a reasonable measure of distinguishablity between distribution.
"""

from __future__ import division

import numpy as np
from six.moves import zip  # pylint: disable=redefined-builtin,import-error

import dit
from ..exceptions import ditException
from ..distconst import mixture_distribution
from ..shannon import entropy as H, entropy_pmf as H_pmf
from ..utils import unitful

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
            raise ditException(msg)
        weights = np.asarray(weights, dtype=float)
        weights /= weights.sum()

    mixture = dit.math.pmfops.convex_combination(pmfs, weights)
    one = H_pmf(mixture)
    entropies = np.apply_along_axis(H_pmf, 1, pmfs)
    two = (entropies * weights).sum()
    return one - two


@unitful
def jensen_shannon_divergence(dists, weights=None):
    """
    The Jensen-Shannon Divergence: H(sum(w_i*P_i)) - sum(w_i*H(P_i)).

    The square root of the Jensen-Shannon divergence is a distance metric.

    Parameters
    ----------
    dists : [Distribution]
        The distributions, P_i, to take the Jensen-Shannon Divergence of.

    weights : [float], None
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


def jensen_divergence(func):
    """
    Construct a Jensen-Shannon-like divergence measure from `func`. In order for this
    resulting divergence to be non-negative, `func` must be convex.

    Parameters
    ----------
    func : function
        A convex function.

    Returns
    -------
    jensen_func_divergence : function
        The divergence based on `func`
    """
    @unitful
    def jensen_blank_divergence(dists, weights=None, *args, **kwargs):
        if weights is None:
            weights = np.array([1 / len(dists)] * len(dists))
        else:
            if hasattr(weights, 'pmf'):
                m = 'Likely user error. Second argument should be weights.'
                raise ditException(m)

        # validation of `weights` is done in mixture_distribution,
        # so we don't need to worry about it for the second part.
        mixture = mixture_distribution(dists, weights, merge=True)
        one = func(mixture, *args, **kwargs)
        two = sum(w * func(d, *args, **kwargs) for w, d in zip(weights, dists))
        jbd = one - two
        return jbd

    docstring = """
        The Jensen-{name} Divergence: {name}(sum(w_i*P_i)) - sum(w_i*{name}(P_i)).

        Parameters
        ----------
        dists : [Distribution]
            The distributions, P_i, to take the Jensen-{name} Divergence of.

        weights : [float], None
            The weights, w_i, to give the distributions. If None, the weights are
            assumed to be uniform.
        
        *args : 

        Returns
        -------
        j{init}d: float
            The Jensen-{name} Divergence

        Raises
        ------
        ditException
            Raised if there `dists` and `weights` have unequal lengths.
        InvalidNormalization
            Raised if the weights do not sum to unity.
        InvalidProbability
            Raised if the weights are not valid probabilities.
        """.format(name=func.__name__, init=func.__name__[0])

    jensen_blank_divergence.__doc__ = docstring

    return jensen_blank_divergence
