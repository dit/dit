"""
Utilities for extracting and normalizing channel matrices from various
representations used throughout dit.
"""

import numpy as np

from ..cdisthelpers import cdist_array
from ..exceptions import ditException

__all__ = (
    "channel_matrix",
    "channels_from_joint",
    "compose_channels",
    "reverse_channels_from_joint",
)


def channel_matrix(channel):
    """
    Convert a channel representation to a 2D numpy stochastic matrix.

    Rows correspond to input values, columns to output values, so
    ``mat[s, y]`` is the probability of output *y* given input *s*.

    Parameters
    ----------
    channel : ndarray, list of Distribution, or Distribution
        One of:
        - A 2D numpy array (returned as-is after validation).
        - A list of conditional distributions ``[P(Y|X=x) for x in X]``.
        - A conditional ``Distribution`` object (``is_conditional() == True``).

    Returns
    -------
    mat : ndarray, shape (n_inputs, n_outputs)
        Row-stochastic matrix.
    """
    from ..distribution import Distribution

    arr = np.asarray(channel, dtype=float) if not isinstance(channel, (list, Distribution)) else None

    if arr is not None and arr.ndim == 2:
        return arr

    if isinstance(channel, Distribution):
        if not channel.is_conditional():
            raise ditException("Distribution passed as channel must be conditional")
        lin = channel._linear_data()
        given_dims = [d for d in channel.dims if d in channel.given_vars]
        free_dims = [d for d in channel.dims if d in channel.free_vars]
        reordered = lin.transpose(*given_dims, *free_dims)
        n_given = int(np.prod([len(channel.data.coords[d]) for d in given_dims]))
        n_free = int(np.prod([len(channel.data.coords[d]) for d in free_dims]))
        return reordered.values.reshape(n_given, n_free)

    if isinstance(channel, list):
        return cdist_array(channel, base="linear", mode="dense")

    raise ditException("channel must be a 2D array, a list of distributions, or a conditional Distribution")


def channels_from_joint(dist, S, Y, Z):
    """
    Extract forward channels and the input marginal from a joint distribution.

    Given a joint distribution over ``(S, Y, Z)``, returns the channel
    matrices ``P(Y|S)`` (kappa) and ``P(Z|S)`` (mu), together with the
    marginal ``P(S)``.

    Parameters
    ----------
    dist : Distribution
        A joint distribution.
    S : list
        Indices or names of the input variable(s).
    Y : list
        Indices or names of the first output variable(s).
    Z : list
        Indices or names of the second output variable(s).

    Returns
    -------
    kappa : ndarray
        Channel matrix ``P(Y|S)``, shape ``(|S|, |Y|)``.
    mu : ndarray
        Channel matrix ``P(Z|S)``, shape ``(|S|, |Z|)``.
    pi_s : ndarray
        Marginal pmf ``P(S)``.
    """
    S = list(S)
    Y = list(Y)
    Z = list(Z)

    sub_sy = dist.marginal(*(S + Y))
    marg_s, cdists_y = sub_sy.condition_on(S, rvs=Y)

    sub_sz = dist.marginal(*(S + Z))
    _, cdists_z = sub_sz.condition_on(S, rvs=Z)

    kappa = cdist_array(cdists_y, base="linear", mode="dense")
    mu = cdist_array(cdists_z, base="linear", mode="dense")
    pi_s = np.array(marg_s.pmf, dtype=float)

    return kappa, mu, pi_s


def reverse_channels_from_joint(dist, S, Y, Z):
    """
    Extract Bayes-inverse channels and output marginals from a joint.

    Given a joint distribution over ``(S, Y, Z)``, returns the reverse
    channel matrices ``P(S|Y)`` (kappa_bar) and ``P(S|Z)`` (mu_bar),
    together with the marginals ``P(Y)`` and ``P(Z)``.

    Parameters
    ----------
    dist : Distribution
        A joint distribution.
    S : list
        Indices or names of the target variable(s).
    Y : list
        Indices or names of the first predictor variable(s).
    Z : list
        Indices or names of the second predictor variable(s).

    Returns
    -------
    kappa_bar : ndarray
        Reverse channel ``P(S|Y)``, shape ``(|Y|, |S|)``.
    mu_bar : ndarray
        Reverse channel ``P(S|Z)``, shape ``(|Z|, |S|)``.
    pi_y : ndarray
        Marginal pmf ``P(Y)``.
    pi_z : ndarray
        Marginal pmf ``P(Z)``.
    """
    S = list(S)
    Y = list(Y)
    Z = list(Z)

    sub_sy = dist.marginal(*(S + Y))
    marg_y, cdists_s_given_y = sub_sy.condition_on(Y, rvs=S)

    sub_sz = dist.marginal(*(S + Z))
    marg_z, cdists_s_given_z = sub_sz.condition_on(Z, rvs=S)

    kappa_bar = cdist_array(cdists_s_given_y, base="linear", mode="dense")
    mu_bar = cdist_array(cdists_s_given_z, base="linear", mode="dense")
    pi_y = np.array(marg_y.pmf, dtype=float)
    pi_z = np.array(marg_z.pmf, dtype=float)

    return kappa_bar, mu_bar, pi_y, pi_z


def compose_channels(post, pre):
    """
    Compose two channels: ``(post ∘ pre)_s(y) = Σ_z post(y|z) pre(z|s)``.

    Parameters
    ----------
    post : ndarray, shape (n_z, n_y)
        The post-processing channel ``P(Y|Z)``.
    pre : ndarray, shape (n_s, n_z)
        The channel being post-processed ``P(Z|S)``.

    Returns
    -------
    composed : ndarray, shape (n_s, n_y)
        The composed channel ``pre @ post``, i.e. ``P(Y|S)``.
    """
    return pre @ post
