"""
Channel helpers for the channel-coding evaluation layer.

A channel is a conditional :class:`~dit.Distribution` ``p(Y | X)``, matching what
:func:`dit.algorithms.channel_capacity` consumes. The erasure symbol of the binary
erasure channel is the integer ``2`` (so that all alphabets stay sortable).
"""

import itertools
from math import log, sqrt

import numpy as np

from ..exceptions import ditException

__all__ = (
    "binary_erasure_channel",
    "binary_symmetric_channel",
    "channel_arrays",
    "log_likelihoods",
)

ERASURE = 2


def binary_symmetric_channel(p):
    """
    The binary symmetric channel with crossover probability ``p``.

    Parameters
    ----------
    p : float
        The probability that a transmitted bit is flipped.

    Returns
    -------
    channel : Distribution
        The conditional distribution ``p(Y | X)`` over ``{0, 1}``.
    """
    from ..distribution import Distribution

    joint = Distribution(
        {(0, 0): 0.5 * (1 - p), (0, 1): 0.5 * p, (1, 0): 0.5 * p, (1, 1): 0.5 * (1 - p)},
        rv_names=["X", "Y"],
    )
    return joint.condition_on("X")


def binary_erasure_channel(epsilon):
    """
    The binary erasure channel with erasure probability ``epsilon``.

    Parameters
    ----------
    epsilon : float
        The probability that a transmitted bit is erased.

    Returns
    -------
    channel : Distribution
        The conditional distribution ``p(Y | X)`` with output alphabet
        ``{0, 1, 2}``, where ``2`` denotes an erasure.
    """
    from ..distribution import Distribution

    joint = Distribution(
        {
            (0, 0): 0.5 * (1 - epsilon),
            (0, ERASURE): 0.5 * epsilon,
            (1, 1): 0.5 * (1 - epsilon),
            (1, ERASURE): 0.5 * epsilon,
        },
        rv_names=["X", "Y"],
    )
    return joint.condition_on("X")


def channel_arrays(channel):
    """
    Extract ``(inputs, outputs, P)`` from a conditional distribution ``p(Y | X)``.

    Parameters
    ----------
    channel : Distribution
        A conditional distribution ``p(Y | X)``.

    Returns
    -------
    inputs : list
        The input alphabet.
    outputs : list
        The output alphabet.
    P : ndarray
        ``P[i, j] = p(Y = outputs[j] | X = inputs[i])``.
    """
    from ..distribution import Distribution

    if not (isinstance(channel, Distribution) and channel.is_conditional()):
        raise ditException("A channel must be a conditional Distribution p(Y | X).")

    lin = channel._linear_data()
    given_dims = [d for d in channel.dims if d in channel.given_vars]
    free_dims = [d for d in channel.dims if d in channel.free_vars]
    reordered = lin.transpose(*given_dims, *free_dims)
    in_coords = [channel.data.coords[d].values for d in given_dims]
    out_coords = [channel.data.coords[d].values for d in free_dims]
    n_in = int(np.prod([len(c) for c in in_coords]))
    n_out = int(np.prod([len(c) for c in out_coords]))
    P = reordered.values.reshape(n_in, n_out)

    def _native(v):
        return v.item() if hasattr(v, "item") else v

    def _combo(coords):
        result = []
        for combo in itertools.product(*coords):
            combo = tuple(_native(v) for v in combo)
            result.append(combo[0] if len(combo) == 1 else combo)
        return result

    return _combo(in_coords), _combo(out_coords), P


def log_likelihoods(channel):
    """
    Per-output-symbol log-likelihood ratios for a binary-input channel.

    Parameters
    ----------
    channel : Distribution
        A conditional distribution ``p(Y | X)`` with binary input ``{0, 1}``.

    Returns
    -------
    llr : dict
        A mapping ``output_symbol -> log p(y|0) / p(y|1)``, clipped to a finite
        range.
    """
    inputs, outputs, P = channel_arrays(channel)
    if list(inputs) != [0, 1]:
        raise ditException("Soft decoding requires a binary-input channel with inputs {0, 1}.")
    clip = 40.0
    llr = {}
    for j, y in enumerate(outputs):
        p0, p1 = P[0, j], P[1, j]
        if p0 <= 0 and p1 <= 0:
            value = 0.0
        elif p1 <= 0:
            value = clip
        elif p0 <= 0:
            value = -clip
        else:
            value = max(-clip, min(clip, log(p0) - log(p1)))
        llr[y] = value
    return llr


def bhattacharyya(channel):
    """
    The Bhattacharyya parameter of a binary-input channel.

    Parameters
    ----------
    channel : Distribution
        A conditional distribution ``p(Y | X)`` with binary input ``{0, 1}``.

    Returns
    -------
    Z : float
    """
    inputs, _, P = channel_arrays(channel)
    if list(inputs) != [0, 1]:
        raise ditException("The Bhattacharyya parameter requires a binary-input channel.")
    return float(sum(sqrt(P[0, j] * P[1, j]) for j in range(P.shape[1])))
