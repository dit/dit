"""
Channel helpers for the channel-coding evaluation layer.

The channel codes in :mod:`dit.coding` talk to a channel through a small
duck-typed protocol, so any object implementing it can be attached to a code:

- ``inputs``, ``outputs`` -- the input and output alphabets, as tuples;
- ``log_likelihood(codeword, received)`` -- :math:`\\log p(y^n \\mid x^n)`;
- ``sample(codeword, rng)`` -- one received word, as a tuple;
- ``symbol_log_likelihoods(received)`` -- an ``(n, len(inputs))`` array whose
  ``[i, a]`` entry is :math:`\\log p(y^n \\mid x_i = a)` up to a per-row
  constant. For a memoryless channel this is :math:`\\log p(y_i \\mid a)`; a
  channel with hidden state can compute it by the forward-backward (BCJR)
  recursions :cite:`Bahl1974` under uniform i.i.d. inputs;
- optionally ``capacity(**kwargs)`` -- a float, or a ``(lower, upper)`` bracket;
- optionally ``bhattacharyya()`` -- required only to construct a polar code;
- optionally ``feedback_initial()`` and ``feedback_step(belief, x)`` -- the
  channel one use at a time, required only by the feedback codes of
  :mod:`dit.coding.feedback`. ``belief`` summarizes everything the transmitter
  knows about the channel's hidden state from the inputs it sent and the
  outputs fed back (``None`` for a memoryless channel), and
  ``feedback_step`` returns every ``(y, p(y | x, belief), next_belief)``
  with nonzero probability;
- optionally ``feedback_capacity(**kwargs)`` -- the capacity with feedback,
  used by :meth:`~dit.coding.feedback.FeedbackCode.capacity_gap`. Without it
  ``capacity()`` is used, which is correct for memoryless channels since
  feedback does not increase their capacity :cite:`Cover2006`.

A conditional :class:`~dit.Distribution` ``p(Y | X)`` is wrapped as a
:class:`MemorylessChannel`. Any other object may either implement the protocol
directly or provide a ``__dit_channel__()`` method returning an object that does.
Concrete example channels (the binary symmetric channel, binary erasure
channel, ...) live in :mod:`dit.example_channels`.
"""

import itertools
from math import log, sqrt

import numpy as np

from ..exceptions import ditException
from ..utils.deprecation import renamed_kwarg

__all__ = (
    "MemorylessChannel",
    "as_channel",
    "channel_arrays",
    "log_likelihoods",
)

_LLR_CLIP = 40.0

_PROTOCOL = ("inputs", "outputs", "log_likelihood", "sample", "symbol_log_likelihoods")


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


class MemorylessChannel:
    """
    A discrete memoryless channel, applying ``p(Y | X)`` independently to each symbol.

    Parameters
    ----------
    dist : Distribution
        A conditional distribution ``p(Y | X)``.
    """

    def __init__(self, dist):
        inputs, outputs, P = channel_arrays(dist)
        self.dist = dist
        self.inputs = tuple(inputs)
        self.outputs = tuple(outputs)
        self.P = P
        self._in_index = {v: i for i, v in enumerate(self.inputs)}
        self._out_index = {v: j for j, v in enumerate(self.outputs)}

    def log_likelihood(self, codeword, received):
        """
        The log-probability ``log p(received | codeword)``.
        """
        score = 0.0
        for x, y in zip(codeword, received, strict=True):
            p = self.P[self._in_index[x], self._out_index[y]]
            if p <= 0:
                return -np.inf
            score += log(p)
        return score

    @renamed_kwarg("prng", "rng")
    def sample(self, codeword, rng):
        """
        Pass `codeword` through the channel once.
        """
        n_out = len(self.outputs)
        return tuple(self.outputs[rng.choice(n_out, p=self.P[self._in_index[x]])] for x in codeword)

    def symbol_log_likelihoods(self, received):
        """
        The ``(n, len(inputs))`` array of ``log p(y_i | x)``.
        """
        cols = [self._out_index[y] for y in received]
        with np.errstate(divide="ignore"):
            return np.log(self.P[:, cols].T)

    def feedback_initial(self):
        """
        The transmitter's initial belief about the channel; ``None``, since there is no state.
        """
        return None

    def feedback_step(self, belief, x):
        """
        The outcomes ``(y, p(y | x), None)`` of one channel use with input `x`.
        """
        row = self.P[self._in_index[x]]
        return [(y, float(p), None) for y, p in zip(self.outputs, row, strict=True) if p > 0]

    def capacity(self):
        """
        The channel capacity, in bits per channel use.
        """
        from ..algorithms import channel_capacity

        capacity, _ = channel_capacity(self.dist)
        return capacity

    def bhattacharyya(self):
        """
        The Bhattacharyya parameter of a binary-input channel.
        """
        if list(self.inputs) != [0, 1]:
            raise ditException("The Bhattacharyya parameter requires a binary-input channel.")
        return float(sum(sqrt(self.P[0, j] * self.P[1, j]) for j in range(self.P.shape[1])))


def as_channel(channel):
    """
    Coerce `channel` to an object implementing the channel protocol.

    Parameters
    ----------
    channel : Distribution or object
        A conditional distribution ``p(Y | X)``; an object with a
        ``__dit_channel__()`` method; or an object already implementing the
        channel protocol described in :mod:`dit.coding._channel`.

    Returns
    -------
    channel : object
        An object implementing the channel protocol.
    """
    from ..distribution import Distribution

    if isinstance(channel, Distribution):
        return MemorylessChannel(channel)
    hook = getattr(channel, "__dit_channel__", None)
    if hook is not None:
        channel = hook()
    if all(hasattr(channel, name) for name in _PROTOCOL):
        return channel
    raise ditException(
        "A channel must be a conditional Distribution p(Y | X) or implement the channel protocol "
        "(inputs, outputs, log_likelihood, sample, symbol_log_likelihoods)."
    )


def _clipped_llr(l0, l1):
    if l0 == -np.inf and l1 == -np.inf:
        return 0.0
    if l1 == -np.inf:
        return _LLR_CLIP
    if l0 == -np.inf:
        return -_LLR_CLIP
    return max(-_LLR_CLIP, min(_LLR_CLIP, l0 - l1))


def symbol_llrs(channel, received):
    """
    Per-position log-likelihood ratios ``log p(y | x_i = 0) / p(y | x_i = 1)``.

    Parameters
    ----------
    channel : Distribution or object
        A binary-input channel (see :func:`as_channel`).
    received : sequence
        The received word.

    Returns
    -------
    llrs : list of float
        One LLR per received position, clipped to a finite range.
    """
    channel = as_channel(channel)
    if list(channel.inputs) != [0, 1]:
        raise ditException("Soft decoding requires a binary-input channel with inputs {0, 1}.")
    S = channel.symbol_log_likelihoods(received)
    return [_clipped_llr(S[i, 0], S[i, 1]) for i in range(len(received))]


def log_likelihoods(channel):
    """
    Per-output-symbol log-likelihood ratios for a binary-input memoryless channel.

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
    with np.errstate(divide="ignore"):
        L = np.log(P)
    return {y: _clipped_llr(L[0, j], L[1, j]) for j, y in enumerate(outputs)}


def bhattacharyya(channel):
    """
    The Bhattacharyya parameter of a binary-input channel.

    Parameters
    ----------
    channel : Distribution or object
        A binary-input channel (see :func:`as_channel`) providing ``bhattacharyya()``.

    Returns
    -------
    Z : float
    """
    channel = as_channel(channel)
    if not hasattr(channel, "bhattacharyya"):
        raise ditException(
            "This channel has no Bhattacharyya parameter, so it cannot design a polar code. "
            "Construct the code on a memoryless design channel and pass this channel to decode()."
        )
    return channel.bhattacharyya()
