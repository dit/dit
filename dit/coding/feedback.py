"""
Channel codes that use a noiseless feedback link.

With feedback the transmitter sees every channel output before choosing the
next input, so a code is a *strategy* ``x_t = f(message, y_{0:t})`` rather than
a fixed codeword, and its block length may depend on what the channel did.
Feedback does not increase the capacity of a memoryless channel
:cite:`Cover2006`, but it makes capacity-achieving codes far simpler and
improves the error exponent :cite:`Burnashev1976`; for channels with memory it
can increase the capacity itself :cite:`Permuter2008`.

A :class:`FeedbackCode` is written in terms of a *shared state*: everything the
receiver knows, which (because the feedback is noiseless) the transmitter also
knows. The transmitter combines it with the message to pick each input, and the
state is updated from each output alone, so the receiver can follow along. The
channel is driven one use at a time through the optional ``feedback_initial``
and ``feedback_step`` methods of the channel protocol (see
:func:`~dit.coding.as_channel`).
"""

import itertools
from abc import ABC, abstractmethod
from functools import lru_cache

import numpy as np

from ..example_channels.binary import ERASURE
from ..exceptions import ditException
from ..utils.deprecation import renamed_kwarg
from . import _gf2

__all__ = (
    "ErasureARQ",
    "FeedbackCode",
    "HorsteinCode",
    "StopAndWaitARQ",
    "erasure_arq",
    "horstein",
    "stop_and_wait_arq",
)

_MAX_USES = 64
_EXACT_BUDGET = 2**20


class _BudgetExceededError(Exception):
    pass


def _merge_key(state, belief, unique):
    """A hashable key for ``(state, belief)``, or the never-repeating `unique` if there is none."""
    if isinstance(belief, np.ndarray):
        belief = belief.tobytes()
    try:
        key = (state, belief)
        hash(key)
    except TypeError:
        return ("unique", unique)
    return key


def _feedback_channel(channel):
    from ._channel import as_channel

    channel = as_channel(channel)
    if not (hasattr(channel, "feedback_initial") and hasattr(channel, "feedback_step")):
        raise ditException(
            "A feedback code needs a channel implementing feedback_initial() and feedback_step(belief, x)."
        )
    return channel


class FeedbackCode(ABC):
    """
    Abstract base class for channel codes with noiseless feedback.

    Subclasses implement :meth:`start`, :meth:`next_input`, :meth:`update` and
    :meth:`decode`. States must be treated as immutable: :meth:`update` returns
    a new state, so that :meth:`probability_of_error` can branch on every
    channel output.

    Parameters
    ----------
    k : int
        The number of message bits.
    """

    def __init__(self, k):
        if k < 1:
            raise ditException("A feedback code needs at least one message bit.")
        self.k = k

    @property
    def message_length(self):
        """The number of message bits ``k``."""
        return self.k

    @abstractmethod
    def start(self, channel):
        """
        The shared state before the first channel use.
        """

    @abstractmethod
    def next_input(self, state, message):
        """
        The next channel input for `message`, or ``None`` to stop transmitting.
        """

    @abstractmethod
    def update(self, state, y):
        """
        The shared state after the channel output `y` is received (and fed back).
        """

    @abstractmethod
    def decode(self, state):
        """
        The receiver's estimate of the message, as a tuple of bits.
        """

    # ── evaluation ───────────────────────────────────────────────────────

    @renamed_kwarg("prng", "rng")
    def transmit(self, message, channel, rng=None, max_uses=_MAX_USES):
        """
        Send `message` across `channel` once.

        Parameters
        ----------
        message : sequence of int
            The ``k`` message bits.
        channel : Distribution or object
            The channel, implementing ``feedback_initial`` and ``feedback_step``.
        rng : numpy.random.Generator, None
            The random number generator.
        max_uses : int
            Stop (and decode) after this many channel uses regardless.

        Returns
        -------
        decoded : tuple
            The receiver's estimate of the message.
        uses : int
            The number of channel uses.
        """
        channel = _feedback_channel(channel)
        if rng is None:
            rng = np.random.default_rng()
        message = tuple(int(b) for b in message)
        state, belief, uses = self.start(channel), channel.feedback_initial(), 0
        while uses < max_uses:
            x = self.next_input(state, message)
            if x is None:
                break
            outcomes = channel.feedback_step(belief, x)
            probs = np.array([p for _, p, _ in outcomes])
            y, _, belief = outcomes[rng.choice(len(outcomes), p=probs / probs.sum())]
            state = self.update(state, y)
            uses += 1
        return tuple(self.decode(state)), uses

    def _exact(self, channel, max_uses, budget):
        # Breadth-first over channel uses, merging branches that reach the same
        # (state, belief): they behave identically from then on.
        errors, uses_total, nodes = 0.0, 0.0, 0
        for message in itertools.product((0, 1), repeat=self.k):
            layer = {None: (self.start(channel), channel.feedback_initial(), 1.0)}
            for uses in range(max_uses + 1):
                following = {}
                for state, belief, prob in layer.values():
                    x = self.next_input(state, message) if uses < max_uses else None
                    if x is None:
                        uses_total += prob * uses
                        if tuple(self.decode(state)) != message:
                            errors += prob
                        continue
                    for y, p, next_belief in channel.feedback_step(belief, x):
                        nodes += 1
                        if nodes > budget:
                            raise _BudgetExceededError
                        next_state = self.update(state, y)
                        key = _merge_key(next_state, next_belief, nodes)
                        if key in following:
                            following[key] = (next_state, next_belief, following[key][2] + prob * p)
                        else:
                            following[key] = (next_state, next_belief, prob * p)
                if not following:
                    break
                layer = following
        return errors / 2**self.k, uses_total / 2**self.k

    def _montecarlo(self, channel, samples, rng, max_uses):
        if rng is None:
            rng = np.random.default_rng()
        errors, uses_total = 0, 0
        for _ in range(samples):
            message = tuple(int(b) for b in rng.integers(0, 2, size=self.k))
            decoded, uses = self.transmit(message, channel, rng=rng, max_uses=max_uses)
            errors += decoded != message
            uses_total += uses
        return errors / samples, uses_total / samples

    @renamed_kwarg("prng", "rng")
    def performance(self, channel, method="auto", samples=10000, rng=None, max_uses=_MAX_USES):
        """
        The block error probability and the expected number of channel uses.

        Parameters
        ----------
        channel : Distribution or object
            The channel, implementing ``feedback_initial`` and ``feedback_step``.
        method : str
            ``'exact'`` enumerates every message and every output sequence;
            ``'montecarlo'`` samples; ``'auto'`` is exact unless the output tree
            exceeds about a million branches.
        samples : int
            The number of Monte Carlo samples.
        rng : numpy.random.Generator, None
            The random number generator for Monte Carlo sampling.
        max_uses : int
            Transmission is cut off (and decoded) after this many channel uses,
            so the exact values are those of the truncated code.

        Returns
        -------
        pe : float
            The probability that the decoded message differs from the sent one.
        uses : float
            The expected number of channel uses.
        """
        channel = _feedback_channel(channel)
        if method in ("auto", "exact"):
            try:
                return self._exact(channel, max_uses, _EXACT_BUDGET if method == "auto" else np.inf)
            except _BudgetExceededError:
                pass
        elif method != "montecarlo":
            raise ditException(f"Unknown method {method!r}; use 'exact', 'montecarlo' or 'auto'.")
        return self._montecarlo(channel, samples, rng, max_uses)

    def probability_of_error(self, channel, **kwargs):
        """
        The block error probability over `channel`; keyword arguments as in :meth:`performance`.
        """
        return self.performance(channel, **kwargs)[0]

    def expected_uses(self, channel, **kwargs):
        """
        The expected number of channel uses; keyword arguments as in :meth:`performance`.
        """
        return self.performance(channel, **kwargs)[1]

    def rate(self, channel, **kwargs):
        """
        The rate ``k / E[N]`` in bits per channel use, where ``N`` is the number of uses.

        The rate of a variable-length code depends on the channel, so unlike
        :meth:`ChannelCoding.rate <dit.coding.ChannelCoding.rate>` it takes one.
        Keyword arguments are as in :meth:`performance`.
        """
        return self.k / self.expected_uses(channel, **kwargs)

    def capacity_gap(self, channel, capacity_kwargs=None, **kwargs):
        """
        The feedback capacity of `channel` minus the code's rate on it.

        The channel's ``feedback_capacity()`` is used when it has one, and
        otherwise its ``capacity()``, which equals the feedback capacity for a
        memoryless channel :cite:`Cover2006`.

        Parameters
        ----------
        channel : Distribution or object
            The channel.
        capacity_kwargs : dict, None
            Passed to the channel's capacity method.
        kwargs : dict
            Passed to :meth:`rate`.

        Returns
        -------
        gap : float or tuple
            ``capacity - rate``, or a ``(lower, upper)`` bracket when the
            channel only brackets its capacity.
        """
        channel = _feedback_channel(channel)
        method = getattr(channel, "feedback_capacity", None) or getattr(channel, "capacity", None)
        if method is None:
            raise ditException("This channel does not report a capacity.")
        capacity = method(**(capacity_kwargs or {}))
        rate = self.rate(channel, **kwargs)
        if isinstance(capacity, tuple):
            lower, upper = capacity
            return (lower - rate, upper - rate)
        return capacity - rate


class ErasureARQ(FeedbackCode):
    """
    Send each bit until it arrives unerased.

    On the binary erasure channel this is zero-error and uses
    ``k / (1 - epsilon)`` channel uses on average, so it achieves the capacity
    ``1 - epsilon`` exactly :cite:`Cover2006`. Any output other than the erasure
    symbol is taken at face value, so on a channel that also flips bits it makes
    errors.

    Parameters
    ----------
    k : int
        The number of message bits.
    erasure : object
        The channel's erasure symbol.
    """

    def __init__(self, k, erasure=ERASURE):
        super().__init__(k)
        self.erasure = erasure

    def __repr__(self):
        return f"ErasureARQ(k={self.k})"

    def start(self, channel):
        """The bits received so far: none."""
        return ()

    def next_input(self, state, message):
        """The first bit not yet received, or ``None`` once all ``k`` have arrived."""
        return message[len(state)] if len(state) < self.k else None

    def update(self, state, y):
        """Keep `y` unless it is an erasure."""
        return state if y == self.erasure else (*state, y)

    def decode(self, state):
        """The received bits, padded with zeros if transmission was cut off."""
        return state + (0,) * (self.k - len(state))


class StopAndWaitARQ(FeedbackCode):
    """
    Stop-and-wait automatic repeat request over a linear block code.

    The codeword is sent; if the received word is not a codeword (a nonzero
    syndrome, or any non-binary output such as an erasure) the receiver asks for
    it again. An error goes undetected only when the error pattern is itself a
    nonzero codeword. After `max_rounds` failed attempts the last received word
    is syndrome-decoded.

    Parameters
    ----------
    code : LinearCode
        The block code used for error detection.
    max_rounds : int, None
        The maximum number of transmissions of the block; unlimited by default.
    """

    def __init__(self, code, max_rounds=None):
        super().__init__(code.k)
        self.code = code
        self.max_rounds = max_rounds
        self._encode = lru_cache(maxsize=None)(code.encode)

    def __repr__(self):
        return f"StopAndWaitARQ({self.code!r}, max_rounds={self.max_rounds})"

    def start(self, channel):
        """``(rounds completed, current block, accepted message)``."""
        return (0, (), None)

    def next_input(self, state, message):
        """The next bit of the codeword, or ``None`` once a block is accepted."""
        _, block, accepted = state
        if accepted is not None:
            return None
        return self._encode(message)[len(block)]

    def _is_codeword(self, block):
        if any(y not in (0, 1) for y in block):
            return False
        return not np.any(_gf2.matvec(self.code.H, np.array(block, dtype=int)))

    def update(self, state, y):
        """Append `y`; at the end of a block, accept it or ask for a retransmission."""
        rounds, block, _ = state
        block = (*block, y)
        if len(block) < self.code.n:
            return (rounds, block, None)
        rounds += 1
        if self._is_codeword(block):
            return (rounds, (), tuple(self.code.decode(block)))
        if self.max_rounds is not None and rounds >= self.max_rounds:
            hard = [y if y in (0, 1) else 0 for y in block]
            return (rounds, (), tuple(self.code.decode(hard)))
        return (rounds, (), None)

    def decode(self, state):
        """The accepted message (all zeros if none was accepted)."""
        accepted = state[2]
        return accepted if accepted is not None else (0,) * self.k


class _HorsteinState:
    __slots__ = ("beliefs", "channel", "inputs", "uses", "weights")

    def __init__(self, channel, weights, beliefs, uses):
        self.channel = channel
        self.weights = weights
        self.beliefs = beliefs
        self.uses = uses
        midpoints = np.cumsum(weights) - weights / 2
        self.inputs = (midpoints >= 0.5).astype(int)


class HorsteinCode(FeedbackCode):
    """
    Horstein's bisection scheme over the ``2^k`` messages.

    Both ends keep the receiver's posterior over the messages, in a fixed
    order. At each use the transmitter sends ``1`` if its message lies above
    the posterior median and ``0`` otherwise, so every output roughly halves
    the receiver's uncertainty :cite:`Horstein1963`. Posterior matching
    :cite:`Shayevitz2011` generalizes the idea; Schalkwijk and Kailath's
    scheme is its Gaussian-channel counterpart :cite:`Schalkwijk1966`.

    The posterior is updated through the channel's ``feedback_step``, as a
    joint belief over the message and the channel's hidden state, so the same
    code runs on finite-state channels. Inputs must be the bits ``{0, 1}``.

    Parameters
    ----------
    k : int
        The number of message bits.
    n : int, None
        Stop after `n` channel uses.
    delta : float, None
        Stop once the largest posterior probability reaches ``1 - delta``.
        At least one of `n` and `delta` must be given; with both, whichever
        comes first.
    """

    def __init__(self, k, n=None, delta=None):
        super().__init__(k)
        if n is None and delta is None:
            raise ditException("Give a block length n, a target error delta, or both.")
        self.n = n
        self.delta = delta

    def __repr__(self):
        return f"HorsteinCode(k={self.k}, n={self.n}, delta={self.delta})"

    def start(self, channel):
        """The uniform prior over messages, with the channel's initial belief for each."""
        if tuple(channel.inputs) != (0, 1):
            raise ditException("Horstein's scheme needs a binary-input channel with inputs {0, 1}.")
        M = 2**self.k
        return _HorsteinState(channel, np.full(M, 1 / M), (channel.feedback_initial(),) * M, 0)

    def _stopped(self, state):
        if self.n is not None and state.uses >= self.n:
            return True
        return self.delta is not None and state.weights.max() >= 1 - self.delta

    def next_input(self, state, message):
        """``1`` if `message` lies above the posterior median, else ``0``; ``None`` once stopped."""
        if self._stopped(state):
            return None
        index = int("".join(map(str, message)), 2)
        return int(state.inputs[index])

    def update(self, state, y):
        """The posterior after output `y`, by Bayes' rule message by message."""
        channel = state.channel
        weights = np.zeros_like(state.weights)
        beliefs = list(state.beliefs)
        memoryless = {}
        for m in np.flatnonzero(state.weights):
            x = int(state.inputs[m])
            belief = state.beliefs[m]
            if belief is None:
                if x not in memoryless:
                    memoryless[x] = {out: (p, b) for out, p, b in channel.feedback_step(None, x)}
                outcomes = memoryless[x]
            else:
                outcomes = {out: (p, b) for out, p, b in channel.feedback_step(belief, x)}
            if y in outcomes:
                p, beliefs[m] = outcomes[y]
                weights[m] = state.weights[m] * p
        total = weights.sum()
        if total <= 0:
            raise ditException(f"The output {y!r} is impossible under every message.")
        return _HorsteinState(channel, weights / total, tuple(beliefs), state.uses + 1)

    def decode(self, state):
        """The maximum a posteriori message."""
        index = int(np.argmax(state.weights))
        return tuple(int(b) for b in format(index, f"0{self.k}b"))


def erasure_arq(k, erasure=ERASURE):
    """
    An :class:`ErasureARQ` code for `k` bits: resend each bit until it is not erased.
    """
    return ErasureARQ(k, erasure=erasure)


def stop_and_wait_arq(code, max_rounds=None):
    """
    A :class:`StopAndWaitARQ` code: resend `code`'s block until it is received as a codeword.
    """
    return StopAndWaitARQ(code, max_rounds=max_rounds)


def horstein(k, n=None, delta=None):
    """
    A :class:`HorsteinCode` for `k` bits, stopping after `n` uses or at posterior ``1 - delta``.
    """
    return HorsteinCode(k, n=n, delta=delta)
