"""
Abstract base classes for coding.

A code is a map between a source (or a channel) and a set of codewords. The two
fundamental families are:

- :class:`SourceCoding` -- lossless representation of a source distribution, the
  goal being to minimize the expected number of code symbols per source symbol
  (the *rate*).
- :class:`ChannelCoding` -- reliable transmission of messages across a noisy
  channel, the goal being to approach the channel capacity while controlling the
  probability of error.

Only source codes are implemented concretely; :class:`ChannelCoding` is provided
as scaffolding for future channel codes (e.g. repetition, polar, LDPC).
"""

from abc import ABC, abstractmethod
from math import log2

from ..exceptions import ditException
from ..shannon import entropy

__all__ = (
    "ChannelCoding",
    "SourceCoding",
)


class SourceCoding(ABC):
    """
    Abstract base class for source codes.

    A source code maps the outcomes of a :class:`~dit.Distribution` to codewords
    over a `radix`-ary code alphabet. Concrete subclasses define how outcomes are
    encoded and decoded, and what the code's `rate` is; this base class provides
    the comparisons to the source entropy that are common to all source codes.

    Parameters
    ----------
    dist : Distribution, None
        The source distribution the code is built for. Required for any of the
        rate-based properties.
    radix : int
        The size of the code alphabet. Default is 2 (binary).
    """

    def __init__(self, dist=None, radix=2):
        self.dist = dist
        self.radix = radix

    @abstractmethod
    def encode(self, source):
        """
        Encode a sequence of source outcomes into a sequence of code symbols.
        """

    @abstractmethod
    def decode(self, encoded):
        """
        Decode a sequence of code symbols back into source outcomes.
        """

    @abstractmethod
    def rate(self):
        """
        The expected number of code symbols emitted per source symbol.
        """

    def source_entropy(self):
        """
        The entropy of the source, in units of `radix`-ary digits.

        This is the fundamental lower bound on the rate of any uniquely decodable
        source code (the source coding theorem; Cover & Thomas, Ch. 5).

        Returns
        -------
        H : float
            The source entropy, ``H[X] / log2(radix)``.
        """
        if self.dist is None:
            raise ditException("A source distribution is required to compute the entropy.")
        dist = self.dist.copy(base="linear") if self.dist.is_log() else self.dist
        return entropy(dist) / log2(self.radix)

    def redundancy(self):
        """
        The rate in excess of the source entropy, ``rate - source_entropy``.

        Returns
        -------
        r : float
            The redundancy, in `radix`-ary digits per source symbol. Always
            non-negative for a uniquely decodable code.
        """
        return self.rate() - self.source_entropy()

    def efficiency(self):
        """
        The ratio of the source entropy to the rate, in ``[0, 1]``.

        Returns
        -------
        e : float
            The coding efficiency, ``source_entropy / rate``.
        """
        return self.source_entropy() / self.rate()


class ChannelCoding(ABC):
    """
    Abstract base class for channel codes.

    A channel code adds redundancy to a message so that it can be recovered after
    transmission across a noisy channel. This class is scaffolding for future
    channel codes (repetition, polar, LDPC, ...); no concrete subclass is provided
    yet.

    Concrete subclasses are expected to implement :meth:`encode` (message ->
    codeword), :meth:`decode` (received word -> message), and :meth:`rate` (message
    symbols per channel use), and may additionally expose channel-code properties
    such as the minimum distance, the block error probability for a given channel,
    and the gap to capacity (via :func:`dit.algorithms.channel_capacity`).

    Parameters
    ----------
    channel : Distribution, None
        The channel, as a conditional distribution ``p(output | input)``.
    radix : int
        The size of the code alphabet. Default is 2 (binary).
    """

    def __init__(self, channel=None, radix=2):
        self.channel = channel
        self.radix = radix

    @abstractmethod
    def encode(self, message):
        """
        Encode a message into a channel codeword.
        """

    @abstractmethod
    def decode(self, received, channel=None):
        """
        Decode a received word back into a message.

        Soft-decision decoders use `channel` (a conditional ``p(Y|X)``
        distribution) to form log-likelihood ratios; hard-decision decoders
        ignore it.
        """

    @abstractmethod
    def rate(self):
        """
        The number of message symbols transmitted per channel use.
        """

    def capacity_gap(self, channel):
        """
        The gap between the channel capacity and the code rate.

        Parameters
        ----------
        channel : Distribution
            The channel, as a conditional distribution ``p(Y|X)``.

        Returns
        -------
        gap : float
            ``capacity(channel) - rate``, in bits per channel use. Positive when
            the code operates below capacity.
        """
        from ..algorithms import channel_capacity

        capacity, _ = channel_capacity(channel)
        return capacity - self.rate()

    def probability_of_error(self, channel, method="auto", samples=10000, prng=None):
        """
        The block error probability of the code over a channel.

        The code's own :meth:`decode` is used (passing `channel` so soft-decision
        decoders engage). With ``method='exact'`` every message and every received
        word is enumerated -- feasible only for small codes -- while
        ``method='montecarlo'`` estimates the probability by sampling.

        Parameters
        ----------
        channel : Distribution
            The channel, as a conditional distribution ``p(Y|X)``.
        method : str
            ``'exact'``, ``'montecarlo'``, or ``'auto'`` (exact when small).
        samples : int
            The number of Monte Carlo samples.
        prng : numpy.random.Generator, None
            The random number generator for Monte Carlo sampling.

        Returns
        -------
        pe : float
            The probability that the decoded message differs from the sent one.
        """
        import itertools

        import numpy as np

        from ._channel import channel_arrays

        inputs, outputs, P = channel_arrays(channel)
        in_index = {v: i for i, v in enumerate(inputs)}
        k = self.message_length
        n = len(self.encode((0,) * k))

        if method == "auto":
            method = "exact" if (2**k) * (len(outputs) ** n) <= 2**20 else "montecarlo"

        if method == "exact":
            total = 0.0
            for message in itertools.product((0, 1), repeat=k):
                codeword = self.encode(message)
                rows = [P[in_index[bit]] for bit in codeword]
                for cols in itertools.product(range(len(outputs)), repeat=n):
                    prob = 1.0
                    for i, j in enumerate(cols):
                        prob *= rows[i][j]
                        if prob == 0.0:
                            break
                    if prob == 0.0:
                        continue
                    received = tuple(outputs[j] for j in cols)
                    decoded = tuple(self.decode(received, channel=channel))
                    if decoded != tuple(message):
                        total += prob
            return total / (2**k)

        if prng is None:
            prng = np.random.default_rng()
        errors = 0
        for _ in range(samples):
            message = tuple(int(b) for b in prng.integers(0, 2, size=k))
            codeword = self.encode(message)
            received = tuple(outputs[prng.choice(len(outputs), p=P[in_index[bit]])] for bit in codeword)
            decoded = tuple(self.decode(received, channel=channel))
            if decoded != message:
                errors += 1
        return errors / samples
