"""
Convolutional codes with Viterbi decoding.

A rate-``1/n`` :class:`ConvolutionalCode` is defined by ``n`` generator
polynomials (given in octal). Encoding runs the input through a shift register
with zero-termination; decoding is the Viterbi algorithm over the trellis
(Viterbi, 1967), hard-decision by default and soft-decision when a channel is
given.
"""

from math import log

from ..exceptions import ditException
from .base import ChannelCoding

__all__ = (
    "ConvolutionalCode",
    "convolutional",
)


def _parity(value):
    return bin(value).count("1") % 2


class ConvolutionalCode(ChannelCoding):
    """
    A rate-``1/n`` convolutional code.

    Parameters
    ----------
    generators : sequence of int
        The generator polynomials, in octal (e.g. ``(0o7, 0o5)``).
    message_length : int
        The number of information bits per encoded block.
    channel : Distribution, None
        A default channel.
    """

    def __init__(self, generators, message_length, channel=None):
        super().__init__(channel=channel, radix=2)
        self.generators = tuple(generators)
        self.n_outputs = len(self.generators)
        self.constraint_length = max(g.bit_length() for g in self.generators)
        self.K = self.constraint_length
        self._message_length = message_length
        self._n_states = 1 << (self.K - 1)
        self._transitions = self._build_transitions()

    @property
    def message_length(self):
        return self._message_length

    def rate(self):
        """The code rate ``1 / n_outputs``."""
        return 1 / self.n_outputs

    def _build_transitions(self):
        """For each (state, input bit): the next state and output bits."""
        transitions = {}
        mask = (1 << (self.K - 1)) - 1
        for state in range(self._n_states):
            for bit in (0, 1):
                reg = (state << 1) | bit
                outputs = tuple(_parity(reg & g) for g in self.generators)
                next_state = reg & mask
                transitions[(state, bit)] = (next_state, outputs)
        return transitions

    def encode(self, message):
        """
        Encode a message, appending ``K - 1`` zeros to terminate the trellis.
        """
        bits = list(message) + [0] * (self.K - 1)
        state = 0
        out = []
        for bit in bits:
            next_state, outputs = self._transitions[(state, bit)]
            out.extend(outputs)
            state = next_state
        return tuple(out)

    def decode(self, received, channel=None):
        """
        Decode by the Viterbi algorithm (soft when a channel is given).
        """
        n = self.n_outputs
        steps = len(received) // n
        chunks = [tuple(received[i * n : (i + 1) * n]) for i in range(steps)]
        metric = self._branch_metric(channel)

        inf = float("inf")
        path = [inf] * self._n_states
        path[0] = 0.0
        backpointers = []
        for chunk in chunks:
            new_path = [inf] * self._n_states
            back = [-1] * self._n_states
            for state in range(self._n_states):
                if path[state] == inf:
                    continue
                for bit in (0, 1):
                    next_state, outputs = self._transitions[(state, bit)]
                    cost = path[state] + metric(outputs, chunk)
                    if cost < new_path[next_state]:
                        new_path[next_state] = cost
                        back[next_state] = state
            path = new_path
            backpointers.append(back)

        # The trellis is terminated, so the final state is 0.
        state = 0
        inputs = []
        for back in reversed(backpointers):
            previous = back[state]
            # Recover the input bit that took `previous` -> `state`.
            bit = 1 if self._transitions[(previous, 1)][0] == state else 0
            inputs.append(bit)
            state = previous
        inputs.reverse()
        return inputs[: self._message_length]

    def _branch_metric(self, channel):
        if channel is None:

            def metric(outputs, chunk):
                return sum(o != r for o, r in zip(outputs, chunk, strict=True))

            return metric

        from ._channel import channel_arrays

        inputs, outputs_alpha, P = channel_arrays(channel)
        in_index = {v: i for i, v in enumerate(inputs)}
        out_index = {v: i for i, v in enumerate(outputs_alpha)}

        def metric(outputs, chunk):
            cost = 0.0
            for o, r in zip(outputs, chunk, strict=True):
                p = P[in_index[o], out_index[r]]
                cost += -log(p) if p > 0 else float("inf")
            return cost

        return metric


def convolutional(generators, message_length, channel=None):
    """
    Build a rate-``1/n`` convolutional code.

    Parameters
    ----------
    generators : sequence of int
        The generator polynomials in octal (e.g. ``(0o7, 0o5)``).
    message_length : int
        The number of information bits per encoded block.
    channel : Distribution, None
        A default channel.

    Returns
    -------
    code : ConvolutionalCode
    """
    if len(generators) < 2:
        raise ditException("A convolutional code needs at least two generators.")
    return ConvolutionalCode(generators, message_length, channel=channel)
