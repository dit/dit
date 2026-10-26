"""
Tests for the dit.coding channel protocol, including channels with hidden state.
"""

import itertools

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from dit.coding import (
    MemorylessChannel,
    as_channel,
    binary_symmetric_channel,
    convolutional,
    hamming,
    ldpc,
    polar,
    repetition,
)
from dit.coding._channel import symbol_llrs
from dit.exceptions import ditException


class HiddenStateChannel:
    """
    A finite-state channel ``T^{(x, y)}[s, s']`` with initial state law ``pi``.
    """

    def __init__(self, T, pi, inputs=(0, 1), outputs=(0, 1), capacity=None):
        self.T = T
        self.pi = np.asarray(pi, dtype=float)
        self.inputs = tuple(inputs)
        self.outputs = tuple(outputs)
        self._capacity = capacity

    def log_likelihood(self, codeword, received):
        alpha = self.pi.copy()
        for x, y in zip(codeword, received, strict=True):
            alpha = alpha @ self.T[(x, y)]
        total = alpha.sum()
        return float(np.log(total)) if total > 0 else -np.inf

    def sample(self, codeword, prng):
        n = len(self.pi)
        s = prng.choice(n, p=self.pi)
        out = []
        for x in codeword:
            weights = np.array([self.T[(x, y)][s] for y in self.outputs])
            j, s = divmod(prng.choice(weights.size, p=weights.ravel()), n)
            out.append(self.outputs[j])
        return tuple(out)

    def symbol_log_likelihoods(self, received):
        n = len(received)
        M = [sum(self.T[(x, y)] for x in self.inputs) / len(self.inputs) for y in received]
        alphas = [self.pi]
        for t in range(n):
            alphas.append(alphas[-1] @ M[t])
        betas = [np.ones(len(self.pi))]
        for t in reversed(range(n)):
            betas.append(M[t] @ betas[-1])
        betas.reverse()
        S = np.empty((n, len(self.inputs)))
        with np.errstate(divide="ignore"):
            for t, y in enumerate(received):
                for a, x in enumerate(self.inputs):
                    S[t, a] = np.log(alphas[t] @ self.T[(x, y)] @ betas[t + 1])
        return S

    def capacity(self):
        return self._capacity


def gilbert_elliott(q_gb=0.1, q_bg=0.2, eps_good=0.01, eps_bad=0.3):
    """A Gilbert-Elliott channel: a BSC whose crossover follows a two-state Markov chain."""
    Q = np.array([[1 - q_gb, q_gb], [q_bg, 1 - q_bg]])
    eps = np.array([eps_good, eps_bad])
    T = {(x, y): Q * np.where(x == y, 1 - eps, eps)[:, None] for x in (0, 1) for y in (0, 1)}
    pi = np.array([q_bg, q_gb]) / (q_gb + q_bg)
    return HiddenStateChannel(T, pi)


def one_state_bsc(p):
    """The BSC as a one-state hidden-state channel, sharing the Distribution's table."""
    P = MemorylessChannel(binary_symmetric_channel(p)).P
    T = {(x, y): np.array([[P[x, y]]]) for x in (0, 1) for y in (0, 1)}
    return HiddenStateChannel(T, [1.0])


def brute_force_likelihood(channel, codeword, received):
    n_states = len(channel.pi)
    total = 0.0
    for path in itertools.product(range(n_states), repeat=len(codeword) + 1):
        prob = channel.pi[path[0]]
        for t, (x, y) in enumerate(zip(codeword, received, strict=True)):
            prob *= channel.T[(x, y)][path[t], path[t + 1]]
        total += prob
    return total


words = st.integers(1, 5).flatmap(
    lambda n: st.tuples(
        st.lists(st.sampled_from((0, 1)), min_size=n, max_size=n),
        st.lists(st.sampled_from((0, 1)), min_size=n, max_size=n),
    )
)


@settings(max_examples=50, deadline=None)
@given(words)
def test_hidden_state_likelihood_matches_path_sum(pair):
    """The forward-algorithm likelihood equals the sum over state paths."""
    codeword, received = pair
    channel = gilbert_elliott()
    expected = brute_force_likelihood(channel, codeword, received)
    assert np.exp(channel.log_likelihood(codeword, received)) == pytest.approx(expected)


@settings(max_examples=30, deadline=None)
@given(st.lists(st.sampled_from((0, 1)), min_size=1, max_size=5))
def test_hidden_state_symbol_likelihoods_match_brute_force(received):
    """BCJR symbol likelihoods equal p(y^n | x_i) under uniform inputs, up to a row constant."""
    channel = gilbert_elliott()
    n = len(received)
    S = channel.symbol_log_likelihoods(received)
    for i in range(n):
        values = []
        for a in (0, 1):
            total = 0.0
            for codeword in itertools.product((0, 1), repeat=n):
                if codeword[i] == a:
                    total += brute_force_likelihood(channel, codeword, received)
            values.append(np.log(total))
        assert S[i, 0] - S[i, 1] == pytest.approx(values[0] - values[1])


@settings(max_examples=20, deadline=None)
@given(st.floats(0.01, 0.4), st.lists(st.sampled_from((0, 1)), min_size=7, max_size=7))
def test_one_state_channel_decodes_like_bsc(p, received):
    """
    A one-state channel decodes like the equivalent memoryless channel.

    Symbol log-likelihoods are only defined up to a per-row constant, so the
    soft decoders see the same LLRs but may break exact ties differently; the
    Viterbi decisions are compared by their likelihood instead.
    """
    stateful, bsc = one_state_bsc(p), binary_symmetric_channel(p)
    received = tuple(received)
    assert symbol_llrs(stateful, received) == pytest.approx(symbol_llrs(bsc, received))
    code = hamming(3)
    assert code.decode(received, channel=stateful) == code.decode(received, channel=bsc)
    conv = convolutional((0o7, 0o5), 2)
    word = received + (1,)
    memoryless = as_channel(bsc)
    scores = [memoryless.log_likelihood(conv.encode(conv.decode(word, channel=c)), word) for c in (stateful, bsc)]
    assert scores[0] == pytest.approx(scores[1])


@pytest.mark.parametrize("p", [0.05, 0.2])
def test_one_state_channel_error_probability_like_bsc(p):
    """The exact block-error probability matches the memoryless computation."""
    code = hamming(3)
    expected = code.probability_of_error(binary_symmetric_channel(p), method="exact")
    assert code.probability_of_error(one_state_bsc(p), method="exact") == pytest.approx(expected)


def test_hidden_state_ml_decoding_is_exact():
    """Maximum-likelihood decoding over a hidden-state channel picks the most likely codeword."""
    channel = gilbert_elliott()
    code = repetition(3)
    for received in itertools.product((0, 1), repeat=3):
        scores = {m: brute_force_likelihood(channel, code.encode(m), received) for m in [(0,), (1,)]}
        assert tuple(code.decode(received, channel=channel)) == max(scores, key=scores.get)


def test_hidden_state_error_probability():
    """The exact error probability matches a direct sum, and Monte Carlo agrees with it."""
    channel = gilbert_elliott()
    code = repetition(3)
    expected = 0.0
    for m in [(0,), (1,)]:
        for received in itertools.product((0, 1), repeat=3):
            if tuple(code.decode(received, channel=channel)) != m:
                expected += brute_force_likelihood(channel, code.encode(m), received) / 2
    exact = code.probability_of_error(channel, method="exact")
    assert exact == pytest.approx(expected)
    estimate = code.probability_of_error(channel, method="montecarlo", samples=4000, prng=np.random.default_rng(0))
    assert estimate == pytest.approx(exact, abs=0.02)


def test_soft_decoders_run_over_memory():
    """The soft decoders run over a channel with memory and recover clean transmissions."""
    channel = gilbert_elliott(eps_good=1e-6, eps_bad=1e-6)
    for code in (hamming(3), ldpc(hamming(3).H), convolutional((0o7, 0o5), 3)):
        for message in itertools.product((0, 1), repeat=code.message_length):
            assert tuple(code.decode(code.encode(message), channel=channel)) == message


def test_capacity_gap_float_and_bracket():
    """capacity_gap subtracts the rate from a float capacity or from each end of a bracket."""
    code = hamming(3)
    rate = code.rate()
    assert code.capacity_gap(HiddenStateChannel({}, [1.0], capacity=0.9)) == pytest.approx(0.9 - rate)
    lower, upper = code.capacity_gap(HiddenStateChannel({}, [1.0], capacity=(0.6, 0.7)))
    assert lower == pytest.approx(0.6 - rate)
    assert upper == pytest.approx(0.7 - rate)


def test_capacity_gap_memoryless_unchanged():
    """A conditional Distribution still uses the Blahut-Arimoto capacity."""
    p = 0.11
    expected = 1 + p * np.log2(p) + (1 - p) * np.log2(1 - p) - hamming(3).rate()
    assert hamming(3).capacity_gap(binary_symmetric_channel(p)) == pytest.approx(expected)


def test_as_channel_wraps_distribution():
    channel = as_channel(binary_symmetric_channel(0.1))
    assert isinstance(channel, MemorylessChannel)
    assert channel.inputs == (0, 1)
    assert channel.outputs == (0, 1)


def test_as_channel_uses_hook():
    target = gilbert_elliott()

    class Wrapper:
        def __dit_channel__(self):
            return target

    assert as_channel(Wrapper()) is target
    assert as_channel(target) is target


def test_as_channel_rejects_other_objects():
    with pytest.raises(ditException):
        as_channel(object())


def test_polar_needs_bhattacharyya():
    """A polar code cannot be designed on a channel without a Bhattacharyya parameter."""
    with pytest.raises(ditException, match="Bhattacharyya"):
        polar(8, 4, gilbert_elliott())


def test_memoryless_sample_is_reproducible():
    channel = as_channel(binary_symmetric_channel(0.3))
    a = channel.sample((0, 1, 0, 1), np.random.default_rng(3))
    b = channel.sample((0, 1, 0, 1), np.random.default_rng(3))
    assert a == b
    assert set(a) <= {0, 1}
