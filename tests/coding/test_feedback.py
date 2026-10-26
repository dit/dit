"""
Tests for the feedback channel codes in dit.coding.feedback.
"""

import itertools

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from dit.coding import (
    MemorylessChannel,
    binary_erasure_channel,
    binary_symmetric_channel,
    erasure_arq,
    hamming,
    horstein,
    repetition,
    stop_and_wait_arq,
)
from dit.exceptions import ditException

from .test_channel_protocol import HiddenStateChannel, gilbert_elliott, one_state_bsc


class FeedbackHiddenStateChannel(HiddenStateChannel):
    """A hidden-state channel driven one use at a time; the belief is the normalized forward vector."""

    def feedback_initial(self):
        return self.pi

    def feedback_step(self, belief, x):
        outcomes = []
        for y in self.outputs:
            alpha = belief @ self.T[(x, y)]
            p = alpha.sum()
            if p > 0:
                outcomes.append((y, float(p), alpha / p))
        return outcomes


def with_feedback(channel):
    return FeedbackHiddenStateChannel(channel.T, channel.pi, channel.inputs, channel.outputs)


def path_log_likelihood(channel, codeword, received):
    belief, total = channel.feedback_initial(), 0.0
    for x, y in zip(codeword, received, strict=True):
        outcomes = {out: (p, b) for out, p, b in channel.feedback_step(belief, x)}
        if y not in outcomes:
            return -np.inf
        p, belief = outcomes[y]
        total += np.log(p)
    return total


# ── the stepwise channel interface ──────────────────────────────────────


@settings(max_examples=30, deadline=None)
@given(st.floats(0.0, 1.0), st.sampled_from((0, 1)))
def test_memoryless_feedback_step_is_a_distribution(epsilon, x):
    channel = MemorylessChannel(binary_erasure_channel(epsilon))
    assert channel.feedback_initial() is None
    outcomes = channel.feedback_step(None, x)
    assert sum(p for _, p, _ in outcomes) == pytest.approx(1.0)
    assert all(p > 0 and b is None for _, p, b in outcomes)


words = st.integers(1, 6).flatmap(
    lambda n: st.tuples(
        st.lists(st.sampled_from((0, 1)), min_size=n, max_size=n),
        st.lists(st.sampled_from((0, 1)), min_size=n, max_size=n),
    )
)


@settings(max_examples=40, deadline=None)
@given(words)
def test_feedback_steps_reproduce_log_likelihood(pair):
    codeword, received = map(tuple, pair)
    for channel in (MemorylessChannel(binary_symmetric_channel(0.2)), with_feedback(gilbert_elliott())):
        assert path_log_likelihood(channel, codeword, received) == pytest.approx(
            channel.log_likelihood(codeword, received)
        )


def test_feedback_code_rejects_channel_without_steps():
    with pytest.raises(ditException):
        erasure_arq(2).probability_of_error(gilbert_elliott())


# ── erasure ARQ ─────────────────────────────────────────────────────────


@settings(max_examples=20, deadline=None)
@given(st.integers(1, 3), st.floats(0.0, 0.5))
def test_erasure_arq_achieves_bec_capacity(k, epsilon):
    pe, uses = erasure_arq(k).performance(binary_erasure_channel(epsilon), method="exact", max_uses=40)
    assert pe == pytest.approx(0.0, abs=1e-6)
    assert uses == pytest.approx(k / (1 - epsilon), rel=1e-4)


def test_erasure_arq_capacity_gap_vanishes_on_the_bec():
    gap = erasure_arq(2).capacity_gap(binary_erasure_channel(0.25), method="exact", max_uses=60)
    assert gap == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("p", [0.05, 0.2])
def test_erasure_arq_is_uncoded_on_the_bsc(p):
    code = erasure_arq(3)
    assert code.probability_of_error(binary_symmetric_channel(p)) == pytest.approx(1 - (1 - p) ** 3)
    assert code.expected_uses(binary_symmetric_channel(p)) == pytest.approx(3)


# ── stop-and-wait ARQ ───────────────────────────────────────────────────


@pytest.mark.parametrize("p", [0.05, 0.2])
def test_stop_and_wait_undetected_error_rate(p):
    code = stop_and_wait_arq(repetition(3))
    accept = (1 - p) ** 3 + p**3
    pe, uses = code.performance(binary_symmetric_channel(p), method="exact", max_uses=90)
    assert pe == pytest.approx(p**3 / accept, rel=1e-4)
    assert uses == pytest.approx(3 / accept, rel=1e-4)


def test_stop_and_wait_is_error_free_on_the_bec():
    epsilon = 0.1
    pe, uses = stop_and_wait_arq(repetition(3)).performance(
        binary_erasure_channel(epsilon), method="exact", max_uses=60
    )
    assert pe == pytest.approx(0.0, abs=1e-8)
    assert uses == pytest.approx(3 / (1 - epsilon) ** 3, rel=1e-6)


def test_stop_and_wait_max_rounds_falls_back_to_correction():
    code = stop_and_wait_arq(hamming(3), max_rounds=1)
    channel = binary_symmetric_channel(0.05)
    assert code.expected_uses(channel, method="montecarlo", samples=200) == 7
    assert code.probability_of_error(channel, method="montecarlo", samples=4000, prng=np.random.default_rng(0)) == (
        pytest.approx(hamming(3).probability_of_error(channel, method="exact"), abs=0.02)
    )


# ── Horstein ────────────────────────────────────────────────────────────


def test_horstein_needs_a_stopping_rule():
    with pytest.raises(ditException):
        horstein(2)


def test_horstein_error_decreases_with_block_length():
    channel = binary_symmetric_channel(0.1)
    errors = [horstein(3, n=n).probability_of_error(channel, method="exact") for n in (3, 6, 9, 12)]
    assert all(a > b for a, b in itertools.pairwise(errors))


def test_horstein_on_a_noiseless_channel_is_binary_search():
    pe, uses = horstein(3, n=3).performance(binary_symmetric_channel(0.0), method="exact")
    assert pe == 0.0
    assert uses == 3


def test_horstein_exact_matches_montecarlo():
    code, channel = horstein(2, n=6), binary_symmetric_channel(0.15)
    exact = code.probability_of_error(channel, method="exact")
    sampled = code.probability_of_error(channel, method="montecarlo", samples=20000, prng=np.random.default_rng(1))
    assert sampled == pytest.approx(exact, abs=0.01)


def test_horstein_posterior_stopping_controls_error_and_respects_capacity():
    p, delta = 0.1, 0.01
    channel = binary_symmetric_channel(p)
    code = horstein(4, delta=delta)
    pe, uses = code.performance(channel, method="montecarlo", samples=3000, prng=np.random.default_rng(2))
    assert pe <= 2 * delta
    capacity = 1 + p * np.log2(p) + (1 - p) * np.log2(1 - p)
    assert 4 / uses < capacity
    assert code.capacity_gap(channel, method="montecarlo", samples=500) > 0


def test_horstein_on_a_one_state_channel_matches_the_memoryless_channel():
    code = horstein(3, n=7)
    memoryless = code.probability_of_error(binary_symmetric_channel(0.1), method="exact")
    hidden = code.probability_of_error(with_feedback(one_state_bsc(0.1)), method="exact")
    assert hidden == pytest.approx(memoryless)


def test_horstein_runs_on_a_channel_with_memory():
    channel = with_feedback(gilbert_elliott())
    errors = [horstein(2, n=n).probability_of_error(channel, method="exact") for n in (2, 8)]
    assert errors[1] < errors[0]
