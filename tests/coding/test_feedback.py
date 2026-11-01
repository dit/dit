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
from dit.coding.feedback import _merge_key
from dit.example_channels import q_ary_symmetric_channel
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
    assert code.probability_of_error(channel, method="montecarlo", samples=4000, rng=np.random.default_rng(0)) == (
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
    sampled = code.probability_of_error(channel, method="montecarlo", samples=20000, rng=np.random.default_rng(1))
    assert sampled == pytest.approx(exact, abs=0.01)


def test_horstein_posterior_stopping_controls_error_and_respects_capacity():
    p, delta = 0.1, 0.01
    channel = binary_symmetric_channel(p)
    code = horstein(4, delta=delta)
    pe, uses = code.performance(channel, method="montecarlo", samples=3000, rng=np.random.default_rng(2))
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


# ── validation and evaluation details ───────────────────────────────────


class ListBeliefChannel(FeedbackHiddenStateChannel):
    """The same hidden-state channel, but its beliefs are plain (unhashable) lists."""

    def feedback_initial(self):
        return list(self.pi)

    def feedback_step(self, belief, x):
        return [(y, p, list(b)) for y, p, b in super().feedback_step(np.asarray(belief), x)]


class BracketedChannel(MemorylessChannel):
    """A memoryless channel that reports only a bracket on its feedback capacity."""

    def feedback_capacity(self, lower, upper):
        return (lower, upper)


class NoCapacityChannel(MemorylessChannel):
    """A memoryless channel that does not report its capacity."""

    capacity = None


def test_feedback_code_needs_a_message_bit():
    """A feedback code must carry at least one message bit."""
    with pytest.raises(ditException, match="at least one message bit"):
        erasure_arq(0)
    with pytest.raises(ditException, match="at least one message bit"):
        horstein(0, n=3)


def test_message_length_is_the_number_of_message_bits():
    """message_length is k, which for stop-and-wait ARQ is the dimension of the block code."""
    assert erasure_arq(3).message_length == 3
    assert stop_and_wait_arq(hamming(3)).message_length == 4
    assert horstein(2, n=5).message_length == 2


def test_transmit_without_an_rng_on_noiseless_channels():
    """Without an rng a fresh generator is used; a noiseless channel delivers every message in k uses."""
    for message in itertools.product((0, 1), repeat=3):
        assert erasure_arq(3).transmit(message, binary_erasure_channel(0.0)) == (message, 3)
        assert horstein(3, n=3).transmit(message, binary_symmetric_channel(0.0)) == (message, 3)


def test_transmission_is_cut_off_after_max_uses():
    """Cut off by max_uses, the receiver decodes what has arrived and pads it with zeros."""
    code, channel = erasure_arq(3), binary_erasure_channel(0.0)
    assert code.transmit((1, 1, 1), channel, rng=np.random.default_rng(0), max_uses=2) == ((1, 1, 0), 2)
    # The third bit never arrives, so it is wrong for half of the messages ...
    assert code.performance(channel, method="exact", max_uses=2) == pytest.approx((0.5, 2.0))
    # ... and without any channel use only the all-zero message is decoded correctly.
    assert code.performance(channel, method="exact", max_uses=0) == pytest.approx((1 - 2**-3, 0.0))


def test_merge_key():
    """Branches merge on equal (state, belief), arrays compared by value; unhashable beliefs never merge."""
    belief = np.array([0.25, 0.75])
    assert _merge_key((1,), belief, 3) == _merge_key((1,), belief.copy(), 4)
    assert _merge_key((1,), belief, 3) != _merge_key((0,), belief, 3)
    assert _merge_key((1,), None, 3) == ((1,), None)
    assert _merge_key((1,), [0.25, 0.75], 3) == ("unique", 3)
    assert _merge_key((1,), [0.25, 0.75], 3) != _merge_key((1,), [0.25, 0.75], 4)


def test_exact_evaluation_with_unhashable_beliefs():
    """Unhashable beliefs disable branch merging but leave the exact performance unchanged."""
    code = stop_and_wait_arq(repetition(3))
    memoryless = code.performance(binary_symmetric_channel(0.1), method="exact", max_uses=12)
    for channel, expected in ((one_state_bsc(0.1), memoryless), (gilbert_elliott(), None)):
        merged = code.performance(with_feedback(channel), method="exact", max_uses=12)
        unmerged = code.performance(
            ListBeliefChannel(channel.T, channel.pi, channel.inputs, channel.outputs), method="exact", max_uses=12
        )
        assert unmerged == pytest.approx(merged, rel=1e-9)
        if expected is not None:
            assert merged == pytest.approx(expected, rel=1e-9)


def test_auto_falls_back_to_montecarlo_beyond_the_exact_budget(monkeypatch):
    """'auto' samples once the exact output tree outgrows its budget; 'exact' never gives up."""
    code, channel = horstein(2, n=6), binary_symmetric_channel(0.15)
    exact = code.performance(channel, method="exact")
    assert code.performance(channel, method="auto") == exact
    # Horstein's posterior states never merge, so the exact tree has 4 * (2 + 4 + ... + 64) = 504 branches.
    monkeypatch.setattr("dit.coding.feedback._EXACT_BUDGET", 100)
    auto = code.performance(channel, method="auto", samples=400, rng=np.random.default_rng(3))
    sampled = code.performance(channel, method="montecarlo", samples=400, rng=np.random.default_rng(3))
    assert auto == sampled
    assert auto[0] == pytest.approx(exact[0], abs=0.05)
    assert auto[1] == 6
    assert code.performance(channel, method="exact") == exact


def test_performance_rejects_an_unknown_method():
    """Only 'exact', 'montecarlo' and 'auto' are evaluation methods."""
    with pytest.raises(ditException, match="Unknown method 'simulate'"):
        erasure_arq(1).performance(binary_erasure_channel(0.1), method="simulate")


def test_capacity_gap_needs_a_capacity():
    """Without feedback_capacity() or capacity() there is no capacity to compare the rate with."""
    with pytest.raises(ditException, match="does not report a capacity"):
        erasure_arq(1).capacity_gap(NoCapacityChannel(binary_erasure_channel(0.25)))


def test_capacity_gap_with_a_bracketed_feedback_capacity():
    """feedback_capacity() takes precedence over capacity(), and bracketing it brackets the gap."""
    channel = BracketedChannel(binary_erasure_channel(0.25))
    gap = erasure_arq(2).capacity_gap(
        channel, capacity_kwargs={"lower": 0.7, "upper": 0.8}, method="exact", max_uses=60
    )
    # Erasure ARQ achieves the BEC capacity 1 - epsilon = 0.75 exactly.
    assert isinstance(gap, tuple)
    assert gap == pytest.approx((0.7 - 0.75, 0.8 - 0.75), abs=1e-6)


def test_horstein_needs_binary_inputs():
    """Horstein's bisection is only defined for the inputs {0, 1}."""
    with pytest.raises(ditException, match="binary-input channel"):
        horstein(1, n=2).probability_of_error(q_ary_symmetric_channel(3, 0.1))


def test_horstein_rejects_an_impossible_output():
    """Once a noiseless channel has fixed the posterior, the other output is impossible under every message."""
    code = horstein(1, n=3)
    state = code.start(MemorylessChannel(binary_symmetric_channel(0.0)))
    assert (code.next_input(state, (0,)), code.next_input(state, (1,))) == (0, 1)
    state = code.update(state, 1)
    assert code.decode(state) == (1,)
    assert code.next_input(state, (1,)) == 1
    with pytest.raises(ditException, match="impossible under every message"):
        code.update(state, 0)
