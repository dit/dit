"""
Tests for dit.multivariate.channel_synthesis_rate.
"""

import numpy as np
import pytest
from hypothesis import given, settings

from dit import Distribution as D
from dit.example_dists import erasure_channel, reverse_erasure_channel, task_assignment
from dit.exceptions import ditException
from dit.multivariate import channel_synthesis_rate, wyner_common_information
from dit.multivariate.common_informations.channel_synthesis import ChannelSynthesis
from dit.multivariate.necessary_conditional_entropy import necessary_conditional_entropy
from dit.shannon import conditional_entropy, entropy, mutual_information
from dit.utils.testing import distributions
from tests._backends import backends

xor = D(["000", "011", "101", "110"], [1 / 4] * 4)


def h(p):
    return entropy(p)


def erasure_rate(t, p):
    """
    The closed-form channel synthesis rate of the erasure channel, from Cuff (2013) Section II-F.
    """
    rs = np.linspace(1 - p, min(2 * (1 - p), 1), 4001)
    return min(max(r, h(p) + r * (1 - h((1 - p) / r)) - t) for r in rs)


def even(m):
    return m + m % 2


@pytest.mark.flaky(reruns=5)
@pytest.mark.parametrize("backend", backends)
@pytest.mark.parametrize("p", [0.2, 0.7])
def test_erasure_wyner(p, backend):
    """
    Without common randomness the rate is the Wyner common information.
    """
    d = erasure_channel(p)
    rate = channel_synthesis_rate(d, [0], [1], backend=backend)
    assert rate == pytest.approx(1 if p < 1 / 2 else h(p), abs=1e-3)


@pytest.mark.flaky(reruns=5)
@pytest.mark.parametrize("p", [0.2, 0.5, 0.7])
@pytest.mark.parametrize("t", [0.1, 0.3, 0.6])
def test_erasure_curve(p, t):
    """
    The rate matches the closed-form trade-off curve.
    """
    d = erasure_channel(p)
    assert channel_synthesis_rate(d, [0], [1], common_randomness=t) == pytest.approx(erasure_rate(t, p), abs=2e-3)


@pytest.mark.flaky(reruns=5)
@pytest.mark.parametrize("p", [0.2, 0.7])
def test_erasure_mutual_information(p):
    """
    With the necessary conditional entropy of common randomness, the rate is the mutual information.
    """
    d = erasure_channel(p)
    t = necessary_conditional_entropy(d, [1], [0])
    assert t == pytest.approx(h(p))
    assert channel_synthesis_rate(d, [0], [1], common_randomness=t) == pytest.approx(1 - p, abs=1e-3)


@pytest.mark.flaky(reruns=5)
@pytest.mark.parametrize("p", [0.3, 0.6])
def test_reverse_erasure(p):
    """
    The reverse erasure channel interpolates between its Wyner common information and mutual information.
    """
    d = reverse_erasure_channel(p)
    assert channel_synthesis_rate(d, [0], [1]) == pytest.approx(wyner_common_information(d), abs=2e-3)
    t = necessary_conditional_entropy(d, [1], [0])
    rate = channel_synthesis_rate(d, [0], [1], common_randomness=t)
    assert rate == pytest.approx(mutual_information(d, [0], [1]), abs=2e-3)


@pytest.mark.flaky(reruns=5)
@pytest.mark.parametrize("m", [3, 4, 5])
def test_task_assignment(m):
    """
    The scatter channel's extreme rates match Cuff, Permuter, and Cover (2010) Example 7.
    """
    d = task_assignment(m)
    wyner = 2 - np.log2(even(m) / (even(m) - 1))
    assert channel_synthesis_rate(d, [0], [1]) == pytest.approx(wyner, abs=2e-3)
    t = necessary_conditional_entropy(d, [1], [0])
    assert t == pytest.approx(np.log2(m - 1))
    rate = channel_synthesis_rate(d, [0], [1], common_randomness=t)
    assert rate == pytest.approx(np.log2(m / (m - 1)), abs=2e-3)


@pytest.mark.flaky(reruns=5)
def test_broadcast():
    """
    Synthesizing a broadcast channel needs the multivariate Wyner common information without common randomness,
    and the mutual information with enough.
    """
    assert channel_synthesis_rate(xor, [0], [[1], [2]]) == pytest.approx(2.0, abs=1e-3)
    assert channel_synthesis_rate(xor, [0], [[1], [2]], common_randomness=2.0) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.flaky(reruns=5)
def test_public():
    """
    A public message needs common randomness of at least the Wyner common information.
    """
    d = erasure_channel(0.3)
    assert channel_synthesis_rate(d, [0], [1], common_randomness=1.0, public=True) == pytest.approx(1.0, abs=1e-3)
    t = entropy(d, [1])
    assert channel_synthesis_rate(d, [0], [1], common_randomness=t, public=True) == pytest.approx(0.7, abs=1e-3)


def test_public_infeasible():
    """
    Too little common randomness for a public channel raises.
    """
    with pytest.raises(ditException):
        channel_synthesis_rate(erasure_channel(0.3), [0], [1], common_randomness=0.5, public=True)


@pytest.mark.flaky(reruns=5)
@pytest.mark.parametrize("t", [0.0, 0.3])
def test_local_randomness(t):
    """
    At a corner point the common and local randomness together supply H[Y|X].
    """
    d = erasure_channel(0.3)
    cs = ChannelSynthesis(d, [[0], [1]], None, common_randomness=t, branch="joint")
    cs.optimize()
    input_term, joint_term = cs._terms(cs._optima)
    r_0 = joint_term + t - input_term
    assert r_0 + cs.local_randomness() == pytest.approx(conditional_entropy(d, [1], [0]), abs=1e-3)


@pytest.mark.flaky(reruns=5)
@settings(max_examples=5, deadline=None)
@given(dist=distributions(alphabets=(2, 2)))
def test_monotone(dist):
    """
    The rate is non-increasing in the common randomness, between the mutual information and the Wyner common
    information.
    """
    rates = [channel_synthesis_rate(dist, [0], [1], common_randomness=t) for t in (0.0, 0.25, 1.0)]
    assert all(a >= b - 1e-3 for a, b in zip(rates, rates[1:], strict=False))
    assert rates[-1] >= mutual_information(dist, [0], [1]) - 1e-6
    assert rates[0] <= wyner_common_information(dist) + 1e-3
