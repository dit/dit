"""
Tests for dit.pid.emergence.
"""

from itertools import product

import numpy as np
import pytest
from hypothesis import given, settings

from dit import Distribution as D
from dit.pid import emergence_delta, emergence_gamma, emergence_psi
from dit.utils.testing import distributions

EPS = 0.1


def _h(p):
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def _noisy_copies(feature, n=2, eps=EPS):
    """
    `n` uniform bits, each copied forward through a binary symmetric channel
    with crossover `eps`, and a feature of them at both times. The variables
    are the n bits at time t, the n bits at time t', then the feature at t and t'.
    """
    outcomes, pmf = [], []
    for bits in product([0, 1], repeat=2 * n):
        x, noise = bits[:n], bits[n:]
        y = tuple(a ^ e for a, e in zip(x, noise, strict=True))
        outcomes.append((*x, *y, feature(x), feature(y)))
        pmf.append(2.0**-n * np.prod([eps if e else 1 - eps for e in noise]))
    return D(outcomes, pmf)


def _parity(bits):
    return sum(bits) % 2


def test_parity_is_causally_decoupled():
    """
    The parity of independent noisy bits predicts its own future but no single
    future bit, though the bits do not interact.
    """
    d = _noisy_copies(_parity)
    assert emergence_psi(d, [[0], [1]], [4], [5]) == pytest.approx(1 - _h(2 * EPS * (1 - EPS)))
    assert emergence_gamma(d, [[2], [3]], [4]) == pytest.approx(0.0, abs=1e-9)
    assert emergence_delta(d, [[0], [1]], [[2], [3]], [4]) == pytest.approx(-(1 - _h(EPS)))


def test_sum_of_independent_chains_is_emergent():
    """
    The sum of independent noisy bits, a discrete centre of mass, passes the
    emergence criterion without any interaction (Liardi et al. 2025).
    """
    d = _noisy_copies(sum)
    assert emergence_psi(d, [[0], [1]], [4], [5]) > 0.1
    assert emergence_delta(d, [[0], [1]], [[2], [3]], [4]) < 0


def test_downward_causation():
    """
    A future bit that copies the parity of the past bits is caused by the parity.
    """
    outcomes = [(a, b, a ^ b, c, a ^ b, a ^ b ^ c) for a, b, c in product([0, 1], repeat=3)]
    d = D(outcomes, [1 / 8] * 8)
    assert emergence_delta(d, [[0], [1]], [[2], [3]], [4]) == pytest.approx(1.0)
    assert emergence_gamma(d, [[2], [3]], [4]) == pytest.approx(1.0)
    assert emergence_psi(d, [[0], [1]], [4], [5]) == pytest.approx(0.0, abs=1e-9)


def test_order():
    """
    The parity of three noisy bits is hidden from every pair of bits, but not
    from all three.
    """
    d = _noisy_copies(_parity, n=3)
    past, future = [[0], [1], [2]], [[3], [4], [5]]
    flips = 3 * EPS * (1 - EPS) ** 2 + EPS**3
    mi = 1 - _h(flips)
    for k in (1, 2):
        assert emergence_psi(d, past, [6], [7], k=k) == pytest.approx(mi)
        assert emergence_gamma(d, future, [6], k=k) == pytest.approx(0.0, abs=1e-9)
    assert emergence_psi(d, past, [6], [7], k=3) == pytest.approx(0.0, abs=1e-9)
    assert emergence_gamma(d, future, [6], k=3) == pytest.approx(mi)


@settings(max_examples=10, deadline=None)
@given(dist=distributions(alphabets=(2, 2, 2, 2)))
def test_whole_system_never_emergent(dist):
    """
    With the whole system as the only group, Psi is at most zero for a feature
    that is a function of the system, by the data processing inequality.
    """
    feature = {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 2}
    d = D([(*o, feature[o[:2]], feature[o[2:]]) for o in dist.outcomes], dist.pmf)
    assert emergence_psi(d, [[0], [1]], [4], [5], k=2) <= 1e-9


def test_units():
    d = _noisy_copies(_parity)
    e = d.copy(base="e")
    assert emergence_psi(e, [[0], [1]], [4], [5]) == pytest.approx(emergence_psi(d, [[0], [1]], [4], [5]) * np.log(2))


@pytest.mark.parametrize("k", [0, 3])
def test_bad_order(k):
    d = _noisy_copies(_parity)
    with pytest.raises(ValueError):
        emergence_psi(d, [[0], [1]], [4], [5], k=k)
    with pytest.raises(ValueError):
        emergence_delta(d, [[0], [1]], [[2], [3]], [4], k=k)
    with pytest.raises(ValueError):
        emergence_gamma(d, [[2], [3]], [4], k=k)
