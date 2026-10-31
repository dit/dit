"""
Tests for dit.pid.measures.iln (less-noisy intersection information I_ln^∩).
"""

import itertools

import numpy as np
import pytest

from dit import Distribution
from dit.pid.distributions import bivariates
from dit.pid.measures.ideg import PID_Deg
from dit.pid.measures.iln import PID_LN
from dit.pid.measures.imc import PID_MC

RED = ((0,), (1,))


@pytest.mark.flaky(reruns=5)
@pytest.mark.parametrize(
    "name, value",
    [("and", 0.3113), ("sum", 0.5), ("unique 1", 0.0), ("redundant", 1.0), ("synergy", 0.0), ("cat", 0.0)],
)
def test_pid_ln_known(name, value):
    """
    Values shared by all three channel-order measures (Gomes & Figueiredo, Table 2).
    """
    pid = PID_LN(bivariates[name], ((0,), (1,)), (2,))
    assert pid[RED] == pytest.approx(value, abs=1e-3)


@pytest.mark.flaky(reruns=5)
def test_pid_ln_diff():
    """
    Opposite Z-channels: the optimal Q is a binary symmetric channel with crossover
    about 0.28; a brute-force search over binary Q gives 0.149.
    """
    pid = PID_LN(bivariates["diff"], ((0,), (1,)), (2,))
    assert pid[RED] == pytest.approx(0.1495, abs=2e-3)


@pytest.mark.flaky(reruns=5)
def test_pid_ln_below_mc():
    """
    Less noisy implies more capable, so I_ln^∩ <= I_mc^∩.
    """
    for name in ["and", "sum", "diff", "reduced or", "pnt. unq."]:
        d = bivariates[name]
        ln = PID_LN(d, ((0,), (1,)), (2,))[RED]
        mc = PID_MC(d, ((0,), (1,)), (2,))[RED]
        assert ln <= mc + 1e-3, f"I_ln^∩ > I_mc^∩ on '{name}': {ln:.4f} > {mc:.4f}"


def test_pid_ln_tied_rows():
    """
    X1 cannot separate Y=1 from Y=2, so neither can Q. A brute-force search gives a
    redundancy of about 0.0216 bits; the optimizer returns a feasible Q, so a value at
    most that, and how close it gets varies by platform. Anything in this range gives
    a negative synergy, since the coinformation is 0.043 bits.
    """
    W0 = np.array([[0.2, 0.5, 0.3], [0, 0, 1], [0.1, 0.9, 0]])
    W1 = np.array([[0.1, 0, 0.9], [0.05, 0.5, 0.45], [0.05, 0.5, 0.45]])
    p = np.array([0.3, 0.55, 0.15])
    outcomes, pmf = [], []
    for y, a, b in itertools.product(range(3), repeat=3):
        w = p[y] * W0[y, a] * W1[y, b]
        if w > 0:
            outcomes.append(f"{a}{b}{y}")
            pmf.append(w)
    d = Distribution(outcomes, pmf)
    pid = PID_LN(d, ((0,), (1,)), (2,), rng=0)
    assert 0.017 <= pid[RED] <= 0.0226
    assert pid[((0, 1),)] < -0.015
    assert pid[RED] >= PID_Deg(d, ((0,), (1,)), (2,))[RED] - 1e-6


def _from_channels(pi_t, *channels):
    """Sources independent given the target, which is the last variable."""
    outcomes, pmf = [], []
    for t, p in enumerate(pi_t):
        for ys in itertools.product(*(range(ch.shape[1]) for ch in channels)):
            w = p * np.prod([ch[t, y] for ch, y in zip(channels, ys, strict=True)])
            if w > 0:
                outcomes.append((*ys, t))
                pmf.append(w)
    return Distribution(outcomes, pmf)


@pytest.mark.flaky(reruns=5)
def test_pid_ln_gomes_figueiredo():
    """
    Tables 3 and 4 of Gomes & Figueiredo (2023). In the first counterexample of Korner &
    Marton the second channel is less noisy than the first without being a degradation,
    so I_ln^∩ is I(T; Y2), about 0.004 bits, while I_d^∩ is about 0.002 bits. For the
    channels of Americo et al. they conjecture that I_ln^∩ vanishes.
    """
    k1 = np.array([[0.25, 0.75], [0.35, 0.65]])
    k2 = np.array([[0.675, 0.325], [0.745, 0.255]])
    d = _from_channels([0.4, 0.6], k1, k2)
    assert PID_LN(d, rng=0)[RED] == pytest.approx(0.00415, abs=5e-5)
    k3 = np.array([[1, 0], [0, 1], [0.5, 0.5]])
    k4 = np.array([[1, 0], [1, 0], [0.5, 0.5]])
    assert PID_LN(_from_channels([0.3, 0.3, 0.4], k3, k4), rng=0)[RED] == pytest.approx(0.0, abs=1e-6)


def test_pid_ln_constant_target():
    """
    A target with a single value, as in a conditional distribution, carries no
    information. Its simplex has no tangent directions, which used to raise an error.
    """
    d = Distribution(["000", "010", "100", "110"], [1 / 4] * 4)
    pid = PID_LN(d, ((0,), (1,)), (2,))
    assert pid[RED] == 0.0
    assert pid.complete
    assert pid.consistent


AND3 = Distribution(["0000", "0010", "0100", "0110", "1000", "1010", "1100", "1111"], [1 / 8] * 8)


def test_pid_ln_three_sources():
    """
    I_ln^∩ is defined for any number of sources. The three channels of the three-way AND
    gate are identical, so the redundancy of all three is I(X_i : Y).
    """
    pid = PID_LN(AND3, [[0], [1], [2]], [3], rng=0)
    assert pid.get_red(((0,), (1,), (2,))) == pytest.approx(0.13793, abs=1e-4)
    assert pid.complete
    assert pid.consistent


@pytest.mark.flaky(reruns=5)
def test_pid_ln_three_sources_ordering():
    """
    I_d^∩ <= I_ln^∩ <= I_mc^∩ with three sources too. Here the less-noisy redundancy,
    about 0.0092 bits, exceeds the degradation redundancy, about 0.0082.
    """
    rng = np.random.default_rng(3)
    outcomes = ["".join(map(str, o)) for o in itertools.product(range(2), repeat=4)]
    d = Distribution(outcomes, rng.dirichlet(np.full(16, 0.5)))
    node = ((0,), (1,), (2,))
    deg = PID_Deg._measure(d, node, (3,))
    ln = PID_LN._measure(d, node, (3,), rng=0)
    mc = PID_MC._measure(d, node, (3,), rng=0)
    assert deg + 5e-4 < ln <= mc + 1e-3
