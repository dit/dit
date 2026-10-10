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
    pid = PID_LN(d, ((0,), (1,)), (2,), seed=0)
    assert 0.017 <= pid[RED] <= 0.0226
    assert pid[((0, 1),)] < -0.015
    assert pid[RED] >= PID_Deg(d, ((0,), (1,)), (2,))[RED] - 1e-6
