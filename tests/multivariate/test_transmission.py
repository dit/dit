"""
Tests for dit.multivariate.transmission.
"""

import numpy as np
import pytest

import dit
from dit.algorithms import maxent_dist
from dit.multivariate import entropy as H
from dit.multivariate import total_correlation, transmission


def test_transmission_independence_is_total_correlation():
    """The independence model's transmission equals the total correlation."""
    d = dit.example_dists.Xor()
    assert transmission(d) == pytest.approx(total_correlation(d), abs=1e-6)


def test_transmission_default_is_independence():
    """The default structure is the independence model."""
    d = dit.example_dists.Xor()
    assert transmission(d) == pytest.approx(transmission(d, [[0], [1], [2]]), abs=1e-6)


def test_transmission_saturated_is_zero():
    """The saturated model captures all constraint, so transmission is zero."""
    d = dit.example_dists.Xor()
    assert transmission(d, [[0, 1, 2]]) == pytest.approx(0.0, abs=1e-6)


def test_transmission_xor_acyclic():
    """Xor concentrates all information in the 3-way interaction, so any proper
    decomposition loses all of it (1 bit)."""
    d = dit.example_dists.Xor()
    assert transmission(d, [[0, 1], [1, 2]]) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize(
    "structure",
    [
        [[0], [1], [2]],
        [[0, 1], [2]],
        [[0, 1], [1, 2]],
        [[0, 1], [0, 2], [1, 2]],
        [[0, 1, 2]],
    ],
)
def test_transmission_equals_entropy_gap(structure):
    """Transmission equals U(model) - U(data) (the constraint the model loses)."""
    rng = np.random.default_rng(2)
    outcomes = [f"{a}{b}{c}" for a in "01" for b in "01" for c in "01"]
    d = dit.Distribution(outcomes, rng.dirichlet(np.ones(8)))
    me = maxent_dist(d, structure)
    assert transmission(d, structure) == pytest.approx(H(me) - H(d), abs=1e-6)


def test_transmission_nonnegative_and_monotone():
    """Transmission is non-negative and the saturated model loses the least."""
    rng = np.random.default_rng(3)
    outcomes = [f"{a}{b}{c}" for a in "01" for b in "01" for c in "01"]
    d = dit.Distribution(outcomes, rng.dirichlet(np.ones(8)))
    t_indep = transmission(d, [[0], [1], [2]])
    t_acyclic = transmission(d, [[0, 1], [1, 2]])
    t_sat = transmission(d, [[0, 1, 2]])
    assert t_sat == pytest.approx(0.0, abs=1e-6)
    assert 0 <= t_acyclic <= t_indep + 1e-9
