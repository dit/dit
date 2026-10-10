"""
Tests for dit.multivariate.secret_key_agreement.one_way_skar.
"""

import itertools

import pytest

from dit import Distribution
from dit.multivariate.secret_key_agreement import one_way_skar


def _erasure_vs_symmetric(p=0.25, e=0.6, s=0.1):
    """
    X ~ Bern(p); Y is X through an erasure channel, Z is X through a binary
    symmetric channel, conditionally independent given X.
    """
    erasure = {0: {"0": 1 - e, "e": e}, 1: {"1": 1 - e, "e": e}}
    symmetric = {0: {"0": 1 - s, "1": s}, 1: {"0": s, "1": 1 - s}}
    outcomes, pmf = [], []
    for x, px in [(0, 1 - p), (1, p)]:
        for (y, py), (z, pz) in itertools.product(erasure[x].items(), symmetric[x].items()):
            outcomes.append(f"{x}{y}{z}")
            pmf.append(px * py * pz)
    return Distribution(outcomes, pmf)


def test_optimum_needs_public_v():
    """
    The optimum here requires a nontrivial V with |U| = 4 = |X|^2 and
    posteriors on the simplex boundary; |U| <= |X| caps the rate at 0.0080.
    """
    d = _erasure_vs_symmetric()
    assert one_way_skar(d, [0], [1], [2]) == pytest.approx(0.017073, abs=1e-5)


def test_reverse_roles():
    """
    Swapping Bob and Eve gives a rate achieved without V.
    """
    d = _erasure_vs_symmetric()
    assert one_way_skar(d, [0], [2], [1]) == pytest.approx(0.095816, abs=1e-5)
