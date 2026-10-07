"""
Tests for dit.multivariate.secret_key_agreement.iterated_discarding.
"""

import numpy as np
import pytest
from hypothesis import given, settings

from dit import Distribution
from dit.example_dists.intrinsic import intrinsic_1, intrinsic_2
from dit.multivariate import coinformation as I
from dit.multivariate.secret_key_agreement import (
    intrinsic_mutual_information,
    iterated_discarding_skar,
    lower_intrinsic_mutual_information,
)
from dit.utils.testing import distributions

W = Distribution(["100", "010", "001"], [1 / 3] * 3)


def _h(p):
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def test_w_one_round():
    """
    One round: Alice discards 2/3 of her zeros, leaving cells (0.6, 0.2, 0.2).
    """
    skar = iterated_discarding_skar(W, [[0], [1]], [2], rounds=1, rng=0)
    assert skar == pytest.approx(5 / 9 * (_h(0.6) - _h(0.2)), abs=1e-6)


@pytest.mark.parametrize(("rounds", "value"), [(2, 0.1644), (3, 0.1737)])
def test_w_more_rounds(rounds, value):
    """
    More rounds tighten the bound on the W distribution.
    """
    skar = iterated_discarding_skar(W, [[0], [1]], [2], rounds=rounds, rng=0)
    assert skar == pytest.approx(value, abs=1e-3)


def test_intrinsic_examples():
    """
    Test against known secret key agreement rates.
    """
    assert iterated_discarding_skar(intrinsic_1, [[0], [1]], [2], rng=0) == pytest.approx(0.0, abs=1e-6)
    assert iterated_discarding_skar(intrinsic_2, [[0], [1]], [2], rng=0) == pytest.approx(1.0, abs=1e-6)


def test_no_eavesdropper():
    """
    Without an eavesdropper the bound is I[X : Y].
    """
    skar = iterated_discarding_skar(W, [[0], [1]], [], rounds=2, rng=0)
    assert skar == pytest.approx(I(W, [[0], [1]]), abs=1e-6)


@pytest.mark.parametrize("rounds", [1, 2])
def test_unequal_alphabets_symmetric(rounds):
    """
    With |X| != |Y|, the bound is defined and does not depend on which party is listed first.
    """
    dist = Distribution(["000", "011", "101", "110", "200", "211"], [0.3, 0.1, 0.1, 0.2, 0.1, 0.2])
    xy = iterated_discarding_skar(dist, [[0], [1]], [2], rounds=rounds, niter=50, rng=0)
    yx = iterated_discarding_skar(dist, [[1], [0]], [2], rounds=rounds, niter=50, rng=0)
    assert xy == pytest.approx(yx, abs=1e-4)
    assert lower_intrinsic_mutual_information(dist, [[0], [1]], [2]) <= xy + 1e-6
    assert xy <= intrinsic_mutual_information(dist, [[0], [1]], [2]) + 1e-4


@settings(max_examples=10)
@given(dist=distributions(alphabets=(2, 2, 2)))
def test_bounds(dist):
    """
    lower_intrinsic_mutual_information <= iterated discarding <= intrinsic mutual information.
    """
    rvs, crvs = [[0], [1]], [2]
    skar = iterated_discarding_skar(dist, rvs, crvs, rounds=2, niter=2, rng=0)
    assert iterated_discarding_skar(dist, rvs, crvs, rounds=0) == pytest.approx(
        lower_intrinsic_mutual_information(dist, rvs, crvs), abs=1e-6
    )
    assert lower_intrinsic_mutual_information(dist, rvs, crvs) <= skar + 1e-6
    assert skar <= intrinsic_mutual_information(dist, rvs, crvs) + 1e-4
