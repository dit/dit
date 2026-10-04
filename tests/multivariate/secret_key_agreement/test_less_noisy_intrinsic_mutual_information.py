"""
Tests for dit.multivariate.secret_key_agreement.less_noisy_intrinsic_mutual_information
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from dit import Distribution
from dit.example_dists.intrinsic import bound_information, intrinsic_1, intrinsic_2, intrinsic_3
from dit.exceptions import ditException
from dit.multivariate import total_correlation
from dit.multivariate.secret_key_agreement import (
    intrinsic_mutual_information,
    less_noisy_intrinsic_mutual_information,
)
from dit.multivariate.secret_key_agreement.less_noisy_intrinsic_mutual_information import (
    LessNoisyIntrinsicMutualInformation,
)
from dit.utils.testing import distributions


def test_bound_information():
    """
    The decoupling J of Pauwels, Gisin & Renner is dominated by Eve, so the bound
    vanishes, although the intrinsic mutual information does not.
    """
    lnimi = less_noisy_intrinsic_mutual_information(bound_information, [[0], [1]], [2], rng=0)
    assert lnimi == pytest.approx(0.0, abs=1e-6)
    assert intrinsic_mutual_information(bound_information, [[0], [1]], [2]) > 1e-4


def test_bound_information_witness():
    """
    The paper's J: Eve's view is less noisy than it, and it decouples X from Y.
    J = X is not dominated.
    """
    opt = LessNoisyIntrinsicMutualInformation(bound_information, [[0], [1]], [2], bound=2, rng=0)
    wj = np.array([[4 / 5, 1 / 5], [1 / 2, 1 / 2], [1 / 2, 1 / 2], [1 / 5, 4 / 5]])
    K = np.linalg.lstsq(opt._wz, wj, rcond=None)[0]
    assert opt._feasible(K, opt._check)
    assert opt.objective(K) == pytest.approx(0.0, abs=1e-9)
    wx = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=float)
    K = np.linalg.lstsq(opt._wz, wx, rcond=None)[0]
    assert not opt._feasible(K, opt._check)


@settings(max_examples=10, deadline=None)
@given(pmf=st.lists(st.floats(min_value=0.05, max_value=1.0), min_size=4, max_size=4))
def test_xor(pmf):
    """
    With Z = X xor Y, every known upper bound reduces to I[X:Y] (Abin & Gohari).
    """
    pmf = np.array(pmf) / sum(pmf)
    d = Distribution(["000", "011", "101", "110"], pmf)
    lnimi = less_noisy_intrinsic_mutual_information(d, [[0], [1]], [2], rng=0)
    assert lnimi == pytest.approx(total_correlation(d, [[0], [1]]), abs=1e-6)


@pytest.mark.parametrize(
    ("dist", "value"),
    [
        (intrinsic_1, 0.0),
        (intrinsic_2, 1.5),
    ],
)
def test_matches_intrinsic(dist, value):
    """
    Test against distributions where a degradation of Z is optimal.
    """
    lnimi = less_noisy_intrinsic_mutual_information(dist, [[0], [1]], [2], rng=0)
    assert lnimi == pytest.approx(value, abs=1e-5)


def test_intrinsic_3():
    """
    Strictly between the secret key rate and the intrinsic mutual information.
    """
    lnimi = less_noisy_intrinsic_mutual_information(intrinsic_3, [[0], [1]], [2], rng=0)
    assert intrinsic_3.secret_rate - 1e-6 <= lnimi < 1.39329 - 1e-3


def test_names():
    """
    Test with random variable names.
    """
    d = bound_information.copy()
    d.set_rv_names("XYZ")
    lnimi = less_noisy_intrinsic_mutual_information(d, ["X", "Y"], "Z", rng=0)
    assert lnimi == pytest.approx(0.0, abs=1e-6)


def test_bound_one():
    """
    A constant J gives the mutual information.
    """
    lnimi = less_noisy_intrinsic_mutual_information(intrinsic_2, [[0], [1]], [2], bound=1)
    assert lnimi == pytest.approx(total_correlation(intrinsic_2, [[0], [1]]))


@settings(max_examples=10, deadline=None)
@given(dist=distributions(alphabets=(2, 2, 3)))
def test_trivial_upper_bound(dist):
    """
    Never more than min(I[X:Y], I[X:Y|Z]).
    """
    lnimi = less_noisy_intrinsic_mutual_information(dist, [[0], [1]], [2], rng=0)
    ub = min(total_correlation(dist, [[0], [1]]), total_correlation(dist, [[0], [1]], [2]))
    assert lnimi <= ub + 1e-6


def test_failures():
    """
    Test that it fails without two parties or without an eavesdropper.
    """
    with pytest.raises(ditException):
        less_noisy_intrinsic_mutual_information(intrinsic_1, [[0], [1], [2]], [2])
    with pytest.raises(ditException):
        less_noisy_intrinsic_mutual_information(intrinsic_1, [[0], [1]], [])
