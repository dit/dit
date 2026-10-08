"""
Tests for dit.multivariate.secret_key_agreement.secret_key_cost
"""

import pytest
from hypothesis import given, settings

from dit import Distribution
from dit.example_dists.intrinsic import bound_information, intrinsic_1, intrinsic_2
from dit.exceptions import ditException
from dit.multivariate import entropy, wyner_common_information
from dit.multivariate.secret_key_agreement import (
    information_of_formation,
    intrinsic_mutual_information,
    secret_key_cost,
)
from dit.multivariate.secret_key_agreement.secret_key_cost import SecretKeyCost
from dit.utils.testing import distributions


@pytest.mark.parametrize(
    "dist",
    [
        Distribution(["000", "001", "110", "111"], [1 / 4] * 4),
        Distribution(["000", "001", "010", "011", "110", "111"], [1 / 6] * 6),
    ],
)
def test_independent_eve(dist):
    """
    With Eve independent of XY, the secret key cost is the Wyner common information.
    """
    skc = secret_key_cost(dist, [[0], [1]], [2], rng=0)
    assert skc == pytest.approx(wyner_common_information(dist, [[0], [1]]), abs=1e-3)


@pytest.mark.parametrize(
    ("dist", "value"),
    [
        (Distribution(["000", "111"], [1 / 2] * 2), 0.0),
        (intrinsic_1, 0.0),
        (intrinsic_2, 1.5),
    ],
)
def test_known(dist, value):
    """
    Test against distributions whose cost meets the intrinsic mutual information.
    """
    skc = secret_key_cost(dist, [[0], [1]], [2], bound_v=4, rng=0)
    assert skc == pytest.approx(value, abs=1e-4)


def test_gap():
    """
    intrinsic_2 costs 1.5 bits to form, but only 1 bit can be extracted from it.
    """
    skc = secret_key_cost(intrinsic_2, [[0], [1]], [2], bound_v=4, rng=0)
    assert skc > intrinsic_2.secret_rate + 0.4


def test_bound_information():
    """
    Bound information has no extractable key, but a positive cost.
    """
    skc = secret_key_cost(bound_information, [[0], [1]], [2], rng=0)
    assert skc > 0
    assert skc >= intrinsic_mutual_information(bound_information, [[0], [1]], [2]) - 1e-6
    assert skc <= wyner_common_information(bound_information, [[0], [1]]) + 1e-4


@settings(max_examples=5, deadline=None)
@given(dist=distributions(alphabets=(2, 2, 2)))
def test_bounds(dist):
    """
    I[X:Y↓Z] <= K_c <= min(C[X:Y], C[X:Y|Z]).
    """
    skc = secret_key_cost(dist, [[0], [1]], [2], rng=0)
    imi = intrinsic_mutual_information(dist, [[0], [1]], [2])
    c = wyner_common_information(dist, [[0], [1]])
    cz = wyner_common_information(dist, [[0], [1]], [2])
    assert imi - 1e-3 <= skc <= min(c, cz) + 1e-3


@pytest.mark.parametrize(("degrade", "crvs"), [(False, [2]), (True, [])])
def test_feasible_initial(degrade, crvs):
    """
    The initial points are feasible, with objectives H[X|Z] and H[X].
    """
    opt = SecretKeyCost(intrinsic_2, [[0], [1]], [2])
    x = opt.construct_feasible_initial(degrade=degrade)
    assert opt.constraint_match_joint(x) == pytest.approx(0.0, abs=1e-12)
    obj = opt._objective()
    assert obj(opt, x) == pytest.approx(entropy(intrinsic_2, [0], crvs))


def test_names():
    """
    Test with random variable names, and the alias.
    """
    d = intrinsic_2.copy()
    d.set_rv_names("XYZ")
    skc = information_of_formation(d, ["X", "Y"], "Z", bound_v=4, rng=0)
    assert skc == pytest.approx(1.5, abs=1e-4)


def test_failures():
    """
    Test that it fails without two parties or without an eavesdropper.
    """
    with pytest.raises(ditException):
        secret_key_cost(intrinsic_1, [[0], [1], [2]], [2])
    with pytest.raises(ditException):
        secret_key_cost(intrinsic_1, [[0], [1]], [])
