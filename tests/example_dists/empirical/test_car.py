"""
Tests for dit.example_dists.empirical.car
"""

import urllib.error

import pytest

from dit.example_dists.empirical import car
from dit.multivariate import coinformation as I
from dit.multivariate import entropy as H


@pytest.fixture(scope="module")
def d():
    """
    The Car Evaluation distribution, fetched once per module. Skips if the
    source data is unreachable (e.g. no network).
    """
    try:
        return car()
    except (RuntimeError, urllib.error.URLError) as e:  # pragma: no cover
        pytest.skip(f"Car Evaluation dataset unavailable: {e}")


def test_normalized(d):
    """
    The empirical pmf sums to one.
    """
    assert sum(d.pmf) == pytest.approx(1.0)


def test_rv_names(d):
    """
    The four random variables are named.
    """
    assert d.get_rv_names() == ("Buying", "Maintenance", "Safety", "Decision")


def test_support(d):
    """
    Only 82 of the 4 x 4 x 3 x 4 = 192 combinations occur; the deterministic
    evaluation rules zero out the rest.
    """
    assert len(d.outcomes) == 82


def test_entropy(d):
    """
    Test the joint entropy against its known value.
    """
    assert H(d) == pytest.approx(6.2013, abs=1e-4)


def test_deterministic_rule(d):
    """
    Safety is strongly informative about the decision, and encodes a hard
    logical rule: when safety is ``Low``, the decision is never ``Good`` or
    ``VeryGood`` -- that mass is exactly zero.
    """
    assert I(d, [["Safety"], ["Decision"]]) == pytest.approx(0.2622, abs=1e-4)

    # No joint state pairs Low safety with a Good/VeryGood decision.
    low_and_good = [
        p
        for (_, _, safety, decision), p in zip(d.outcomes, d.pmf, strict=True)
        if safety == "Low" and decision in ("Good", "VeryGood")
    ]
    assert sum(low_and_good) == pytest.approx(0.0)
