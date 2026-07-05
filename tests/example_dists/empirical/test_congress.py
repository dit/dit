"""
Tests for dit.example_dists.empirical.congress
"""

import urllib.error

import pytest

from dit.example_dists.empirical import congress
from dit.multivariate import coinformation as I
from dit.multivariate import entropy as H


@pytest.fixture(scope="module")
def d():
    """
    The congressional voting distribution, fetched once per module. Skips if the
    source data is unreachable (e.g. no network).
    """
    try:
        return congress()
    except (RuntimeError, urllib.error.URLError) as e:  # pragma: no cover
        pytest.skip(f"Congressional voting dataset unavailable: {e}")


def test_normalized(d):
    """
    The empirical pmf sums to one.
    """
    assert sum(d.pmf) == pytest.approx(1.0)


def test_rv_names(d):
    """
    The seventeen random variables are named: Party plus the sixteen votes.
    """
    names = d.get_rv_names()
    assert names[0] == "Party"
    assert names[4] == "PhysicianFeeFreeze"
    assert len(names) == 17


def test_support(d):
    """
    342 distinct voting patterns occur among the 435 members.
    """
    assert len(d.outcomes) == 342


def test_party_entropy(d):
    """
    Party affiliation carries ~0.962 bits (267 democrats, 168 republicans).
    """
    assert H(d, ["Party"]) == pytest.approx(0.9623, abs=1e-4)


def test_entropy(d):
    """
    Test the joint entropy against its known value.
    """
    assert H(d) == pytest.approx(8.1811, abs=1e-4)


def test_party_information(d):
    """
    The sixteen votes jointly resolve essentially all of party affiliation,
    while individual votes vary from strong party proxies to near-uninformative.
    """
    votes = [nm for nm in d.get_rv_names() if nm != "Party"]
    assert I(d, [["Party"], votes]) == pytest.approx(0.9623, abs=1e-4)
    assert I(d, [["Party"], ["PhysicianFeeFreeze"]]) == pytest.approx(0.7400, abs=1e-4)
    assert I(d, [["Party"], ["WaterProject"]]) == pytest.approx(0.0004, abs=1e-4)
