"""
Tests for dit.example_dists.empirical.penguins
"""

import urllib.error

import pytest

from dit.example_dists.empirical import penguins
from dit.multivariate import coinformation as I
from dit.multivariate import entropy as H


@pytest.fixture(scope="module")
def d():
    """
    The Palmer Penguins distribution, fetched once per module. Skips if the
    source data is unreachable (e.g. no network).
    """
    try:
        return penguins()
    except (RuntimeError, urllib.error.URLError) as e:  # pragma: no cover
        pytest.skip(f"Palmer Penguins dataset unavailable: {e}")


def test_normalized(d):
    """
    The empirical pmf sums to one.
    """
    assert sum(d.pmf) == pytest.approx(1.0)


def test_rv_names(d):
    """
    The three random variables are named.
    """
    assert d.get_rv_names() == ("Species", "Island", "Sex")


def test_support(d):
    """
    Only 10 of the 18 possible species-island-sex combinations occur.
    """
    assert len(d.outcomes) == 10


def test_entropy(d):
    """
    Test the joint entropy against its known value.
    """
    assert H(d) == pytest.approx(3.2119, abs=1e-4)


def test_geographic_entanglement(d):
    """
    Species and island are strongly entangled (ecological sorting), while sex is
    essentially independent of species.
    """
    assert I(d, [["Species"], ["Island"]]) == pytest.approx(0.7419, abs=1e-4)
    assert I(d, [["Species"], ["Sex"]]) == pytest.approx(0.0001, abs=1e-4)
