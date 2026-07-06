"""
Tests for dit.example_dists.empirical.corelli
"""

import urllib.error

import pytest

from dit.multivariate import o_information

music21 = pytest.importorskip("music21")

from dit.example_dists.empirical import corelli  # noqa: E402


@pytest.fixture(scope="module")
def d():
    """
    The Corelli sonatas distribution, built once per module from a capped number
    of movements to keep the test fast. Skips if music21 is unavailable or the
    source scores cannot be fetched (e.g. no network).
    """
    try:
        return corelli(limit=15)
    except (RuntimeError, urllib.error.URLError) as e:  # pragma: no cover
        pytest.skip(f"Corelli sonatas unavailable: {e}")


def test_normalized(d):
    """
    The empirical pmf sums to one.
    """
    assert sum(d.pmf) == pytest.approx(1.0)


def test_rv_names(d):
    """
    The four voices are named.
    """
    assert d.get_rv_names() == ("Violin1", "Violin2", "Violone", "Organo")


def test_redundancy(d):
    """
    The sonatas are redundancy-dominated: the O-information is positive.
    """
    assert o_information(d) > 0
