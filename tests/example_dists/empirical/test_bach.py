"""
Tests for dit.example_dists.empirical.bach
"""

import pytest

from dit.multivariate import o_information

music21 = pytest.importorskip("music21")

from dit.example_dists.empirical import bach  # noqa: E402


@pytest.fixture(scope="module")
def d():
    """
    The Bach chorales distribution, built once per module from a capped number of
    chorales to keep the test fast. Skips if music21 or its corpus is unavailable.
    """
    try:
        return bach(limit=40)
    except Exception as e:  # pragma: no cover
        pytest.skip(f"Bach chorales unavailable: {e}")


def test_normalized(d):
    """
    The empirical pmf sums to one.
    """
    assert sum(d.pmf) == pytest.approx(1.0)


def test_rv_names(d):
    """
    The four voices are named.
    """
    assert d.get_rv_names() == ("Soprano", "Alto", "Tenor", "Bass")


def test_synergy(d):
    """
    The chorales are synergy-dominated: the O-information is negative.
    """
    assert o_information(d) < 0
