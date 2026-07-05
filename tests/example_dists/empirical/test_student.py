"""
Tests for dit.example_dists.empirical.student
"""

import urllib.error

import pytest

from dit.example_dists.empirical import student
from dit.multivariate import coinformation as I
from dit.multivariate import entropy as H


@pytest.fixture(scope="module")
def d():
    """
    The Student Performance distribution, fetched once per module. Skips if the
    source data is unreachable (e.g. no network).
    """
    try:
        return student()
    except (RuntimeError, urllib.error.URLError) as e:  # pragma: no cover
        pytest.skip(f"Student Performance dataset unavailable: {e}")


def test_normalized(d):
    """
    The empirical pmf sums to one.
    """
    assert sum(d.pmf) == pytest.approx(1.0)


def test_rv_names(d):
    """
    The four random variables are named.
    """
    assert d.get_rv_names() == ("Internet", "Romantic", "WeekendAlcohol", "Grade")


def test_support(d):
    """
    All 40 (2 x 2 x 5 x 2) combinations occur in the Portuguese course file.
    """
    assert len(d.outcomes) == 40


def test_entropy(d):
    """
    Test the joint entropy against its known value.
    """
    assert H(d) == pytest.approx(4.4323, abs=1e-4)


def test_grade_information(d):
    """
    The lifestyle attributes are only weakly informative about passing: even
    jointly they carry only a few hundredths of a bit about the grade outcome.
    """
    assert I(d, [["Grade"], ["WeekendAlcohol"]]) == pytest.approx(0.0102, abs=1e-4)
    assert I(d, [["Grade"], ["Internet", "Romantic", "WeekendAlcohol"]]) == pytest.approx(0.0349, abs=1e-4)
