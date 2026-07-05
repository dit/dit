"""
Tests for dit.multivariate.cross_mutual_information.
"""

import pytest

from dit import Distribution
from dit.distconst import random_distribution
from dit.multivariate import caekl_mutual_information as J
from dit.multivariate import coinformation as I
from dit.multivariate import cross_caekl_mutual_information as CJ
from dit.multivariate import cross_coinformation as CI
from dit.multivariate import cross_dual_total_correlation as CB
from dit.multivariate import cross_total_correlation as CT
from dit.multivariate import dual_total_correlation as B
from dit.multivariate import total_correlation as T
from dit.shannon import mutual_information as MI

# A test distribution with a positive X-Y association and a full-support
# reference so that all outcomes of the test data are supported.
POS = Distribution(["00", "01", "10", "11"], [0.4, 0.1, 0.1, 0.4])
NEG = Distribution(["00", "01", "10", "11"], [0.1, 0.4, 0.4, 0.1])
INDEP = Distribution(["00", "01", "10", "11"], [0.25] * 4)


@pytest.mark.parametrize(
    "cross, base",
    [
        (CI, I),
        (CT, T),
        (CB, B),
        (CJ, J),
    ],
)
def test_reduces_to_base(cross, base):
    """When p == q, the cross measure equals the conventional measure."""
    d = random_distribution(3, 2, alpha=(1,) * 8)
    assert cross(d, d) == pytest.approx(base(d))


@pytest.mark.parametrize(
    "cross, base",
    [
        (CI, I),
        (CT, T),
        (CB, B),
        (CJ, J),
    ],
)
def test_reduces_to_base_conditional(cross, base):
    """The reduction to the base measure holds for conditional forms too."""
    d = random_distribution(3, 2, alpha=(1,) * 8)
    assert cross(d, d, [[0], [1]], [2]) == pytest.approx(base(d, [[0], [1]], [2]))


def test_bivariate_equals_mutual_information():
    """For two variables with p == q, the cross MI is the mutual information."""
    assert CI(POS, POS) == pytest.approx(MI(POS, [0], [1]))


@pytest.mark.parametrize("cross", [CI, CT, CB])
def test_independent_reference_is_zero(cross):
    """An independent reference distribution gives a cross measure of zero."""
    assert cross(POS, INDEP) == pytest.approx(0.0)


def test_cross_mi_can_be_negative():
    """
    The cross MI can be negative when the test dependence is surprising
    relative to the reference dependence.
    """
    assert CI(NEG, POS) < 0


def test_rv_names():
    """The cross measures respect random variable names."""
    p = random_distribution(3, 2, alpha=(1,) * 8)
    p.set_rv_names("XYZ")
    q = random_distribution(3, 2, alpha=(1,) * 8)
    q.set_rv_names("XYZ")
    assert CT(p, q, [["X"], ["Y"]], ["Z"]) == pytest.approx(CT(p, q, [[0], [1]], [2]))


def test_cross_total_correlation_decomposition():
    """
    The cross total correlation obeys the same additive decomposition as the
    total correlation: CT[AB] == CT[A] + CT[B] + CI[A:B].
    """
    p = random_distribution(4, 2, alpha=(1,) * 16)
    q = random_distribution(4, 2, alpha=(1,) * 16)
    whole = CT(p, q)
    parts = CT(p, q, [[0], [1]]) + CT(p, q, [[2], [3]]) + CI(p, q, [[0, 1], [2, 3]])
    assert whole == pytest.approx(parts)


def test_cross_caekl_bivariate():
    """The cross CAEKL mutual information reduces to the cross MI bivariately."""
    p = random_distribution(2, 3, alpha=(1,) * 9)
    q = random_distribution(2, 3, alpha=(1,) * 9)
    assert CJ(p, q) == pytest.approx(CI(p, q))
