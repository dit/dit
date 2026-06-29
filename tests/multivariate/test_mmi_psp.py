"""
Tests for dit.multivariate.mmi_psp.
"""

import pytest
from hypothesis import given, settings

from dit import Distribution as D
from dit.distconst import random_distribution
from dit.multivariate import caekl_mutual_information as J
from dit.multivariate.caekl_mutual_information import _caekl_by_partitions
from dit.multivariate.mmi_psp import caekl_mutual_information_psp
from dit.utils.testing import distributions


def test_mmi_psp_xor():
    d = D(["000", "011", "101", "110"], [1 / 4] * 4)
    assert caekl_mutual_information_psp(d, [[0], [1], [2]], None) == pytest.approx(0.5)


@pytest.mark.parametrize("n", [3, 4, 5, 6, 7])
@pytest.mark.parametrize("trial", range(5))
def test_mmi_psp_matches_partitions(n, trial):
    d = random_distribution(n, 2, alpha=(0.5,) * 2**n)
    rvs = [[i] for i in range(n)]
    assert caekl_mutual_information_psp(d, rvs, None) == pytest.approx(_caekl_by_partitions(d, rvs, None))


def test_mmi_psp_matches_partitions_n8():
    n = 8
    d = random_distribution(n, 2, alpha=(0.5,) * 2**n)
    rvs = [[i] for i in range(n)]
    assert caekl_mutual_information_psp(d, rvs, None) == pytest.approx(_caekl_by_partitions(d, rvs, None))


@given(dist=distributions(alphabets=(4, 4, 4, 4, 4)))
@settings(max_examples=50, deadline=None)
def test_caekl_psp_matches_partitions_five_quaternary(dist):
    """
    PSP and partition enumeration agree on random 5-var, alphabet-4 distributions.
    """
    rvs = [[i] for i in range(5)]
    assert caekl_mutual_information_psp(dist, rvs, None) == pytest.approx(_caekl_by_partitions(dist, rvs, None))


def test_caekl_psp_near_deterministic():
    """
    Near-degenerate distributions: tie-breaking among equal Fuse candidates
    must not merge a zero-entropy singleton with the wrong cluster.
    """
    dist = D(
        [(0, 0, 0, 0, 0), (0, 0, 0, 1, 1)],
        [9.999999900000002e-09, 0.9999999900000002],
    )
    rvs = [[i] for i in range(5)]
    assert caekl_mutual_information_psp(dist, rvs, None) == pytest.approx(_caekl_by_partitions(dist, rvs, None))


def test_caekl_uses_psp():
    d = random_distribution(5, 4, alpha=(0.5,) * 4**5)
    rvs = [[i] for i in range(5)]
    assert J(d) == pytest.approx(caekl_mutual_information_psp(d, rvs, None))
