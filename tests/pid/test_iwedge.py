"""
Tests for dit.pid.iwedge.
"""

import numpy as np
import pytest

from dit import Distribution
from dit.algorithms import insert_meet
from dit.multivariate import coinformation
from dit.pid.distributions import bivariates, trivariates
from dit.pid.measures.iwedge import PID_GK, _meet_information


def test_pid_gk1():
    """
    Test iwedge on a generic distribution.
    """
    d = bivariates["imp. rdn"]
    pid = PID_GK(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(0.0)
    assert pid[((0,),)] == pytest.approx(1.0)
    assert pid[((1,),)] == pytest.approx(0.98959)
    assert pid[((0, 1),)] == pytest.approx(-0.98959)


def test_pid_gk2():
    """
    Test iwedge on another generic distribution.
    """
    d = trivariates["sum"]
    pid = PID_GK(d, [[0], [1], [2]], [3])
    for atom in pid._lattice:
        if atom in [((0, 1, 2),), ((0,),), ((1,),), ((2,),)]:
            assert pid[atom] == pytest.approx(0.31127812445913294)
        elif atom in [((0, 1),), ((0, 2),), ((1, 2),)]:
            assert pid[atom] == pytest.approx(0.18872187554086706)
        else:
            assert pid[atom] == pytest.approx(0.0)


def test_meet_matches_sigma_algebra():
    """
    The union-find meet agrees with the sigma-algebra meet.
    """
    rng = np.random.default_rng(0)
    outcomes = [f"{a}{b}{c}{t}" for a in "012" for b in "01" for c in "012" for t in "01"]
    for _ in range(20):
        k = rng.integers(3, 10)
        d = Distribution(list(rng.choice(outcomes, size=k, replace=False)), rng.dirichlet(np.ones(k)))
        for node in PID_GK(d, [[0], [1], [2]], [3])._lattice:
            coalesced = d.coalesce(node + ((3,),))
            coalesced = Distribution(coalesced.outcomes, coalesced.pmf)
            coalesced = insert_meet(coalesced, -1, coalesced.rvs[:-1], support_only=True)
            expected = coinformation(coalesced, [coalesced.rvs[-2], coalesced.rvs[-1]])
            assert _meet_information(d, node, (3,)) == pytest.approx(expected, abs=1e-12)


def test_pid_gk_large_alphabets():
    """
    Sources with many values no longer need exponentially large sigma-algebras.
    """
    d = bivariates["rdn xor"]
    pairs = [(a, b) for a in d.outcomes for b in d.outcomes]
    big = Distribution([(a[0] + b[0], a[1] + b[1], a[2] + b[2]) for a, b in pairs], [1 / len(pairs)] * len(pairs))
    assert PID_GK(big)[((0,), (1,))] == pytest.approx(2.0)
