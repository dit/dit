"""
Regression tests for PID bugs found in review.
"""

import numpy as np
import pytest

from dit import Distribution
from dit.multivariate import coinformation
from dit.pid import PID_GH, PID_MES, PID_MMI, PID_PM, PID_RR, PID_SX, PID_Do
from dit.pid.distributions import bivariates, trivariates

AND_TARGET_LAST = Distribution(["000", "010", "100", "111"], [1 / 4] * 4)
AND_TARGET_FIRST = Distribution(["000", "001", "010", "111"], [1 / 4] * 4)


@pytest.mark.parametrize("measure", [PID_SX, PID_PM])
def test_target_position_does_not_matter(measure):
    last = measure(AND_TARGET_LAST, [[0], [1]], [2])
    first = measure(AND_TARGET_FIRST, [[1], [2]], [0])
    for node_last, node_first in [
        (((0, 1),), ((1, 2),)),
        (((0,),), ((1,),)),
        (((1,),), ((2,),)),
        (((0,), (1,)), ((1,), (2,))),
    ]:
        assert last[node_last] == pytest.approx(first[node_first], abs=1e-9)


def test_mes_ignores_extra_variables():
    d = bivariates["redundant"]
    extra = Distribution([(*o, n) for o in d.outcomes for n in "01"], [p / 2 for p in d.pmf for _ in "01"])
    pid = PID_MES(extra, [[0], [1]], [2])
    assert pid.get_red(((0,), (1,))) == pytest.approx(1.0, abs=1e-6)


def test_mes_trivariate_consistent():
    pid = PID_MES(trivariates["redundant"])
    assert pid.get_red(((0,), (1,), (2,))) == pytest.approx(1.0, abs=1e-4)


def test_gh_redundancy_bounded_by_mi():
    d = Distribution(["0000", "0010", "0100", "0110", "1000", "1010", "1100", "1111"], [1 / 8] * 8)
    red = PID_GH._measure(d, ((0,), (1,)), (3,))
    assert red <= coinformation(d, [[0], [3]]) + 1e-6


def test_rr_constant_source():
    d = Distribution(["000", "011"], [1 / 2, 1 / 2])
    pid = PID_RR(d, [[0], [1]], [2])
    assert not any(np.isnan(pid[node]) for node in pid._lattice)


def test_incomplete_pid_atoms_sum_to_total():
    """
    With two sources the redundancy fixes the whole lattice; atoms must sum to I.
    """
    d = bivariates["simple"]
    pid = PID_Do(d, [[0], [1]], [2])
    total = coinformation(d, [[0, 1], [2]])
    assert sum(pid[node] for node in pid._lattice) == pytest.approx(total, abs=1e-6)


def test_reduced_pid_forwards_kwargs():
    seen = []

    class Spy(PID_MMI):
        @staticmethod
        def _measure(d, sources, target, flag=None):
            seen.append(flag)
            return PID_MMI._measure(d, sources, target)

    Spy(trivariates["sum"], flag="x")
    assert set(seen) == {"x"}
