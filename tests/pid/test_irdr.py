"""
Tests for dit.pid.irdr.
"""

import math
from fractions import Fraction
from itertools import combinations, product

import numpy as np
import pytest
from hypothesis import given, settings

from dit import Distribution
from dit.pid.distributions import bivariates, trivariates
from dit.pid.measures.irdr import PID_RDR
from dit.shannon import mutual_information
from dit.utils.testing import distributions


def _sparse_dist(seed, sizes, sparsity=0.4):
    """A random distribution with structural zeros and non-dyadic probabilities."""
    rng = np.random.default_rng(seed)
    outcomes = ["".join(map(str, o)) for o in product(*(range(k) for k in sizes))]
    pmf = rng.dirichlet(np.full(len(outcomes), 0.7))
    pmf[rng.random(len(outcomes)) < sparsity] = 0
    pmf[np.argmax(pmf)] += 1 - pmf.sum()
    keep = pmf > 0
    return Distribution([o for o, k in zip(outcomes, keep, strict=True) if k], pmf[keep])


def _copy_of_target(seed, k_other, k_target, copy_first, alpha):
    """A distribution in which one source is a copy of the target."""
    rng = np.random.default_rng(seed)
    pmf = rng.dirichlet(np.full(k_other * k_target, alpha)).reshape(k_other, k_target)
    outcomes = [f"{t}{x}{t}" if copy_first else f"{x}{t}{t}" for x in range(k_other) for t in range(k_target)]
    return Distribution(outcomes, pmf.ravel())


def _exact_union(d, collection, target):
    """
    Union information of a collection of sources, with the ROC curves and their upper
    concave envelope computed in exact rational arithmetic.
    """
    tables = []
    for source in collection:
        table = {}
        for (s, t), p in d.coalesce([list(source), list(target)]).to_dict().items():
            table[s, t] = table.get((s, t), 0) + Fraction(p)
        tables.append(table)
    total = sum(tables[0].values())
    p_t = {}
    for (_, t), p in tables[0].items():
        p_t[t] = p_t.get(t, 0) + p / total
    value = 0.0
    for t, p in p_t.items():
        if not 0 < p < 1:
            continue
        points = []
        for table in tables:
            hit, miss = {}, {}
            for (s, u), q in table.items():
                if u == t:
                    hit[s] = hit.get(s, 0) + q / (p * total)
                else:
                    miss[s] = miss.get(s, 0) + q / ((1 - p) * total)
            vectors = [(hit.get(s, 0), miss.get(s, 0)) for s in set(hit) | set(miss)]
            vectors.sort(key=lambda v: (v[1] != 0, -v[0] / v[1] if v[1] else 0))
            fpr = tpr = Fraction(0)
            points.append((fpr, tpr))
            for x, y in vectors:
                fpr, tpr = fpr + y, tpr + x
                points.append((fpr, tpr))
        hull = []
        for q in sorted(set(points)):
            while len(hull) >= 2 and (hull[-1][0] - hull[-2][0]) * (q[1] - hull[-2][1]) >= (
                hull[-1][1] - hull[-2][1]
            ) * (q[0] - hull[-2][0]):
                hull.pop()
            hull.append(q)
        for (f0, r0), (f1, r1) in zip(hull, hull[1:], strict=False):
            x, y = r1 - r0, f1 - f0
            if x > 0:
                value += float(p * x) * math.log2(x / (p * x + (1 - p) * y))
    return value


def test_pid_rdr1():
    """
    Test rdr on a generic distribution (same as test_ibroja)
    """
    d = bivariates["diff"]
    pid = PID_RDR(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(0.12255624891826589, abs=1e-4)
    assert pid[((0,),)] == pytest.approx(0.18872187554086706, abs=1e-4)
    assert pid[((1,),)] == pytest.approx(0.18872187554086706, abs=1e-4)
    assert pid[((0, 1),)] == pytest.approx(0.0, abs=1e-4)
    assert pid.complete
    assert pid.nonnegative
    assert pid.consistent


def test_pid_rdr2():
    """
    Test irdr on reduced or. (same as broja)
    """
    d = bivariates["reduced or"]
    pid = PID_RDR(d)
    assert pid[((0,), (1,))] == pytest.approx(0.31127812445913305, abs=1e-4)
    assert pid[((0,),)] == pytest.approx(0.0, abs=1e-4)
    assert pid[((1,),)] == pytest.approx(0.0, abs=1e-4)
    assert pid[((0, 1),)] == pytest.approx(0.6887218755408672, abs=1e-4)
    assert pid.complete
    assert pid.nonnegative
    assert pid.consistent


def test_pid_rdr3():
    """
    Test that xor cat is complete, nonnegative and consistent.
    """
    d = trivariates["xor cat"]
    pid = PID_RDR(d)
    assert pid.complete
    assert pid.nonnegative
    assert pid.consistent


def test_pid_rdr4():
    """
    Test that dbl xor is complete, nonnegative and consistent.
    """
    d = trivariates["dbl xor"]
    pid = PID_RDR(d)
    assert pid.complete
    assert pid.nonnegative
    assert pid.consistent


def test_pid_rdr_source_copies_target():
    """
    When a source is a relabelled copy of the target, the redundancy is the information
    of the other source. These probabilities are not dyadic, which used to scramble the
    vertices of the join and gave a redundancy of 1.12 bits.
    """
    d = Distribution(["010", "101", "110"], [0.22424101414401437, 0.7399015190383124, 0.03585746681767324])
    pid = PID_RDR(d)
    assert pid.get_red(((0,), (1,))) == pytest.approx(mutual_information(d, [0], [2]), abs=1e-12)
    assert pid.nonnegative


@pytest.mark.parametrize(
    ("seed", "k_other", "k_target", "copy_first"),
    [(140, 2, 4, False)] + [(seed, 2 + seed % 3, 2 + seed % 2, seed % 4 < 2) for seed in range(12)],
)
def test_pid_rdr_copy_of_target(seed, k_other, k_target, copy_first):
    """A source which copies the target has redundancy I(other source; target)."""
    d = _copy_of_target(seed, k_other, k_target, copy_first, alpha=[0.3, 1.0][seed % 2])
    other = [1] if copy_first else [0]
    pid = PID_RDR(d)
    assert pid.get_red(((0,), (1,))) == pytest.approx(mutual_information(d, other, [2]), abs=1e-12)
    assert pid.nonnegative


@pytest.mark.parametrize(
    ("seed", "sizes"), [(seed, (2, 3, 3)) for seed in range(5)] + [(seed, (2, 2, 2, 3)) for seed in range(5)]
)
def test_pid_rdr_exact_arithmetic(seed, sizes):
    """Every redundancy agrees with exact rational arithmetic on sparse distributions."""
    d = _sparse_dist(seed, sizes)
    n = len(sizes) - 1
    pid = PID_RDR(d)
    for node in pid._lattice:
        expected = sum(
            (-1) ** (k + 1) * _exact_union(d, sub, (n,))
            for k in range(1, len(node) + 1)
            for sub in combinations(node, k)
        )
        assert pid.get_red(node) == pytest.approx(expected, abs=1e-12)


def test_pid_rdr_tuple_outcomes():
    """Outcomes written as tuples give the same decomposition as strings."""
    d = _sparse_dist(3, (2, 3, 3))
    d_tuples = Distribution([tuple(o) for o in d.outcomes], d.pmf)
    pid, pid_tuples = PID_RDR(d), PID_RDR(d_tuples)
    for node in pid._lattice:
        assert pid_tuples.get_red(node) == pytest.approx(pid.get_red(node), abs=1e-12)


def test_pid_rdr_multivariate_target():
    """A target of several variables gives the same decomposition as its coalesced variable."""
    d = _sparse_dist(4, (2, 2, 2, 2), sparsity=0.2)
    pid = PID_RDR(d, [[0], [1]], [2, 3])
    pid_coalesced = PID_RDR(d.coalesce([[0], [1], [2, 3]]), [[0], [1]], [2])
    for node in pid._lattice:
        assert pid.get_red(node) == pytest.approx(pid_coalesced.get_red(node), abs=1e-12)


def test_pid_rdr_constant_target():
    """A constant target carries no information, rather than raising ZeroDivisionError."""
    d = Distribution(["000", "110", "100"], [0.2, 0.3, 0.5])
    pid = PID_RDR(d)
    for node in pid._lattice:
        assert pid.get_red(node) == 0


@settings(max_examples=50, deadline=None)
@given(dist=distributions(alphabets=(2, 2, 2, 2)))
def test_pid_rdr_nonnegative(dist):
    """The decomposition is non-negative and consistent for any number of sources."""
    pid = PID_RDR(dist)
    assert pid.nonnegative
    assert pid.consistent
