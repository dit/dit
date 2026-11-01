"""
Partial information decompositions on degenerate inputs: constant sources or targets,
duplicated variables, a single source, and malformed sources.
"""

import itertools

import numpy as np
import pytest

from dit import Distribution
from dit.exceptions import ditException
from dit.multivariate import coinformation
from dit.pid import (
    PID_BROJA,
    PID_CCS,
    PID_CI,
    PID_CT,
    PID_GH,
    PID_GK,
    PID_IG,
    PID_IPID,
    PID_LN,
    PID_MC,
    PID_MES,
    PID_MMI,
    PID_PM,
    PID_RA,
    PID_RAV,
    PID_RDR,
    PID_RR,
    PID_SX,
    PID_WB,
    PID_Deg,
    PID_Delta,
    PID_DeltaLambda,
    PID_dep,
    PID_dep_b,
    PID_Do,
    PID_Prec,
    PID_Proj,
    PID_SKAR_nw,
    PID_SKAR_owa,
    PID_SKAR_owb,
    PID_SKAR_tw,
)
from dit.pid.distributions import bivariates, trivariates
from dit.pid.measures import __all_pids
from dit.pid.pid import BaseIncompletePID

# options keeping the optimization-based measures fast and reproducible
OPTIONS = {
    PID_MC: {"niter": 3, "rng": 0},
    PID_LN: {"niter": 3, "rng": 0},
    PID_IPID: {"niter": 2, "rng": 0},
    PID_DeltaLambda: {"niter": 2, "rng": 0},
}

MEASURES = [pytest.param(measure, id=measure.__name__) for measure in __all_pids]

# measures known to give inconsistent unique informations (secret key rates and
# reconstructability analysis), so that the values filled into a lattice depend on
# which of the inconsistent equations are used
INCONSISTENT = {PID_SKAR_nw, PID_SKAR_owa, PID_SKAR_owb, PID_SKAR_tw, PID_RA, PID_dep_b}

# measures whose redundancy of two copies of a source is less than its information:
# they fail subset equality (I_do, and I_ME* which equals it)
NOT_SUBSET_EQUAL = {PID_Do, PID_MES}

ATOL = {PID_MC: 1e-3, PID_LN: 1e-3, PID_IPID: 1e-3, PID_DeltaLambda: 1e-3, PID_Delta: 1e-4, PID_dep: 1e-4}


def _from_table(table):
    table = {outcome: p for outcome, p in table.items() if p > 0}
    return Distribution(list(table), list(table.values()))


def _pid(measure, dist, sources=None, target=None):
    return measure(dist, sources, target, **OPTIONS.get(measure, {}))


def _atol(measure):
    return ATOL.get(measure, 1e-6)


def _mi(dist, rvs, target):
    return coinformation(dist, [list(rvs), list(target)])


CONSTANT_TARGET = {
    "independent sources": _from_table({(a, b, "0"): 1 / 4 for a in "01" for b in "01"}),
    "correlated sources": _from_table(
        {("0", "0", "0"): 0.4, ("0", "1", "0"): 0.1, ("1", "0", "0"): 0.1, ("1", "1", "0"): 0.4}
    ),
    "constant sources": _from_table({("0", "0", t): 1 / 2 for t in "01"}),
    "one outcome": _from_table({("0", "0", "0"): 1.0}),
}

# X0 is constant, and X1 a noisy copy of T, or a ternary noisy relative of a ternary T
CONSTANT_SOURCE = {
    "binary": _from_table({("0", x, t): 0.5 * (0.7 if x == t else 0.3) for x in "01" for t in "01"}),
    "ternary": _from_table(
        {("0", x, t): (0.5 if x == "0" else 0.25) * (0.8 if x == t else 0.1) for x in "012" for t in "012"}
    ),
}

# X1 is a copy of X0, or its negation, and T a noisy copy of X0
DUPLICATE = {
    "copy": _from_table({(x, x, t): (0.6 if x == "0" else 0.4) * (0.8 if x == t else 0.2) for x in "01" for t in "01"}),
    "negation": _from_table(
        {(x, str(1 - int(x)), t): (0.6 if x == "0" else 0.4) * (0.8 if x == t else 0.2) for x in "01" for t in "01"}
    ),
}


@pytest.mark.parametrize("measure", MEASURES)
@pytest.mark.parametrize("name", list(CONSTANT_TARGET))
def test_constant_target(measure, name):
    """
    A constant target carries no information, so every redundancy and atom vanishes.
    """
    pid = _pid(measure, CONSTANT_TARGET[name], [[0], [1]], [2])
    for node in pid._lattice:
        assert pid.get_red(node) == pytest.approx(0.0, abs=_atol(measure))
        assert pid.get_pi(node) == pytest.approx(0.0, abs=_atol(measure))


@pytest.mark.parametrize("measure", MEASURES)
@pytest.mark.parametrize("name", list(CONSTANT_SOURCE))
@pytest.mark.parametrize("position", [0, 1])
def test_constant_source(measure, name, position):
    """
    A constant source shares nothing, so the other source's information is all unique.
    """
    dist = CONSTANT_SOURCE[name]
    sources = [[0], [1]] if position == 0 else [[1], [0]]
    pid = _pid(measure, dist, sources, [2])
    atol = _atol(measure)
    assert pid.get_red(((0,), (1,))) == pytest.approx(0.0, abs=atol)
    assert pid.get_pi(((0,),)) == pytest.approx(0.0, abs=atol)
    if measure not in INCONSISTENT:
        assert pid.get_pi(((1,),)) == pytest.approx(_mi(dist, [1], [2]), abs=atol)
        assert pid.get_pi(((0, 1),)) == pytest.approx(0.0, abs=atol)


@pytest.mark.parametrize("measure", MEASURES)
@pytest.mark.parametrize("name", list(DUPLICATE))
def test_duplicate_source(measure, name):
    """
    Two copies of a source share all of their information, except for measures which
    fail subset equality.
    """
    dist = DUPLICATE[name]
    pid = _pid(measure, dist, [[0], [1]], [2])
    mi = _mi(dist, [0], [2])
    atol = _atol(measure)
    if measure in NOT_SUBSET_EQUAL:
        assert pid.get_red(((0,), (1,))) < mi - 1e-3
    elif measure not in INCONSISTENT:
        assert pid.get_red(((0,), (1,))) == pytest.approx(mi, abs=atol)
        assert pid.get_pi(((0,),)) == pytest.approx(0.0, abs=atol)
        assert pid.get_pi(((1,),)) == pytest.approx(0.0, abs=atol)
        assert pid.get_pi(((0, 1),)) == pytest.approx(0.0, abs=atol)


@pytest.mark.parametrize("measure", MEASURES)
def test_constant_third_source(measure):
    """
    Adding a constant source changes no redundancy: an element made of it alone shares
    nothing, and it adds nothing to any other element.
    """
    table = {}
    rng = np.random.default_rng(3)
    for (a, b, t), p in zip(itertools.product("01", repeat=3), rng.dirichlet(np.ones(8)), strict=True):
        table[(a, b, "0", t)] = p
    dist = _from_table(table)
    if measure in INCONSISTENT:
        pytest.skip("inconsistent unique informations")
    pid = _pid(measure, dist, [[0], [1], [2]], [3])
    two = _pid(measure, dist, [[0], [1]], [3])
    atol = _atol(measure)
    for node in pid._lattice:
        red = pid.get_red(node)
        if any(element == (2,) for element in node):
            assert red == pytest.approx(0.0, abs=atol), node
            continue
        reduced = {tuple(v for v in element if v != 2) for element in node}
        if all(not set(a) < set(b) for a in reduced for b in reduced):
            expected = two.get_red(tuple(sorted(reduced, key=lambda e: (len(e), e))))
        elif not isinstance(pid, BaseIncompletePID):
            reduced = tuple(sorted(reduced, key=lambda e: (len(e), e)))
            expected = measure._measure(dist, reduced, (3,), **OPTIONS.get(measure, {}))
        else:
            continue
        if not (np.isnan(red) or np.isnan(expected)):
            assert red == pytest.approx(expected, abs=atol), node


@pytest.mark.parametrize("measure", MEASURES)
def test_dense_and_log(measure):
    """
    Zero-probability outcomes change nothing, and a log base only changes the units.
    """
    reference = _pid(measure, bivariates["and"])
    dense = bivariates["and"].copy()
    dense.make_dense()
    dense = _pid(measure, dense)
    natural = _pid(measure, bivariates["and"].copy(base=np.e))
    atol = _atol(measure)
    for node in reference._lattice:
        assert dense.get_pi(node) == pytest.approx(reference.get_pi(node), abs=atol)
        assert natural.get_pi(node) == pytest.approx(np.log(2) * reference.get_pi(node), abs=atol)


@pytest.mark.parametrize("measure", MEASURES)
def test_integer_symbols(measure):
    """
    Outcomes need not be strings.
    """
    reference = _pid(measure, bivariates["and"])
    ints = _pid(measure, _from_table({(a, b, a & b): 1 / 4 for a in (0, 1) for b in (0, 1)}))
    for node in reference._lattice:
        assert ints.get_pi(node) == pytest.approx(reference.get_pi(node), abs=_atol(measure))


@pytest.mark.parametrize("measure", MEASURES)
def test_single_source(measure):
    """
    A single source has one atom, its mutual information with the target.
    """
    dist = bivariates["and"]
    pid = _pid(measure, dist, [[0]], [2])
    assert list(pid._lattice) == [((0,),)]
    assert pid.get_red(((0,),)) == pytest.approx(_mi(dist, [0], [2]), abs=1e-6)
    assert pid.get_pi(((0,),)) == pytest.approx(_mi(dist, [0], [2]), abs=1e-6)


@pytest.mark.parametrize(
    "sources, target, match",
    [
        ([[0], [0]], [2], "same source twice"),
        ([[0], [1, 0]], [2], "contain another source"),
        ([[0], [1], [0, 1]], [2], "union of other sources"),
        ([[0], [1]], [1], "shares variables with the target"),
        ([[0], []], [2], "nonempty"),
        ([], [2], "at least one source"),
        ([[0], [1]], [], "target must be nonempty"),
    ],
)
def test_malformed_sources(sources, target, match):
    """
    Sources which do not give a well-formed lattice are rejected.
    """
    with pytest.raises(ditException, match=match):
        PID_WB(bivariates["and"], sources, target)


@pytest.mark.parametrize("measure", MEASURES)
def test_unknown_option(measure):
    """
    An option the measure does not take is an error, rather than silently ignored.
    """
    with pytest.raises(TypeError, match=measure.__name__):
        measure(bivariates["and"], methd="exact", **OPTIONS.get(measure, {}))


def test_node_outside_lattice():
    """
    Asking for a node outside the lattice is an error that says so.
    """
    pid = PID_WB(bivariates["and"])
    with pytest.raises(ditException, match="not in the lattice"):
        pid.get_red(((0,), (2,)))


@pytest.mark.parametrize("measure", [PID_WB, PID_RR, PID_BROJA, PID_CI, PID_RDR, PID_SX])
def test_compute_false(measure):
    """
    Redundancies computed on demand equal those of the full computation, whatever the
    base class (a synergy measure used to return its synergy here).
    """
    full = measure(bivariates["and"])
    lazy = measure(bivariates["and"], compute=False)
    for node in full._lattice:
        assert lazy.get_red(node) == pytest.approx(full.get_red(node), abs=1e-6)
        assert lazy.get_pi(node) == pytest.approx(full.get_pi(node), abs=1e-6)


def test_no_recomputation():
    """
    Printing a decomposition, or checking its properties, does not recompute it.
    """
    calls = []

    class Spy(PID_BROJA):
        @staticmethod
        def _measure(d, sources, target, **kwargs):
            calls.append(1)
            return PID_BROJA._measure(d, sources, target, **kwargs)

    pid = Spy(bivariates["and"])
    count = len(calls)
    str(pid)
    assert pid.consistent
    assert pid.nonnegative
    assert pid.complete
    assert len(calls) == count


@pytest.mark.parametrize("measure", [PID_RR, PID_Do, PID_CT, PID_IG, PID_MES])
def test_four_sources(measure):
    """
    Two-source measures extend to four sources without exhaustive searches.
    """
    table = {(*bits, str(int(all(b == "1" for b in bits)))): 1 / 16 for bits in itertools.product("01", repeat=4)}
    pid = measure(_from_table(table))
    assert len(pid._lattice) == 166
    assert pid.consistent


@pytest.mark.parametrize("measure", [PID_WB, PID_SX, PID_MMI, PID_GK, PID_PM, PID_CCS])
def test_overlapping_sources(measure):
    """
    Sources may share variables: a node of two collections has no repeated variable.
    """
    dist = _from_table(
        {("0", "0", "0", "0"): 0.25, ("1", "0", "0", "1"): 0.25, ("1", "0", "1", "0"): 0.25, ("1", "1", "1", "1"): 0.25}
    )
    pid = measure(dist, [[0, 2], [1, 2]], [3])
    assert ((0, 1, 2),) in pid._lattice
    assert pid.get_red(((0, 1, 2),)) == pytest.approx(_mi(dist, [0, 1, 2], [3]), abs=1e-6)


def test_measures_registry():
    """
    The tests above cover every measure.
    """
    assert {
        PID_MMI,
        PID_GK,
        PID_WB,
        PID_RR,
        PID_CCS,
        PID_PM,
        PID_SX,
        PID_Proj,
        PID_GH,
        PID_BROJA,
        PID_MES,
        PID_dep,
        PID_RA,
        PID_RAV,
        PID_SKAR_nw,
        PID_SKAR_owa,
        PID_SKAR_owb,
        PID_SKAR_tw,
        PID_Prec,
        PID_CT,
        PID_IG,
        PID_IPID,
        PID_Delta,
        PID_DeltaLambda,
        PID_RDR,
        PID_Deg,
        PID_MC,
        PID_LN,
        PID_Do,
        PID_CI,
    } <= set(__all_pids)


@pytest.mark.parametrize("measure", [PID_Do, PID_MES])
def test_incomplete_lattice_is_consistent(measure):
    """
    Atoms are filled only as far as the computed redundancies determine them. The earlier
    search filled the lattice of three sources with atoms contradicting the computed
    redundancies (by 0.23 bits for I({0, 2} : T) here).
    """
    pid = measure(trivariates["anddup"])
    assert pid.consistent
    for node in pid._lattice:
        if len(node) == 1:
            computed = coinformation(pid._dist, [node[0], pid._target])
        elif len(node) == 2:
            computed = measure._measure(pid._dist, node, pid._target)
        else:
            continue
        parts = [pid.get_pi(n) for n in pid._lattice.descendants(node, include=True)]
        if not any(np.isnan(parts)):
            assert sum(parts) == pytest.approx(computed, abs=1e-9)


def test_incomplete_lattice_exact_atom():
    """
    The unique information of a source with a single child in the lattice is the difference
    of two computed redundancies.
    """
    d = trivariates["anddup"]
    pid = PID_MES(d)
    expected = PID_MES._measure(d, ((1,),), (3,)) - PID_MES._measure(d, ((1,), (0, 2)), (3,))
    assert pid.get_pi(((1,),)) == pytest.approx(expected, abs=1e-12)
