"""
Tests for dit.pid.bounds: intervals over the nonnegative decompositions, and forced overlap.
"""

from itertools import product

import numpy as np
import pytest

from dit import Distribution
from dit.exceptions import ditException
from dit.multivariate import coinformation, entropy
from dit.pid import (
    PID_MMI,
    PID_WB,
    ForcedOverlap,
    Interval,
    PID_dep_b,
    PIDBounds,
    forced_overlap,
    hifi,
    pid_overlap,
    pid_overlaps,
    zero_overlap_conditions,
)
from dit.pid.bounds import _antichains, _conditions, _Lattice
from dit.pid.distributions import bivariates, trivariates
from dit.pid.pid import _lattice_of


def _uniform(outcomes):
    return Distribution([tuple(outcome) for outcome in outcomes], [1 / len(outcomes)] * len(outcomes))


def _random(n_sources, seed, concentration=1.0):
    rng = np.random.default_rng(seed)
    size = 2 ** (n_sources + 1)
    return Distribution.from_ndarray(rng.dirichlet(np.full(size, concentration)).reshape((2,) * (n_sources + 1)))


PARITY = _uniform([(a, b, c, a ^ b ^ c) for a, b, c in product((0, 1), repeat=3)])
NOISY_COPY = _uniform([(a, a ^ n, n, a) for a, n in product((0, 1), repeat=2)])

# arXiv:2611.08131, Appendix: f({i}) = 0 and f({1, 3}) = 0 (here 0-based), 1 on the other
# pairs and on every triple, and 2 on all four sources
APPENDIX_SET_FUNCTION = np.array([0.0] + [{1: 0.0, 2: 1.0, 3: 1.0, 4: 2.0}[bin(g).count("1")] for g in range(1, 16)])
APPENDIX_SET_FUNCTION[0b1010] = 0.0
FOUR = tuple((i,) for i in range(4))


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_lattice_matches_dit(n):
    """The bitmask lattice has the nodes of dit's redundancy lattice."""
    sources = tuple((i,) for i in range(n))
    assert set(_Lattice(sources).labels) == set(_lattice_of(sources)) if n > 1 else len(_antichains(1)) == 1


def test_lattice_sizes():
    """D(n) - 2 nodes, and the 2^n - 1 Shannon equations are independent."""
    assert [len(_antichains(n)) for n in (2, 3, 4, 5)] == [4, 18, 166, 7579]
    lattice = _Lattice(tuple((i,) for i in range(5)))
    assert np.linalg.matrix_rank(lattice.incidence) == 31


def test_lattice_composite_sources():
    """Groups of composite sources are labelled by their variables."""
    lattice = _Lattice(((0, 1), (2,)))
    assert set(lattice.labels) == {((2,), (0, 1)), ((0, 1),), ((2,),), ((0, 1, 2),)}


def test_lattice_indistinct_groups():
    with pytest.raises(ditException, match="distinct"):
        _Lattice(((0, 1), (1,), (0,)))


@pytest.mark.parametrize("name", sorted(bivariates))
def test_two_sources_segment(name):
    """For two sources every interval is the segment from Hi-Fi to MMI (Proposition 1)."""
    d = bivariates[name]
    bounds = PIDBounds(d)
    overlap = pid_overlap(PID_MMI(d))
    assert bounds.get_pi(((0,), (1,))) == pytest.approx(overlap.redundancy_bounds, abs=1e-9)
    assert bounds.get_pi(((0, 1),)) == pytest.approx(overlap.synergy_bounds, abs=1e-9)
    for source, unique in zip(((0,), (1,)), overlap.unique_bounds, strict=True):
        assert bounds.get_pi((source,)) == pytest.approx(unique, abs=1e-9)


@pytest.mark.parametrize("seed", range(4))
def test_extremes_three_sources(seed):
    """MMI attains the largest redundancy of every node and Hi-Fi the largest unique
    information of every source (Theorem 1); nonnegative measures lie inside."""
    d = _random(3, seed)
    bounds = PIDBounds(d)
    mmi, wb = PID_MMI(d), PID_WB(d)
    drivers = hifi(d)
    for node in bounds.nodes:
        atom, red = bounds.get_pi(node), bounds.get_red(node)
        assert red.upper == pytest.approx(mmi.get_red(node), abs=1e-9)
        assert red.lower <= mmi.get_red(node) + 1e-9
        for pid in (mmi, wb):
            assert atom.lower - 1e-8 <= pid.get_pi(node) <= atom.upper + 1e-8
    for i in range(3):
        assert bounds.get_pi(((i,),)).upper == pytest.approx(drivers[(i,)].unique, abs=1e-8)
    synergy = min(coinformation(d, [[i], [3]], [j for j in range(3) if j != i]) for i in range(3))
    assert bounds.get_pi(((0, 1, 2),)).upper == pytest.approx(synergy, abs=1e-8)


def test_extremes_four_sources():
    d = _random(4, 7)
    bounds = PIDBounds(d)
    drivers = hifi(d)
    for i in range(4):
        assert bounds.get_pi(((i,),)).upper == pytest.approx(drivers[(i,)].unique, abs=1e-8)
    mmi = PID_MMI(d)
    for node in bounds.nodes[:30]:
        assert bounds.get_red(node).upper == pytest.approx(mmi.get_red(node), abs=1e-9)


def test_five_sources():
    """Five sources are evaluated node by node."""
    d = _random(5, 3, concentration=0.5)
    bounds = PIDBounds(d)
    drivers = hifi(d)
    for i in range(5):
        assert bounds[((i,),)].upper == pytest.approx(drivers[(i,)].unique, abs=1e-8)
    assert bounds.get_red(((0,), (1,), (2,), (3,), (4,))).lower == pytest.approx(0.0, abs=1e-12)
    with pytest.warns(UserWarning, match="minutes"):
        bounds._warn_full_table()


def test_six_sources():
    d = _random(6, 0)
    with pytest.raises(ValueError, match="at most 5"):
        PIDBounds(d)


@pytest.mark.parametrize("seed", range(3))
def test_identity_target_zero_overlap(seed):
    """With an identity target and a nonnegative information diagram, the decomposition with
    zero overlap on every pair Hi-Fi examines is unique: the diagram (Proposition 9)."""
    rng = np.random.default_rng(seed)
    pmf = rng.dirichlet(np.ones(8)).reshape((2, 2, 2))
    while coinformation(Distribution.from_ndarray(pmf)) < 0:
        pmf = rng.dirichlet(np.ones(8)).reshape((2, 2, 2))
    sources = Distribution.from_ndarray(pmf)
    outcomes = [(*o, "".join(map(str, o))) for o in sources.outcomes]
    d = Distribution(outcomes, sources.pmf)
    bounds = PIDBounds(d, zero_overlap="hifi")
    assert bounds.feasible
    assert bounds.identified == frozenset(bounds.nodes)
    for i in range(3):
        j, k = (m for m in range(3) if m != i)
        assert bounds[((i,),)].lower == pytest.approx(entropy(d, [i], [j, k]), abs=1e-8)
        if i < j:
            assert bounds[((i,), (j,))].lower == pytest.approx(coinformation(d, [[i], [j]], [k]), abs=1e-8)
    assert bounds[((0,), (1,), (2,))].lower == pytest.approx(coinformation(d, [[0], [1], [2]]), abs=1e-8)


def test_identity_target_negative_diagram():
    """For Rauh et al.'s distribution, C(X0 : X1 : X2) = -1, so zero overlap is impossible."""
    bounds = PIDBounds(trivariates["xor cat"], zero_overlap="hifi")
    assert not bounds.feasible
    assert "No nonnegative decomposition" in str(bounds)
    with pytest.raises(ditException, match="No nonnegative decomposition"):
        bounds.get_pi(((0,), (1,)))


@pytest.mark.parametrize(
    "name, d, total, pair",
    [
        ("copy partner", trivariates["copy partner"], 1.0, ((0,), (1, 2))),
        ("xor cat", trivariates["xor cat"], 1.0, None),
        ("parity", PARITY, 0.0, None),
        ("noisy copy", NOISY_COPY, 0.0, None),
    ],
)
def test_forced_overlap(name, d, total, pair):
    """Examples 1 and 2 force a bit of overlap; parity and the noisy copy force none."""
    result = forced_overlap(d)
    assert isinstance(result, ForcedOverlap)
    assert result.total == pytest.approx(total, abs=1e-9)
    assert sum(result.overlaps.values()) == pytest.approx(total, abs=1e-9)
    assert len(result.overlaps) == 6
    if pair is not None:
        assert result.overlaps[pair] == pytest.approx(1.0, abs=1e-9)
    assert min(result.witness.values()) > 0


def test_forced_overlap_given_pairs():
    """On a single pair of Example 1 the overlap can vanish, except on the forced one."""
    d = trivariates["copy partner"]
    assert forced_overlap(d, pairs=[((0,), (1,))]).total == pytest.approx(0.0, abs=1e-9)
    assert forced_overlap(d, pairs=[((0,), (1, 2))]).total == pytest.approx(1.0, abs=1e-9)
    with pytest.raises(ditException, match="disjoint"):
        forced_overlap(d, pairs=[((0, 1), (1, 2))])


@pytest.mark.parametrize("seed", range(24))
def test_conditions_decide_three_sources(seed):
    """For three sources, zero overlap on all six pairs is possible exactly when no condition
    (M) or (S) is violated (Theorem 2)."""
    d = _random(3, 100 + seed, concentration=0.4)
    violations = zero_overlap_conditions(d)
    bounds = PIDBounds(d, zero_overlap="hifi")
    assert bounds.feasible == (not violations)
    assert (forced_overlap(d).total <= 1e-9) == (not violations)
    if violations:
        # every condition is a valid lower bound on the forced overlap
        assert forced_overlap(d).total >= max(v.amount for v in violations) - 1e-9


def test_conditions_examples():
    """Example 1 violates (M), Example 2 violates (S)."""
    (copy,) = zero_overlap_conditions(trivariates["copy partner"])
    assert copy.condition == "M"
    assert copy.pairs == (((0,), (1, 2)),)
    assert copy.amount == pytest.approx(1.0)
    rauh = zero_overlap_conditions(trivariates["xor cat"])
    assert {v.condition for v in rauh} == {"S"}
    assert all(v.amount == pytest.approx(1.0) for v in rauh)
    assert zero_overlap_conditions(bivariates["and"]) == ()


def test_appendix_set_function():
    """A four-source set function that satisfies (M) and (S) yet forces two bits of overlap
    over all 25 pairs, while zero overlap on the 22 pairs Hi-Fi examines is possible."""
    assert _conditions(APPENDIX_SET_FUNCTION, FOUR, 1.0, 1e-9) == ()
    bounds = PIDBounds._from_informations(APPENDIX_SET_FUNCTION, FOUR)
    assert bounds._forced_overlap("all").total == pytest.approx(2.0, abs=1e-9)
    assert bounds._forced_overlap("hifi").total == pytest.approx(0.0, abs=1e-9)
    assert PIDBounds._from_informations(APPENDIX_SET_FUNCTION, FOUR, zero_overlap="hifi").feasible
    assert not PIDBounds._from_informations(APPENDIX_SET_FUNCTION, FOUR, zero_overlap="all").feasible


def test_fixed():
    """Fixing a redundancy narrows the other intervals; an impossible value is infeasible."""
    d = bivariates["and"]
    bounds = PIDBounds(d, fixed={((0,), (1,)): 0.1})
    assert bounds.get_red(((0,), (1,))) == pytest.approx((0.1, 0.1))
    assert bounds.get_pi(((0, 1),)) == pytest.approx((0.1 + 0.188722, 0.1 + 0.188722), abs=1e-6)
    assert not PIDBounds(d, fixed={((0,), (1,)): 0.5}).feasible


def test_zero_overlap_two_sources_is_hifi():
    """With zero overlap the two-source decomposition is Hi-Fi's."""
    bounds = PIDBounds(bivariates["and"], zero_overlap="all")
    assert bounds.identified == frozenset(bounds.nodes)
    assert bounds[((0,), (1,))] == pytest.approx((0.0, 0.0), abs=1e-9)
    assert bounds.get_red(((0,), (1,))) == pytest.approx((0.0, 0.0), abs=1e-9)


def test_positions():
    """MMI sits at the upper end of the redundancy interval of two sources, and atoms that
    every decomposition shares have no position."""
    d = bivariates["and"]
    bounds = PIDBounds(d)
    positions = bounds.positions(PID_MMI(d))
    assert positions[((0,), (1,))] == pytest.approx(1.0)
    assert positions[((0,),)] == pytest.approx(0.0)
    assert np.isnan(PIDBounds(bivariates["redundant"]).positions(PID_MMI(bivariates["redundant"]))[((0,), (1,))])


def test_interval():
    interval = Interval(0.25, 1.0)
    assert interval.width == pytest.approx(0.75)
    assert interval == (0.25, 1.0)


def test_units():
    reference = PIDBounds(bivariates["and"])
    scaled = PIDBounds(bivariates["and"].copy(base="e"))
    assert scaled[((0, 1),)] == pytest.approx(tuple(v * np.log(2) for v in reference[((0, 1),)]))
    assert scaled.get_red(((0,),)).lower == pytest.approx(reference.get_red(((0,),)).lower * np.log(2))
    assert forced_overlap(trivariates["copy partner"].copy(base="e")).total == pytest.approx(np.log(2))


def test_table():
    table = PIDBounds(bivariates["and"]).to_string()
    assert "{0}{1}" in table
    assert "0.3113" in table


def test_unknown_node():
    with pytest.raises(ditException, match="not a node"):
        PIDBounds(bivariates["and"]).get_pi(((0,), (2,)))


def test_symbolic():
    sympy = pytest.importorskip("sympy")
    p = sympy.Symbol("p", positive=True)
    with pytest.raises(NotImplementedError, match="symbolic"):
        PIDBounds(Distribution(["000", "111"], [p, 1 - p]))


def test_pid_overlaps_mmi():
    """MMI has the largest overlap on every pair at once (Theorem 1(i))."""
    for name in ("anddup", "sum", "copy partner"):
        overlaps = pid_overlaps(PID_MMI(trivariates[name]), pairs="all")
        assert len(overlaps) == 6
        for overlap in overlaps.values():
            assert overlap.delta == pytest.approx(overlap.delta_max, abs=1e-9)


def test_pid_overlap_pair_two_sources():
    """For two sources the pair of the sources gives the default."""
    pid = PID_MMI(bivariates["and"])
    assert pid_overlap(pid, pair=((0,), (1,))) == pid_overlap(pid)


def test_pid_overlap_pair_errors():
    pid = PID_MMI(trivariates["anddup"])
    with pytest.raises(ValueError, match="two sources, or a pair"):
        pid_overlap(pid)
    with pytest.raises(ValueError, match="disjoint groups"):
        pid_overlap(pid, pair=((0, 1), (1, 2)))
    with pytest.raises(ValueError, match="disjoint groups"):
        pid_overlap(pid, pair=((0,), (0, 1)))
    with pytest.raises(TypeError, match="target"):
        pid_overlaps(object())


def test_pid_overlap_pair_requires_finite_consistent():
    pid = PID_MMI(trivariates["anddup"])
    pid._reds[((0,), (1,))] = np.nan
    with pytest.raises(ValueError, match="finite"):
        pid_overlap(pid, pair=((0,), (1,)))
    with pytest.raises(ValueError, match="consistent"):
        pid_overlap(PID_dep_b(bivariates["simple"]), pair=((0,), (1,)))


def test_bad_pairs():
    d = trivariates["copy partner"]
    with pytest.raises(ValueError, match="kind of pairs"):
        forced_overlap(d, pairs="some")
    with pytest.raises(ditException, match="disjoint groups"):
        forced_overlap(d, pairs=[((0,), (5,))])
    with pytest.raises(ditException, match="disjoint groups"):
        forced_overlap(d, pairs=[((0,), (1,), (2,))])


def test_cached_intervals():
    bounds = PIDBounds(trivariates["anddup"])
    node = ((0,), (1,))
    assert bounds.get_red(node) is bounds.get_red(node)
    assert bounds.get_pi(node) is bounds[node]
