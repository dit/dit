"""
Tests for dit.pid.measures.ici (conditional-independence union information I_∪^CI).

Values are from Gomes & Figueiredo, "A Measure of Synergy Based on Union
Information", Entropy 26, 271, 2024.
"""

import pytest
from hypothesis import given, settings

from dit import Distribution
from dit.multivariate import coinformation
from dit.pid.distributions import bivariates, trivariates
from dit.pid.measures.ici import PID_CI, _prune_sources, synergy_ci, union_ci
from dit.utils.testing import distributions


def _uniform(outcomes):
    """A uniform distribution over tuple outcomes."""
    return Distribution([tuple(o) for o in outcomes], [1 / len(outcomes)] * len(outcomes))


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("synergy", 1.0),
        ("and", 0.2704),
        ("cat", 0.0),
        ("rdn xor", 1.0),
    ],
)
def test_table9_bivariate(name, expected):
    """Paper Table 9: S^CI on bivariate distributions."""
    assert synergy_ci(bivariates[name], [[0], [1]], [2]) == pytest.approx(expected, abs=1e-3)


def test_table9_rdnunqxor():
    """Paper Table 9 / A2: S^CI(RDNUNQXOR) = 1 out of I(Y;T) = 4."""
    rows = [
        (0, 0, 0), (1, 0, 1), (1, 1, 0), (0, 1, 1), (2, 0, 2), (3, 0, 3), (3, 1, 2), (2, 1, 3),
        (4, 2, 0), (5, 2, 1), (5, 3, 0), (4, 3, 1), (6, 2, 2), (7, 2, 3), (7, 3, 2), (6, 3, 3),
        (8, 4, 4), (9, 4, 5), (9, 5, 4), (8, 5, 5), (10, 4, 6), (11, 4, 7), (11, 5, 6), (10, 5, 7),
        (12, 6, 4), (13, 6, 5), (13, 7, 4), (12, 7, 5), (14, 6, 6), (15, 6, 7), (15, 7, 6), (14, 7, 7),
    ]  # fmt: skip
    d = _uniform(rows)
    assert coinformation(d, [[1, 2], [0]]) == pytest.approx(4.0)
    assert synergy_ci(d, [[1], [2]], [0]) == pytest.approx(1.0, abs=1e-6)


XOR_DUPLICATE = [(0, 0, 0, 0), (0, 1, 1, 1), (1, 0, 0, 1), (1, 1, 1, 0)]  # (Y1, Y2, Y3 = Y2, T)
XOR_LOSES = [(0, 0, 0, 0), (1, 0, 1, 1), (1, 1, 0, 1), (0, 1, 1, 0)]  # (T, Y1, Y2, Y3 = T)
XOR_MULTICOAL = [
    (0, 0, 0, 0), (0, 1, 1, 1), (0, 2, 2, 2), (0, 3, 3, 3), (1, 2, 1, 0), (1, 3, 0, 1), (1, 0, 3, 2), (1, 1, 2, 3),
]  # fmt: skip


@pytest.mark.parametrize(
    ("outcomes", "sources", "target", "expected"),
    [
        (XOR_DUPLICATE, [[0], [1], [2]], [3], 1.0),
        (XOR_LOSES, [[1], [2], [3]], [0], 0.0),
        (XOR_MULTICOAL, [[1], [2], [3]], [0], 1.0),
    ],
)
def test_table9_trivariate(outcomes, sources, target, expected):
    """Paper Table 9 / A1: S^CI on trivariate distributions."""
    assert synergy_ci(_uniform(outcomes), sources, target) == pytest.approx(expected, abs=1e-6)


def test_table9_anddup():
    """Paper Table 9: ANDDUPLICATE has the same synergy as AND."""
    d = trivariates["anddup"]
    assert synergy_ci(d, [[0], [1], [2]], [3]) == pytest.approx(0.2704, abs=1e-3)


def test_and_decomposition():
    """Bivariate AND: I_∪ = 0.5409, R = 0.0817, U_i = 0.2296, S = 0.2704."""
    pid = PID_CI(bivariates["and"])
    assert pid.get_union(((0,), (1,))) == pytest.approx(0.5409, abs=1e-3)
    assert pid.get_syn(((0,), (1,))) == pytest.approx(0.2704, abs=1e-3)
    assert pid[((0,), (1,))] == pytest.approx(0.0817, abs=1e-3)
    assert pid[((0,),)] == pytest.approx(0.2296, abs=1e-3)
    assert pid[((1,),)] == pytest.approx(0.2296, abs=1e-3)
    assert pid[((0, 1),)] == pytest.approx(0.2704, abs=1e-3)


def _adapted_reduced_or(r):
    """Paper Table 3, outcomes (T, Y1, Y2)."""
    outcomes = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 0, 1), (1, 1, 1)]
    return Distribution(outcomes, [1 / 2, r / 4, (1 - r) / 4, (1 - r) / 4, r / 4])


def test_adapted_reduced_or_clipped():
    """Paper Table 3: for r > 1/2, I_q > I_p so the union is capped and synergy is 0."""
    d = _adapted_reduced_or(0.75)
    assert union_ci(d, [[1], [2]], [0]) == pytest.approx(coinformation(d, [[1, 2], [0]]))
    assert synergy_ci(d, [[1], [2]], [0]) == pytest.approx(0.0, abs=1e-9)


def test_adapted_reduced_or_iq():
    """Paper Table 3: I_q(Y;T) = 0.549 regardless of r."""
    d = _adapted_reduced_or(0.0)
    assert union_ci(d, [[1], [2]], [0]) == pytest.approx(0.549, abs=1e-3)


def test_deterministic_source_removed():
    """Paper Sec. 3.3: with Y3 = Y2 and T = Y1 AND Y2, I_∪(Y1, Y2, Y3) = I_∪(Y1, Y2) = 0.5409."""
    d = _uniform([(0, 0, 0, 0), (0, 1, 1, 0), (1, 0, 0, 0), (1, 1, 1, 1)])
    assert union_ci(d, [[0], [1], [2]], [3]) == pytest.approx(0.5409, abs=1e-3)
    assert union_ci(d, [[0], [1]], [3]) == pytest.approx(0.5409, abs=1e-3)


def test_overlapping_sources_kept():
    """
    Overlapping sources are not dropped merely for sharing variables. For the
    three-way XOR every pair is independent of T, so I_∪((Y1,Y2),(Y1,Y3),(Y2,Y3)) = 0.
    """
    d = trivariates["synergy"]
    sources = [[0, 1], [0, 2], [1, 2]]
    assert _prune_sources(d, sources) == [(0, 1), (0, 2), (1, 2)]
    assert union_ci(d, sources, [3]) == pytest.approx(0.0, abs=1e-9)


def test_target_monotonicity_fails():
    """Paper Table 5: I_∪(Y1,Y2 -> T) = 0.91 > 0.90 = I_∪(Y1,Y2 -> (T,Z))."""
    outcomes = [(0, 0, 1, 0), (1, 1, 2, 1), (2, 1, 3, 0), (0, 0, 3, 1), (2, 2, 4, 4)]
    d = Distribution(outcomes, [0.419, 0.203, 0.007, 0.346, 0.025])
    assert union_ci(d, [[2], [3]], [0]) == pytest.approx(0.91, abs=5e-3)
    assert union_ci(d, [[2], [3]], [0, 1]) == pytest.approx(0.90, abs=5e-3)


def test_copy_vs_xor_target():
    """Paper Table 6: T1 = COPY gives no synergy; T2 = XOR (a function of T1) gives 1 bit."""
    d = _uniform([(0, 0, 0, 0), (1, 1, 0, 1), (1, 2, 1, 0), (0, 3, 1, 1)])
    assert synergy_ci(d, [[2], [3]], [1]) == pytest.approx(0.0, abs=1e-9)
    assert synergy_ci(d, [[2], [3]], [0]) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("name", ["and", "sum", "redundant", "synergy", "unique 1", "cat", "reduced or", "boom"])
def test_bivariate_consistent(name):
    """PID_CI fills the bivariate lattice consistently."""
    pid = PID_CI(bivariates[name])
    assert pid.consistent
    assert pid.complete


@pytest.mark.parametrize("name", ["anddup", "cat", "dbl xor", "redundant", "sum", "synergy", "xor cat"])
def test_trivariate_consistent(name):
    """PID_CI fills the trivariate lattice consistently, and the atoms sum to I(Y;T)."""
    pid = PID_CI(trivariates[name])
    assert pid.consistent
    assert pid.complete
    assert sum(pid[node] for node in pid._lattice) == pytest.approx(pid._total, abs=1e-6)


def test_get_syn_full_sources():
    """get_syn of all singletons is the paper's S^CI."""
    d = trivariates["anddup"]
    pid = PID_CI(d)
    assert pid.get_syn(((0,), (1,), (2,))) == pytest.approx(synergy_ci(d, [[0], [1], [2]], [3]))


@settings(deadline=None, max_examples=50)
@given(dist=distributions(alphabets=(2, 2, 2)))
def test_union_bounds(dist):
    """max_i I(Y_i;T) <= I_∪^CI <= I(Y;T), so synergy is non-negative."""
    union = union_ci(dist, [[0], [1]], [2])
    whole = coinformation(dist, [[0, 1], [2]])
    single = max(coinformation(dist, [[0], [2]]), coinformation(dist, [[1], [2]]))
    assert single - 1e-9 <= union <= whole + 1e-9
    assert synergy_ci(dist, [[0], [1]], [2]) >= -1e-9


@settings(deadline=None, max_examples=25)
@given(dist=distributions(alphabets=(2, 2, 2)))
def test_self_redundancy(dist):
    """A single source's union information is its mutual information with the target."""
    assert union_ci(dist, [[0]], [2]) == pytest.approx(coinformation(dist, [[0], [2]]), abs=1e-9)


def test_tiny_probabilities():
    """
    p(a) p(t) underflowed to 0 when both are about 1e-182, so the union information was
    infinite and the decomposition undefined.
    """
    d = Distribution([(0, 0, 0), (1, 0, 1), (0, 1, 0)], [0.5, 1e-182, 0.5])
    assert union_ci(d, [[0]], [2]) == pytest.approx(coinformation(d, [[0], [2]]), abs=1e-12)
    assert synergy_ci(d, [[0], [1]], [2]) == pytest.approx(0.0, abs=1e-12)
    assert PID_CI(d).complete
