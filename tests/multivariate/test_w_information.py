"""
Tests for dit.multivariate.w_information.
"""

import importlib
from itertools import product

import numpy as np
import pytest
from hypothesis import given, settings

from dit import Distribution
from dit.multivariate import coinformation, m_information, w_information
from dit.pid import PID_BROJA
from dit.pid.distributions import bivariates
from dit.utils.testing import distributions

XOR = Distribution(["000", "011", "101", "110"], [1 / 4] * 4)


def system(f, nfree):
    """
    Uniform over `nfree` free bits, each mapped to sources (0, 1) and targets (2, 3) by `f`.
    """
    pmf = {}
    for bits in product((0, 1), repeat=nfree):
        outcome = "".join(map(str, f(*bits)))
        pmf[outcome] = pmf.get(outcome, 0) + 1 / 2**nfree
    return Distribution(list(pmf), list(pmf.values()))


# Liardi et al.'s toy systems (App. E), with bits and xor for their Gaussians and sums:
# each puts one bit in one PhiID atom, so M = 1 exactly when that atom involves synergy.
SYSTEMS = {
    "R->R": system(lambda z: (z, z, z, z), 1),
    "R->U": system(lambda z, n: (z, z, z, n), 2),
    "R->S": system(lambda a, b: (a ^ b, a ^ b, a, b), 2),
    "U->R": system(lambda z, n: (z, n, z, z), 2),
    "U->U": system(lambda z, m, n: (z, m, n, z), 3),
    "U->S": system(lambda a, b, n: (a ^ b, n, a, b), 3),
    "S->R": system(lambda a, b: (a, b, a ^ b, a ^ b), 2),
    "S->U": system(lambda a, b, n: (a, b, a ^ b, n), 3),
    "S->S": system(lambda a, b, c: (a, b, c, a ^ b ^ c), 3),
}

# Three binary sources and three binary targets whose W-information has a local minimum that
# is not global: a run from the uniform marginals ends at 0.46526431, above the minimum.
SPURIOUS = [
    3.50541337611e-07, 0.000316186619567, 0.00441996694801, 0.0539917353404, 2.28402324939e-08,
    2.70769203633e-06, 0.0134746079764, 7.15770247448e-05, 1.32152105695e-07, 0.00264049125861,
    0.00023489697789, 3.36581045336e-07, 0.040686834005, 2.97624047331e-10, 0.165521754378,
    4.71073343479e-06, 0.023586731118, 2.81534016777e-06, 1.13188510423e-05, 4.63253427331e-05,
    0.0420475869717, 1.24077339274e-07, 0.0409283133258, 0.00232168228581, 5.10089506488e-05,
    0.00653145476703, 0.0107378624082, 1.03758906893e-05, 2.49264884252e-06, 0.00102863073637,
    0.000133115631771, 0.000458276463104, 2.87064852508e-08, 0.00208369935597, 5.0323745049e-05,
    0.00613859973575, 0.0061979204906, 0.0165097353083, 0.00122861760905, 2.58265753218e-08,
    0.0272454193896, 1.98557039639e-05, 9.82422139353e-05, 3.00287359841e-13, 1.8467922225e-06,
    3.35514398767e-06, 1.09459976548e-05, 1.41714580889e-06, 4.22195304445e-05, 0.368809326284,
    0.0233935051276, 2.96939759819e-13, 0.0102765939403, 0.0260540151015, 7.24141492212e-08,
    4.8399034901e-05, 0.000772704977152, 1.20434073777e-05, 0.000892877405095, 0.0119369997049,
    0.000339698933428, 9.44763371469e-05, 0.0249866067087, 0.063486001752,
]  # fmt: skip


# Two sources and a target whose W-minimizer gives every value of the sources mass: a run that
# starts without the value (0, 0) has to restore it.
RESTORED = [0.0722948, 0.0203774, 0.0889678, 0.00978255, 0.522315, 0.014631, 0.153671, 0.117961]


# Two sources and a target for which the outcome (1, 1, 1) has positive pairwise marginals, but
# zero probability under every distribution sharing them.
INDUCED = [0.208297, 0.0423601, 0.330672, 0.108707, 0.193282, 3.94043e-09, 0.116682, 0.0]


# Two sources and a target for which the outcomes without the value (0, 0) of the sources carry
# the pairwise marginals only in the limit, so a run that starts without it has to restore it.
DIVERGENT = [0.106361, 0.0305745, 0.532825, 0.036321, 0.0114349, 0.1784, 0.0494262, 0.0546581]


def union_information(d, sources, target):
    """
    The BROJA union information I(X_0 : Y) + I(X_1 : Y) - R of two sources.
    """
    pid = PID_BROJA(d, sources, target)
    mis = sum(coinformation(d, [source, target]) for source in sources)
    return mis - pid.get_red(tuple(map(tuple, sources)))


@pytest.mark.parametrize("name", list(bivariates))
def test_single_target_is_broja_union(name):
    """
    With one target, W is the union information of BROJA, and M its synergy.
    """
    d = bivariates[name]
    expected = union_information(d, [[0], [1]], [2])
    assert w_information(d, [[0], [1]], [[2]]) == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize(
    ("name", "w"),
    [("synergy", 0.0), ("redundant", 1.0), ("cat", 2.0), ("and", 0.3112781244591328), ("sum", 0.5)],
)
def test_single_target_values(name, w):
    """
    Exact values, to the certified tolerance of the convex case.
    """
    d = bivariates[name]
    assert w_information(d, [[0], [1]], [[2]]) == pytest.approx(w, abs=1e-9)
    total = coinformation(d, [[0, 1], [2]])
    assert m_information(d, [[0], [1]], [[2]]) == pytest.approx(total - w, abs=1e-9)


def test_single_source_is_broja_union():
    """
    The problem is symmetric: one source and two targets give the union information of the
    targets about the source.
    """
    d = bivariates["and"]
    expected = union_information(d, [[0], [1]], [2])
    assert w_information(d, [[2]], [[0], [1]]) == pytest.approx(expected, abs=1e-4)


def test_one_group_each_side():
    """
    Keeping the only pair keeps the joint distribution.
    """
    d = bivariates["jeff"]
    assert w_information(d, [[0, 1]], [[2]]) == pytest.approx(coinformation(d, [[0, 1], [2]]))
    assert m_information(d, [[0, 1]], [[2]]) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("name", list(SYSTEMS))
def test_toy_systems(name):
    """
    Each toy system carries one bit, beyond the pairs exactly when its atom involves synergy.
    """
    d = SYSTEMS[name]
    m = 1.0 if "S" in name else 0.0
    assert w_information(d, rng=0) == pytest.approx(1 - m, abs=1e-8)
    assert m_information(d, rng=0) == pytest.approx(m, abs=1e-8)


def test_xor_example():
    assert w_information(XOR, [[0], [1]], [[2]]) == pytest.approx(0.0, abs=1e-12)
    assert m_information(XOR, [[0], [1]], [[2]]) == pytest.approx(1.0)


def test_independent_pairs():
    """
    The uniform distribution on four bits is its own minimizer, and so is any product of a
    distribution of the sources with one of the targets.
    """
    d = Distribution(["".join(map(str, x)) for x in product((0, 1), repeat=4)], [1 / 16] * 16)
    assert w_information(d, rng=0) == pytest.approx(0.0, abs=1e-12)


def test_groups():
    """
    Groups of variables count as one source or target, and default to the halves of the variables.
    """
    d = SYSTEMS["S->S"]
    assert w_information(d, [[0, 1]], [[2, 3]]) == pytest.approx(1.0)
    assert w_information(d, [[0], [1]], [[2], [3]], rng=0) == pytest.approx(w_information(d, rng=0), abs=1e-10)


def test_spurious_local_minimum():
    """
    The minimization is not convex: a single run can stop above the minimum, and random
    restarts find it.
    """
    outcomes = list(product((0, 1), repeat=6))
    d = Distribution(outcomes, np.array(SPURIOUS) / sum(SPURIOUS))
    sources, targets = [[0], [1], [2]], [[3], [4], [5]]
    assert w_information(d, sources, targets, niter=0) == pytest.approx(0.46526431, abs=1e-6)
    assert w_information(d, sources, targets, niter=1, rng=0) == pytest.approx(0.46394524, abs=1e-6)


@settings(max_examples=10, deadline=None)
@given(dist=distributions(alphabets=(2, 2, 2, 2)))
def test_bounds(dist):
    """
    max_ij I(X_i : Y_j) <= W <= I(X : Y), so that M is nonnegative.
    """
    w = w_information(dist, rng=0)
    pairs = max(coinformation(dist, [[i], [j]]) for i in (0, 1) for j in (2, 3))
    total = coinformation(dist, [[0, 1], [2, 3]])
    assert pairs - 1e-8 <= w <= total + 1e-8
    assert m_information(dist, rng=0) >= -1e-8


def test_log_base():
    """
    Values are reported in the base of the distribution.
    """
    d = bivariates["and"].copy(base="e")
    expected = w_information(bivariates["and"], [[0], [1]], [[2]]) * np.log(2)
    assert w_information(d, [[0], [1]], [[2]]) == pytest.approx(expected)


def test_unconverged_projections(monkeypatch):
    """
    A projection that has not converged is never used, so the result is still an upper bound
    on W, and the run reports that it stopped short.
    """
    module = importlib.import_module("dit.multivariate.w_information")
    monkeypatch.setattr(module, "_MAX_NEWTON", 1)
    d = bivariates["and"]
    with pytest.warns(RuntimeWarning, match="did not converge"):
        w = w_information(d, [[0], [1]], [[2]])
    assert 0.3112781244591328 - 1e-12 <= w <= coinformation(d, [[0, 1], [2]]) + 1e-12


def test_dropped_value_restored():
    """
    A value dropped too early is restored once the rest has converged, and the run still
    reaches the minimum.
    """
    module = importlib.import_module("dit.multivariate.w_information")
    p = np.array(RESTORED).reshape(2, 2, 2) / sum(RESTORED)
    polytope = module._Polytope(p, 2)
    uniform_x = polytope.values_x / polytope.values_x.sum()
    uniform_y = polytope.values_y / polytope.values_y.sum()
    expected, _, _ = polytope.polish(uniform_x, uniform_y, np.zeros(len(polytope.targets)), 1e-10, 10_000)
    start = uniform_x.copy()
    start[0] = 0
    value, gap, _ = polytope.polish(start, uniform_y.copy(), np.zeros(len(polytope.targets)), 1e-10, 10_000)
    assert gap <= 1e-10
    assert value == pytest.approx(expected, abs=1e-12)


def test_needed_value_restored():
    """
    A dropped value without which the polytope cannot be reached is restored, and the run still
    reaches the minimum.
    """
    module = importlib.import_module("dit.multivariate.w_information")
    p = np.array(DIVERGENT).reshape(2, 2, 2) / sum(DIVERGENT)
    polytope = module._Polytope(p, 2)
    uniform_x = polytope.values_x / polytope.values_x.sum()
    uniform_y = polytope.values_y / polytope.values_y.sum()
    expected, _, _ = polytope.polish(uniform_x, uniform_y, np.zeros(len(polytope.targets)), 1e-10, 10_000)
    start = uniform_x.copy()
    start[0] = 0
    value, gap, _ = polytope.polish(start, uniform_y.copy(), np.zeros(len(polytope.targets)), 1e-10, 10_000)
    assert gap <= 1e-10
    assert value == pytest.approx(expected, abs=1e-12)


def test_projection_restarts_from_zero():
    """
    A projection whose warm start is too far off for Newton's method starts again from zero
    multipliers.
    """
    module = importlib.import_module("dit.multivariate.w_information")
    polytope = module._Polytope(module._joint(bivariates["and"], [(0,), (1,), (2,)]), 2)
    r_x = polytope.values_x / polytope.values_x.sum()
    r_y = polytope.values_y / polytope.values_y.sum()
    cold, _, _ = polytope.project(r_x, r_y, np.zeros(len(polytope.targets)))
    wild = 100 * np.random.default_rng(0).standard_normal(len(polytope.targets))
    q, _, _ = polytope.project(r_x, r_y, wild)
    assert q == pytest.approx(cold, abs=1e-12)


def test_induced_zeros():
    """
    Outcomes that the pairwise marginals allow but force to zero are left out, so the
    projections onto the polytope are attained instead of only approached.
    """
    module = importlib.import_module("dit.multivariate.w_information")
    p = np.array(INDUCED).reshape(2, 2, 2) / sum(INDUCED)
    polytope = module._Polytope(p, 2)
    assert not polytope.allowed[1, 1, 1]
    assert polytope.allowed.sum() == 7
    d = Distribution(list(product((0, 1), repeat=3)), p.ravel())
    expected = union_information(d, [[0], [1]], [2])
    assert w_information(d, [[0], [1]], [[2]]) == pytest.approx(expected, abs=1e-4)


def test_not_converged():
    d = bivariates["not two"]
    with pytest.warns(RuntimeWarning, match="did not converge"):
        w_information(d, [[0], [1]], [[2]], maxiter=2)


@pytest.mark.parametrize(
    ("sources", "targets"),
    [
        ([[0]], None),
        (None, [[2]]),
        ([[0], [1]], [[1], [2]]),
        ([[0], []], [[2]]),
        ([], [[2]]),
    ],
)
def test_errors(sources, targets):
    with pytest.raises(ValueError):
        w_information(bivariates["and"], sources, targets)


def test_odd_number_of_variables():
    with pytest.raises(ValueError):
        w_information(XOR)
