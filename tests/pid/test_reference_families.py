"""
Theorem-based regression tests from the reference families of two-source PIDs.

Van Santen, Rosseel & Marinazzo (arXiv:2611.08131) write several redundancies as the
largest positive co-information over nested families of reference distributions, which
orders them on every distribution, and tabulate their values on named distributions.
"""

from itertools import product

import numpy as np
import pytest

from dit import Distribution
from dit.example_dists import dyadic, triadic
from dit.multivariate import coinformation
from dit.pid import (
    PID_BROJA,
    PID_CCS,
    PID_MES,
    PID_MMI,
    PID_PM,
    PID_RA,
    PID_RAV,
    PID_WB,
    PID_dep,
    PID_dep_c,
    PID_Do,
    PID_Proj,
)
from dit.pid.distributions import bivariates

RED = ((0,), (1,))

# without ECOS (as on Windows CI) BROJA is computed by admUI, accurate to ~1e-5
ATOL = 1e-4


def _joint(table):
    outcomes = list(table)
    return Distribution(outcomes, [table[outcome] for outcome in outcomes])


def _bsc(x, eps):
    return {x: 1 - eps, 1 - x: eps}


def _noisy_copies(eps_1, eps_2):
    """Copies of a uniform bit through binary symmetric channels."""
    table = {}
    for y in (0, 1):
        for x1, p1 in _bsc(y, eps_1).items():
            for x2, p2 in _bsc(y, eps_2).items():
                table[(x1, x2, y)] = table.get((x1, x2, y), 0) + p1 * p2 / 2
    return _joint(table)


def _chain(eps_1, eps_2):
    """Y -> X1 -> X2 through binary symmetric channels."""
    table = {}
    for y in (0, 1):
        for x1, p1 in _bsc(y, eps_1).items():
            for x2, p2 in _bsc(x1, eps_2).items():
                table[(x1, x2, y)] = table.get((x1, x2, y), 0) + p1 * p2 / 2
    return _joint(table)


def _noisy_and(eps):
    table = {}
    for x1, x2 in product((0, 1), repeat=2):
        for y, p in _bsc(x1 & x2, eps).items():
            table[(x1, x2, y)] = p / 4
    return _joint(table)


def _correlated_copy(p_equal):
    """Y = (X1, X2) for correlated uniform bits."""
    return _joint(
        {(x1, x2, 2 * x1 + x2): (p_equal if x1 == x2 else 1 - p_equal) / 2 for x1, x2 in product((0, 1), repeat=2)}
    )


# Table II of arXiv:2611.08131 (bits): C, C2, I3, delta_max, the redundancies of Hi-Fi,
# the pairwise reference (PID_dep_c), I_dep, BROJA and MMI, and the positions of
# I_min, I_ccs and I_pm (None where delta_max = 0).
TABLE_II = {
    "XOR": (bivariates["synergy"], -1.0, 0.0, 1.0, 0.0, (0, 0, 0, 0, 0), None),
    "AND": (bivariates["and"], -0.189, -0.189, 0.0, 0.311, (0, 0, 0.08, 0.31, 0.31), (1.0, 0.33, 1.80)),
    "SUM": (bivariates["sum"], -0.5, -0.5, 0.0, 0.5, (0, 0, 0.19, 0.50, 0.50), (1.0, 0.0, 1.0)),
    "RDN": (bivariates["redundant"], 1.0, 1.0, 0.0, 0.0, (1, 1, 1, 1, 1), None),
    "UNQ": (bivariates["unique 1"], 0.0, 0.0, 0.0, 0.0, (0, 0, 0, 0, 0), None),
    "COPY": (bivariates["cat"], 0.0, 0.0, 0.0, 1.0, (0, 0, 0, 0, 1), (1.0, 0.0, 1.0)),
    "COPY, corr.": (_correlated_copy(0.8), 0.278, 0.278, 0.0, 0.722, (0.28, 0.28, 0.28, 0.28, 1), (1.0, 0.37, 1.0)),
    "RdnXor": (bivariates["rdn xor"], 0.0, 1.0, 1.0, 1.0, (0, 1, 1, 1, 1), (1.0, 1.0, 1.0)),
    "chain": (_chain(0.1, 0.2), 0.173, 0.173, 0.0, 0.0, (0.17,) * 5, None),
    "common driver": (
        _noisy_copies(0.15, 0.25),
        0.090,
        0.090,
        0.0,
        0.098,
        (0.09, 0.09, 0.09, 0.19, 0.19),
        (1.0, 1.89, 5.98),
    ),
    "noisy AND": (_noisy_and(0.1), -0.119, -0.041, 0.077, 0.147, (0, 0, 0.03, 0.15, 0.15), (1.0, 0.44, 2.70)),
    "noisy copies": (
        _noisy_copies(0.1, 0.1),
        0.320,
        0.320,
        0.0,
        0.211,
        (0.32, 0.32, 0.32, 0.53, 0.53),
        (1.0, 1.22, 2.35),
    ),
    "dyadic": (dyadic, 0.0, 0.0, 0.0, 1.0, (0, 0, 0, 0, 1), (1.0, 0.0, 1.0)),
    "triadic": (triadic, 0.0, 1.0, 1.0, 1.0, (0, 1, 1, 1, 1), (1.0, 1.0, 1.0)),
    "PwUnq": (bivariates["pnt. unq."], 0.0, 0.0, 0.0, 0.5, (0, 0, 0.25, 0.50, 0.50), (1.0, 0.0, 0.0)),
}


@pytest.mark.parametrize("name", list(TABLE_II))
def test_table_ii(name):
    """dit reproduces the named distributions of Table II, to the table's precision."""
    d, c, c2, i3, delta_max, reds, positions = TABLE_II[name]
    d = d.coalesce([[0], [1], [2]])
    mis = [coinformation(d, [[i], [2]]) for i in (0, 1)]
    segment = min(mis) - max(0.0, coinformation(d))
    assert coinformation(d) == pytest.approx(c, abs=6e-4)
    assert segment == pytest.approx(delta_max, abs=6e-4)
    pairwise = PID_RA(d).get_red(RED)
    assert pairwise == pytest.approx(c2, abs=6e-4)
    assert pairwise - coinformation(d) == pytest.approx(i3, abs=6e-4)
    hifi = max(0.0, coinformation(d))
    actual = [hifi] + [measure(d).get_red(RED) for measure in (PID_dep_c, PID_dep, PID_BROJA, PID_MMI)]
    assert actual == pytest.approx(reds, abs=6e-3)
    if positions is not None:
        actual = [(measure(d).get_red(RED) - hifi) / segment for measure in (PID_WB, PID_CCS, PID_PM)]
        assert actual == pytest.approx(positions, abs=6e-3)


def _random_joints(shape, count, seed, sparsity=0.0):
    rng = np.random.default_rng(seed)
    joints = []
    for _ in range(count):
        p = rng.dirichlet(np.ones(int(np.prod(shape))))
        if sparsity:
            p[rng.random(p.shape) < sparsity] = 0.0
            p /= p.sum()
        joints.append(Distribution.from_ndarray(p.reshape(shape)))
    return joints


RANDOM = {
    **{f"binary {i}": d for i, d in enumerate(_random_joints((2, 2, 2), 10, 26110811))},
    **{f"ternary {i}": d for i, d in enumerate(_random_joints((3, 3, 3), 4, 26110812))},
    **{f"sparse {i}": d for i, d in enumerate(_random_joints((2, 2, 2), 6, 26110813, sparsity=0.3))},
}


@pytest.mark.parametrize("name", sorted(RANDOM))
def test_reference_family_order(name):
    """Hi-Fi <= pairwise reference <= I_dep <= BROJA <= MMI (Proposition 4)."""
    d = RANDOM[name]
    reds = [max(0.0, coinformation(d))] + [m(d).get_red(RED) for m in (PID_dep_c, PID_dep, PID_BROJA, PID_MMI)]
    for lower, upper in zip(reds, reds[1:], strict=False):
        assert lower <= upper + ATOL


TWO_SOURCE = {**RANDOM, **{f"bivariates[{name!r}]": d for name, d in bivariates.items()}}


@pytest.mark.parametrize("name", sorted(TWO_SOURCE))
def test_dep_is_larger_of_ra_and_mes(name):
    """For two sources, I_dep's redundancy is the larger of I_RA's and I_ME*'s (= I_do's)."""
    d = TWO_SOURCE[name]
    dep = PID_dep(d).get_red(RED)
    ra = PID_RA(d).get_red(RED)
    assert dep == pytest.approx(max(ra, PID_MES(d).get_red(RED)), abs=1e-9)
    assert dep == pytest.approx(max(ra, PID_Do(d).get_red(RED)), abs=1e-9)


@pytest.mark.parametrize("name", ["and", "sum", "cat", "rdn xor", "pnt. unq.", "reduced or", "imp. rdn", "w", "boom"])
def test_marginal_measures_above_broja(name):
    """Measures that depend on the source-target marginals, here I_min and I_red, sit at or
    above BROJA, and monotonicity caps them by MMI (Lemma 1 and Eq. (17))."""
    d = bivariates[name]
    broja = PID_BROJA(d).get_red(RED)
    mmi = PID_MMI(d).get_red(RED)
    for measure in (PID_WB, PID_Proj):
        red = measure(d).get_red(RED)
        assert broja - ATOL <= red <= mmi + ATOL


@pytest.mark.parametrize(
    "name", sorted(name for name in RANDOM if name.startswith("binary")) + ["bivariates['and']", "bivariates['sum']"]
)
def test_rav_between_hifi_and_broja(name):
    """0 <= delta_RAV <= delta_BROJA (Proposition 10)."""
    d = TWO_SOURCE[name]
    rav = PID_RAV(d).get_red(RED)
    assert max(0.0, coinformation(d)) - ATOL <= rav <= PID_BROJA(d).get_red(RED) + ATOL


@pytest.mark.parametrize("n", [2, 3, 4])
def test_mmi_closed_form_atoms(n):
    """MMI's synergy is min_i I(Y : X_i | X_rest) and its unique atoms are
    [I(Y : X_i) - I(Y : X_rest)]_+ (Theorem 1(i))."""
    rng = np.random.default_rng(2611 + n)
    for _ in range(3):
        d = Distribution.from_ndarray(rng.dirichlet(np.ones(2 ** (n + 1))).reshape((2,) * (n + 1)))
        pid = PID_MMI(d)
        everything = list(range(n))
        total = coinformation(d, [everything, [n]])
        rests = [[j for j in everything if j != i] for i in everything]
        synergy = min(total - coinformation(d, [rest, [n]]) for rest in rests)
        assert pid.get_pi((tuple(everything),)) == pytest.approx(synergy, abs=1e-9)
        for i, rest in enumerate(rests):
            unique = max(0.0, coinformation(d, [[i], [n]]) - coinformation(d, [rest, [n]]))
            assert pid.get_pi(((i,),)) == pytest.approx(unique, abs=1e-9)
        # MMI's redundancy of every node is the smallest mutual information of its collections
        for node in pid._lattice:
            assert pid.get_red(node) == pytest.approx(min(coinformation(d, [list(c), [n]]) for c in node), abs=1e-9)
