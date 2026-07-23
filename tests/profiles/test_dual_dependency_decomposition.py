"""
Tests for DualDependencyDecomposition (m-flat dependency lattice).
"""

import numpy as np
import pytest

from dit import Distribution
from dit.algorithms import m_projection, m_projection_eps_limit, symmetric_smooth
from dit.algorithms.mprojection import mflat_subsets_from_dependency
from dit.divergences import kullback_leibler_divergence as D
from dit.example_dists import n_mod_m
from dit.profiles import DependencyDecomposition, DualDependencyDecomposition
from dit.shannon import entropy as H


def test_mflat_subsets_downward_closure():
    node = frozenset([frozenset([0, 1]), frozenset([2])])
    subs = mflat_subsets_from_dependency(node)
    assert (0, 1) in subs
    assert (0,) in subs and (1,) in subs and (2,) in subs
    assert (0, 2) not in subs


def test_symmetric_smooth_preserves_w_symmetry():
    w = Distribution(["001", "010", "100"], [1 / 3] * 3)
    pe = symmetric_smooth(w, 1e-6)
    pm = {"".join(o) if not isinstance(o, str) else o: float(p) for o, p in zip(pe.outcomes, pe.pmf)}
    wt1 = [pm["001"], pm["010"], pm["100"]]
    zeros = [pm[x] for x in ("000", "011", "101", "110", "111")]
    assert max(wt1) - min(wt1) < 1e-15
    assert max(zeros) - min(zeros) < 1e-15


def test_dual_dd_top_zero_rkl():
    d = n_mod_m(3, 2)
    dd = DualDependencyDecomposition(d, nrestarts=4, maxiter=800)
    top = frozenset([frozenset([0, 1, 2])])
    assert dd[top]["rKL"] == pytest.approx(0.0, abs=1e-8)


def test_dual_dd_xor_needs_triple():
    """XOR: any node without the full triple keeps large reverse KL."""
    d = Distribution(["000", "011", "101", "110"], [0.25] * 4)
    dd = DualDependencyDecomposition(d, nrestarts=4, maxiter=800)
    pairs = frozenset([frozenset([0, 1]), frozenset([0, 2]), frozenset([1, 2])])
    top = frozenset([frozenset([0, 1, 2])])
    assert dd[pairs]["rKL"] > 1.0
    assert dd[top]["rKL"] == pytest.approx(0.0, abs=1e-8)


def test_dual_dd_giant_bit_pairs_suffice():
    """Giant Bit lies in the pairwise additive family."""
    d = Distribution(["000", "111"], [0.5, 0.5])
    dd = DualDependencyDecomposition(d, nrestarts=4, maxiter=800)
    pairs = frozenset([frozenset([0, 1]), frozenset([0, 2]), frozenset([1, 2])])
    assert dd[pairs]["rKL"] == pytest.approx(0.0, abs=1e-3)


def test_dual_dd_w_pairs_symmetric():
    """W pair-cover projection should respect S3 symmetry under symmetric P_ε."""
    w = Distribution(["001", "010", "100"], [1 / 3] * 3)
    dd = DualDependencyDecomposition(w, nrestarts=6, maxiter=1500)
    pairs = frozenset([frozenset([0, 1]), frozenset([0, 2]), frozenset([1, 2])])
    q = dd.dists[pairs]
    pm = {"".join(o) if not isinstance(o, str) else o: float(p) for o, p in zip(q.outcomes, q.pmf)}
    wt1 = np.array([pm.get("001", 0), pm.get("010", 0), pm.get("100", 0)])
    wt2 = np.array([pm.get("011", 0), pm.get("101", 0), pm.get("110", 0)])
    assert np.ptp(wt1) < 1e-3
    assert np.ptp(wt2) < 1e-3
    assert dd[pairs]["rKL"] > 1.0  # pairs do not recover W in m-flat


def test_dual_dd_pairs_match_order2_eps_limit():
    d = Distribution(["000", "111"], [0.5, 0.5])
    schedule = (1e-4, 1e-6, 1e-8)
    dd = DualDependencyDecomposition(d, eps_schedule=schedule, nrestarts=4, maxiter=800)
    pairs = frozenset([frozenset([0, 1]), frozenset([0, 2]), frozenset([1, 2])])
    q2 = m_projection_eps_limit(d, order=2, eps_schedule=schedule, nrestarts=4, maxiter=800)["dist"]
    assert dd.dists[pairs].is_approx_equal(q2, rtol=1e-5, atol=1e-5)
    assert dd[pairs]["rKL"] == pytest.approx(D(q2, dd.target), abs=1e-5)


def test_dual_dd_single_eps():
    d = Distribution(["000", "111"], [0.5, 0.5])
    dd = DualDependencyDecomposition(d, eps=1e-6, eps_schedule=None, nrestarts=4, maxiter=800)
    top = frozenset([frozenset([0, 1, 2])])
    assert dd[top]["rKL"] == pytest.approx(0.0, abs=1e-8)


def test_dual_dd_differs_from_dependency_decomposition():
    d = n_mod_m(3, 2)
    dual = DualDependencyDecomposition(d, nrestarts=4, maxiter=800)
    primal = DependencyDecomposition(d, measures={"H": H})
    assert dual.get_dependencies() == primal.get_dependencies()
    pairs = frozenset([frozenset([0, 1]), frozenset([0, 2]), frozenset([1, 2])])
    assert dual[pairs]["rKL"] != pytest.approx(primal[pairs]["H"], abs=0.1)


def test_dual_dd_named_rvs():
    d = n_mod_m(3, 2)
    d.set_rv_names("XYZ")
    dd = DualDependencyDecomposition(d, nrestarts=4, maxiter=800)
    top = frozenset([frozenset(["X", "Y", "Z"])])
    assert dd[top]["rKL"] == pytest.approx(0.0, abs=1e-8)


def test_dual_dd_custom_measure():
    d = Distribution(["000", "111"], [0.5, 0.5])
    dd = DualDependencyDecomposition(
        d,
        measures={"H": H, "rKL": DualDependencyDecomposition.REVERSE_KL},
        nrestarts=4,
        maxiter=800,
    )
    bottom = frozenset([frozenset([0]), frozenset([1]), frozenset([2])])
    assert "H" in dd[bottom] and "rKL" in dd[bottom]
    assert dd[bottom]["H"] > 0


def test_m_projection_eps_limit_w_order2():
    w = Distribution(["001", "010", "100"], [1 / 3] * 3)
    res = m_projection_eps_limit(w, order=2, eps_schedule=(1e-4, 1e-6), nrestarts=8, maxiter=1500)
    q = res["dist"]
    pm = {"".join(o) if not isinstance(o, str) else o: float(p) for o, p in zip(q.outcomes, q.pmf)}
    wt2 = [pm.get(x, 0) for x in ("011", "101", "110")]
    assert np.ptp(wt2) < 1e-3
    assert res["rKL"] > 1.0
