"""
Fast smoke tests for dit.pid.syndisc.

The full SynDisc decomposition is optimization-heavy (see the @slow tests in
test_syndisc.py).  These tests exercise the optimization-free machinery -- the
constraint lattice construction and node mapping -- and construct the optimizer
to evaluate its objective/constraints *once* (without basin-hopping), so the
code paths are executed and confirmed syntactically correct quickly.
"""

import numpy as np

from dit.example_dists import Xor
from dit.pid.syndisc import (
    ModifiedSynDisc,
    SynDisc,
    SyndiscOptimizer,
    _build_constraint_lattice,
    _node_to_alpha,
    _transform_constraint,
)
from lattices.orderings import constraint_le

# ── Pure lattice machinery (no optimization) ───────────────────────────────


def test_constraint_le():
    a = frozenset({frozenset({(0,)})})
    b = frozenset({frozenset({(0,), (1,)})})
    assert constraint_le(a, b)
    assert not constraint_le(b, a)
    assert constraint_le(frozenset(), a)  # empty alpha <= anything
    assert not constraint_le(a, frozenset())  # nothing <= empty beta


def test_build_constraint_lattice():
    lat = _build_constraint_lattice(((0,), (1,)))
    assert len(list(lat)) == 5


def test_transform_constraint():
    lat = _transform_constraint(_build_constraint_lattice(((0,), (1,))))
    assert lat.top == ()
    assert lat.bottom == ((0, 1),)


def test_node_to_alpha():
    assert _node_to_alpha(((0,), (1,)), ((0,), (1,))) == ((0,), (1,))
    assert _node_to_alpha(((0, 1),), ((0,), (1,))) == ((0, 1),)


# ── Optimizer smoke test (construct + evaluate, no optimize) ────────────────


def test_syndisc_optimizer_smoke():
    """Build the optimizer and evaluate its objective once, without optimizing."""
    d = Xor()
    opt = SyndiscOptimizer(d, [[0], [1]], [2], alpha=((0,),))
    assert opt._compute_bound() == 5

    x = opt.construct_random_initial()
    obj = opt._objective()(opt, x)
    assert np.isfinite(obj)

    disclosure = opt.synergistic_disclosure(x)
    assert np.isfinite(disclosure)
    assert disclosure >= -1e-9

    for fn in opt._mi_subgroups.values():
        assert np.isfinite(opt._squared_mi(x, fn))


# ── Decomposition / display machinery (per-node optimization stubbed out) ───


def _exercise(sd):
    """Run the lattice/Mobius/backbone/display code paths over a built object."""
    assert np.isfinite(sd[()])
    for node in sd._lattice:
        assert np.isfinite(sd.get_synergy(node))
        assert np.isfinite(sd.get_atom(node))
    n = len(sd._sources)
    for m in range(n + 1):
        assert np.isfinite(sd.get_backbone(m))
    for m in range(1, n + 1):
        assert np.isfinite(sd.get_backbone_atom(m))
    assert "S_" in sd.to_string()
    assert "backbone" in sd.backbone_to_string()
    assert isinstance(repr(sd), str)
    assert isinstance(str(sd), str)


def test_syndisc_decomposition_smoke(monkeypatch):
    """Exercise the full SynDisc decomposition with the optimizer stubbed out.

    parallel_sweep uses threads, so the stub is visible to its workers.
    """
    monkeypatch.setattr(SynDisc, "_compute_s_alpha", lambda self, node, rng=None: float(len(node)))
    _exercise(SynDisc(Xor()))


def test_syndisc_real_compute_s_alpha(monkeypatch):
    """Run the *real* _compute_s_alpha for every lattice node, stubbing only the
    basin-hopping optimize() so each node does a single fast evaluation.

    This covers the trivial-node short circuits (empty / unconstrained / fully
    constrained) and the optimizer-backed branch without the slow optimization.
    """

    def fake_optimize(self, niter=None, rng=None, **kwargs):
        self._optima = self.construct_random_initial()

    monkeypatch.setattr(SyndiscOptimizer, "optimize", fake_optimize)
    sd = SynDisc(Xor())
    for node in sd._lattice:
        assert np.isfinite(sd.get_synergy(node))


def test_modified_syndisc_smoke(monkeypatch):
    """ModifiedSynDisc runs its real singleton/CMI logic; only the multi-source
    fallback (the slow optimizer) is stubbed.

    Regression guard: ModifiedSynDisc.__init__ previously raised TypeError
    because _compute_s_alpha did not accept the rng kwarg threaded through by
    parallel_sweep.
    """
    monkeypatch.setattr(SynDisc, "_compute_s_alpha", lambda self, node, rng=None: 0.5)
    _exercise(ModifiedSynDisc(Xor()))
