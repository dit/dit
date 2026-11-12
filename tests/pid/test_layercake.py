"""
Tests for dit.pid.layercake: MMI and maximal unique information from the layer cake.
"""

import numpy as np
import pytest

from dit import Distribution
from dit.multivariate import coinformation
from dit.pid import PID_MMI, PID_HiFi, PIDBounds, hifi, max_unique_atoms, mmi_atoms
from dit.pid.distributions import bivariates, trivariates
from dit.pid.layercake import _layer_cake


def _random(n_sources, seed):
    rng = np.random.default_rng(seed)
    size = 2 ** (n_sources + 1)
    return Distribution.from_ndarray(rng.dirichlet(np.ones(size)).reshape((2,) * (n_sources + 1)))


CASES = {
    **{f"bivariates[{name!r}]": d for name, d in bivariates.items()},
    **{f"trivariates[{name!r}]": d for name, d in trivariates.items()},
    **{f"random 3 sources {seed}": _random(3, seed) for seed in range(3)},
    **{f"random 4 sources {seed}": _random(4, seed) for seed in range(2)},
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_mmi_atoms(name):
    """The layer cake of the mutual informations is PID_MMI (Lemma 2)."""
    d = CASES[name]
    pid = PID_MMI(d)
    atoms = mmi_atoms(d)
    for node in pid._lattice:
        assert atoms.get(node, 0.0) == pytest.approx(pid.get_pi(node), abs=1e-9)


@pytest.mark.parametrize("name", sorted(CASES))
def test_max_unique_atoms(name):
    """The decomposition with every unique information at Hi-Fi's bound is nonnegative and
    reproduces every mutual information (Theorem 1(ii))."""
    d = CASES[name]
    atoms = max_unique_atoms(d)
    drivers = hifi(d)
    assert all(atom > 0 for atom in atoms.values())
    bounds = PIDBounds(d)
    pi = np.array([atoms.get(node, 0.0) for node in bounds.nodes])
    assert bounds._lattice.incidence @ pi == pytest.approx(bounds._informations[1:], abs=1e-9)
    for source in drivers.drivers:
        assert atoms.get((source,), 0.0) == pytest.approx(drivers[source].unique, abs=1e-9)


@pytest.mark.parametrize("name", sorted(bivariates))
def test_max_unique_atoms_two_sources_is_hifi(name):
    d = bivariates[name]
    pid = PID_HiFi(d)
    atoms = max_unique_atoms(d)
    for node in pid._lattice:
        assert atoms.get(node, 0.0) == pytest.approx(pid.get_pi(node), abs=1e-9)


def test_chain():
    """The atoms of the layer cake lie on a chain: each node's up-set contains the next's."""
    d = _random(4, 5)
    lattice = PIDBounds(d)._lattice
    nodes = list(mmi_atoms(d))
    for lower, upper in zip(nodes, nodes[1:], strict=False):
        up_lower, up_upper = lattice.up_sets[lattice.find(lower)], lattice.up_sets[lattice.find(upper)]
        assert up_upper & ~up_lower == 0
        assert up_upper != up_lower


def test_top_atom():
    """MMI's top atom is the smallest conditional information of a source given the rest."""
    d = _random(4, 2)
    synergy = min(coinformation(d, [[i], [4]], [j for j in range(4) if j != i]) for i in range(4))
    assert mmi_atoms(d).get(((0, 1, 2, 3),), 0.0) == pytest.approx(synergy, abs=1e-9)


def test_six_sources():
    """No redundancy lattice is needed, so six sources are cheap."""
    d = _random(6, 1)
    atoms = mmi_atoms(d)
    assert sum(atoms.values()) == pytest.approx(coinformation(d, [list(range(6)), [6]]), abs=1e-9)
    unique = max_unique_atoms(d)
    assert sum(unique.values()) == pytest.approx(coinformation(d, [list(range(6)), [6]]), abs=1e-9)
    assert min(unique.values()) > 0


def test_layer_cake_rounding():
    """Values within the tolerance form one layer, and rounding errors that break
    monotonicity are absorbed."""
    values = np.array([0.0, 1.0, 1.0 + 1e-13, 2.0 - 1e-13])
    atoms = _layer_cake(values, 2, 1e-9)
    assert [node for node, _ in atoms] == [(1, 2), (3,)]
    assert [atom for _, atom in atoms] == pytest.approx([1.0, 1.0], abs=1e-9)
    assert _layer_cake(np.array([0.0, 1.0, 0.0, 1.0 - 1e-14]), 2, 1e-9) == [((1,), pytest.approx(1.0))]
    assert _layer_cake(np.zeros(4), 2, 1e-9) == []


def test_units():
    reference = mmi_atoms(bivariates["and"])
    scaled = mmi_atoms(bivariates["and"].copy(base="e"))
    assert scaled == pytest.approx({node: atom * np.log(2) for node, atom in reference.items()})
    reference = max_unique_atoms(bivariates["and"])
    scaled = max_unique_atoms(bivariates["and"].copy(base="e"))
    assert scaled == pytest.approx({node: atom * np.log(2) for node, atom in reference.items()})
