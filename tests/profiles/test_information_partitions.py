"""
Tests for dit.util.information_partitions.
"""

from itertools import islice, product

import numpy as np
import pytest

from dit import Distribution
from dit.algorithms import degrees_of_freedom
from dit.example_dists import n_mod_m
from dit.multivariate import coinformation as I
from dit.multivariate import dual_total_correlation as B
from dit.multivariate import total_correlation, transmission
from dit.profiles.information_partitions import (
    DependencyDecomposition,
    DualDependencyDecomposition,
    ExtropyPartition,
    ShannonPartition,
    ShapleyDecomposition,
)
from dit.shannon import entropy
from dit.utils import partitions, powerset


def all_info_measures(rvs):
    """ """
    for stuff in islice(powerset(rvs), 1, None):
        others = set(rvs) - set(stuff)
        for part in partitions(stuff, tuples=True):
            for cond in powerset(others):
                yield (part, cond)


@pytest.mark.parametrize("meas", all_info_measures(range(4)))
def test_sp1(meas):
    """Test all possible info measures"""
    d = n_mod_m(4, 2)
    ip = ShannonPartition(d)
    assert ip[meas] == pytest.approx(I(d, meas[0], meas[1]))


@pytest.mark.parametrize("meas", all_info_measures("xyzw"))
def test_sp2(meas):
    """Test all possible info measures, with rv_names"""
    d = n_mod_m(4, 2)
    d.set_rv_names("xyzw")
    ip = ShannonPartition(d)
    assert ip[meas] == pytest.approx(I(d, meas[0], meas[1]))


def test_sp3():
    """Test get_atoms()"""
    d = n_mod_m(3, 2)
    ip = ShannonPartition(d)
    atoms1 = {"H[0|1,2]", "H[1|0,2]", "H[2|0,1]", "I[0:1:2]", "I[0:1|2]", "I[0:2|1]", "I[1:2|0]"}
    atoms2 = ip.get_atoms()
    assert (atoms1 - atoms2) | (atoms2 - atoms1) == set()


def test_sp4():
    """Test printing"""
    d = n_mod_m(3, 2)
    ip = ShannonPartition(d)
    string = """\
+---------------------+
|  Shannon Partition  |
+-----------+---------+
|  measure  |   bits  |
+-----------+---------+
|  H[0|1,2] |   0.000 |
|  H[1|0,2] |   0.000 |
|  H[2|0,1] |   0.000 |
|  I[0:1|2] |   1.000 |
|  I[0:2|1] |   1.000 |
|  I[1:2|0] |   1.000 |
|  I[0:1:2] |  -1.000 |
+-----------+---------+"""
    assert str(ip) == string


def test_ep1():
    """
    Test against known values.
    """
    d = n_mod_m(3, 2)
    ep = ExtropyPartition(d)
    string = """\
+---------------------+
|  Extropy Partition  |
+-----------+---------+
|  measure  |  exits  |
+-----------+---------+
|  X[0|1,2] |   0.000 |
|  X[1|0,2] |   0.000 |
|  X[2|0,1] |   0.000 |
|  X[0:1|2] |   0.245 |
|  X[0:2|1] |   0.245 |
|  X[1:2|0] |   0.245 |
|  X[0:1:2] |   0.510 |
+-----------+---------+"""
    assert str(ep) == string


def test_dd1():
    """
    Test against known values.
    """
    d = n_mod_m(3, 2)
    ep = DependencyDecomposition(d, measures={"B": B})
    string = """\
+--------------------------+
| Dependency Decomposition |
+---------------+----------+
|   dependency  |    B     |
+---------------+----------+
|      012      |   2.000  |
|    01:02:12   |   0.000  |
|     01:02     |   0.000  |
|     01:12     |   0.000  |
|     02:12     |   0.000  |
|      01:2     |   0.000  |
|      02:1     |   0.000  |
|      12:0     |   0.000  |
|     0:1:2     |   0.000  |
+---------------+----------+"""
    assert str(ep) == string


def test_dd2():
    """
    Test against known values.
    """
    d = n_mod_m(3, 2)
    d.set_rv_names("XYZ")
    dd = DependencyDecomposition(d)
    assert dd[frozenset([frozenset(["X", "Y", "Z"])])]["H"] == pytest.approx(2.0)


def test_dd3():
    """
    Test dependencies.
    """
    d = n_mod_m(3, 2)
    dd = DependencyDecomposition(d)
    deps = dd.get_dependencies()
    true_deps = {"012", "01:02:12", "01:02", "01:12", "02:12", "01:2", "02:1", "12:0", "0:1:2"}
    assert deps == true_deps


def test_dd4():
    """
    Test dependencies.
    """
    d = n_mod_m(3, 2)
    dd = DependencyDecomposition(d)
    deps = dd.get_dependencies(string=False)
    a = frozenset([0])
    b = frozenset([1])
    c = frozenset([2])
    true_deps = {
        frozenset([a | b | c]),
        frozenset([a | b, a | c, b | c]),
        frozenset([a | b, a | c]),
        frozenset([a | b, b | c]),
        frozenset([a | c, b | c]),
        frozenset([a | b, c]),
        frozenset([a | c, b]),
        frozenset([b | c, a]),
        frozenset([a, b, c]),
    }
    assert deps == true_deps


# Reconstructability measures and the Shapley decomposition


def _node(spec, label=int):
    """The dependency written like '01:2', each character labelling a variable; '' is the empty dependency."""
    if not spec:
        return frozenset()
    return frozenset(frozenset(label(c) for c in part) for part in spec.split(":"))


def _shapley(decomposition, measure="H"):
    """The Shapley value of each atom (a frozenset holding the dependency it adds) for `measure`."""
    return {atom: values[measure] for atom, values in decomposition._shapleys.items()}


def _generic_three_bits():
    """Three bits in which every pair of variables is dependent, each pair differently."""
    outcomes = ["000", "001", "010", "011", "100", "101", "110", "111"]
    return Distribution(outcomes, [0.27, 0.04, 0.09, 0.12, 0.05, 0.15, 0.17, 0.11])


def _exchangeable_pair_with_child():
    """X0 and X1 are i.i.d. bits and X2 depends on their sum, so swapping X0 and X1 is a symmetry."""
    p, q = 0.3, (0.1, 0.6, 0.85)
    outcomes, pmf = [], []
    for x0, x1, x2 in product((0, 1), repeat=3):
        outcomes.append(f"{x0}{x1}{x2}")
        pmf.append((p if x0 else 1 - p) * (p if x1 else 1 - p) * (q[x0 + x1] if x2 else 1 - q[x0 + x1]))
    return Distribution(outcomes, pmf)


def test_dd_degrees_of_freedom():
    """The degrees-of-freedom measure counts each structure's free log-linear parameters."""
    d = n_mod_m(3, 3)
    dd = DependencyDecomposition(d, measures={"dof": degrees_of_freedom})
    # Each ternary variable has 2 free parameters, each pairwise interaction 4, and the saturated model 3**3 - 1.
    expected = {
        "012": 26,
        "01:02:12": 18,
        "01:02": 14,
        "01:12": 14,
        "02:12": 14,
        "01:2": 10,
        "02:1": 10,
        "12:0": 10,
        "0:1:2": 6,
    }
    assert {node: values["dof"] for node, values in dd.atoms.items()} == {_node(k): v for k, v in expected.items()}
    d.set_rv_names("XYZ")
    named = DependencyDecomposition(d, measures={"dof": degrees_of_freedom})
    assert named[_node("XY:Z", str)]["dof"] == 10
    assert named[_node("XYZ", str)]["dof"] == 26


def test_dd_transmission():
    """The transmission atom is the KL divergence from the data to each node's MaxEnt reconstruction."""
    d = _generic_three_bits()
    dd = DependencyDecomposition(d, measures={"H": entropy, "T": transmission})
    for node, values in dd.atoms.items():
        # Pythagorean identity of the I-projection: D(P || Q) = H(Q) - H(P).
        assert values["T"] == pytest.approx(values["H"] - entropy(d), abs=1e-6)
        assert values["T"] == pytest.approx(transmission(d, [sorted(part) for part in node]), abs=1e-6)
    assert dd[_node("012")]["T"] == pytest.approx(0.0, abs=1e-9)
    assert dd[_node("0:1:2")]["T"] == pytest.approx(total_correlation(d), abs=1e-6)


def test_dual_dd_degrees_of_freedom_and_forward_kl():
    """In the dual decomposition transmission is the forward KL D(P || Q), unlike the default reverse KL."""
    d = Distribution(["00", "01", "10", "11"], [0.4, 0.1, 0.1, 0.4])
    measures = {"dof": degrees_of_freedom, "rKL": DualDependencyDecomposition.REVERSE_KL, "T": transmission}
    dual = DualDependencyDecomposition(d, measures=measures, nrestarts=1, maxiter=200)
    top, bottom = _node("01"), _node("0:1")
    assert dual[top] == pytest.approx({"dof": 3, "rKL": 0.0, "T": 0.0}, abs=1e-9)
    # P is invariant under swapping the bits and under flipping both, so its m-projection onto
    # h0(x0) + h1(x1) is the uniform distribution.
    np.testing.assert_allclose(dual.dists[bottom].pmf, 0.25, atol=1e-6)
    assert dual[bottom]["dof"] == 2
    assert dual[bottom]["T"] == pytest.approx(2 - entropy(d), abs=1e-6)  # D(P || U)
    assert dual[bottom]["rKL"] == pytest.approx(-2 - np.mean(np.log2(d.pmf)), abs=1e-6)  # D(U || P)


def test_dual_dd_reconstructs_the_mixture_family_exactly():
    """A distribution h0(x0) + h1(x1) has no dual transmission at 0:1, though its MaxEnt transmission is I[X0:X1]."""
    d = Distribution(["00", "01", "10", "11"], [0.48, 0.26, 0.24, 0.02])  # P00 + P11 == P01 + P10
    dual = DualDependencyDecomposition(d, measures={"T": transmission}, nrestarts=1, maxiter=200)
    primal = DependencyDecomposition(d, measures={"T": transmission})
    bottom = _node("0:1")
    assert dual[bottom]["T"] == pytest.approx(0.0, abs=1e-6)
    assert primal[bottom]["T"] == pytest.approx(I(d, [[0], [1]]), abs=1e-6)
    assert primal[bottom]["T"] > 0.05


def test_shapley_two_variables_by_hand():
    """With two variables the Shapley values are H[X_i] - 1 for each variable and -I[X0:X1] for the pair."""
    d = Distribution(["00", "01", "10", "11"], [0.5, 0.2, 0.1, 0.2])
    sd = ShapleyDecomposition(d)
    expected = {_node("0"): entropy(d, [0]) - 1, _node("1"): entropy(d, [1]) - 1, _node("01"): -I(d, [[0], [1]])}
    assert _shapley(sd) == pytest.approx(expected, abs=1e-6)
    # Indexing still gives the node values: the lattice runs from the unconstrained (uniform) distribution to P.
    assert sd[_node("")]["H"] == pytest.approx(2.0)
    assert sd[_node("01")]["H"] == pytest.approx(entropy(d))


def test_shapley_table():
    """The table lists the Shapley value of each atom."""
    d = Distribution(["00", "01", "10", "11"], [0.5, 0.2, 0.1, 0.2])
    string = """\
+------------------------+
| Shapley Decomposition  |
+--------------+---------+
|  dependency  |    H    |
+--------------+---------+
|      01      |  -0.091 |
|      0       |  -0.119 |
|      1       |  -0.029 |
+--------------+---------+"""
    assert str(ShapleyDecomposition(d)) == string


def test_shapley_values_are_efficient():
    """The Shapley values sum to the change in entropy from the bottom of the lattice to the top."""
    d = _generic_three_bits()
    full = ShapleyDecomposition(d)
    assert len(full._shapleys) == 7  # one atom per nonempty subset of the variables
    # Without the cover restriction the bottom of the lattice is the uniform distribution on three bits ...
    assert sum(_shapley(full).values()) == pytest.approx(entropy(d) - 3, abs=1e-6)
    covers = ShapleyDecomposition(d, cover=True)
    assert len(covers._shapleys) == 4  # the singletons are already in the bottom
    # ... and with it, the product of the marginals.
    assert sum(_shapley(covers).values()) == pytest.approx(-total_correlation(d), abs=1e-6)


def test_shapley_values_respect_symmetry():
    """Exchangeable variables get equal Shapley values; single-variable and top atoms have closed forms."""
    d = _exchangeable_pair_with_child()
    phi = _shapley(ShapleyDecomposition(d))
    assert phi[_node("0")] == pytest.approx(phi[_node("1")], abs=1e-6)
    assert phi[_node("02")] == pytest.approx(phi[_node("12")], abs=1e-6)
    # A variable's own atom is added before any atom containing it, and the full joint is always added last.
    for i in range(3):
        assert phi[_node(str(i))] == pytest.approx(entropy(d, [i]) - 1, abs=1e-6)
    assert phi[_node("012")] == pytest.approx(-transmission(d, [[0, 1], [0, 2], [1, 2]]), abs=1e-6)


def test_shapley_aggregation_function():
    """Every maximal chain adds each atom exactly once, and agg_func reduces the per-chain contributions."""
    d = _generic_three_bits()
    contributions = _shapley(ShapleyDecomposition(d, agg_func=list))
    phi = _shapley(ShapleyDecomposition(d))
    for atom, values in contributions.items():
        # The maximal chains are the 48 linear extensions of the Boolean lattice on three elements.
        assert len(values) == 48
        assert np.mean(values) == pytest.approx(phi[atom], abs=1e-9)
    # Adding a single variable, or the full joint, has the same effect in every chain; a pairwise term does not.
    assert np.ptp(contributions[_node("0")]) == pytest.approx(0.0, abs=1e-6)
    assert np.ptp(contributions[_node("012")]) == pytest.approx(0.0, abs=1e-6)
    assert np.ptp(contributions[_node("01")]) > 1e-3


def test_shapley_with_reconstructability_measures():
    """Each interaction term adds prod(|A_i| - 1) parameters, and transmission changes exactly as entropy does."""
    d = Distribution(["00", "01", "10", "11", "20", "21"], [0.3, 0.1, 0.05, 0.2, 0.15, 0.2])
    sd = ShapleyDecomposition(d, measures={"H": entropy, "dof": degrees_of_freedom, "T": transmission})
    assert _shapley(sd, "dof") == {_node("0"): 2, _node("1"): 1, _node("01"): 2}
    assert sum(_shapley(sd, "dof").values()) == 3 * 2 - 1
    # T(node) = H(node) - H(P) at every node, so the two measures change identically along every edge.
    assert _shapley(sd, "T") == pytest.approx(_shapley(sd, "H"), abs=1e-9)
