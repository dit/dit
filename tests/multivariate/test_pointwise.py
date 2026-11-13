"""
Tests for dit.multivariate.pointwise.
"""

import itertools

import numpy as np
import pytest

from dit import Distribution
from dit.example_dists import Xor
from dit.exceptions import InvalidOutcome, ditException
from dit.multivariate import (
    PointwiseMeasure,
    coinformation,
    dual_total_correlation,
    entropy,
    interaction_information,
    o_information,
    pointwise_coinformation,
    pointwise_dual_total_correlation,
    pointwise_entropy,
    pointwise_interaction_information,
    pointwise_mutual_information,
    pointwise_o_information,
    pointwise_residual_entropy,
    pointwise_s_information,
    pointwise_total_correlation,
    residual_entropy,
    s_information,
    total_correlation,
)

PAIRS = [
    (pointwise_entropy, entropy),
    (pointwise_coinformation, coinformation),
    (pointwise_total_correlation, total_correlation),
    (pointwise_dual_total_correlation, dual_total_correlation),
    (pointwise_residual_entropy, residual_entropy),
    (pointwise_interaction_information, interaction_information),
    (pointwise_o_information, o_information),
    (pointwise_s_information, s_information),
]

POS = Distribution(["00", "01", "10", "11"], [0.4, 0.1, 0.1, 0.4])


def random_sparse(rng, n=4, k=3, keep=0.5):
    """A random distribution over `n` variables with `k` symbols and structural zeros."""
    outcomes = ["".join(o) for o in itertools.product("012"[:k], repeat=n)]
    pmf = rng.dirichlet(np.ones(len(outcomes))) * (rng.random(len(outcomes)) < keep)
    pmf[0] += 0.01
    return Distribution(outcomes, pmf / pmf.sum())


def ising(coupling):
    """Three spins with p(s) proportional to exp(J (s0 s1 + s0 s2 + s1 s2))."""
    outcomes = ["".join(o) for o in itertools.product("01", repeat=3)]
    spins = [[2 * int(c) - 1 for c in o] for o in outcomes]
    weights = np.array([np.exp(coupling * (s[0] * s[1] + s[0] * s[2] + s[1] * s[2])) for s in spins])
    return Distribution(outcomes, weights / weights.sum())


@pytest.mark.parametrize("pointwise, measure", PAIRS)
@pytest.mark.parametrize(
    "rvs, crvs",
    [
        (None, None),
        ([[0], [1]], None),
        ([[0], [1], [2]], [3]),
        ([[0, 1], [2]], [3]),
        ([[0], [2]], [1, 3]),
    ],
)
def test_expectation_is_measure(pointwise, measure, rvs, crvs):
    """Averaging a pointwise measure gives the measure."""
    rng = np.random.default_rng(0)
    for _ in range(3):
        d = random_sparse(rng)
        assert pointwise(d, rvs, crvs).expectation() == pytest.approx(measure(d, rvs, crvs), abs=1e-10)


@pytest.mark.parametrize("pointwise, measure", PAIRS)
def test_expectation_in_base(pointwise, measure):
    """Values are in the base of the distribution, as the measures are."""
    d = random_sparse(np.random.default_rng(1), n=3).copy(base="e")
    linear = pointwise(d.copy(base="linear"), [[0], [1]], [2])
    natural = pointwise(d, [[0], [1]], [2])
    assert natural.expectation() == pytest.approx(measure(d, [[0], [1]], [2]))
    np.testing.assert_allclose(natural.data, linear.data * np.log(2))


def test_pointwise_mutual_information():
    """The pointwise mutual information is log2 p(x, y) / (p(x) p(y))."""
    pmi = pointwise_mutual_information(POS)
    assert pmi["00"] == pytest.approx(np.log2(0.4 / 0.25))
    assert pmi["01"] == pytest.approx(np.log2(0.1 / 0.25))
    assert pmi["01"] < 0
    assert pmi.name == "i[X0:X1]"
    assert pointwise_mutual_information is pointwise_coinformation


def test_pointwise_entropy():
    """The pointwise entropy is the surprisal, -log2 p(x|z)."""
    d = Distribution(["A", "B", "C"], [1 / 2, 1 / 4, 1 / 4])
    h = pointwise_entropy(d)
    assert h.to_dict() == {("A",): 1.0, ("B",): 2.0, ("C",): 2.0}
    d = random_sparse(np.random.default_rng(2), n=3)
    h = pointwise_entropy(d, [0, 2], [1])
    assert h.name == "h[X0,X2|X1]"
    for outcome, value in h.to_dict().items():
        p_xyz = d[outcome]
        p_y = d.marginal([1])[(outcome[1],)]
        assert value == pytest.approx(-np.log2(p_xyz / p_y))


def test_xor():
    """Every outcome of xor has the same local values."""
    d = Xor()
    assert set(pointwise_coinformation(d).to_dict().values()) == {-1.0}
    assert set(pointwise_coinformation(d, [[0], [1]], [2]).to_dict().values()) == {1.0}
    assert set(pointwise_coinformation(d, [[0], [1]]).to_dict().values()) == {0.0}
    assert set(pointwise_total_correlation(d).to_dict().values()) == {1.0}
    assert set(pointwise_dual_total_correlation(d).to_dict().values()) == {2.0}


def test_local_o_information_identity():
    """
    The local O-information is (n - 2) h(x) + sum_j h(x_j) - h(x_{-j}), eq. (5) of
    Scagliarini et al. (2022).
    """
    d = random_sparse(np.random.default_rng(3))
    omega = pointwise_o_information(d)

    def h(rvs, outcome):
        return -np.log2(d.marginal(rvs)[tuple(outcome[i] for i in rvs)])

    for outcome, value in omega.to_dict().items():
        rest = [[i for i in range(4) if i != j] for j in range(4)]
        expected = 2 * h([0, 1, 2, 3], outcome) + sum(h([j], outcome) - h(rest[j], outcome) for j in range(4))
        assert value == pytest.approx(expected)


@pytest.mark.parametrize("coupling", [1.0, -1.0, 0.3, -0.3])
def test_ising_o_information(coupling):
    """
    Aligned spins are redundancy-dominated and the others synergy-dominated, whatever
    the sign of the coupling (Scagliarini et al., 2022, Fig. 1).
    """
    omega = pointwise_o_information(ising(coupling))
    for outcome, value in omega.to_dict().items():
        assert (value > 0) == (len(set(outcome)) == 1)
    assert (omega.expectation() > 0) == (coupling > 0)


def test_lookup():
    """Values can be looked up by outcome in several forms."""
    d = Xor()
    d.set_rv_names("XYZ")
    pmi = pointwise_mutual_information(d, ["X", "Z"])
    assert pmi.dist.dims == ("X", "Z")
    assert pmi[("0", "1")] == pmi["01"] == pmi[["0", "1"]] == pmi[np.array(["0", "1"])] == 0.0
    assert pmi["011"] == pmi[{"X": "0", "Z": "1"}] == pmi[{"X": "0", "Y": "1", "Z": "1"}] == 0.0
    assert np.isnan(pointwise_coinformation(d)["001"])
    for outcome in ["2", "02", ("0", "2"), {"X": "0"}, ["0"], 1]:
        with pytest.raises(InvalidOutcome):
            pmi[outcome]


def test_lookup_one_variable():
    """A measure of one variable accepts bare symbols, including multi-character ones."""
    d = Distribution([(0, "red"), (1, "blue")], [1 / 4, 3 / 4])
    h = pointwise_entropy(d, [1])
    assert h["red"] == h[("red",)] == h[(0, "red")] == 2.0
    h = pointwise_entropy(Distribution([0, 1, 2], [1 / 2, 1 / 4, 1 / 4]))
    assert h[2] == h[(2,)] == 2.0
    with pytest.raises(InvalidOutcome):
        h[3]


def test_rv_names():
    """Variables can be given by name or by index."""
    d = random_sparse(np.random.default_rng(4), n=3)
    by_index = pointwise_coinformation(d, [[0], [1]], [2])
    d.set_rv_names("XYZ")
    by_name = pointwise_coinformation(d, "XY", "Z")
    assert by_name.name == "i[X:Y|Z]"
    np.testing.assert_array_equal(by_name.data, by_index.data)


def test_off_support():
    """Values are nan exactly off the support."""
    d = random_sparse(np.random.default_rng(5), n=3)
    pw = pointwise_dual_total_correlation(d)
    support = d.copy()
    support.make_dense()
    for outcome, p in support.zipped(mode="atoms"):
        assert np.isnan(pw[outcome]) == (p == 0)


def test_specific_surprise():
    """
    Averaging the pointwise mutual information over X gives the specific surprise
    sum_x p(x|y) log2 p(x|y) / p(x) of DeWeese & Meister (1999).
    """
    d = random_sparse(np.random.default_rng(6), n=2)
    surprise = pointwise_mutual_information(d).expectation(over=[0])
    assert isinstance(surprise, PointwiseMeasure)
    assert surprise.name == "E[i[X0:X1] | X1]"
    assert surprise.dist.dims == ("X1",)
    p_x = d.marginal([0])
    p_y = d.marginal([1])
    for (y,), value in surprise.to_dict().items():
        expected = sum(
            d[(x, y)] / p_y[(y,)] * np.log2(d[(x, y)] / p_y[(y,)] / p_x[(x,)]) for (x,) in p_x.outcomes if d[(x, y)] > 0
        )
        assert value == pytest.approx(expected)
        assert value >= 0
    assert surprise.expectation() == pytest.approx(coinformation(d))


def test_partial_expectation():
    """Partially averaging over variables, conditional on the rest."""
    d = random_sparse(np.random.default_rng(7), n=3)
    d.set_rv_names("XYZ")
    cmi = pointwise_coinformation(d, "XY", "Z")
    per_z = cmi.expectation(over="XY")
    marginal, conditionals = d.condition_on(crvs=["Z"], rvs=["X", "Y"])
    for (z,), conditional in zip(marginal.outcomes, conditionals, strict=True):
        assert per_z[z] == pytest.approx(coinformation(conditional, "XY"))
    assert cmi.expectation(over=[0, 1, 2]) == cmi.expectation(over="XYZ") == pytest.approx(cmi.expectation())
    assert cmi.expectation(over=2).expectation() == pytest.approx(cmi.expectation())
    pmi = pointwise_mutual_information(d, "XY")
    np.testing.assert_allclose(pmi.expectation(over="Z").data, pmi.data)
    with pytest.raises(ditException):
        pmi.expectation(over="W")
    with pytest.raises(ditException):
        pmi.expectation(over=[3])


def test_to_string():
    """The table lists each outcome, its probability, and its value."""
    pmi = pointwise_mutual_information(POS)
    assert repr(pmi) == "<PointwiseMeasure i[X0:X1]>"
    table = str(pmi)
    assert "i[X0:X1]" in table
    assert "-1.3219" in table
    assert "0.6781" in table
    assert "0.400" in pmi.to_string(digits=3)


def test_symbolic():
    """Symbolic distributions are not supported."""
    pytest.importorskip("sympy")
    from dit.symbolic import symbolic_distribution, symbols

    p = symbols("p")
    d = symbolic_distribution(["00", "11"], [p, 1 - p])
    with pytest.raises(NotImplementedError):
        pointwise_entropy(d)
