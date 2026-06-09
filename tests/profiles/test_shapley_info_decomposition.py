"""
Tests for dit.profiles.shapley_info_decomposition.
"""

import pytest

from dit import Distribution
from dit.example_dists import Xor, n_mod_m
from dit.params import ditParams
from dit.profiles.shapley_info_decomposition import (
    ShapleyDependencyDecomposition,
    ShapleyShannonDecomposition,
)


def _labelled(decomp):
    """Map each predictor to a readable label for easy assertions."""
    return {ShapleyDependencyDecomposition._stringify_predictor(k): v for k, v in decomp.contributions.items()}


def test_xor_is_pure_synergy():
    """XOR with default sources/target: all information is in {0,1}."""
    s = ShapleyDependencyDecomposition(Xor())
    contribs = _labelled(s)
    assert contribs["{0}"] == pytest.approx(0.0, abs=1e-6)
    assert contribs["{1}"] == pytest.approx(0.0, abs=1e-6)
    assert contribs["{0,1}"] == pytest.approx(1.0, abs=1e-6)
    assert sum(s.contributions.values()) == pytest.approx(1.0, abs=1e-6)


def test_explicit_sources_and_target():
    """A redundant copy (X=Y=Z) splits its 1 bit equally between sources."""
    d = Distribution(["000", "111"], [0.5, 0.5])
    s = ShapleyDependencyDecomposition(d, sources=[[0], [1]], target=[2])
    contribs = _labelled(s)
    assert contribs["{0}"] == pytest.approx(0.5, abs=1e-6)
    assert contribs["{1}"] == pytest.approx(0.5, abs=1e-6)
    assert contribs["{0,1}"] == pytest.approx(0.0, abs=1e-6)


def test_three_sources_parity():
    """3-bit parity: the full triple carries all the information."""
    s = ShapleyDependencyDecomposition(n_mod_m(4, 2))
    contribs = _labelled(s)
    assert contribs["{0,1,2}"] == pytest.approx(1.0, abs=1e-6)
    assert sum(s.contributions.values()) == pytest.approx(1.0, abs=1e-6)


def test_getitem():
    """__getitem__ retrieves a predictor's contribution."""
    s = ShapleyDependencyDecomposition(Xor())
    key = frozenset({frozenset({0, 1})})
    assert s[key] == pytest.approx(1.0, abs=1e-6)


def test_str_table():
    """str() renders the table with each predictor and rounds near-zero to 0."""
    s = ShapleyDependencyDecomposition(Xor())
    out = str(s)
    assert "Shapley Dependency Decomposition" in out
    for label in ("{0}", "{1}", "{0,1}"):
        assert label in out


def test_repr_respects_ditparams():
    """__repr__ prints the table only when repr.print is set."""
    s = ShapleyDependencyDecomposition(Xor())
    original = ditParams["repr.print"]
    try:
        ditParams["repr.print"] = True
        assert repr(s) == s.to_string()
        ditParams["repr.print"] = False
        assert repr(s).startswith("<")
    finally:
        ditParams["repr.print"] = original


def test_shannon_alias_is_dependency_decomposition():
    """ShapleyShannonDecomposition aliases the Shannon DependencyDecomposition."""
    from dit.profiles.information_partitions import ShapleyDecomposition

    assert ShapleyShannonDecomposition is ShapleyDecomposition
