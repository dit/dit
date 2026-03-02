"""
Tests for dit.algorithms
"""

import numpy as np
import pytest

import dit


def test_pruned_samplespace_scalar():
    """Prune a sample space from a Distribution."""
    pmf = [1 / 2, 0, 1 / 2]
    d = dit.Distribution(pmf)
    d2 = dit.algorithms.pruned_samplespace(d)
    assert d2.outcomes == (0, 2)
    assert np.allclose(d2.pmf, [1 / 2, 1 / 2])


def test_pruned_samplespace():
    """Prune a sample space from a Distribution."""
    outcomes = ["0", "1", "2"]
    pmf = [1 / 2, 0, 1 / 2]
    d = dit.Distribution(outcomes, pmf)
    d2 = dit.algorithms.pruned_samplespace(d)
    assert d2.outcomes == (("0",), ("2",))
    assert np.allclose(d2.pmf, [1 / 2, 1 / 2])


def test_pruned_samplespace2():
    """Prune a sample space while specifying a desired sample space."""
    outcomes = ["0", "1", "2", "3"]
    pmf = [1 / 2, 0, 1 / 2, 0]
    d = dit.Distribution(outcomes, pmf)
    d2 = dit.algorithms.pruned_samplespace(d, sample_space=[("0",), ("1",), ("2",)])
    d2.make_dense()
    assert d2.outcomes == (("0",), ("1",), ("2",))
    assert np.allclose(d2.pmf, [1 / 2, 0, 1 / 2])


def test_expanded_samplespace():
    """Expand a sample space from a Distribution."""
    outcomes = ["01", "10"]
    pmf = [1 / 2, 1 / 2]
    d = dit.Distribution(outcomes, pmf)
    d2 = dit.algorithms.expanded_samplespace(d)
    d2.make_dense()
    ss = [("0", "0"), ("0", "1"), ("1", "0"), ("1", "1")]
    assert list(d2.sample_space()) == ss


def test_expanded_samplespace2():
    """Expand a sample space from a Distribution."""
    pmf = [1 / 2, 1 / 2]
    d = dit.Distribution(pmf)
    assert d.outcomes == (0, 1)
    d2 = dit.algorithms.expanded_samplespace(d, [0, 1, 2])
    d2.make_dense()
    assert d2.outcomes == (0, 1, 2)


def test_expanded_samplespace3():
    """Expand a sample space without unioning the alphabets."""
    outcomes = ["01a", "10a"]
    pmf = [1 / 2, 1 / 2]
    d = dit.Distribution(outcomes, pmf)
    d2 = dit.algorithms.expanded_samplespace(d, union=False)
    d2.make_dense()
    ss_ = [("0", "0", "a"), ("0", "1", "a"), ("1", "0", "a"), ("1", "1", "a")]
    assert list(d2.sample_space()) == ss_


def test_expanded_samplespace_bad():
    """Expand a sample space with wrong number of alphabets."""
    outcomes = ["01", "10"]
    pmf = [1 / 2, 1 / 2]
    d = dit.Distribution(outcomes, pmf)
    alphabets = ["01"]
    assert d.outcome_length() == 2
    with pytest.raises(Exception, match="You need to provide 1 alphabets"):
        dit.algorithms.expanded_samplespace(d, alphabets)


def test_expanded_samplespace_bad2():
    """Expand a sample space with incompatible alphabet."""
    outcomes = "01"
    pmf = [1 / 2, 1 / 2]
    d = dit.Distribution(outcomes, pmf)
    alphabets = [["0"]]
    assert d.outcome_length() == 1
    with pytest.raises(dit.exceptions.InvalidOutcome):
        dit.algorithms.expanded_samplespace(d, alphabets)
