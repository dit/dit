from __future__ import division

import pytest

import numpy as np

import dit

def test_pruned_samplespace_scalar():
    """Prune a sample space from a ScalarDistribution."""
    pmf = [1/2, 0, 1/2]
    d = dit.ScalarDistribution(pmf)
    d2 = dit.algorithms.pruned_samplespace(d)
    ss2_ = [0, 2]
    ss2 = list(d2.sample_space())
    assert ss2 == ss2_
    assert np.allclose(d2.pmf, [1/2, 1/2])

def test_pruned_samplespace():
    """Prune a sample space from a Distribution."""
    outcomes = ['0', '1', '2']
    pmf = [1/2, 0, 1/2]
    d = dit.ScalarDistribution(outcomes, pmf)
    d2 = dit.algorithms.pruned_samplespace(d)
    ss2_ = ['0', '2']
    ss2 = list(d2.sample_space())
    assert ss2 == ss2_
    assert np.allclose(d2.pmf, [1/2, 1/2])

def test_pruned_samplespace2():
    """Prune a sample space while specifying a desired sample space."""
    outcomes = ['0', '1', '2', '3']
    pmf = [1/2, 0, 1/2, 0]
    ss2_ = ['0', '1', '2']
    d = dit.ScalarDistribution(outcomes, pmf)
    d2 = dit.algorithms.pruned_samplespace(d, sample_space=ss2_)
    # We must make it dense, since the zero element will not appear in pmf.
    d2.make_dense()
    ss2 = list(d2.sample_space())
    assert ss2 == ss2_
    assert np.allclose(d2.pmf, [1/2, 0, 1/2])

def test_expanded_samplespace():
    """Expand a sample space from a Distribution."""
    outcomes = ['01', '10']
    pmf = [1/2, 1/2]
    d = dit.Distribution(outcomes, pmf, sample_space=outcomes)
    assert list(d.sample_space()) == ['01', '10']
    d2 = dit.algorithms.expanded_samplespace(d)
    ss = ['00', '01', '10', '11']
    assert list(d2.sample_space()) == ss

def test_expanded_samplespace2():
    """Expand a sample space from a ScalarDistribution."""
    pmf = [1/2, 1/2]
    ss = [0, 1]
    d = dit.ScalarDistribution(pmf)
    assert list(d.sample_space()) == ss
    ss2 = [0, 1, 2]
    d2 = dit.algorithms.expanded_samplespace(d, ss2)
    assert list(d2.sample_space()) == ss2

def test_expanded_samplespace3():
    """Expand a sample space without unioning the alphabets."""
    outcomes = ['01a', '10a']
    pmf = [1/2, 1/2]
    d = dit.Distribution(outcomes, pmf, sample_space=outcomes)
    d2 = dit.algorithms.expanded_samplespace(d, union=False)
    ss_ = ['00a', '01a', '10a', '11a']
    assert list(d2.sample_space()) == ss_

def test_expanded_samplespace_bad():
    """Expand a sample space with wrong number of alphabets."""
    outcomes = ['01', '10']
    pmf = [1/2, 1/2]
    d = dit.Distribution(outcomes, pmf)
    alphabets = ['01']
    assert d.outcome_length() == 2
    # This fails because we need to specify two alphabets, not one.
    with pytest.raises(Exception):
        dit.algorithms.expanded_samplespace(d, alphabets)

def test_expanded_samplespace_bad2():
    """Expand a sample space with wrong number of alphabets."""
    outcomes = '01'
    pmf = [1/2, 1/2]
    d = dit.Distribution(outcomes, pmf)
    alphabets = '0'
    assert d.outcome_length() == 1
    # This fails because the sample space is too small, doesn't contain '1'.
    with pytest.raises(dit.exceptions.InvalidOutcome):
        dit.algorithms.expanded_samplespace(d, alphabets)
