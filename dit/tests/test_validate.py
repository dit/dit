from nose.tools import *

import numpy as np

import dit
from dit.math import LinearOperations, LogOperations
import dit.validate as v

def test_is_pmf():
    ops = LinearOperations()

    pmf = np.asarray([.5, .5])
    assert_true(v.is_pmf(pmf, ops))

    pmf = np.array([.6, .7])
    assert_false(v.is_pmf(pmf, ops))

    pmf = np.array([1.6, -.6])
    assert_false(v.is_pmf(pmf, ops))

def test_validate_normalization():
    ops = LinearOperations()

    pmf = np.asarray([.6, .4])
    assert_true(v.validate_normalization(pmf, ops))

    pmf = np.asarray([.6, .6])
    assert_raises(v.InvalidNormalization, v.validate_normalization, pmf, ops)

def test_validate_outcomes():
    outcomes = [1,2,3]
    sample_space = [1,2,3,4]
    assert_true(v.validate_outcomes(outcomes, sample_space))

    # One bad outcome
    outcomes = [1,2,3]
    sample_space = [1,2,4]
    assert_raises(v.InvalidOutcome, v.validate_outcomes, outcomes, sample_space)

    # Multiple bad outcomes
    outcomes = [1,2,3]
    sample_space = [1]
    assert_raises(v.InvalidOutcome, v.validate_outcomes, outcomes, sample_space)

def test_validate_pmf():
    # Already covered by other tests
    ops = LinearOperations()
    pmf = np.asarray([.5, .5])
    assert_true(v.is_pmf(pmf, ops))

def test_validate_probabilities():
    # Already covered by other tests
    ops = LinearOperations()
    pmf = np.asarray([.5, .5])
    assert_true(v.validate_probabilities(pmf, ops))

    ops = LogOperations(2)
    pmf = np.array([.1, .23, -.523])
    assert_raises(v.InvalidProbability, v.validate_probabilities, pmf, ops)

def test_validate_sequence():
    x = '101'
    assert_true(v.validate_sequence(x))
    x = 3
    assert_raises(dit.exceptions.ditException, v.validate_sequence, x)

def test_validate_outcome_class():
    x = [1,2,3]
    assert_true(v.validate_outcome_class(x))
    x = [1,2,'3']
    assert_raises(dit.exceptions.ditException, v.validate_outcome_class, x)

def test_validate_outcome_length():
    # Is a sequence
    x = ['1','2','3']
    assert_true(v.validate_outcome_length(x))
    # Not a sequence
    x = ['1','2',3]
    assert_raises(dit.exceptions.ditException, v.validate_outcome_length, x)
    # Unequal lengths
    x = ['1','2','33']
    assert_raises(dit.exceptions.ditException, v.validate_outcome_length, x)
