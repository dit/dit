"""
Tests for dit.validate.
"""

import pytest

import numpy as np

import dit
from dit.math import LinearOperations, LogOperations
import dit.validate as v

def test_is_pmf():
    ops = LinearOperations()

    pmf = np.asarray([0.5, 0.5])
    assert v.is_pmf(pmf, ops)

    pmf = np.array([0.6, 0.7])
    assert not v.is_pmf(pmf, ops)

    pmf = np.array([1.6, -0.6])
    assert not v.is_pmf(pmf, ops)

def test_validate_normalization():
    ops = LinearOperations()

    pmf = np.asarray([0.6, 0.4])
    assert v.validate_normalization(pmf, ops)

    pmf = np.asarray([0.6, 0.6])
    with pytest.raises(v.InvalidNormalization):
        v.validate_normalization(pmf, ops)

def test_validate_outcomes():
    outcomes = [1, 2, 3]
    sample_space = [1, 2, 3, 4]
    assert v.validate_outcomes(outcomes, sample_space)

    # One bad outcome
    outcomes = [1, 2, 3]
    sample_space = [1, 2, 4]
    with pytest.raises(v.InvalidOutcome):
        v.validate_outcomes(outcomes, sample_space)

    # Multiple bad outcomes
    outcomes = [1, 2, 3]
    sample_space = [1]
    with pytest.raises(v.InvalidOutcome):
        v.validate_outcomes(outcomes, sample_space)

def test_validate_pmf():
    # Already covered by other tests
    ops = LinearOperations()
    pmf = np.asarray([0.5, 0.5])
    assert v.is_pmf(pmf, ops)

def test_validate_probabilities():
    # Already covered by other tests
    ops = LinearOperations()
    pmf = np.asarray([0.5, 0.5])
    assert v.validate_probabilities(pmf, ops)

    ops = LogOperations(2)
    pmf = np.array([0.1, 0.23, -0.523])
    with pytest.raises(v.InvalidProbability):
        v.validate_probabilities(pmf, ops)

def test_validate_sequence():
    x = '101'
    assert v.validate_sequence(x)
    x = 3
    with pytest.raises(dit.exceptions.ditException):
        v.validate_sequence(x)

def test_validate_outcome_class():
    x = [1, 2, 3]
    assert v.validate_outcome_class(x)
    x = [1, 2, '3']
    with pytest.raises(dit.exceptions.ditException):
        v.validate_outcome_class(x)

def test_validate_outcome_length():
    # Is a sequence
    x = ['1', '2', '3']
    assert v.validate_outcome_length(x)
    # Not a sequence
    x = ['1', '2', 3]
    with pytest.raises(dit.exceptions.ditException):
        v.validate_outcome_length(x)
    # Unequal lengths
    x = ['1', '2', '33']
    with pytest.raises(dit.exceptions.ditException):
        v.validate_outcome_length(x)
