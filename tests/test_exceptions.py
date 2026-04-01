"""
Tests for dit.exceptions.

Verifies that __str__ on each exception subclass returns a proper string,
not the exception object itself (regression test for GitHub issue #201).
"""

import numpy as np
import pytest

from dit.exceptions import (
    IncompatibleDistribution,
    InvalidBase,
    InvalidDistribution,
    InvalidNormalization,
    InvalidOutcome,
    InvalidProbability,
    OptimizationException,
    ditException,
)
from dit.math import LinearOperations


class TestDitException:
    def test_str_with_message(self):
        e = ditException("something went wrong")
        assert str(e) == "something went wrong"
        assert isinstance(str(e), str)

    def test_str_empty(self):
        e = ditException()
        assert str(e) == ""

    def test_str_with_kwarg_msg(self):
        e = ditException(msg="custom message")
        assert str(e) == "custom message"

    def test_repr(self):
        e = ditException("oops")
        assert "ditException" in repr(e)


class TestIncompatibleDistribution:
    def test_str(self):
        e = IncompatibleDistribution()
        result = str(e)
        assert isinstance(result, str)
        assert "not compatible" in result

    def test_str_with_extra_args(self):
        e = IncompatibleDistribution("extra detail")
        result = str(e)
        assert isinstance(result, str)
        assert "not compatible" in result


class TestInvalidBase:
    def test_str(self):
        e = InvalidBase(-1)
        result = str(e)
        assert isinstance(result, str)
        assert "-1" in result
        assert "logarithm base" in result

    def test_str_no_args(self):
        e = InvalidBase()
        result = str(e)
        assert isinstance(result, str)


class TestInvalidDistribution:
    def test_str(self):
        e = InvalidDistribution("bad dist")
        result = str(e)
        assert isinstance(result, str)
        assert result == "bad dist"


class TestInvalidOutcome:
    def test_str_single(self):
        e = InvalidOutcome("X")
        result = str(e)
        assert isinstance(result, str)
        assert "not in the sample space" in result

    def test_str_multiple(self):
        e = InvalidOutcome(["X", "Y"], single=False)
        result = str(e)
        assert isinstance(result, str)
        assert "not in the sample space" in result

    def test_str_custom_msg(self):
        e = InvalidOutcome(msg="custom")
        result = str(e)
        assert isinstance(result, str)
        assert result == "custom"


class TestInvalidNormalization:
    def test_str(self):
        e = InvalidNormalization(1.5)
        result = str(e)
        assert isinstance(result, str)
        assert "Bad normalization" in result
        assert "1.5" in result

    def test_summation_attribute(self):
        e = InvalidNormalization(0.99)
        assert e.summation == 0.99

    def test_raised_and_caught(self):
        with pytest.raises(InvalidNormalization, match="Bad normalization"):
            raise InvalidNormalization(0.5)


class TestInvalidProbability:
    def test_str_single(self):
        ops = LinearOperations()
        e = InvalidProbability(np.array([-0.1]), ops=ops)
        result = str(e)
        assert isinstance(result, str)
        assert "Probability" in result

    def test_str_multiple(self):
        ops = LinearOperations()
        e = InvalidProbability(np.array([-0.1, 1.5]), ops=ops)
        result = str(e)
        assert isinstance(result, str)
        assert "Probabilities" in result


class TestOptimizationException:
    def test_str(self):
        e = OptimizationException("optimizer failed")
        result = str(e)
        assert isinstance(result, str)
        assert result == "optimizer failed"
