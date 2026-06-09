"""
Tests for dit.helpers.
"""

import numpy as np
import pytest

from dit import Distribution
from dit.exceptions import InvalidOutcome, ditException
from dit.helpers import construct_alphabets, copypmf, numerical_test, parse_rvs, reorder


def _dist_with_null():
    d = Distribution(["0", "1", "2"], [0.5, 0.5, 0.0])
    d.make_dense()
    return d


def test_copypmf_sparse_drops_nulls():
    assert np.allclose(copypmf(_dist_with_null(), mode="sparse"), [0.5, 0.5])


def test_copypmf_dense_keeps_nulls():
    assert np.allclose(copypmf(_dist_with_null(), mode="dense"), [0.5, 0.5, 0.0])


def test_copypmf_log_to_log_base_change():
    d = _dist_with_null()
    d.set_base(2)
    converted = copypmf(d, base="e")
    assert converted[0] == pytest.approx(np.log(0.5))


def test_copypmf_log_to_linear():
    d = _dist_with_null()
    d.set_base(2)
    assert np.allclose(copypmf(d, base="linear"), [0.5, 0.5, 0.0])


def test_copypmf_linear_to_log():
    converted = copypmf(_dist_with_null(), base=2)
    assert converted[0] == pytest.approx(-1.0)
    assert np.isneginf(converted[2])


def test_construct_alphabets1():
    outcomes = ["00", "01", "10", "11"]
    alphas = construct_alphabets(outcomes)
    assert alphas == (("0", "1"), ("0", "1"))


def test_construct_alphabets2():
    outcomes = 3
    with pytest.raises(TypeError):
        construct_alphabets(outcomes)


def test_construct_alphabets3():
    outcomes = [0, 1, 2]
    with pytest.raises(ditException):
        construct_alphabets(outcomes)


def test_construct_alphabets4():
    outcomes = ["0", "1", "01"]
    with pytest.raises(ditException):
        construct_alphabets(outcomes)


def test_parse_rvs1():
    outcomes = ["00", "11"]
    pmf = [1 / 2] * 2
    d = Distribution(outcomes, pmf)
    with pytest.raises(ditException):
        parse_rvs(d, [0, 0, 1])


def test_parse_rvs2():
    outcomes = ["00", "11"]
    pmf = [1 / 2] * 2
    d = Distribution(outcomes, pmf)
    d.set_rv_names("XY")
    with pytest.raises(ditException):
        parse_rvs(d, ["X", "Y", "Z"])


def test_parse_rvs3():
    outcomes = ["00", "11"]
    pmf = [1 / 2] * 2
    d = Distribution(outcomes, pmf)
    with pytest.raises(ditException):
        parse_rvs(d, [0, 1, 2])


def test_reorder1():
    outcomes = ["00", "11", "01"]
    pmf = [1 / 3] * 3
    sample_space = ("00", "01", "10", "11")
    new = reorder(outcomes, pmf, sample_space)
    assert new[0] == ["00", "01", "11"]


def test_reorder2():
    outcomes = ["00", "11", "22"]
    pmf = [1 / 3] * 3
    sample_space = ("00", "01", "10", "11")
    with pytest.raises(InvalidOutcome):
        reorder(outcomes, pmf, sample_space)


def test__numerical_test1():
    """test _numerical_test on a good distribution"""
    d = Distribution([(0, 0), (1, 0), (2, 1), (3, 1)], [1 / 8, 1 / 8, 3 / 8, 3 / 8])
    assert numerical_test(d) is None


def test__numerical_test2():
    """Test _numerical_test on a bad distribution"""
    # A bad distribution is one with a non-numerical alphabet
    d = Distribution([(0, "0"), (1, "0"), (2, "1"), (3, "1")], [1 / 8, 1 / 8, 3 / 8, 3 / 8])
    with pytest.raises(TypeError):
        numerical_test(d)
