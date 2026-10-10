"""
Regression tests for core Distribution / math fixes.
"""

import numpy as np
import pytest

import dit
from dit.cdisthelpers import joint_from_factors
from dit.distconst import _combine_scalar_dists
from dit.helpers import copypmf
from dit.math import LinearOperations, LogOperations
from dit.math.pmfops import replace_zeros
from dit.math.sampling import _samples_discrete__python
from dit.validate import validate_probabilities

UNIFORM = dit.Distribution(["0", "1"], [0.5, 0.5])


def test_scalar_mul_log_base():
    lin = UNIFORM * 0.5
    log = UNIFORM.copy(base=2) * 0.5
    assert np.allclose(log.copy(base="linear").pmf, lin.pmf)


def test_scalar_div_log_base():
    lin = UNIFORM / 2
    log = UNIFORM.copy(base=2) / 2
    assert np.allclose(log.copy(base="linear").pmf, lin.pmf)


def test_add_log_base():
    lin = UNIFORM + UNIFORM
    log = UNIFORM.copy(base=2) + UNIFORM.copy(base=2)
    assert np.allclose(log.copy(base="linear").pmf, lin.pmf)


def test_add_keeps_disjoint_outcomes():
    p = dit.Distribution(["00", "11"], [0.5, 0.5])
    q = dit.Distribution(["00", "22"], [0.5, 0.5])
    total = (p + q).copy(base="linear")
    assert np.isclose(np.sum(total.pmf), 2.0)


def test_matmul_log_base():
    a = UNIFORM.copy(base=2)
    prod = a @ a
    assert prod.get_base() == 2
    assert np.allclose(prod.copy(base="linear").pmf, [0.25] * 4)


def test_combine_scalar_dists_log_base():
    s = dit.Distribution([0, 1], [0.5, 0.5]).copy(base=2)
    d = _combine_scalar_dists(s, s, lambda a, b: a + b)
    assert np.allclose(d.copy(base="linear").pmf, [0.25, 0.5, 0.25])


def test_is_approx_equal_checks_alphabet_values():
    a = dit.Distribution(["00", "11"], [0.5, 0.5], rv_names=["X", "Y"])
    b = dit.Distribution(["aa", "bb"], [0.5, 0.5], rv_names=["X", "Y"])
    assert not a.is_approx_equal(b)


def test_is_approx_equal_across_bases():
    a = dit.Distribution(["00", "11"], [0.25, 0.75])
    assert a.is_approx_equal(a.copy(base=2))


def test_normalize_axis():
    x = np.array([[1.0, 3.0], [1.0, 1.0]])
    assert np.allclose(LinearOperations().normalize(x, axis=-1), [[0.25, 0.75], [0.5, 0.5]])
    ops = LogOperations(2)
    z = ops.normalize(ops.log(x), axis=-1)
    assert np.allclose(ops.exp(z), [[0.25, 0.75], [0.5, 0.5]])


def test_replace_zeros_rows():
    pmf = np.array([[0.5, 0.5, 0.0], [0.25, 0.25, 0.5]])
    out = replace_zeros(pmf, delta=0.1, rand=False)
    assert np.allclose(out.sum(axis=1), 1.0)
    assert np.allclose(out[1], pmf[1])


def test_python_sampler_rounding():
    pmf = np.array([0.3, 0.3, 0.4 - 1e-12])
    assert list(_samples_discrete__python(pmf, np.array([0.1, 0.9999999999999]))) == [0, 2]


def test_joint_from_factors_numeric_scalar():
    d = dit.Distribution([(0, "a"), (1, "b")], [0.5, 0.5])
    m, cds = d.condition_on([0])
    assert joint_from_factors(m, cds).is_approx_equal(d)


def test_validate_probabilities_base_below_one():
    assert validate_probabilities(np.array([-1e-15]), LogOperations(0.5))


def test_copypmf_bad_mode():
    with pytest.raises(ValueError, match="Invalid mode"):
        copypmf(UNIFORM, mode="bogus")


def test_len_symbolic():
    sympy = pytest.importorskip("sympy")
    from dit.symbolic import symbolic_distribution

    p = sympy.Symbol("p", positive=True)
    assert len(symbolic_distribution(["0", "1"], [p, 1 - p])) == 2
