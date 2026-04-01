"""
Tests for dit.example_dists.numeric.
"""

import numpy as np
import pytest

from dit.example_dists import bernoulli, binomial, hypergeometric, multinomial, uniform
from dit.shannon import entropy


def test_bernoulli1():
    """Test bernoulli distribution"""
    d = bernoulli(1 / 2)
    assert d.outcomes == (0, 1)
    assert sum(d.pmf) == pytest.approx(1)


@pytest.mark.parametrize("p", [i / 10 for i in range(0, 11)])
def test_bernoulli2(p):
    """Test bernoulli distribution"""
    d = bernoulli(p)
    assert d[0] == pytest.approx(1 - p)
    assert d[1] == pytest.approx(p)


@pytest.mark.parametrize("p", [-1, 1.5, "a", int, []])
def test_bernoulli3(p):
    """Test bernoulli distribution failures"""
    with pytest.raises(ValueError, match="is not a valid probability."):
        bernoulli(p)


@pytest.mark.parametrize("n", range(1, 10))
def test_binomial1(n):
    """Test binomial distribution"""
    d = binomial(n, 1 / 2)
    assert d.outcomes == tuple(range(n + 1))
    assert sum(d.pmf) == pytest.approx(1)


@pytest.mark.parametrize("n", [-1, 1.5, "a", int, []])
def test_binomial2(n):
    """Test binomial distribution failures"""
    with pytest.raises(ValueError, match="is not a positive integer"):
        binomial(n, 1 / 2)


def test_uniform1():
    """Test uniform distribution"""
    for n in range(2, 10):
        d = uniform(n)
        assert d.outcomes == tuple(range(n))
        assert d[0] == pytest.approx(1 / n)
        assert entropy(d) == pytest.approx(np.log2(n))


@pytest.mark.parametrize("v", [-1, 1.5, "a", int, []])
def test_uniform2(v):
    """Test uniform distribution failures"""
    with pytest.raises(ValueError, match="is not a"):
        uniform(v)


@pytest.mark.parametrize(("a", "b"), zip([1, 2, 3, 4, 5], [5, 7, 9, 11, 13], strict=True))
def test_uniform3(a, b):
    """Test uniform distribution construction"""
    d = uniform(a, b)
    assert len(d.outcomes) == b - a
    assert d[a] == pytest.approx(1 / (b - a))


@pytest.mark.parametrize(("a", "b"), [(2, 0), (0, [])])
def test_uniform4(a, b):
    """Test uniform distribution failures"""
    with pytest.raises(ValueError, match="is not an integer"):
        uniform(a, b)


@pytest.mark.parametrize("n", range(1, 6))
def test_multinomial_pmf_sums_to_one(n):
    """Test that the multinomial PMF sums to 1 for various n."""
    d = multinomial(n, [1 / 3, 1 / 3, 1 / 3])
    assert sum(d.pmf) == pytest.approx(1)


def test_multinomial_specific_values():
    """Test specific multinomial probabilities against known values."""
    d = multinomial(3, [0.5, 0.3, 0.2])
    # P(3, 0, 0) = 1 * 0.5^3 = 0.125
    assert d[(3, 0, 0)] == pytest.approx(0.125)
    # P(0, 3, 0) = 1 * 0.3^3 = 0.027
    assert d[(0, 3, 0)] == pytest.approx(0.027)
    # P(1, 1, 1) = 3! / (1!1!1!) * 0.5 * 0.3 * 0.2 = 6 * 0.03 = 0.18
    assert d[(1, 1, 1)] == pytest.approx(0.18)


def test_multinomial_agrees_with_binomial():
    """Test that multinomial with k=2 agrees with binomial."""
    n, p = 5, 0.4
    d_multi = multinomial(n, [p, 1 - p])
    d_binom = binomial(n, p)
    for k in range(n + 1):
        assert d_multi[(k, n - k)] == pytest.approx(d_binom[k])


def test_multinomial_n_zero():
    """Test multinomial with n=0 gives a single outcome."""
    d = multinomial(0, [0.5, 0.5])
    assert d.outcomes == ((0, 0),)
    assert d[(0, 0)] == pytest.approx(1.0)


def test_multinomial_deterministic():
    """Test multinomial with a deterministic category."""
    d = multinomial(4, [1.0, 0.0])
    assert d[(4, 0)] == pytest.approx(1.0)


@pytest.mark.parametrize("n", [-1, 1.5, "a", int, []])
def test_multinomial_invalid_n(n):
    """Test multinomial rejects invalid n."""
    with pytest.raises(ValueError, match="is not a positive integer"):
        multinomial(n, [0.5, 0.5])


@pytest.mark.parametrize("ps", [[-0.1, 1.1], [0.5, "a"], [0.5, int]])
def test_multinomial_invalid_ps(ps):
    """Test multinomial rejects invalid probabilities."""
    with pytest.raises(ValueError, match="is not a valid probability"):
        multinomial(2, ps)


def test_multinomial_ps_not_summing_to_one():
    """Test multinomial rejects ps that don't sum to 1."""
    with pytest.raises(ValueError, match="ps must sum to 1"):
        multinomial(2, [0.5, 0.3])


def test_multinomial_too_few_categories():
    """Test multinomial rejects fewer than 2 categories."""
    with pytest.raises(ValueError, match="ps must have at least 2 categories"):
        multinomial(2, [1.0])


def test_hypergeometric1():
    """Test hypergeometric distribution"""
    d = hypergeometric(50, 5, 10)
    assert d[4] == pytest.approx(0.003964583)
    assert d[5] == pytest.approx(0.0001189375)


@pytest.mark.parametrize(
    "vals",
    [
        (50, 5, -1),
        (50, -1, 10),
        (-1, 5, 10),
        (50, 5, 1.5),
        (50, 1.5, 10),
        (1.5, 5, 10),
        (50, 5, "a"),
        (50, "a", 10),
        ("a", 5, 10),
        (50, 5, int),
        (50, int, 10),
        (int, 5, 10),
        (50, 5, []),
        (50, [], 10),
        ([], 5, 10),
    ],
)
def test_hypergeometric2(vals):
    """Test hypergeometric distribution failures"""
    with pytest.raises(ValueError, match="is not a positive integer"):
        hypergeometric(*vals)
