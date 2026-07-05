"""
Tests for dit.other.negentropy.
"""

import pytest

from dit import Distribution
from dit.example_dists import Xor, giant_bit, uniform
from dit.other import negentropy as N


@pytest.mark.parametrize("i", range(2, 10))
def test_negentropy1(i):
    """Test that uniform distributions have zero negentropy."""
    assert N(Distribution([str(_) for _ in range(i)], [1 / i] * i)) == pytest.approx(0)


@pytest.mark.parametrize("i", range(2, 10))
def test_negentropy2(i):
    """Test that a peaked distribution has negentropy equal to log2 of its size."""
    from numpy import log2

    d = Distribution([str(_) for _ in range(i)], [1] + [0] * (i - 1))
    d.make_dense()
    assert N(d) == pytest.approx(log2(i))


def test_negentropy3():
    """Test the negentropy of the Xor distribution."""
    assert N(Xor()) == pytest.approx(1)


def test_negentropy4():
    """Test that the negentropy is base-aware."""
    from numpy import log

    d = Xor()
    d.set_base("e")
    assert N(d) == pytest.approx(log(2))


@pytest.mark.parametrize("n", range(2, 8))
def test_negentropy5(n):
    """Test the negentropy of the giant bit distribution."""
    assert N(giant_bit(n, 2)) == pytest.approx(n - 1)


def test_negentropy6():
    """Test the negentropy over a subset of random variables."""
    d = Distribution(["00", "01", "11"], [1 / 3] * 3)
    # marginal over rv [0]: p(0)=2/3, p(1)=1/3, so negentropy = 1 - H(1/3)
    from dit.shannon import entropy as H

    assert N(d, [0]) == pytest.approx(1 - H(d.marginal([0])))
    # marginal over rv [1]: p(1)=2/3, p(0)=1/3
    assert N(d, [1]) == pytest.approx(1 - H(d.marginal([1])))


@pytest.mark.parametrize("n", range(2, 8))
def test_negentropy7(n):
    """Test that negentropy is non-negative."""
    assert N(uniform(n)) == pytest.approx(0)
