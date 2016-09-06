"""
Tests for dit.other.perplexity.
"""

from __future__ import division

import pytest

from dit import (ScalarDistribution as SD,
                 Distribution as D)
from dit.other import perplexity as P
from six.moves import range # pylint: disable=redefined-builtin


@pytest.mark.parametrize('i', range(2, 10))
def test_p1(i):
    """ Test some simple base cases using SD """
    assert P(SD([1/i]*i)) == pytest.approx(i)

@pytest.mark.parametrize('i', range(2, 10))
def test_p2(i):
    """ Test some simple base cases using SD with varying bases """
    d = SD([1/i]*i)
    d.set_base(i)
    assert P(d) == pytest.approx(i)

@pytest.mark.parametrize('i', range(2, 10))
def test_p3(i):
    """ Test some simple base cases using D """
    d = D([str(_) for _ in range(i)], [1/i]*i)
    assert P(d) == pytest.approx(i)

@pytest.mark.parametrize('i', range(2, 10))
def test_p4(i):
    """ Test some simple base cases using D with varying bases """
    d = D([str(_) for _ in range(i)], [1/i]*i)
    d.set_base(i)
    assert P(d) == pytest.approx(i)

def test_p5():
    """ Test some joint, marginal, and conditional perplexities """
    d = D(['00', '01', '10', '11'], [1/4]*4)
    assert P(d) == pytest.approx(4)
    assert P(d, [0]) == pytest.approx(2)
    assert P(d, [1]) == pytest.approx(2)
    assert P(d, [0], [1]) == pytest.approx(2)
    assert P(d, [1], [0]) == pytest.approx(2)

def test_p6():
    """ Test some joint and conditional perplexities """
    d = D(['00', '11'], [1/2]*2)
    assert P(d) == pytest.approx(2)
    assert P(d, [0], [1]) == pytest.approx(1)
    assert P(d, [1], [0]) == pytest.approx(1)
