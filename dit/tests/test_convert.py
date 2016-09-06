"""
Tests for dit.convert.
"""

from __future__ import division

import pytest

from dit import Distribution, ScalarDistribution
from dit.convert import DtoSD, SDtoD
from dit.exceptions import InvalidDistribution

def test_DtoSD1():
    outcomes = ['00', '01', '10', '11']
    pmf = [1/4]*4
    d = Distribution(outcomes, pmf)
    sd = DtoSD(d, False)
    assert type(d) is Distribution
    assert type(sd) is ScalarDistribution

def test_DtoSD2():
    outcomes = [(0,), (2,), (4,)]
    pmf = [1/3]*3
    d = Distribution(outcomes, pmf)
    sd = DtoSD(d, True)
    assert type(sd) is ScalarDistribution
    assert sd.outcomes == (0, 2, 4)

def test_SDtoD1():
    sd = ScalarDistribution([1/4]*4)
    d = SDtoD(sd)
    assert type(sd) is ScalarDistribution
    assert type(d) is Distribution

def test_SDtoD2():
    sd = ScalarDistribution([1])
    sd[0] = 0
    sd.make_sparse()
    with pytest.raises(InvalidDistribution):
        SDtoD(sd)

def test_SDtoD3():
    sd = ScalarDistribution([(0, 1), (2, 3), (4, 5)], [1/3]*3)
    d = SDtoD(sd)
    assert type(d) is Distribution
    assert d.outcome_length() == 2
