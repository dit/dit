from __future__ import division

from nose.tools import assert_is_instance

from dit import Distribution, ScalarDistribution
from dit.convert import DtoSD, SDtoD

def test_DtoSD():
    outcomes = ['00', '01', '10', '11']
    pmf = [1/4]*4
    d = Distribution(outcomes, pmf)
    sd = DtoSD(d, False)
    assert(type(d) is Distribution)
    assert(type(sd) is ScalarDistribution)

def test_SDtoD():
    sd = ScalarDistribution([1/4]*4)
    d = SDtoD(sd)
    assert(type(sd) is ScalarDistribution)
    assert(type(d) is Distribution)