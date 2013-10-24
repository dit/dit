from __future__ import division

from nose.tools import assert_almost_equal, assert_in, assert_raises, \
                       assert_true

from dit import Distribution
from dit.distribution import BaseDistribution
from dit.exceptions import ditException, InvalidNormalization

def test_dist_iter1():
    outcomes = ['00', '01', '10', '11']
    pmf = [1/4]*4
    d = Distribution(outcomes, pmf)
    for o in d:
        assert_in(o, outcomes)


def test_dist_iter2():
    outcomes = ['00', '01', '10', '11']
    pmf = [1/4]*4
    d = Distribution(outcomes, pmf)
    for o in reversed(d):
        assert_in(o, outcomes)


def test_numerical():
    outcomes = ['00', '01', '10', '11']
    pmf = [1/4]*4
    d = Distribution(outcomes, pmf)
    assert_true(d.is_numerical())


def test_rand():
    outcomes = ['00', '01', '10', '11']
    pmf = [1/4]*4
    d = Distribution(outcomes, pmf)
    for _ in range(10):
        assert_in(d.rand(), outcomes)


def test_to_dict():
    outcomes = ['00', '01', '10', '11']
    pmf = [1/4]*4
    d = Distribution(outcomes, pmf)
    dd = d.to_dict()
    for o, p in dd.items():
        assert_almost_equal(d[o], p)

def test_validate1():
    outcomes = ['00', '01', '10', '11']
    pmf = [1/4]*4
    d = Distribution(outcomes, pmf)
    assert_true(d.validate())
    assert_true(BaseDistribution.validate(d))

def test_validate2():
    outcomes = ['00', '01', '10', '11']
    pmf = [1/4]*4
    d = Distribution(outcomes, pmf)
    d['00'] = 0
    assert_raises(InvalidNormalization, d.validate)
    assert_raises(InvalidNormalization, BaseDistribution.validate, d)

def test_zipped1():
    outcomes = ['00', '01', '10', '11']
    pmf = [1/4]*4
    d = Distribution(outcomes, pmf)
    zipped = d.zipped(mode='pants')
    assert_raises(ditException, list, zipped)