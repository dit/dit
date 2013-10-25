#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division

from nose.tools import *
from numpy.testing import assert_array_almost_equal

from six.moves import map, range, zip

from dit.npdist import Distribution, ScalarDistribution, _make_distribution
from dit.exceptions import *

import numpy as np

def test_init1():
    # Invalid initializations.
    assert_raises(InvalidDistribution, Distribution, [])
    assert_raises(InvalidDistribution, Distribution, [], [])
    Distribution([], [], sample_space=[(0,1)], validate=False)

def test_init2():
    # Cannot initialize with an iterator.
    # Must pass in a sequence for outcomes.
    outcomes = map(int, ['0','1','2','3','4'])
    pmf = [1/5] * 5
    assert_raises(TypeError, Distribution, outcomes, pmf)

def test_init3():
    dist = {'0': 1/2, '1': 1/2}
    d = Distribution(dist)
    assert_equal(d.outcomes, ('0', '1'))
    assert_array_almost_equal(d.pmf, [1/2, 1/2])

def test_init4():
    dist = {'0': 1/2, '1': 1/2}
    pmf = [1/2, 1/2]
    assert_raises(InvalidDistribution, Distribution, dist, pmf)

def test_init5():
    outcomes = ['0', '1', '2']
    pmf = [1/2, 1/2]
    assert_raises(InvalidDistribution, Distribution, outcomes, pmf)

def test_init6():
    outcomes = set(['0', '1', '2'])
    pmf = [1/3]*3
    assert_raises(ditException, Distribution, outcomes, pmf)

def test_init7():
    outcomes = ['0', '1']
    pmf = [1/2, 1/2]
    d1 = Distribution(outcomes, pmf)
    d2 = Distribution.from_distribution(d1)
    assert_true(d1.is_approx_equal(d2))

def test_init8():
    outcomes = [(0,), (1,)]
    pmf = [1/2, 1/2]
    d1 = ScalarDistribution(pmf)
    d2 = Distribution.from_distribution(d1)
    d3 = Distribution(outcomes, pmf)
    assert_true(d2.is_approx_equal(d3))

def test_init9():
    outcomes = [(0,), (1,)]
    pmf = [1/2, 1/2]
    d1 = ScalarDistribution(pmf)
    d2 = Distribution.from_distribution(d1, base=10)
    d3 = Distribution(outcomes, pmf)
    d3.set_base(10)
    assert_true(d2.is_approx_equal(d3))

def test_atoms():
    pmf = [.125, .125, .125, .125, .25, 0, .25]
    outcomes = ['000', '011', '101', '110', '222', '321', '333']
    d = Distribution(outcomes, pmf)

    atoms = outcomes
    assert_equal( list(d.atoms()), atoms)

    patoms = ['000', '011', '101', '110', '222', '333']
    assert_equal( list(d.atoms(patoms=True)), patoms)

    d = Distribution(outcomes, pmf, sample_space=outcomes + ['444'])
    atoms = outcomes + ['444']
    assert_equal( list(d.atoms()), atoms)

def test_zipped():
    pmf = [.125, .125, .125, .125, .25, 0, .25]
    outcomes = ['000', '011', '101', '110', '222', '321', '333']
    d = Distribution(outcomes, pmf)

    outcomes_, pmf_ = list(zip(*d.zipped()))
    d2 = Distribution(outcomes_, pmf_)
    assert_true(d.is_approx_equal(d2))

    outcomes_, pmf_ = list(zip(*d.zipped(mode='atoms')))
    d3 = Distribution(outcomes_, pmf_)
    assert_true(d.is_approx_equal(d3))

    outcomes_, pmf_ = list(zip(*d.zipped(mode='patoms')))
    d4 = Distribution(outcomes_, pmf_)
    d.make_sparse()
    np.testing.assert_allclose(d.pmf, d4.pmf)

def test_make_distribution():
    outcomes = ['0', '1']
    pmf = [1/2, 1/2]
    d = _make_distribution(outcomes, pmf, None)
    assert_true(type(d) is Distribution)
    assert_equal(d.outcomes, ('0', '1'))

def test_setitem1():
    d = Distribution(['0', '1'], [1/2, 1/2])
    assert_raises(InvalidOutcome, d.__setitem__, '2', 0)

def test_setitem2():
    d = Distribution(['00', '11'], [1, 0])
    d.make_sparse()
    d['11'] = 1/2
    d.normalize()
    assert('11' in d)
    assert_almost_equal(d['11'], 1/3)

def test_coalesce():
    outcomes = ['000', '011', '101', '110']
    pmf = [1/4]*4
    d = Distribution(outcomes, pmf)
    d = d.coalesce([[0, 1], [2]])
    assert_equal(d.outcome_length(), 2)

def test_copy():
    outcomes = ['0', '1']
    pmf = [1/2, 1/2]
    d1 = Distribution(outcomes, pmf)
    d2 = d1.copy(base=10)
    d3 = Distribution(outcomes, pmf)
    d3.set_base(10)
    assert_true(d2.is_approx_equal(d3))

def test_outcome_length():
    outcomes = ['000', '011', '101', '110']
    pmf = [1/4]*4
    d = Distribution(outcomes, pmf)
    d = d.marginal([0, 2])
    assert_equal(d.outcome_length(), 2)
    assert_equal(d.outcome_length(masked=True), 3)