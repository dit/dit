#!/usr/bin/env python
# -*- coding: utf-8 -*-

from nose.tools import *

from six.moves import map, range, zip

from dit.npdist import Distribution
from dit.exceptions import *

import numpy as np

def test_init1():
    # Invalid initializations.
    assert_raises(InvalidDistribution, Distribution, [])
    assert_raises(InvalidDistribution, Distribution, [], [])
    Distribution([], [], sample_space=[(0,1)], validate=False)

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

def test_init2():
    # Cannot initialize with an iterator.
    # Must pass in a sequence for outcomes.
    outcomes = map(int, ['0','1','2','3','4'])
    pmf = [1/5] * 5
    assert_raises(TypeError, Distribution, outcomes, pmf)
