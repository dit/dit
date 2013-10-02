#!/usr/bin/env python
# -*- coding: utf-8 -*-

from nose.tools import *

from dit.npdist import Distribution
from dit.exceptions import *

import numpy as np

def test_init1():
    # Invalid initializations.
    assert_raises(InvalidDistribution, Distribution, [])
    assert_raises(InvalidDistribution, Distribution, [], [])
    assert_raises(InvalidDistribution, Distribution, [], [], alphabet=[(0,1)], validate=False)

def test_atoms():
    pmf = [.125, .125, .125, .125, .25, 0, .25]
    outcomes = ['000', '011', '101', '110', '222', '321', '333']
    d = Distribution(pmf, outcomes)
    atoms = [
        '000', '001', '002', '003', '010', '011', '012', '013', '020', '021',
        '022', '023', '030', '031', '032', '033', '100', '101', '102', '103',
        '110', '111', '112', '113', '120', '121', '122', '123', '130', '131',
        '132', '133', '200', '201', '202', '203', '210', '211', '212', '213',
        '220', '221', '222', '223', '230', '231', '232', '233', '300', '301',
        '302', '303', '310', '311', '312', '313', '320', '321', '322', '323',
        '330', '331', '332', '333'
    ]
    assert_equal( list(d.atoms()), atoms)
    patoms = ['000', '011', '101', '110', '222', '333']
    assert_equal( list(d.atoms(patoms=True)), patoms)

def test_zipped():
    pmf = [.125, .125, .125, .125, .25, 0, .25]
    outcomes = ['000', '011', '101', '110', '222', '321', '333']
    d = Distribution(pmf, outcomes)

    outcomes_, pmf_ = zip(*d.zipped())
    d2 = Distribution(pmf_, outcomes_)
    assert_true(d.is_approx_equal(d2))

    outcomes_, pmf_ = zip(*d.zipped(mode='atoms'))
    d3 = Distribution(pmf_, outcomes_)
    assert_true(d.is_approx_equal(d3))

    outcomes_, pmf_ = zip(*d.zipped(mode='patoms'))
    d4 = Distribution(pmf_, outcomes_)
    d.make_sparse()
    np.testing.assert_allclose(d.pmf, d4.pmf)

