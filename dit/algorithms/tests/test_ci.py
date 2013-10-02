from __future__ import division

from nose.tools import *

from dit import Distribution
from dit.algorithms import common_information as K

def test_K1():
	outcomes = ['00', '11']
	pmf = [1/2, 1/2]
	d = Distribution(pmf, outcomes)
	assert_almost_equal(K(d, [[0],[1]]), 1.0)

def test_K2():
	outcomes = ['00', '01', '11']
	pmf = [1/3, 1/3, 1/3]
	d = Distribution(pmf, outcomes)
	assert_almost_equal(K(d, [[0],[1]]), 0.0)

def test_K3():
	outcomes = ['00', '01', '10', '11', '22', '33']
	pmf = [1/8, 1/8, 1/8, 1/8, 1/4, 1/4]
	d = Distribution(pmf, outcomes)
	assert_almost_equal(K(d, [[0],[1]]), 1.5)

def test_K4():
	outcomes = ['000', '010', '100', '110', '221', '331']
	pmf = [1/8, 1/8, 1/8, 1/8, 1/4, 1/4]
	d = Distribution(pmf, outcomes)
	assert_almost_equal(K(d, [[0],[1]]), 1.5)
	assert_almost_equal(K(d, [[0],[1],[2]]), 1.0)