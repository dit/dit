from nose.tools import assert_almost_equal, assert_raises

from dit import Distribution
from dit.exceptions import ditException
from dit.divergences import jensen_shannon_divergence as JSD

def test_jsd1():
	d1 = Distribution("AB", [0.5, 0.5])
	jsd = JSD([d1, d1])
	assert_almost_equal(jsd, 0)

def test_jsd2():
	d1 = Distribution("AB", [0.5, 0.5])
	d2 = Distribution("BC", [0.5, 0.5])
	jsd = JSD([d1, d2])
	assert_almost_equal(jsd, 0.5)

def test_jsd3():
	d1 = Distribution("AB", [0.5, 0.5])
	d2 = Distribution("CD", [0.5, 0.5])
	jsd = JSD([d1, d2])
	assert_almost_equal(jsd, 1.0)

def test_jsd4():
	d1 = Distribution("AB", [0.5, 0.5])
	d2 = Distribution("BC", [0.5, 0.5])
	jsd = JSD([d1, d2], [0.25, 0.75])
	assert_almost_equal(jsd, 0.40563906222956625)

def test_jsd5():
	d1 = Distribution("AB", [0.5, 0.5])
	d2 = Distribution("BC", [0.5, 0.5])
	assert_raises(ditException, JSD, [d1, d2], [0.1, 0.6, 0.3])
