from nose.tools import assert_almost_equal, assert_raises

from dit.exceptions import ditException
import dit

def test_jsd1():
	d1 = dit.Distribution("AB", [0.5, 0.5])
	jsd = dit.algorithms.jensen_shannon_divergence([d1, d1])
	assert_almost_equal(jsd, 0)

def test_jsd2():
	d1 = dit.Distribution("AB", [0.5, 0.5])
	d2 = dit.Distribution("BC", [0.5, 0.5])
	jsd = dit.algorithms.jensen_shannon_divergence([d1, d2])
	assert_almost_equal(jsd, 0.5)

def test_jsd3():
	d1 = dit.Distribution("AB", [0.5, 0.5])
	d2 = dit.Distribution("CD", [0.5, 0.5])
	jsd = dit.algorithms.jensen_shannon_divergence([d1, d2])
	assert_almost_equal(jsd, 1.0)

def test_jsd4():
	d1 = dit.Distribution("AB", [0.5, 0.5])
	d2 = dit.Distribution("BC", [0.5, 0.5])
	jsd = dit.algorithms.jensen_shannon_divergence([d1, d2], [0.25, 0.75])
	assert_almost_equal(jsd, 0.40563906222956625)

def test_jsd5():
	d1 = dit.Distribution("AB", [0.5, 0.5])
	d2 = dit.Distribution("BC", [0.5, 0.5])
	assert_raises(ditException, dit.algorithms.jensen_shannon_divergence, [d1, d2], [0.1, 0.6, 0.3])
