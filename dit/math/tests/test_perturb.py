# coding: utf-8

from __future__ import division
from __future__ import print_function

import dit
import numpy as np

from nose.tools import *

def test_perturb_pmf():
	# Smoke test
	d = np.array([0, .5, .5])
	d2 = dit.math.perturb_pmf(d, .00001)
	d3 = d2.round(2)
	assert_true(np.allclose(d, d3))

