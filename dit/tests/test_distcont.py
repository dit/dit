from nose.tools import *

import dit
import numpy as np
import numpy.testing as npt

def test_mixture_distribution():
    d = dit.Distribution([.5, .5], ['A','B'])
    d2 = dit.Distribution([1, 0], ['A', 'B'])
    d3 = dit.mixture_distribution([d,d2], [.5, .5])
    pmf = np.array([.75, .25])
    npt.assert_allclose(pmf, d3.pmf)

def test_mixture_distribution_log():
    d = dit.Distribution([.5, .5], ['A','B'])
    d2 = dit.Distribution([1, 0], ['A', 'B'])
    d.set_base(2)
    d2.set_base(2)
    weights = np.log2(np.array([.5, .5]))
    d3 = dit.mixture_distribution([d,d2], weights)
    pmf = np.log2(np.array([.75, .25]))
    npt.assert_allclose(pmf, d3.pmf)
