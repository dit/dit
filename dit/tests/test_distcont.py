from nose.tools import *

import numpy as np
import numpy.testing as npt

from dit.exceptions import *
import dit

def test_mixture_distribution():
    d = dit.Distribution(['A','B'], [.5, .5])
    d2 = dit.Distribution(['A', 'B'], [1, 0])
    pmf = np.array([.75, .25])

    d3 = dit.mixture_distribution([d,d2], [.5, .5])
    npt.assert_allclose(pmf, d3.pmf)

def test_mixture_distribution_log():
    d = dit.Distribution(['A', 'B'], [.5, .5])
    d2 = dit.Distribution(['A', 'B'], [1, 0])
    d.set_base(2)
    d2.set_base(2)
    weights = np.log2(np.array([.5, .5]))
    pmf = np.log2(np.array([.75, .25]))

    d3 = dit.mixture_distribution([d,d2], weights)
    npt.assert_allclose(pmf, d3.pmf)

def test_mixture_distribution2():
    # Test when sample spaces are incompatible.
    d = dit.Distribution(['A','B'], [.5, .5])
    d2 = dit.Distribution(['A', 'B'], [1, 0], sort=True, trim=True)
    pmf = np.array([.25, .75])

    # Fails when it tries to get d2['A']
    assert_raises(InvalidOutcome, dit.mixture_distribution, [d,d2], [.5,.5])
    # Fails when it checks that all pmfs have the same length.
    assert_raises(ValueError, dit.mixture_distribution2, [d,d2], [.5,.5])

def test_mixture_distribution3():
    # Sample spaces are compatible.
    # But pmfs have a different order.
    d = dit.Distribution(['A','B'], [.5, .5])
    d2 = dit.Distribution(['B', 'A'], [1, 0], sort=False, trim=False, sparse=False)
    pmf = np.array([.25, .75])

    d3 = dit.mixture_distribution([d,d2], [.5, .5])
    assert_true(np.allclose(pmf, d3.pmf))
    d3 = dit.mixture_distribution2([d,d2], [.5, .5])
    assert_false(np.allclose(pmf, d3.pmf))

def test_mixture_distribution4():
    # Sample spaces are compatible.
    # But pmfs have a different lengths and orders.
    d = dit.Distribution(['A','B'], [.5, .5])
    d2 = dit.Distribution(['B', 'A'], [1, 0], sort=False, trim=False, sparse=True)
    d2.make_sparse(trim=True)
    pmf = np.array([.25, .75])

    d3 = dit.mixture_distribution([d,d2], [.5, .5])
    assert_true(np.allclose(pmf, d3.pmf))
    assert_raises(ValueError, dit.mixture_distribution2, [d,d2], [.5,.5])

def test_mixture_distribution5():
    # Incompatible sample spaces.
    d1 = dit.Distribution(['A', 'B'], [.5, .5])
    d2 = dit.Distribution(['B', 'C'], [.5, .5])
    d3 = dit.mixture_distribution([d1, d2], [.5, .5], merge=True)
    pmf = np.array([.25, .5, .25])
    assert_true(np.allclose(pmf, d3.pmf))
