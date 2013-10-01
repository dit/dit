from nose.tools import *

import numpy as np
import numpy.testing as npt

from dit.exceptions import *
import dit

def test_mixture_distribution():
    d = dit.Distribution([.5, .5], ['A','B'])
    d2 = dit.Distribution([1, 0], ['A', 'B'])
    pmf = np.array([.75, .25])

    d3 = dit.mixture_distribution([d,d2], [.5, .5])
    npt.assert_allclose(pmf, d3.pmf)

def test_mixture_distribution_log():
    d = dit.Distribution([.5, .5], ['A','B'])
    d2 = dit.Distribution([1, 0], ['A', 'B'])
    d.set_base(2)
    d2.set_base(2)
    weights = np.log2(np.array([.5, .5]))
    pmf = np.log2(np.array([.75, .25]))

    d3 = dit.mixture_distribution([d,d2], weights)
    npt.assert_allclose(pmf, d3.pmf)

def test_mixture_distribution2():
    # Test when sample spaces are incompatible.
    d = dit.Distribution([.5, .5], ['A','B'])
    d2 = dit.Distribution([0, 1], ['A', 'B'], sort=True, trim=True)
    pmf = np.array([.25, .75])

    # Fails when it tries to get d2['A']
    assert_raises(InvalidOutcome, dit.mixture_distribution, [d,d2], [.5,.5])
    # Fails when it checks that all pmfs have the same length.
    assert_raises(ValueError, dit.mixture_distribution2, [d,d2], [.5,.5])

def test_mixture_distribution3():
    # Sample spaces are compatible.
    # But pmfs have a different order.
    d = dit.Distribution([.5, .5], ['A','B'])
    d2 = dit.Distribution([1, 0], ['B', 'A'], sort=False, trim=False, sparse=False)
    pmf = np.array([.25, .75])

    d3 = dit.mixture_distribution([d,d2], [.5, .5])
    assert_true(np.allclose(pmf, d3.pmf))
    d3 = dit.mixture_distribution2([d,d2], [.5, .5])
    print d3.pmf, d2.pmf, .5 * d.pmf + .5 * d2.pmf
    assert_false(np.allclose(pmf, d3.pmf))

def test_mixture_distribution4():
    # Sample spaces are compatible.
    # But pmfs have a different lengths and orders.
    d = dit.Distribution([.5, .5], ['A','B'])
    d2 = dit.Distribution([1, 0], ['B', 'A'], sort=False, trim=False, sparse=True)
    d2.make_sparse(trim=True)
    pmf = np.array([.25, .75])

    d3 = dit.mixture_distribution([d,d2], [.5, .5])
    assert_true(np.allclose(pmf, d3.pmf))
    assert_raises(ValueError, dit.mixture_distribution2, [d,d2], [.5,.5])
