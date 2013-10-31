from __future__ import division

from nose.tools import *

import numpy as np
import numpy.testing as npt

from dit.exceptions import *
import dit

def test_mixture_distribution_weights():
    d = dit.Distribution(['A','B'], [.5, .5])
    d2 = dit.Distribution(['A', 'B'], [1, 0])

    assert_raises(ditException, dit.mixture_distribution, [d, d2], [1])
    assert_raises(ditException, dit.mixture_distribution2, [d, d2], [1])

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

def test_random_scalar_distribution():
    # Test with no alpha and only an integer
    pmf = np.array([0.297492727853, 0.702444212002, 6.30601451072e-05])
    for prng in [None, dit.math.prng]:
        dit.math.prng.seed(1)
        d = dit.random_scalar_distribution(3, prng=prng)
        assert_equal(d.outcomes, (0,1,2))
        np.testing.assert_allclose(d.pmf, pmf)

    # Test with outcomes specified
    dit.math.prng.seed(1)
    d = dit.random_scalar_distribution([0,1,2])
    assert_equal(d.outcomes, (0,1,2))
    np.testing.assert_allclose(d.pmf, pmf)

    # Test with concentration parameters
    pmf = np.array([0.34228708,  0.52696865,  0.13074428])
    dit.math.prng.seed(1)
    d = dit.random_scalar_distribution(3, alpha=[1,2,1])
    assert_equal(d.outcomes, (0,1,2))
    np.testing.assert_allclose(d.pmf, pmf)
    assert_raises(ditException, dit.random_scalar_distribution, 3, alpha=[1])

def test_random_distribution():
    # Test with no alpha
    pmf = np.array([2.48224944e-01, 5.86112396e-01, 5.26167518e-05, 1.65610043e-01])
    outcomes = ((0,0),(0,1),(1,0),(1,1))
    for prng in [None, dit.math.prng]:
        dit.math.prng.seed(1)
        d = dit.random_distribution(2, 2, prng=prng)
        assert_equal(d.outcomes, outcomes)
        np.testing.assert_allclose(d.pmf, pmf)

    # Test with a single alphabet specified
    dit.math.prng.seed(1)
    d = dit.random_distribution(2, [[0,1]])
    assert_equal(d.outcomes, outcomes)
    np.testing.assert_allclose(d.pmf, pmf)

    # Test with two alphabets specified
    dit.math.prng.seed(1)
    d = dit.random_distribution(2, [[0,1],[0,1]])
    assert_equal(d.outcomes, outcomes)
    np.testing.assert_allclose(d.pmf, pmf)

    # Test with invalid number of alphabets
    assert_raises(TypeError, dit.random_distribution, 3, [3,2])
    assert_raises(TypeError, dit.random_distribution, 3, [3,2,3])

    # Test with concentration parameters
    pmf = np.array([ 0.15092872,  0.23236257,  0.05765063,  0.55905808])
    dit.math.prng.seed(1)
    d = dit.random_distribution(2, 2, alpha=[1,2,1,3])
    assert_equal(d.outcomes, outcomes)
    np.testing.assert_allclose(d.pmf, pmf)
    assert_raises(ditException, dit.random_distribution, 2, 2, alpha=[1])

# These can be simple smoke test, since the random* tests hit all the branches.

def test_uniform_scalar_distribution():
    pmf = np.array([1/3] * 3)
    outcomes = (0,1,2)
    dit.math.prng.seed(1)
    d = dit.uniform_scalar_distribution(len(outcomes))
    assert_equal(d.outcomes, outcomes)
    np.testing.assert_allclose(d.pmf, pmf)

    dit.math.prng.seed(1)
    d = dit.uniform_scalar_distribution(outcomes)
    assert_equal(d.outcomes, outcomes)
    np.testing.assert_allclose(d.pmf, pmf)


def test_uniform_distribution():
    pmf = np.array([1/4] * 4)
    dit.math.prng.seed(1)
    d = dit.uniform_distribution(2, 2)
    assert_equal(d.outcomes, ((0,0),(0,1),(1,0),(1,1)))
    np.testing.assert_allclose(d.pmf, pmf)
