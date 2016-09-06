"""
Tests for dit.distconst.
"""

from __future__ import division

import pytest

import itertools

from six.moves import range

import numpy as np

from dit.exceptions import ditException, InvalidOutcome
import dit

def test_mixture_distribution_weights():
    d = dit.Distribution(['A', 'B'], [0.5, 0.5])
    d2 = dit.Distribution(['A', 'B'], [1, 0])

    with pytest.raises(ditException):
        dit.mixture_distribution([d, d2], [1])
    with pytest.raises(ditException):
        dit.mixture_distribution2([d, d2], [1])

def test_mixture_distribution():
    d = dit.Distribution(['A', 'B'], [0.5, 0.5])
    d2 = dit.Distribution(['A', 'B'], [1, 0])
    pmf = np.array([0.75, 0.25])

    d3 = dit.mixture_distribution([d, d2], [0.5, 0.5])
    assert np.allclose(pmf, d3.pmf)

def test_mixture_distribution_log():
    d = dit.Distribution(['A', 'B'], [0.5, 0.5])
    d2 = dit.Distribution(['A', 'B'], [1, 0])
    d.set_base(2)
    d2.set_base(2)
    weights = np.log2(np.array([0.5, 0.5]))
    pmf = np.log2(np.array([0.75, 0.25]))

    d3 = dit.mixture_distribution([d, d2], weights)
    assert np.allclose(pmf, d3.pmf)

def test_mixture_distribution2():
    # Test when pmfs are different lengths.
    d = dit.Distribution(['A', 'B'], [0.5, 0.5])
    d2 = dit.Distribution(['A', 'B'], [1, 0], sort=True, trim=True)

    # Fails when it checks that all pmfs have the same length.
    with pytest.raises(ValueError):
        dit.mixture_distribution2([d, d2], [0.5, 0.5])

def test_mixture_distribution3():
    # Sample spaces are compatible.
    # But pmfs have a different order.
    d = dit.Distribution(['A', 'B'], [0.5, 0.5])
    d2 = dit.Distribution(['B', 'A'], [1, 0], sort=False, trim=False, sparse=False)
    pmf = np.array([0.25, 0.75])

    d3 = dit.mixture_distribution([d, d2], [0.5, 0.5])
    assert np.allclose(pmf, d3.pmf)
    d3 = dit.mixture_distribution2([d, d2], [0.5, 0.5])
    assert not np.allclose(pmf, d3.pmf)

def test_mixture_distribution4():
    # Sample spaces are compatible.
    # But pmfs have a different lengths and orders.
    d = dit.Distribution(['A', 'B'], [0.5, 0.5])
    d2 = dit.Distribution(['B', 'A'], [1, 0], sort=False, trim=False, sparse=True)
    d2.make_sparse(trim=True)
    pmf = np.array([0.25, 0.75])

    d3 = dit.mixture_distribution([d, d2], [0.5, 0.5])
    assert np.allclose(pmf, d3.pmf)
    with pytest.raises(ValueError):
        dit.mixture_distribution2([d, d2], [0.5, 0.5])

def test_mixture_distribution5():
    # Incompatible sample spaces.
    d1 = dit.Distribution(['A', 'B'], [0.5, 0.5])
    d2 = dit.Distribution(['B', 'C'], [0.5, 0.5])
    d3 = dit.mixture_distribution([d1, d2], [0.5, 0.5], merge=True)
    pmf = np.array([0.25, 0.5, 0.25])
    assert np.allclose(pmf, d3.pmf)

def test_random_scalar_distribution():
    # Test with no alpha and only an integer
    pmf = np.array([0.297492727853, 0.702444212002, 6.30601451072e-05])
    for prng in [None, dit.math.prng]:
        dit.math.prng.seed(1)
        d = dit.random_scalar_distribution(3, prng=prng)
        assert d.outcomes == (0, 1, 2)
        assert np.allclose(d.pmf, pmf)

    # Test with outcomes specified
    dit.math.prng.seed(1)
    d = dit.random_scalar_distribution([0, 1, 2])
    assert d.outcomes == (0, 1, 2)
    assert np.allclose(d.pmf, pmf)

    # Test with concentration parameters
    pmf = np.array([0.34228708, 0.52696865, 0.13074428])
    dit.math.prng.seed(1)
    d = dit.random_scalar_distribution(3, alpha=[1, 2, 1])
    assert d.outcomes == (0, 1, 2)
    assert np.allclose(d.pmf, pmf)
    with pytest.raises(ditException):
        dit.random_scalar_distribution(3, alpha=[1])

def test_random_distribution():
    # Test with no alpha
    pmf = np.array([2.48224944e-01, 5.86112396e-01, 5.26167518e-05, 1.65610043e-01])
    outcomes = ((0, 0), (0, 1), (1, 0), (1, 1))
    for prng in [None, dit.math.prng]:
        dit.math.prng.seed(1)
        d = dit.random_distribution(2, 2, prng=prng)
        assert d.outcomes == outcomes
        assert np.allclose(d.pmf, pmf)

    # Test with a single alphabet specified
    dit.math.prng.seed(1)
    d = dit.random_distribution(2, [[0, 1]])
    assert d.outcomes == outcomes
    assert np.allclose(d.pmf, pmf)

    # Test with two alphabets specified
    dit.math.prng.seed(1)
    d = dit.random_distribution(2, [[0, 1], [0, 1]])
    assert d.outcomes == outcomes
    assert np.allclose(d.pmf, pmf)

    # Test with invalid number of alphabets
    with pytest.raises(TypeError):
        dit.random_distribution(3, [3, 2])
    with pytest.raises(TypeError):
        dit.random_distribution(3, [3, 2, 3])

    # Test with concentration parameters
    pmf = np.array([0.15092872, 0.23236257, 0.05765063, 0.55905808])
    dit.math.prng.seed(1)
    d = dit.random_distribution(2, 2, alpha=[1, 2, 1, 3])
    assert d.outcomes == outcomes
    assert np.allclose(d.pmf, pmf)
    with pytest.raises(ditException):
        dit.random_distribution(2, 2, alpha=[1])

def test_simplex_grid1():
    # Test with tuple
    dists = np.asarray(list(dit.simplex_grid(2, 2**2, using=tuple)))
    dists_ = np.asarray([(0.0, 1.0), (0.25, 0.75), (0.5, 0.5),
                         (0.75, 0.25), (1.0, 0.0)])
    assert np.allclose(dists, dists_)

def test_simplex_grid2():
    # Test with ScalarDistribution
    dists = np.asarray([d.pmf for d in dit.simplex_grid(2, 2**2)])
    dists_ = np.asarray([(0.0, 1.0), (0.25, 0.75), (0.5, 0.5),
                         (0.75, 0.25), (1.0, 0.0)])
    assert np.allclose(dists, dists_)

def test_simplex_grid3():
    # Test with Distribution
    d = dit.random_distribution(1, 2)
    dists = np.asarray([x.pmf for x in dit.simplex_grid(2, 2**2, using=d)])
    dists_ = np.asarray([(0.0, 1.0), (0.25, 0.75), (0.5, 0.5),
                         (0.75, 0.25), (1.0, 0.0)])
    assert np.allclose(dists, dists_)

def test_simplex_grid4():
    # Test with Distribution but with wrong length specified.
    d = dit.random_distribution(2, 2)
    g = dit.simplex_grid(5, 2**2, using=d)
    with pytest.raises(Exception):
        next(g)

def test_simplex_grid5():
    # Test with ScalarDistribution with inplace=True
    # All final dists should be the same.
    dists = np.asarray([d.pmf for d in dit.simplex_grid(2, 2**2, inplace=True)])
    dists_ = np.asarray([(1.0, 0.0)]*5)
    assert np.allclose(dists, dists_)

def test_simplex_grid6():
    # Test using NumPy arrays and a base of 3.
    d_ = np.array([
        [ 0.        ,  0.        ,  1.        ],
        [ 0.        ,  0.33333333,  0.66666667],
        [ 0.        ,  0.66666667,  0.33333333],
        [ 0.        ,  1.        ,  0.        ],
        [ 0.33333333,  0.        ,  0.66666667],
        [ 0.33333333,  0.33333333,  0.33333333],
        [ 0.33333333,  0.66666667,  0.        ],
        [ 0.66666667,  0.        ,  0.33333333],
        [ 0.66666667,  0.33333333,  0.        ],
        [ 1.        ,  0.        ,  0.        ]
    ])
    d = np.array(list(dit.simplex_grid(3, 3, using=np.array)))
    assert np.allclose(d, d_)

def test_simplex_grid_bad_n():
    x = dit.simplex_grid(0, 3)
    with pytest.raises(ditException):
        list(x)

def test_simplex_grid_bad_subdivisions():
    x = dit.simplex_grid(3, 0)
    with pytest.raises(ditException):
        list(x)

# These can be simple smoke test, since the random* tests hit all the branches.

def test_uniform_scalar_distribution():
    pmf = np.array([1/3] * 3)
    outcomes = (0, 1, 2)
    dit.math.prng.seed(1)
    d = dit.uniform_scalar_distribution(len(outcomes))
    assert d.outcomes == outcomes
    assert np.allclose(d.pmf, pmf)

    dit.math.prng.seed(1)
    d = dit.uniform_scalar_distribution(outcomes)
    assert d.outcomes == outcomes
    assert np.allclose(d.pmf, pmf)

def test_uniform_distribution():
    pmf = np.array([1/4] * 4)
    dit.math.prng.seed(1)
    d = dit.uniform_distribution(2, 2)
    assert d.outcomes == ((0, 0), (0, 1), (1, 0), (1, 1))
    assert np.allclose(d.pmf, pmf)

def test_rvfunctions1():
    # Smoke test with strings
    d = dit.Distribution(['00', '01', '10', '11'], [1/4]*4)
    bf = dit.RVFunctions(d)
    d = dit.insert_rvf(d, bf.xor([0,1]))
    d = dit.insert_rvf(d, bf.xor([1,2]))
    assert d.outcomes == ('0000', '0110', '1011', '1101')

def test_rvfunctions2():
    # Smoke test with int tuples
    d = dit.Distribution([(0,0), (0,1), (1,0), (1,1)], [1/4]*4)
    bf = dit.RVFunctions(d)
    d = dit.insert_rvf(d, bf.xor([0,1]))
    d = dit.insert_rvf(d, bf.xor([1,2]))
    assert d.outcomes == ((0,0,0,0), (0,1,1,0), (1,0,1,1), (1,1,0,1))

def test_rvfunctions3():
    # Smoke test strings with from_hexes
    outcomes = ['000', '001', '010', '011', '100', '101', '110', '111']
    pmf = [1/8] * 8
    d = dit.Distribution(outcomes, pmf)
    bf = dit.RVFunctions(d)
    d = dit.insert_rvf(d, bf.from_hexes('27'))
    outcomes = ('0000', '0010', '0101', '0110', '1000', '1010', '1100', '1111')
    assert d.outcomes == outcomes

def test_rvfunctions4():
    # Smoke test int tuples from_hexes
    outcomes = ['000', '001', '010', '011', '100', '101', '110', '111']
    outcomes = [tuple(map(int, o)) for o in outcomes]
    pmf = [1/8] * 8
    d = dit.Distribution(outcomes, pmf)
    bf = dit.RVFunctions(d)
    d = dit.insert_rvf(d, bf.from_hexes('27'))
    outcomes = ('0000', '0010', '0101', '0110', '1000', '1010', '1100', '1111')
    outcomes = tuple(tuple(map(int, o)) for o in outcomes)
    assert d.outcomes == outcomes

def test_rvfunctions_scalardist():
    d = dit.ScalarDistribution(range(5), [1/5] * 5)
    with pytest.raises(ditException):
        dit.RVFunctions(d)

def test_rvfunctions_ints():
    d = dit.uniform_distribution(2, 2)
    rvf = dit.RVFunctions(d)
    partition = [(d.outcomes[i],) for i in range(len(d))]
    mapping = rvf.from_partition(partition)
    d2 = dit.insert_rvf(d, mapping)
    outcomes = ((0,0,0), (0,1,1), (1,0,2), (1,1,3))
    assert d2.outcomes == outcomes

def test_rvfunctions_toolarge():
    letters = 'abcd'
    outcomes = itertools.product(letters, repeat=3)
    outcomes = list(map(''.join, outcomes))
    d = dit.Distribution(outcomes, [1/64]*64, validate=False)
    rvf = dit.RVFunctions(d)
    partition = [(d.outcomes[i],) for i in range(len(d))]
    with pytest.raises(NotImplementedError):
        rvf.from_partition(partition)

def test_insert_rvf1():
    # Test multiple insertion.
    d = dit.uniform_distribution(2, 2)
    def xor(outcome):
        o = int(outcome[0] != outcome[1])
        # Here we are returning 2 random variables
        return (o,o)
    # We are also inserting two times simultaneously.
    d2 = dit.insert_rvf(d, [xor, xor])
    outcomes = (
        (0, 0, 0, 0, 0, 0),
        (0, 1, 1, 1, 1, 1),
        (1, 0, 1, 1, 1, 1),
        (1, 1, 0, 0, 0, 0)
    )
    assert d2.outcomes == outcomes

def test_insert_rvf2():
    # Test multiple insertion.
    d = dit.uniform_distribution(2, 2)
    d = dit.modify_outcomes(d, lambda x: ''.join(map(str, x)))
    def xor(outcome):
        o = str(int(outcome[0] != outcome[1]))
        # Here we are returning 2 random variables
        return o*2
    # We are also inserting two times simultaneously.
    d2 = dit.insert_rvf(d, [xor, xor])
    outcomes = ('000000', '011111', '101111', '110000')
    assert d2.outcomes == outcomes

def test_RVFunctions_from_mapping1():
    d = dit.Distribution(['00', '01', '10', '11'], [1/4]*4)
    bf = dit.RVFunctions(d)
    mapping = {'00': '0', '01': '1', '10': '1', '11': '0'}
    d = dit.insert_rvf(d, bf.from_mapping(mapping))
    outcomes = ('000', '011', '101', '110')
    assert d.outcomes == outcomes

def test_RVFunctions_from_mapping2():
    d = dit.Distribution([(0,0), (0,1), (1,0), (1,1)], [1/4]*4)
    bf = dit.RVFunctions(d)
    mapping = {(0,0): 0, (0,1): 1, (1,0): 1, (1,1): 0}
    d = dit.insert_rvf(d, bf.from_mapping(mapping, force=True))
    outcomes = ((0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0))
    assert d.outcomes == outcomes

def test_RVFunctions_from_partition():
    d = dit.Distribution(['00', '01', '10', '11'], [1/4]*4)
    bf = dit.RVFunctions(d)
    partition = (('00','11'), ('01', '10'))
    d = dit.insert_rvf(d, bf.from_partition(partition))
    outcomes = ('000', '011', '101', '110')
    assert d.outcomes == outcomes

def test_product():
    """
    Smoke test for product_distribution().

    """
    d = dit.example_dists.Xor()
    d_iid = dit.product_distribution(d)
    d_truth = dit.uniform_distribution(3, ['01'])
    d_truth = dit.modify_outcomes(d_truth, lambda x: ''.join(x))
    assert d_truth.is_approx_equal(d_iid)

def test_product_nonjoint():
    """
    Test product_distribution() from a ScalarDistribution.

    """
    d = dit.ScalarDistribution([.5, .5])
    with pytest.raises(Exception):
        dit.product_distribution(d)

def test_product_with_rvs1():
    """
    Test product_distribution() with an rvs specification.

    """
    d = dit.example_dists.Xor()
    d_iid = dit.product_distribution(d, [[0,1], [2]])
    d_truth = dit.uniform_distribution(3, ['01'])
    d_truth = dit.modify_outcomes(d_truth, lambda x: ''.join(x))
    assert d_truth.is_approx_equal(d_iid)

def test_product_with_rvs2():
    """
    Test product_distribution() with an rvs specification.

    """
    d = dit.example_dists.Xor()
    d_iid = dit.product_distribution(d, [[0,1]])
    d_truth = dit.uniform_distribution(2, ['01'])
    d_truth = dit.modify_outcomes(d_truth, lambda x: ''.join(x))
    assert d_truth.is_approx_equal(d_iid)

def test_product_with_badrvs():
    """
    Test product_distribution() with overlapping rvs specification.

    """
    d = dit.example_dists.Xor()
    with pytest.raises(Exception):
        dit.product_distribution(d, [[0,1], [0]])

@pytest.mark.parametrize(('n', 'k'), [(2, 2), (2, 3), (3, 2)])
def test_all_dist_structures(n, k):
    """
    Test all_dist_structures().

    """
    num_dists = len(list(dit.all_dist_structures(n, k)))
    assert num_dists == 2**(k**n)-1

@pytest.mark.parametrize('d', [ dit.random_dist_structure(3, 3) for _ in range(10) ])
def test_random_dist_structure(d):
    """
    Test random_dist_structure()

    """
    words = {''.join(word) for word in itertools.product('012', repeat=3)}
    diff = set(d.outcomes) - words
    assert diff == set()
    assert 0 < len(d.outcomes) <= 3**3

def test_coarsegrain():
    d = dit.example_dists.Xor()
    d2 = dit.modify_outcomes(d, lambda x: '1' if '1' in x else '0')
    assert d2.outcomes == ('0', '1')
    assert np.allclose(d2.pmf, [.25, .75])
