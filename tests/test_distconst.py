"""
Tests for dit.distconst.
"""

import itertools

import numpy as np
import pytest

import dit
from dit.exceptions import ditException


def test_mixture_distribution_weights():
    d = dit.Distribution(["A", "B"], [0.5, 0.5])
    d2 = dit.Distribution(["A", "B"], [1, 0])

    with pytest.raises(ditException):
        dit.mixture_distribution([d, d2], [1])
    with pytest.raises(ditException):
        dit.mixture_distribution2([d, d2], [1])


def test_mixture_distribution():
    d = dit.Distribution(["A", "B"], [0.5, 0.5])
    d2 = dit.Distribution(["A", "B"], [1, 0])
    pmf = np.array([0.75, 0.25])

    d3 = dit.mixture_distribution([d, d2], [0.5, 0.5])
    assert np.allclose(pmf, d3.pmf)


def test_mixture_distribution_log():
    d = dit.Distribution(["A", "B"], [0.5, 0.5])
    d2 = dit.Distribution(["A", "B"], [1, 0])
    d.set_base(2)
    d2.set_base(2)
    weights = np.log2(np.array([0.5, 0.5]))
    pmf = np.log2(np.array([0.75, 0.25]))

    d3 = dit.mixture_distribution([d, d2], weights)
    assert np.allclose(pmf, d3.pmf)


def test_mixture_distribution2():
    # Test when pmfs are different lengths.
    d = dit.Distribution(["A", "B"], [0.5, 0.5])
    d2 = dit.Distribution(["A", "B"], [1, 0], sort=True, trim=True)

    # Fails when it checks that all pmfs have the same length.
    with pytest.raises(ValueError, match="All pmfs must have the same length."):
        dit.mixture_distribution2([d, d2], [0.5, 0.5])


def test_mixture_distribution3():
    # Sample spaces are compatible.
    # Outcomes are given in a different order, but alphabets are canonicalized.
    d = dit.Distribution(["A", "B"], [0.5, 0.5])
    d2 = dit.Distribution(["B", "A"], [1, 0], sort=False, trim=False, sparse=False)
    pmf = np.array([0.25, 0.75])

    d3 = dit.mixture_distribution([d, d2], [0.5, 0.5])
    assert np.allclose(pmf, d3.pmf)
    d3 = dit.mixture_distribution2([d, d2], [0.5, 0.5])
    assert np.allclose(pmf, d3.pmf)


def test_mixture_distribution2_uses_weights():
    d = dit.Distribution(["00", "01", "10", "11"], [1, 0, 0, 0], sparse=False, trim=False)
    d2 = dit.Distribution(["00", "01", "10", "11"], [0, 0, 0, 1], sparse=False, trim=False)
    d3 = dit.mixture_distribution2([d, d2], [0.5, 0.5])
    assert np.allclose(d3.pmf, [0.5, 0, 0, 0.5])


def test_mixture_distribution4():
    # Sample spaces are compatible.
    # But pmfs have a different lengths and orders.
    d = dit.Distribution(["A", "B"], [0.5, 0.5])
    d2 = dit.Distribution(["B", "A"], [1, 0], sort=False, trim=False, sparse=True)
    d2.make_sparse(trim=True)
    pmf = np.array([0.25, 0.75])

    d3 = dit.mixture_distribution([d, d2], [0.5, 0.5])
    assert np.allclose(pmf, d3.pmf)
    with pytest.raises(ValueError, match="All pmfs must have the same length."):
        dit.mixture_distribution2([d, d2], [0.5, 0.5])


def test_mixture_distribution5():
    # Incompatible sample spaces.
    d1 = dit.Distribution(["A", "B"], [0.5, 0.5])
    d2 = dit.Distribution(["B", "C"], [0.5, 0.5])
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
    dists_ = np.asarray([(0, 1), (0.25, 0.75), (0.5, 0.5), (0.75, 0.25), (1, 0)])
    assert np.allclose(dists, dists_)


def test_simplex_grid2():
    # Test with Distribution
    dists = np.asarray([d.pmf for d in dit.simplex_grid(2, 2**2)])
    dists_ = np.asarray([(0, 1), (0.25, 0.75), (0.5, 0.5), (0.75, 0.25), (1, 0)])
    assert np.allclose(dists, dists_)


def test_simplex_grid3():
    # Test with Distribution
    d = dit.random_distribution(1, 2)
    dists = np.asarray([x.pmf for x in dit.simplex_grid(2, 2**2, using=d)])
    dists_ = np.asarray([(0, 1), (0.25, 0.75), (0.5, 0.5), (0.75, 0.25), (1, 0)])
    assert np.allclose(dists, dists_)


def test_simplex_grid4():
    # Test with Distribution but with wrong length specified.
    d = dit.random_distribution(2, 2)
    g = dit.simplex_grid(5, 2**2, using=d)
    with pytest.raises(Exception, match="`length` must match the length of pmf"):
        next(g)


def test_simplex_grid5():
    # Test with Distribution with inplace=True
    # With Distribution, pmf returns a fresh array, so inplace mode
    # still yields the correct individual grid-point pmfs.
    dists = np.asarray([d.pmf for d in dit.simplex_grid(2, 2**2, inplace=True)])
    dists_ = np.asarray([(0, 1), (0.25, 0.75), (0.5, 0.5), (0.75, 0.25), (1, 0)])
    assert np.allclose(dists, dists_)


def test_simplex_grid6():
    # Test using NumPy arrays and a base of 3.
    d_ = np.array(
        [
            [0, 0, 1],
            [0, 0.33333333, 0.66666667],
            [0, 0.66666667, 0.33333333],
            [0, 1, 0],
            [0.33333333, 0, 0.66666667],
            [0.33333333, 0.33333333, 0.33333333],
            [0.33333333, 0.66666667, 0],
            [0.66666667, 0, 0.33333333],
            [0.66666667, 0.33333333, 0],
            [1, 0, 0],
        ]
    )
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
    pmf = np.array([1 / 3] * 3)
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
    pmf = np.array([1 / 4] * 4)
    dit.math.prng.seed(1)
    d = dit.uniform_distribution(2, 2)
    assert d.outcomes == ((0, 0), (0, 1), (1, 0), (1, 1))
    assert np.allclose(d.pmf, pmf)


def test_rvfunctions1():
    # Smoke test with strings
    d = dit.Distribution(["00", "01", "10", "11"], [1 / 4] * 4)
    bf = dit.RVFunctions(d)
    d = dit.insert_rvf(d, bf.xor([0, 1]))
    d = dit.insert_rvf(d, bf.xor([1, 2]))
    assert d.outcomes == (("0", "0", "0", "0"), ("0", "1", "1", "0"), ("1", "0", "1", "1"), ("1", "1", "0", "1"))


def test_rvfunctions2():
    # Smoke test with int tuples
    d = dit.Distribution([(0, 0), (0, 1), (1, 0), (1, 1)], [1 / 4] * 4)
    bf = dit.RVFunctions(d)
    d = dit.insert_rvf(d, bf.xor([0, 1]))
    d = dit.insert_rvf(d, bf.xor([1, 2]))
    assert d.outcomes == ((0, 0, 0, 0), (0, 1, 1, 0), (1, 0, 1, 1), (1, 1, 0, 1))


def test_rvfunctions3():
    # Smoke test strings with from_hexes — appended value is int (0/1)
    outcomes = ["000", "001", "010", "011", "100", "101", "110", "111"]
    pmf = [1 / 8] * 8
    d = dit.Distribution(outcomes, pmf)
    bf = dit.RVFunctions(d)
    d = dit.insert_rvf(d, bf.from_hexes("27"))
    expected = (
        ("0", "0", "0", 0),
        ("0", "0", "1", 0),
        ("0", "1", "0", 1),
        ("0", "1", "1", 0),
        ("1", "0", "0", 0),
        ("1", "0", "1", 0),
        ("1", "1", "0", 0),
        ("1", "1", "1", 1),
    )
    assert d.outcomes == expected


def test_rvfunctions4():
    # Smoke test int tuples from_hexes
    outcomes = ["000", "001", "010", "011", "100", "101", "110", "111"]
    outcomes = [tuple(map(int, o)) for o in outcomes]
    pmf = [1 / 8] * 8
    d = dit.Distribution(outcomes, pmf)
    bf = dit.RVFunctions(d)
    d = dit.insert_rvf(d, bf.from_hexes("27"))
    outcomes = ("0000", "0010", "0101", "0110", "1000", "1010", "1100", "1111")
    outcomes = tuple(tuple(map(int, o)) for o in outcomes)
    assert d.outcomes == outcomes


def test_rvfunctions_scalardist():
    d = dit.Distribution(range(5), [1 / 5] * 5)
    with pytest.raises(ditException):
        dit.RVFunctions(d)


def test_rvfunctions_ints():
    d = dit.uniform_distribution(2, 2)
    rvf = dit.RVFunctions(d)
    partition = [(d.outcomes[i],) for i in range(len(d))]
    mapping = rvf.from_partition(partition)
    d2 = dit.insert_rvf(d, mapping)
    outcomes = ((0, 0, 0), (0, 1, 1), (1, 0, 2), (1, 1, 3))
    assert d2.outcomes == outcomes


def test_rvfunctions_toolarge():
    # With Distribution, from_partition uses integer values so there's no
    # 62-character limit.  Just verify it completes without error.
    letters = "abcd"
    outcomes = list(itertools.product(letters, repeat=3))
    d = dit.Distribution(outcomes, [1 / 64] * 64, validate=False)
    rvf = dit.RVFunctions(d)
    partition = [(d.outcomes[i],) for i in range(len(d))]
    f = rvf.from_partition(partition)
    assert callable(f)


def test_insert_rvf1():
    # Test multiple insertion.
    d = dit.uniform_distribution(2, 2)

    def xor(outcome):
        o = int(outcome[0] != outcome[1])
        # Here we are returning 2 random variables
        return (o, o)

    # We are also inserting two times simultaneously.
    d2 = dit.insert_rvf(d, [xor, xor])
    outcomes = ((0, 0, 0, 0, 0, 0), (0, 1, 1, 1, 1, 1), (1, 0, 1, 1, 1, 1), (1, 1, 0, 0, 0, 0))
    assert d2.outcomes == outcomes


def test_insert_rvf2():
    # Test multiple insertion.
    d = dit.uniform_distribution(2, 2)
    d = dit.modify_outcomes(d, lambda x: "".join(map(str, x)))

    def xor(outcome):
        o = str(int(outcome[0] != outcome[1]))
        # Here we are returning 2 random variables
        return o * 2

    # We are also inserting two times simultaneously.
    d2 = dit.insert_rvf(d, [xor, xor])
    outcomes = (
        ("0", "0", "0", "0", "0", "0"),
        ("0", "1", "1", "1", "1", "1"),
        ("1", "0", "1", "1", "1", "1"),
        ("1", "1", "0", "0", "0", "0"),
    )
    assert d2.outcomes == outcomes


def test_RVFunctions_from_mapping1():
    d = dit.Distribution([("0", "0"), ("0", "1"), ("1", "0"), ("1", "1")], [1 / 4] * 4)
    bf = dit.RVFunctions(d)
    mapping = {("0", "0"): "0", ("0", "1"): "1", ("1", "0"): "1", ("1", "1"): "0"}
    d = dit.insert_rvf(d, bf.from_mapping(mapping))
    outcomes = (("0", "0", "0"), ("0", "1", "1"), ("1", "0", "1"), ("1", "1", "0"))
    assert d.outcomes == outcomes


def test_RVFunctions_from_mapping2():
    d = dit.Distribution([(0, 0), (0, 1), (1, 0), (1, 1)], [1 / 4] * 4)
    bf = dit.RVFunctions(d)
    mapping = {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 0}
    d = dit.insert_rvf(d, bf.from_mapping(mapping, force=True))
    outcomes = ((0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0))
    assert d.outcomes == outcomes


def test_RVFunctions_from_partition():
    d = dit.Distribution([("0", "0"), ("0", "1"), ("1", "0"), ("1", "1")], [1 / 4] * 4)
    bf = dit.RVFunctions(d)
    partition = ((("0", "0"), ("1", "1")), (("0", "1"), ("1", "0")))
    d = dit.insert_rvf(d, bf.from_partition(partition))
    outcomes = (("0", "0", 0), ("0", "1", 1), ("1", "0", 1), ("1", "1", 0))
    assert d.outcomes == outcomes


def test_product():
    """
    Smoke test for product_distribution().
    """
    d = dit.example_dists.Xor()
    d_iid = dit.product_distribution(d)
    d_truth = dit.uniform_distribution(3, ["01"])
    d_truth = dit.modify_outcomes(d_truth, lambda x: "".join(x))
    assert d_truth.is_approx_equal(d_iid)


def test_product_nonjoint():
    """
    Test product_distribution() from a Distribution.
    """
    d = dit.Distribution([0.5, 0.5])
    with pytest.raises(Exception, match="A joint distribution is required."):
        dit.product_distribution(d)


def test_product_with_rvs1():
    """
    Test product_distribution() with an rvs specification.
    """
    d = dit.example_dists.Xor()
    d_iid = dit.product_distribution(d, [[0, 1], [2]])
    d_truth = dit.uniform_distribution(3, ["01"])
    d_truth = dit.modify_outcomes(d_truth, lambda x: "".join(x))
    assert d_truth.is_approx_equal(d_iid)


def test_product_with_rvs2():
    """
    Test product_distribution() with an rvs specification.
    """
    d = dit.example_dists.Xor()
    d_iid = dit.product_distribution(d, [[0, 1]])
    d_truth = dit.uniform_distribution(2, ["01"])
    d_truth = dit.modify_outcomes(d_truth, lambda x: "".join(x))
    assert d_truth.is_approx_equal(d_iid)


def test_product_with_badrvs():
    """
    Test product_distribution() with overlapping rvs specification.
    """
    d = dit.example_dists.Xor()
    with pytest.raises(Exception, match="The elements of `rvs` have nonzero intersection."):
        dit.product_distribution(d, [[0, 1], [0]])


@pytest.mark.parametrize(("n", "k"), [(2, 2), (2, 3), (3, 2)])
def test_all_dist_structures(n, k):
    """
    Test all_dist_structures().
    """
    num_dists = len(list(dit.all_dist_structures(n, k)))
    assert num_dists == 2 ** (k**n) - 1


@pytest.mark.parametrize("d", [dit.random_dist_structure(3, 3) for _ in range(10)])
def test_random_dist_structure(d):
    """
    Test random_dist_structure()
    """
    words = {tuple(word) for word in itertools.product("012", repeat=3)}
    diff = set(d.outcomes) - words
    assert diff == set()
    assert 0 < len(d.outcomes) <= 3**3


def test_coarsegrain():
    d = dit.example_dists.Xor()
    d2 = dit.modify_outcomes(d, lambda x: "1" if "1" in x else "0")
    assert d2.outcomes == (("0",), ("1",))
    assert np.allclose(d2.pmf, [0.25, 0.75])


def test_noisy():
    """
    Test noisy
    """
    d = dit.example_dists.giant_bit(2, 2)
    d2 = dit.noisy(d, noise=0.5)
    print(d2)
    d3 = dit.Distribution(["00", "01", "10", "11"], [3 / 8, 1 / 8, 1 / 8, 3 / 8])
    assert d2.is_approx_equal(d3)


@pytest.mark.parametrize(("n", "k", "expected"), [(1, 2, 2), (2, 2, 5)])
def test_distribution_enumerator(n, k, expected):
    """
    DistributionEnumerator yields exactly the non-isomorphic uniform
    distributions over ``n`` variables of cardinality ``k``.
    """
    from dit.distconst import DistributionEnumerator

    dists = list(DistributionEnumerator(n, k))
    assert len(dists) == expected
    for d in dists:
        assert d.outcome_length() == n
        # Uniform: all probabilities equal.
        assert np.allclose(d.pmf, d.pmf[0])


def test_combine_scalar_dists():
    """
    _combine_scalar_dists treats the inputs as independent and combines their
    outcomes with the supplied operator.
    """
    import operator

    from dit.distconst import _combine_scalar_dists

    d1 = dit.Distribution([(1,), (2,)], [0.5, 0.5])
    d2 = dit.Distribution([(0,), (1,)], [0.5, 0.5])
    combined = _combine_scalar_dists(d1, d2, operator.add)
    result = {o: p for o, p in zip(combined.outcomes, combined.pmf, strict=True)}
    assert result == pytest.approx({1: 0.25, 2: 0.5, 3: 0.25})


def weighted(n):
    """A fixed full-support distribution over n bits."""
    outcomes = ["".join(o) for o in itertools.product("01", repeat=n)]
    weights = np.arange(1, 2**n + 1, dtype=float) ** 1.5
    return dit.Distribution(outcomes, list(weights / weights.sum()))


def dense_array(d):
    """The dense pmf of `d` with one axis per variable."""
    d = d.copy()
    d.make_dense()
    return d.pmf.reshape([len(a) for a in d.alphabet])


def marginal_array(p, S):
    """The marginal of the array `p` onto the axes `S`, keeping all dimensions."""
    return p.sum(axis=tuple(i for i in range(p.ndim) if i not in S), keepdims=True)


def product_array(p):
    """The product of the single-variable marginals of the array `p`."""
    return np.prod(np.broadcast_arrays(*[marginal_array(p, (i,)) for i in range(p.ndim)]), axis=0)


@pytest.mark.parametrize("n", range(3, 6))
def test_kirkwood_approximation_giant_bit(n):
    """The giant bit's approximation is (1/2)^(1 + (-1)^n) on each diagonal outcome."""
    q = dit.kirkwood_approximation(dit.example_dists.giant_bit(n, 2))
    expected = np.zeros(2**n)
    expected[[0, -1]] = 0.5 ** (1 + (-1) ** n)
    assert q.pmf == pytest.approx(expected)


def test_kirkwood_approximation_three_variables():
    """For three variables the approximation is p(xy) p(xz) p(yz) / (p(x) p(y) p(z))."""
    d = weighted(3)
    p = dense_array(d)
    pairs = [marginal_array(p, S) for S in itertools.combinations(range(3), 2)]
    expected = np.prod(np.broadcast_arrays(*pairs), axis=0) / product_array(p)
    assert dense_array(dit.kirkwood_approximation(d)) == pytest.approx(expected)


def test_kirkwood_approximation_order_one_is_product():
    """The order-1 approximation is the product of marginals."""
    d = weighted(3)
    assert dense_array(dit.kirkwood_approximation(d, order=1)) == pytest.approx(product_array(dense_array(d)))


def test_kirkwood_approximation_order_zero_is_one():
    """The order-0 approximation is the empty product."""
    assert dit.kirkwood_approximation(weighted(3), order=0).pmf == pytest.approx(np.ones(8))


def test_kirkwood_approximation_grouped():
    """With two groups the approximation is the product of the group marginals."""
    d = weighted(3)
    p = dense_array(d)
    expected = (marginal_array(p, (0,)) * marginal_array(p, (1, 2))).ravel()
    assert dit.kirkwood_approximation(d, [[0], [1, 2]]).pmf == pytest.approx(expected)


@pytest.mark.parametrize("approximation", [dit.kirkwood_approximation, dit.lancaster_approximation])
@pytest.mark.parametrize("order", [-1, 4])
def test_approximation_bad_order(approximation, order):
    """The order must lie in [0, n - 1]."""
    with pytest.raises(ditException, match="order must satisfy"):
        approximation(weighted(4), order=order)


def test_lancaster_three_variables():
    """p_L = p(xy) p(z) + p(xz) p(y) + p(yz) p(x) - 2 p(x) p(y) p(z)."""
    d = weighted(3)
    p = dense_array(d)
    px, py, pz = (marginal_array(p, (i,)) for i in range(3))
    expected = (
        marginal_array(p, (0, 1)) * pz
        + marginal_array(p, (0, 2)) * py
        + marginal_array(p, (1, 2)) * px
        - 2 * px * py * pz
    )
    assert dense_array(dit.lancaster_approximation(d)) == pytest.approx(expected)


def test_lancaster_bivariate_is_product():
    """For two variables the Lancaster approximation is the product of marginals."""
    d = weighted(2)
    assert dense_array(dit.lancaster_approximation(d)) == pytest.approx(product_array(dense_array(d)))


@pytest.mark.parametrize("order", [0, 1])
def test_lancaster_low_orders_are_product(order):
    """Orders 0 and 1 are both the product of marginals."""
    d = weighted(4)
    q = dit.lancaster_approximation(d, order=order)
    assert dense_array(q) == pytest.approx(product_array(dense_array(d)))


def test_lancaster_four_variables_order_two():
    """The order-2 approximation keeps only the pairwise Lancaster interactions."""
    d = weighted(4)
    p = dense_array(d)
    prod = product_array(p)
    expected = prod.copy()
    for S in itertools.combinations(range(4), 2):
        others = [marginal_array(p, (i,)) for i in range(4) if i not in S]
        expected = expected + marginal_array(p, S) * np.prod(np.broadcast_arrays(*others), axis=0) - prod
    assert dense_array(dit.lancaster_approximation(d, order=2)) == pytest.approx(expected)


@pytest.mark.parametrize("n", range(2, 6))
def test_lancaster_default_order(n):
    """The default order is n - 1."""
    d = weighted(n)
    q = dit.lancaster_approximation(d)
    assert q.pmf == pytest.approx(dit.lancaster_approximation(d, order=n - 1).pmf)


@pytest.mark.parametrize("k", range(4))
def test_lancaster_marginals(k):
    """The order-k approximation has total mass one and matches every k-marginal of p."""
    d = weighted(4)
    p = dense_array(d)
    q = dense_array(dit.lancaster_approximation(d, order=k))
    assert q.sum() == pytest.approx(1.0)
    for S in itertools.combinations(range(4), k):
        assert marginal_array(q, S) == pytest.approx(marginal_array(p, S))


@pytest.mark.parametrize("split", [[[0], [1], [2]], [[0], [2], [1]], [[1], [2], [0]]])
def test_lancaster_factorization(split):
    """With three variables, p_L = p whenever one variable is independent of the rest."""
    d = weighted(2) @ dit.Distribution(["0", "1"], [1 / 3, 2 / 3])
    expected = dense_array(d.coalesce(split))
    assert dense_array(dit.lancaster_approximation(d, split)) == pytest.approx(expected)


def test_lancaster_no_factorization():
    """p_L = p does not imply a factorization (Sejdinovic et al. 2013, Table 4)."""
    d = dit.Distribution(["000", "001", "010", "011", "100", "101", "110", "111"], [0.2] + [0.1] * 6 + [0.2])
    assert dit.lancaster_approximation(d).pmf == pytest.approx(dense_array(d).ravel())
    assert dit.multivariate.total_correlation(d) > 0.01


def test_lancaster_four_variables_misses_factorization():
    """For four variables, (X0, X1) independent of (X2, X3) need not give p_L = p."""
    giant_bit = dit.example_dists.giant_bit(2, 2)
    d = giant_bit @ giant_bit
    difference = dit.lancaster_approximation(d).pmf - dense_array(d).ravel()
    assert np.abs(difference) == pytest.approx(np.full(16, 1 / 16))


def test_lancaster_signed():
    """The Lancaster approximation can be negative."""
    d = dit.Distribution(["000", "010", "100", "111"], [1 / 4] * 4)
    q = dit.lancaster_approximation(d)
    assert q.pmf == pytest.approx([0.3125, -0.0625, 0.1875, 0.0625, 0.1875, 0.0625, 0.0625, 0.1875])
