from nose.tools import assert_almost_equal

from dit.example_dists import Unq, Rdn, Xor, RdnXor, ImperfectRdn, Subtle

from dit.algorithms import entropy, insert_meet, mutual_information

def test_unq():
    d = Unq()
    d = insert_meet(d, -1, [[0], [1]])
    i1 = mutual_information(d, [0], [2])
    i2 = mutual_information(d, [1], [2])
    i12 = mutual_information(d, [0,1], [2])
    r = mutual_information(d, [2], [3])
    assert_almost_equal(i1, 1)
    assert_almost_equal(i2, 1)
    assert_almost_equal(i12, 2)
    assert_almost_equal(r, 0)

def test_rdn():
    d = Rdn()
    d = insert_meet(d, -1, [[0], [1]])
    i1 = mutual_information(d, [0], [2])
    i2 = mutual_information(d, [1], [2])
    i12 = mutual_information(d, [0,1], [2])
    r = mutual_information(d, [2], [3])
    assert_almost_equal(i1, 1)
    assert_almost_equal(i2, 1)
    assert_almost_equal(i12, 1)
    assert_almost_equal(r, 1)

def test_xor():
    d = Xor()
    d = insert_meet(d, -1, [[0], [1]])
    i1 = mutual_information(d, [0], [2])
    i2 = mutual_information(d, [1], [2])
    i12 = mutual_information(d, [0,1], [2])
    r = mutual_information(d, [2], [3])
    assert_almost_equal(i1, 0)
    assert_almost_equal(i2, 0)
    assert_almost_equal(i12, 1)
    assert_almost_equal(r, 0)

def test_rdnxor():
    d = RdnXor()
    d = insert_meet(d, -1, [[0], [1]])
    i1 = mutual_information(d, [0], [2])
    i2 = mutual_information(d, [1], [2])
    i12 = mutual_information(d, [0,1], [2])
    r = mutual_information(d, [2], [3])
    assert_almost_equal(i1, 1)
    assert_almost_equal(i2, 1)
    assert_almost_equal(i12, 2)
    assert_almost_equal(r, 1)

def test_imperfectrdn():
    d = ImperfectRdn()
    d = insert_meet(d, -1, [[0], [1]])
    i1 = mutual_information(d, [0], [2])
    i2 = mutual_information(d, [1], [2])
    i12 = mutual_information(d, [0,1], [2])
    r = mutual_information(d, [2], [3])
    assert_almost_equal(i1, 1)
    assert_almost_equal(i2, 0.98959007894024409)
    assert_almost_equal(i12, 1)
    assert_almost_equal(r, 0)

def test_subtle():
    d = Subtle()
    d = insert_meet(d, -1, [[0], [1]])
    i1 = mutual_information(d, [0], [2])
    i2 = mutual_information(d, [1], [2])
    i12 = mutual_information(d, [0,1], [2])
    r = mutual_information(d, [2], [3])
    assert_almost_equal(i1, 0.91829583405448956)
    assert_almost_equal(i2, 0.91829583405448956)
    assert_almost_equal(i12, 1.5849625007211561)
    assert_almost_equal(r, 0)