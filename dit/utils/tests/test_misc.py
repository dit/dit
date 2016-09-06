"""
Tests for dit.utils.misc.
"""

import pytest

from dit.utils.misc import flatten, is_string_like, partitions, partitions2, \
                           ordered_partitions, require_keys, partition_set, \
                           abstract_method, digits
from six import u

def test_flatten1():
    x = [[[0], 1, 2], 3, [4]]
    fx = list(flatten(x))
    y = [0, 1, 2, 3, 4]
    assert len(fx) == len(y)
    for i, j in zip(fx, y):
        assert i == j

def test_is_string_like1():
    ys = ['', 'hi', "pants", '"test"', u('pants'), r'pants']
    ns = [1, [], int, {}, ()]
    for y in ys:
        assert is_string_like(y)
    for n in ns:
        assert not is_string_like(n)

def test_partitions1():
    bells = [1, 1, 2, 5, 15, 52]
    for i, b in enumerate(bells):
        parts = list(partitions(range(i)))
        assert len(parts) == b

def test_partitions2():
    bells = [1, 1, 2, 5, 15, 52]
    for i, b in enumerate(bells):
        parts = list(partitions(range(i), tuples=True))
        assert len(parts) == b

def test_partitions3():
    bells = [1, 1, 2, 5, 15, 52]
    for i, b in enumerate(bells):
        parts = list(partitions2(i))
        assert len(parts) == b

def test_ordered_partitions1():
    obells = [1, 1, 3, 13, 75]
    for i, b in enumerate(obells):
        oparts = list(ordered_partitions(range(i)))
        assert len(oparts) == b


def test_ordered_partitions2():
    obells = [1, 1, 3, 13, 75]
    for i, b in enumerate(obells):
        oparts = list(ordered_partitions(range(i), tuples=True))
        assert len(oparts) == b

def test_require_keys1():
    d = {0: '', '0': '', 'pants': ''}
    required = [0, '0']
    assert require_keys(required, d) is None

def test_require_keys2():
    d = {0: '', '0': '', 'pants': ''}
    required = [0, '0', 'jeans']
    with pytest.raises(Exception):
        require_keys(required, d)

def test_partition_set1():
    stuff = ['0', '1', '00', '11', '000', '111', [0, 1, 2]]
    fn = lambda a, b: len(a) == len(b)
    _, lookup = partition_set(stuff, fn)
    assert lookup == [0, 0, 1, 1, 2, 2, 2]

def test_partition_set2():
    stuff = ['0', '1', '00', '11', '000', '111', [0, 1, 2]]
    fn = lambda a, b: len(a) == len(b)
    _, lookup = partition_set(stuff, fn, reflexive=True, transitive=True)
    assert lookup == [0, 0, 1, 1, 2, 2, 2]

def test_abstract_method():
    @abstract_method
    def an_abstract_method():
        pass
    with pytest.raises(NotImplementedError):
        an_abstract_method()

class TestDigits():
    def test_bad_base(self):
        with pytest.raises(ValueError):
            digits(3, 1)

    def test_nonint_base(self):
        with pytest.raises(ValueError):
            digits(3, 3.2)

    def test_bad_alphabet(self):
        with pytest.raises(ValueError):
            digits(3, 2, alphabet=[1,2,3])

    def test_with_alphabet(self):
        x = digits(3, 2, alphabet=[0, 1])
        assert x == [1, 1]

    def test_with_pad(self):
        x = digits(3, 2, pad=4, alphabet=[0, 1])
        assert x == [0, 0, 1, 1]

    def test_little_endian(self):
        x = digits(2, 2)
        assert x == [1, 0]

        x = digits(2, 2, big_endian=False)
        assert x == [0, 1]
