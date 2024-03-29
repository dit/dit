"""
Tests for dit.math.misc.
"""

import pytest

from dit.math.misc import combinations, is_integer, is_number, factorial


@pytest.mark.parametrize('n', range(-10, 10))
def test_number1(n):
        assert is_number(n)


@pytest.mark.parametrize('n', range(-10, 10))
def test_number2(n):
    assert is_number(n / 10)


@pytest.mark.parametrize('n', ['a', int, []])
def test_number3(n):
    assert not is_number(n)


@pytest.mark.parametrize('n', range(-10, 10))
def test_integer1(n):
    assert is_integer(n)


@pytest.mark.parametrize('n', range(-10, 10))
def test_integer2(n):
    assert not is_integer(n / 10)


@pytest.mark.parametrize('n', ['a', int, []])
def test_integer3(n):
    assert not is_integer(n)


@pytest.mark.parametrize(('n', 'expected'), [
    (0, 1),
    (1, 1),
    (2, 2),
    (3, 6),
    (4, 24),
    (5, 120),
])
def test_factorial1(n, expected):
    assert factorial(n) == expected


@pytest.mark.parametrize('n', [-1, 0.5, 1 + 2j])
def test_factorial2(n):
    with pytest.raises(ValueError, match="is not a positive integer."):
        factorial(n)


@pytest.mark.parametrize('n', ['a', int, []])
def test_factorial3(n):
    with pytest.raises(TypeError, match="(is not a number.|unhashable type)"):
        factorial(n)


@pytest.mark.parametrize(('k', 'c'), [
    (0, 1),
    (1, 3),
    (2, 3),
    (3, 1),
])
def test_combinations1(k, c):
    n = 3
    assert combinations(n, k) == c


def test_combinations2():
    with pytest.raises(ValueError, match="is larger than"):
        combinations(5, 7)
