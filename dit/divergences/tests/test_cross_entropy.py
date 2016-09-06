"""
Tests for dit.divergences.cross_entropy.
"""

from __future__ import division

import pytest

import numpy as np

from dit import Distribution
from dit.divergences import cross_entropy
from dit.exceptions import ditException
from dit.multivariate import entropy


d1 = Distribution(['0', '1'], [1/2, 1/2])
d2 = Distribution(['0', '2'], [1/2, 1/2])
d3 = Distribution(['0', '1', '2'], [1/3, 1/3, 1/3])
d4 = Distribution(['00', '11'], [2/5, 3/5])
d5 = Distribution(['00', '11'], [1/2, 1/2])


@pytest.mark.parametrize(('args', 'expected'), [
    ([d1, d3], 1.5849625007211563),
    ([d1, d4], 1.0294468445267841),
    ([d1, d3, [0]], 1.5849625007211563),
    ([d1, d4, [0]], 1.0294468445267841),
    ([d4, d1, [0]], 1),
    ([d4, d5], 1),
    ([d5, d4], 1.0294468445267841),
    ([d4, d5, [0], [1]], 0),
    ([d4, d5, [1], [0]], 0),
    ([d1, d2], np.inf),
    ([d2, d1], np.inf),
    ([d3, d1], np.inf),
])
def test_cross_entropy_1(args, expected):
    """
    Test against several known values.
    """
    assert cross_entropy(*args) ==  pytest.approx(expected)


@pytest.mark.parametrize('d', [d1, d2, d3, d4, d5])
def test_cross_entropy_2(d):
    """
    Test that xH(d, d) = H(d).
    """
    assert cross_entropy(d, d) == pytest.approx(entropy(d))


@pytest.mark.parametrize('args', [
    [d4, d1, None, None],
    [d4, d2, None, None],
    [d4, d3, None, None],
    [d1, d2, [0, 1], None],
    [d3, d4, [1], None],
    [d5, d1, [0], [1]],
    [d4, d3, [1], [0]],
])
def test_cross_entropy_3(args):
    """
    Test that when p has outcomes that q doesn't have, that we raise an exception.
    """
    first, second, rvs, crvs = args
    with pytest.raises(ditException):
        cross_entropy(first, second, rvs, crvs)
