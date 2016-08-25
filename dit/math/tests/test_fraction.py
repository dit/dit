from __future__ import division

import pytest

from fractions import Fraction
from dit.math.fraction import *

def test_fraction():
    """Smoke tests to convert float to fraction."""
    numerators = range(10)
    denominator = 10
    xvals = [x / denominator for x in numerators]
    af = lambda x: approximate_fraction(x, .01)
    yvals = map(af, xvals)
    yvals_ = map(lambda x: Fraction(x, denominator), numerators)
    for y, y_ in zip(yvals, yvals_):
        assert y == y_

    # Negative values
    af = lambda x: approximate_fraction(-x, .01)
    yvals = map(af, xvals)
    yvals_ = map(lambda x: Fraction(-x, denominator), numerators)
    for y, y_ in zip(yvals, yvals_):
        assert y == y_

def test_fraction_zero():
    """Convert float to fraction when closer to 0."""
    x = .1
    y = approximate_fraction(x, .2)
    y_ = Fraction(0, 1)
    assert y == y_
    y = approximate_fraction(-x, .2)
    assert y == -y_

def test_fraction_emptyinterval():
    with pytest.raises(ValueError):
        approximate_fraction(0.1, 0)
