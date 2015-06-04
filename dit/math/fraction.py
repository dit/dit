#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Function for calculating the simplest faction from a float.

An alternative approach is to use:

    >>> from fractions import Fraction
    >>> x = .12345
    >>> y = Fraction(x)
    >>> y.limit_denominator(10)
    Fraction(1, 8)
    >>> y.limit_denominator(100)
    Fraction(10, 81)

But usually, we are interested in a fraction that matches within some tolerance
and the max denominator that gives a particular tolerance is not obvious.

"""
from __future__ import division

from fractions import Fraction
from math import modf

__all__ = ['approximate_fraction']

def simplest_fraction_in_interval(x, y):
    """
    Return the fraction with the lowest denominator in the interval [x, y].

    """

    # http://stackoverflow.com/questions/4266741/check-if-a-number-is-rational-in-python

    if x == y:
        # The algorithm will not terminate if x and y are equal.
        raise ValueError("Equal arguments.")
    elif x < 0 and y < 0:
        # Handle negative arguments by solving positive case and negating.
        return -simplest_fraction_in_interval(-y, -x)
    elif x <= 0 or y <= 0:
        # One argument is 0, or arguments are on opposite sides of 0, so
        # the simplest fraction in interval is 0 exactly.
        return Fraction(0)
    else:
        # Remainder and Coefficient of continued fractions for x and y.
        xr, xc = modf(1 / x)
        yr, yc = modf(1 / y)
        if xc < yc:
            return Fraction(1, int(xc) + 1)
        elif yc < xc:
            return Fraction(1, int(yc) + 1)
        else:
            return 1 / (int(xc) + simplest_fraction_in_interval(xr, yr))

def approximate_fraction(x, e):
    """
    Return an approxite rational fraction of x.

    The returned Fraction instance is the fraction with the lowest denominator
    that differs from `x` by no more than `e`.

    Examples
    --------
    >>> x = 1/3
    >>> y = approximate_fraction(x, 1e-9)
    >>> print y
    1/3

    """
    return simplest_fraction_in_interval(x - e, x + e)
