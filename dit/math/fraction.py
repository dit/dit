#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Function for calculating the simplest faction from a float.

http://stackoverflow.com/questions/4266741/check-if-a-number-is-rational-in-python

"""
from fractions import Fraction
from math import modf

__all__ = ['approximate_fraction']

def simplest_fraction_in_interval(x, y):
    """Return the fraction with the lowest denominator in [x,y]."""
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
        xr, xc = modf(1/x)
        yr, yc = modf(1/y)
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

