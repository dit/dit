#!/usr/bin/env python
#cython: language_level=3

"""
Low level implementation of scalar `allclose`.
"""

from libc.math cimport isnan, isinf, fabs



def close(double x, double y, double rtol, double atol):
    """Returns True if the scalars x and y are close.

    The relative error rtol must be positive and << 1.0
    The absolute error atol usually comes into play when y is very small or
    zero; it says how small x must be also.

    """
    # Test for nan
    if isnan(x) or isnan(y):
        return False

    # Make sure they are both inf or non-inf
    cdef int xinf, yinf
    xinf = isinf(x)
    yinf = isinf(y)

    if not xinf == yinf:
        return False

    if xinf:
        # If they are both inf, make sure the signs are the same.
        return (x > 0) == (y > 0)
    else:
        # Otherwise, make sure they are close.
        return fabs(x-y) <= atol + rtol * fabs(y)
