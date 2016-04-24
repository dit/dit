#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Low level implementation of scalar `allclose`.

"""
import sys

cdef extern from "math.h":
    double fabs(double)
    int isinf(double)
    int isnan(double)

cdef extern from "float.h":
    double fabs(double)
    int _isnan(double)
    int _finite(double)


if sys.platform in ('win32', 'cygwin'):
    def isnan(double x):
        return _isnan(x)
    def isinf(double x):
        return 1 - _finite(x)


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
