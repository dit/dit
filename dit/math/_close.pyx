#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Low level implementation of scalar `allclose`.

"""

cdef extern from "math.h":
    double fabs(double)
    int isinf(double)
    int isnan(double)

def close(double x, double y, double rtol, double atol):
    """Returns True if the scalars x and y are close.

    The relative error rtol must be positive and << 1.0
    The absolute error atol usually comes into play when y is very small or
    zero; it says how small x must be also.

    If rtol or atol are unspecified, they are taken from cmpyParams['rtol']
    and cmpyParams['atol'].

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

    cdef bint xgz, ygz
    if xinf:
        # If they are inf, make sure the signs are the same.
        xgz = x>0
        ygz = y>0
        if (xgz and not ygz) or (not xgz and ygz):
            return False
        else:
            return True
    else:
        # Otherwise, make sure they are close.
        return fabs(x-y) <= atol + rtol * fabs(y)

