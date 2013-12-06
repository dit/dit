#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Functions to test if two floats are equal to within relative and absolute
tolerances. This dynamically chooses a cython implementation if available.
"""

from __future__ import absolute_import

from numpy import allclose as _allclose, isinf

from dit import ditParams

__all__ = ['close', 'allclose']

def close__cython(x, y, rtol=None, atol=None): # pylint: disable=missing-docstring
    if rtol is None:
        rtol = ditParams['rtol']
    if atol is None:
        atol = ditParams['atol']
    return close_(x, y, rtol, atol)

def close__python(x, y, rtol=None, atol=None): # pylint: disable=missing-docstring
    if rtol is None:
        rtol = ditParams['rtol']
    if atol is None:
        atol = ditParams['atol']

    # Make sure they are both inf or non-inf
    xinf = isinf(x)
    yinf = isinf(y)
    if not xinf == yinf:
        return False

    if xinf:
        # If they are inf, make sure the signs are the same.
        xgz = x > 0
        ygz = y > 0
        if (xgz and not ygz) or (not xgz and ygz):
            return False
    else:
        # Otherwise, make sure they are close.
        return abs(x-y) <= atol + rtol * abs(y)

    return True

close_docstring = \
"""Returns True if the scalars x and y are close.

The relative error rtol must be positive and << 1.0
The absolute error atol usually comes into play when y is very small or
zero; it says how small x must be also.

If rtol or atol are unspecified, they are taken from ditParams['rtol']
and ditParams['atol'].

"""

cython_doc = "\nNote: This version is cythonified.\n"

close__python.__doc__ = close_docstring
close__cython.__doc__ = close_docstring + cython_doc

# Load the cython function if possible
try: # pragma: no cover
    from ._close import close as close_
    close = close__cython
except ImportError: # pragma: no cover
    close = close__python

def allclose(x, y, rtol=None, atol=None):
    """Returns True if all components of x and y are close.

    The relative error rtol must be positive and << 1.0
    The absolute error atol usually comes into play for those elements of y that
    are very small or zero; it says how small x must be also.

    If rtol or atol are unspecified, they are taken from ditParams['rtol']
    and ditParams['atol'].

    """
    if rtol is None:
        rtol = ditParams['rtol']
    if atol is None:
        atol = ditParams['atol']

    return _allclose(x, y, rtol=rtol, atol=atol)

