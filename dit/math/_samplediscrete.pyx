#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Low-level functions for sampling from discrete distributions.

"""

import cython
cimport cython

import numpy as np
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
def sample(np.ndarray[np.float_t, ndim=1] pmf, np.float_t rand):
    """Returns a sample from a discrete distribution.

    Note: This version has no bells or whistles.

    Parameters
    ----------
    pmf : NumPy array, shape (n,)
        An array of floats representing the probability mass function. The
        events will be the indices of the list. The floats should represent
        probabilities (and not log probabilities).
    rand : float
        The sample is drawn using the passed number.

    Returns
    -------
    s : int
        The index of the sampled event.

    Notes
    -----
    This version is Cythonified.

    """
    cdef np.float_t total = 0
    cdef Py_ssize_t i
    for i in range(pmf.shape[0]):
        total += pmf[i]
        if rand < total:
            return i

@cython.boundscheck(False)
@cython.wraparound(False)
def samples(np.ndarray[np.float_t, ndim=1] pmf,
            np.ndarray[np.float_t, ndim=1] rands,
            np.ndarray[np.int_t, ndim=1] out = None):
    """Returns samples from a discrete distribution.

    Note: This version has no bells or whistles.

    Parameters
    ----------
    pmf : NumPy float array, shape (n,)
        An array of floats representing the probability mass function. The
        events will be the indices of the list. The floats should represent
        probabilities (and not log probabilities).
    rand : NumPy float array, shape (k,)
        The k samples are drawn using the passed in random numbers.  Each
        element should be between 0 and 1, inclusive.
    out : NumPy int array, shape (k,)
        The samples from `pmf`.  Each element will be filled with an integer i
        representing a sampling of the event with probability `pmf[i]`.

    Returns
    -------
    out : NumPy int array, shape (k,)

    Notes
    -----
    This version is Cythonified.

    """
    cdef np.float_t total, rand
    cdef Py_ssize_t i,j,n,m

    m = rands.shape[0]
    if out is None:
        out = np.empty(m, dtype=int)

    n = pmf.shape[0]
    for i in range(m):
        rand = rands[i]
        total = 0
        for j in range(n):
            total += pmf[j]
            if rand < total:
                out[i] = j
                break

    return out

