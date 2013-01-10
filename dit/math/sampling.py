#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Functions related to sampling from discrete distributions.

"""

import operator
from itertools import izip

import numpy as np

import dit.exceptions

__all__ = (
    'sample',
)


def sample(dist, size=None, rand=None, prng=None):
    """Returns a sample from a discrete distribution.

    Parameters
    ----------
    dist : Dit distribution
        The distribuion from which the sample is drawn.
    size : int or None
        The number of samples to draw from the distribution. If `None`, then
        a single sample is returned.  Otherwise, a list of samples if returned.
    rand : float or NumPy array
        When `size` is `None`, `rand` should be a random number drawn uniformly
        from the interval [0,1]. When `size` is not `None`, then this should be
        a NumPy array of random numbers.  In either situation, if `rand` is
        `None`, then the random number(s) will be drawn from a pseudo random
        number generator.
    prng : random number generator
        A random number generator with a `rand' method that returns a random
        number between 0 and 1 when called with no arguments. If unspecified,
        then we use the random number generator on the distribution.

    Returns
    -------
    s : sample
        The sample drawn from the distribution.

    """
    if size is None:
        n = 1
    else:
        n = size

    if rand is None:
        if prng is None:
            prng = dist.prng
        try:
            rand = prng.rand(n)
        except AttributeError:
            msg = "The random number generator must support a `rand()' call."
            e = dit.exceptions.ditException(msg)
            raise(e)
    elif n != len(rand):
        msg = "The number of random numbers must equal n."
        e = dit.exceptions.ditException(msg)
        raise(e)

    indexes = _samples(dist._pmf, rand)
    events = dist._events
    s = [events[i] for i in indexes]
    if size is None:
        s = s[0]

    return s

def _sample_discrete__python(pmf, rand):
    """Returns a sample from a discrete distribution.

    Note: This version has no bells or whistles.

    Parameters
    ----------
    pmf : list of floats
        A list of floats representing the probability mass function. The events
        will be the indices of the list. The floats should represent
        probabilities (and not log probabilities).
    rand : float
        The sample is drawn using the passed number.

    Returns
    -------
    s : int
        The index of the sampled event.

    """
    total = 0
    for i,prob in enumerate(pmf):
        total += prob
        if rand < total:
            return i

def _samples_discrete__python(pmf, rands, out=None):
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
    None

    """
    L = rands.shape[0]
    if out is None:
        out = np.empty(L, dtype=int)

    n = pmf.shape[0]
    for i in range(L):
        rand = rands[i]
        total = 0
        for j in range(n):
            total += pmf[j]
            if rand < total:
                out[i] = j
                break

# Load the cython function if possible
try:
    from ._samplediscrete import sample as _sample_discrete__cython
    _sample = _sample_discrete__cython
except ImportError:
    _sample = _sample_discrete__python
try:
    from ._samplediscrete import samples as _samples_discrete__cython
    _samples = _samples_discrete__cython
except ImportError:
    _samples = _samples_discrete__python

