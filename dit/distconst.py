#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np

from .npdist import Distribution
from .npjdist import JointDistribution

__all__ = [
    'modify_outcomes',
    'random_distribution',
    'random_joint_distribution',
    'uniform_distribution',
    'uniform_joint_distribution',
]

def modify_outcomes(dist, ctor):
    """
    Returns `dist` but with modified outcomes, after passing them to `ctor`.

    Parameters
    ----------
    dist : Distribution, JointDistribution
        The distribution to be modified.

    ctor : callable
        The constructor that receives an existing outcome and returns a new
        modified outcome.

    Returns
    -------
    d : Distribution, JointDistribution
        The modified distribution.

    Examples
    --------
    Increment the outcomes by 1.
    >>> d = dit.uniform_distribution(5)
    >>> d2 = dit.modify_outcomes(d, lambda x: x + 1)

    Convert the outcomes to strings.
    >>> d = dit.uniform_joint_distribution(3, ['0', '1'])
    >>> d2 = dit.modify_outcomes(d, lambda x: ''.join(x))

    """
    outcomes = map(ctor, dist.outcomes)
    d = dist.__class__(dist.pmf, outcomes, base=dist.get_base())
    return d

def random_distribution(n, prng=None):
    """
    Returns a random distribution over `n` outcomes.

    The distribution is sampled uniformly over the space of distributions.

    """
    import dit.math

    if prng is None:
        prng = dit.math.prng

    d = uniform_distribution(n)
    pmf = prng.dirichlet( np.ones(len(d)) )
    d.pmf = pmf
    return d

def random_joint_distribution(word_length, alphabet_size, prng=None):
    """
    Returns a uniform joint distribution.

    The distribution is sampled uniformly over the space of distributions.

    Parameters
    ----------
    word_length : int
        The length of the outcomes.

    alphabet_size : int, list
        The alphabet used to construct the outcomes of the distribution. If an
        integer, then the alphabet will consist of integers from 0 to k-1 where
        k is the alphabet size.  If a list, then the elements are used as the
        alphabet.

    Returns
    -------
    d : JointDistribution.
        A uniform joint distribution.

    """
    import dit.math

    if prng is None:
        prng = dit.math.prng

    d = uniform_joint_distribution(word_length, alphabet_size)
    pmf = prng.dirichlet( np.ones(len(d)) )
    d.pmf = pmf
    return d

def uniform_distribution(n):
    """
    Returns a uniform distribution over `n` outcomes.

    Parameters
    ----------
    n : int, list
        If an integer, then the outcomes are integers from 0 to n-1. If a list
        then the elements are treated as the outcomes.

    Returns
    -------
    d : Distribution
        A uniform distribution.

    """
    try:
        nOutcomes = len(n)
        outcomes = n
    except TypeError:
        nOutcomes = n
        outcomes = range(n)

    pmf = [1/nOutcomes] * nOutcomes
    d = Distribution(pmf, outcomes, base='linear')

    return d

def uniform_joint_distribution(word_length, alphabet_size):
    """
    Returns a uniform joint distribution.

    Parameters
    ----------
    word_length : int
        The length of the outcomes.

    alphabet_size : int, list
        The alphabet used to construct the outcomes of the distribution. If an
        integer, then the alphabet will consist of integers from 0 to k-1 where
        k is the alphabet size.  If a list, then the elements are used as the
        alphabet.

    Returns
    -------
    d : JointDistribution.
        A uniform joint distribution.

    """
    from itertools import product

    try:
        nSymbols = len(alphabet_size)
        alphabet = tuple(alphabet_size)
    except TypeError:
        nSymbols = alphabet_size
        alphabet = range(nSymbols)

    Z = nSymbols**word_length
    pmf = [1/Z] * Z
    outcomes = tuple( product(alphabet, repeat=word_length) )
    d = JointDistribution(pmf, outcomes, base='linear')

    return d
