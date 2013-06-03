#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np

from .npdist import Distribution
from .npscalardist import ScalarDistribution

__all__ = [
    'modify_outcomes',
    'random_scalar_distribution',
    'random_distribution',
    'uniform_distribution',
    'uniform_scalar_distribution',
]

def modify_outcomes(dist, ctor):
    """
    Returns `dist` but with modified outcomes, after passing them to `ctor`.

    Parameters
    ----------
    dist : Distribution, ScalarDistribution
        The distribution to be modified.

    ctor : callable
        The constructor that receives an existing outcome and returns a new
        modified outcome.

    Returns
    -------
    d : Distribution, ScalarDistribution
        The modified distribution.

    Examples
    --------
    Convert joint tuple outcomes to strings.
    >>> d = dit.uniform_distribution(3, ['0', '1'])
    >>> d2 = dit.modify_outcomes(d, lambda x: ''.join(x))

    Increment scalar outcomes by 1.
    >>> d = dit.uniform_scalar_distribution(5)
    >>> d2 = dit.modify_outcomes(d, lambda x: x + 1)
    """
    outcomes = map(ctor, dist.outcomes)
    d = dist.__class__(dist.pmf, outcomes, base=dist.get_base())
    return d

def random_scalar_distribution(n, prng=None):
    """
    Returns a random scalar distribution over `n` outcomes.

    The distribution is sampled uniformly over the space of distributions.

    """
    import dit.math

    if prng is None:
        prng = dit.math.prng

    d = uniform_distribution(n)
    pmf = prng.dirichlet( np.ones(len(d)) )
    d.pmf = pmf
    return d

def random_distribution(outcome_length, alphabet_size, prng=None):
    """
    Returns a uniform distribution.

    The distribution is sampled uniformly over the space of distributions.

    Parameters
    ----------
    outcome_length : int
        The length of the outcomes.

    alphabet_size : int, list
        The alphabet used to construct the outcomes of the distribution. If an
        integer, then the alphabet will consist of integers from 0 to k-1 where
        k is the alphabet size.  If a list, then the elements are used as the
        alphabet.

    Returns
    -------
    d : Distribution.
        A uniform sampled distribution.

    """
    import dit.math

    if prng is None:
        prng = dit.math.prng

    d = uniform_distribution(outcome_length, alphabet_size)
    pmf = prng.dirichlet( np.ones(len(d)) )
    d.pmf = pmf
    return d

def uniform_scalar_distribution(n):
    """
    Returns a uniform scalar distribution over `n` outcomes.

    Parameters
    ----------
    n : int, list
        If an integer, then the outcomes are integers from 0 to n-1. If a list
        then the elements are treated as the outcomes.

    Returns
    -------
    d : ScalarDistribution
        A uniform scalar distribution.

    """
    try:
        nOutcomes = len(n)
        outcomes = n
    except TypeError:
        nOutcomes = n
        outcomes = range(n)

    pmf = [1/nOutcomes] * nOutcomes
    d = ScalarDistribution(pmf, outcomes, base='linear')

    return d

def uniform_distribution(outcome_length, alphabet_size):
    """
    Returns a uniform distribution.

    Parameters
    ----------
    outcome_length : int
        The length of the outcomes.

    alphabet_size : int, list of lists
        The alphabets used to construct the outcomes of the distribution. If an
        integer, then the alphabet for each random variable will be the same,
        consisting of integers from 0 to k-1 where k is the alphabet size.
        If a list, then the elements are used as the alphabet for each random
        variable.  If the list has a single element, then it will be used
        as the alphabet for each random variable.

    Returns
    -------
    d : Distribution.
        A uniform distribution.

    Examples
    --------
    Each random variable has the same standardized alphabet: [0,1]
    >>> d = dit.uniform_distribution(2, 2)

    Each random variable has its own alphabet.
    >>> d = dit.uniform_distribution(2, [[0,1],[1,2]])

    Both random variables have ['H','T'] as an alphabet.
    >>> d = dit.uniform_distribution(2, [['H','T']])

    """
    from itertools import product

    try:
        int(alphabet_size)
    except TypeError:
        # Assume it is a list of lists.
        alphabet = alphabet_size

        # Autoextend if only one alphabet is provided.
        if len(alphabet) == 1:
            alphabet = [alphabet[0]] * word_length
        elif len(alphabet) != word_length:
            raise TypeError("word_length does not match number of rvs.")
    else:
        # Build the standard alphabet.
        alphabet = [range(alphabet_size)] * word_length

    try:
        Z = np.prod(map(len, alphabet))
    except TypeError:
        raise TypeError("alphabet_size must be an int or list of lists.")

    pmf = [1/Z] * Z
    outcomes = tuple( product(*alphabet) )
    d = Distribution(pmf, outcomes, base='linear')

    return d
