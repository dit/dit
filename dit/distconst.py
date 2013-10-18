#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
from six.moves import map, range, zip

from .exceptions import ditException
from .npdist import Distribution
from .npscalardist import ScalarDistribution
from .validate import validate_pmf

__all__ = [
    'mixture_distribution',
    'mixture_distribution2',
    'modify_outcomes',
    'random_scalar_distribution',
    'random_distribution',
    'uniform_distribution',
    'uniform_scalar_distribution',
]

def mixture_distribution(dists, weights, merge=False):
    """
    Create a mixture distribution: $\sum p_i d_i$

    Parameters
    ----------
    dists: [Distribution]
        List of distributions to mix.  Each distribution is assumed to have
        the same base and sample space.

    weights: [float]
        List of weights to use while mixing `dists`.  The weights are assumed
        to be probability represented in the base of the distributions.

    merge: bool
        If `True` then distributions will be mixed even if they do not share
        the same sample space. The idea is that each of the input distributions
        is reinterpreted on a common, merged sample space. If `False`, then
        an exception will be raised if incompatible distributions are mixed.

    Returns
    -------
    mix: Distribution
        The mixture distribution.

    Raises
    ------
    DitException
        Raised if there `dists` and `weights` have unequal lengths.
    InvalidNormalization
        Raised if the weights do not sum to unity.
    InvalidProbability
        Raised if the weights are not valid probabilities.
    IncompatibleOutcome
        Raised if the sample spaces for each distribution are not compatible.

    """
    weights = np.asarray(weights)
    if len(dists) != len(weights):
        msg = "Length of `dists` and `weights` must be equal."
        raise ditException(msg)

    ops = dists[0].ops
    validate_pmf(weights, ops)

    if merge:
        vals = lambda o: [(ops.mult(w, d[o]) if o in d else 0)
                          for w, d in zip(weights, dists)]
    else:
        vals = lambda o: [ops.mult(w, d[o])
                          for w, d in zip(weights, dists)]

    outcomes = set().union(*[ d.outcomes for d in dists ])
    pmf = [ops.add_reduce(np.array(vals(o))) for o in outcomes]
    mix = Distribution(tuple(outcomes), pmf, base=ops.get_base())
    return mix

def mixture_distribution2(dists, weights):
    """
    Create a mixture distribution: $\sum p_i d_i$

    This version assumes that the pmf for each distribution is of the same
    form, and as a result, will be faster than `mixture_distribution`.
    Explicitly, it assumes that the sample space is ordered exactly the same
    for each distribution and that the outcomes currently represented in the
    pmf are the same as well. Using it in any other case will result in
    incorrect output or an exception.

    Parameters
    ----------
    dists: [Distribution]
        List of distributions to mix.  Each distribution is assumed to have
        the same base and sample space.

    weights: [float]
        List of weights to use while mixing `dists`.  The weights are assumed
        to be probability represented in the base of the distributions.

    Returns
    -------
    mix: Distribution
        The mixture distribution.

    Raises
    ------
    DitException
        Raised if there `dists` and `weights` have unequal lengths.
    InvalidNormalization
        Raised if the weights do not sum to unity.
    InvalidProbability
        Raised if the weights are not valid probabilities.
    IncompatibleDistribution
        Raised if the sample spaces for each distribution are not compatible.

    """
    weights = np.asarray(weights)
    if len(dists) != len(weights):
        msg = "Length of `dists` and `weights` must be equal."
        raise ditException(msg)

    # Also just quickly make sure that the pmfs have the same length. In
    # general, NumPy should give a value error complaining that it cannot
    # broadcast the smaller array. But if a pmf has length 1, then it can
    # be broadcast. This would make it harder to detect errors.
    shapes = set([dist.pmf.shape for dist in dists])
    if len(shapes) != 1:
        raise ValueError('All pmfs must have the same length.')

    ops = dists[0].ops
    validate_pmf(weights, ops)

    mix = dists[0].copy()
    ops.mult_inplace(mix.pmf, weights[0])
    for dist, weight in zip(dists[1:], weights[1:]):
        ops.add_inplace(mix.pmf, ops.mult(dist.pmf, weight))
    return mix

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
    outcomes = tuple(map(ctor, dist.outcomes))
    d = dist.__class__(outcomes, dist.pmf, base=dist.get_base())
    return d

def random_scalar_distribution(n, alpha=None, prng=None):
    """
    Returns a random scalar distribution over `n` outcomes.

    The distribution is sampled uniformly over the space of distributions on
    the `n`-simplex. If `alpha` is not `None`, then the distribution is
    sampled from the Dirichlet distribution with parameter `alpha`.

    Parameters
    ----------
    n : int | list
        The number of outcomes, or a list containing the outcomes.

    alpha : list | None
        The concentration parameters defining that the Dirichlet distribution
        used to draw the random distribution. If `None`, then each of the
        concentration parameters are set equal to 1.

    """
    import dit.math

    if prng is None:
        prng = dit.math.prng

    try:
        nOutcomes = len(n)
    except TypeError:
        nOutcomes = n

    d = uniform_scalar_distribution(nOutcomes)
    if alpha is None:
        alpha = np.ones(len(d))
    elif len(alpha) != nOutcomes:
        raise ditException('Number of concentration parameters must be `n`.')

    pmf = prng.dirichlet(alpha)
    d.pmf = pmf
    return d

def random_distribution(outcome_length, alphabet_size, alpha=None, prng=None):
    """
    Returns a uniform distribution.

    The distribution is sampled uniformly over the space of distributions on
    the `n`-simplex, where `n` is equal to `alphabet_size**outcome_length`.
    If `alpha` is not `None`, then the distribution is sampled from the
    Dirichlet distribution with parameter `alpha`.

    Parameters
    ----------
    outcome_length : int
        The length of the outcomes.

    alphabet_size : int, list
        The alphabet used to construct the outcomes of the distribution. If an
        integer, then the alphabet will consist of integers from 0 to k-1 where
        k is the alphabet size.  If a list, then the elements are used as the
        alphabet.

    alpha : list | None
        The concentration parameters defining that the Dirichlet distribution
        used to draw the random distribution. If `None`, then each of the
        concentration parameters are set equal to 1.

    Returns
    -------
    d : Distribution.
        A uniform sampled distribution.

    """
    import dit.math

    if prng is None:
        prng = dit.math.prng

    d = uniform_distribution(outcome_length, alphabet_size)

    if alpha is None:
        alpha = np.ones(len(d))
    elif len(alpha) != len(d):
        raise ditException('Invalid number of concentration parameters.')

    pmf = prng.dirichlet(alpha)
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
        outcomes = tuple(range(n))

    pmf = [1/nOutcomes] * nOutcomes
    d = ScalarDistribution(outcomes, pmf, base='linear')

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
            alphabet = [alphabet[0]] * outcome_length
        elif len(alphabet) != outcome_length:
            raise TypeError("outcome_length does not match number of rvs.")
    else:
        # Build the standard alphabet.
        alphabet = [tuple(range(alphabet_size))] * outcome_length

    try:
        Z = np.prod( list(map(len, alphabet)) )
    except TypeError:
        raise TypeError("alphabet_size must be an int or list of lists.")

    pmf = [1/Z] * Z
    outcomes = tuple( product(*alphabet) )
    d = Distribution(outcomes, pmf, base='linear')

    return d
