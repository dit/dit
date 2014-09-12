#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Specialized distribution constructors.

"""

from __future__ import division

import numpy as np
from six.moves import map, range, zip # pylint: disable=redefined-builtin

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
    'simplex_grid',
    'uniform_distribution',
    'uniform_scalar_distribution',
    'insert_frv',
    'BooleanFunctions'
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

    outcomes = set().union(*[d.outcomes for d in dists])
    pmf = [ops.add_reduce(np.array(vals(o))) for o in outcomes]
    mix = dists[0].__class__(tuple(outcomes), pmf, base=ops.get_base())
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
    Returns a random distribution drawn uniformly from the simplex.

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

def simplex_grid(length, depth, base=2, using=None, inplace=False):
    """Returns a generator over distributions, determined by a grid.

    The grid is "triangular" in Euclidean space.

    Parameters
    ----------
    length : int
        The number of elements in each distribution. The dimensionality
        of the simplex is length-1.
    depth : int
        Controls the density of the grid.  The number of points on the simplex
        is given by:
            (base**depth + length - 1)! / (base**depth)! / (length-1)!
        At each depth, we exponentially increase the number of points.
    base : int
        The rate at which we divide probabilities.
    using : None or distribution
        If not `None`, then each yielded distribution is a copy of `using`
        with its pmf set appropriately.  If `using` is equal to the tuple
        type, then only tuples are yielded.
    inplace : bool
        If `True`, then each yielded distribution is the same Python object,
        but with a new probability mass function. If `False`, then each yielded
        distribution is a unique Python object and can be safely stored for
        other calculations after the generator has finished. When `using` is
        equal to `tuple`, this option has no effect---all yielded tuples are
        distinct.

    Examples
    --------
    >>> list(dit.simplex_grid(2, 2, using=tuple))
    [(0.0, 1.0), (0.25, 0.75), (0.5, 0.5), (0.75, 0.25), (1.0, 0.0)]

    """
    from dit.math.combinatorics import slots

    gen = slots(int(base)**int(depth), int(length), normalized=True)

    if using == tuple:
        for pmf in gen:
            yield pmf

    else:
        if using is None:
            using = random_scalar_distribution(length)
        elif length != len(using.pmf):
            raise Exception('`length` must match the length of pmf')

        if inplace:
            d = using
            for pmf in gen:
                d.pmf[:] = pmf
                yield d
        else:
            for pmf in gen:
                d = using.copy()
                d.pmf[:] = pmf
                yield d

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
        Z = np.prod(list(map(len, alphabet)))
    except TypeError:
        raise TypeError("alphabet_size must be an int or list of lists.")

    pmf = [1/Z] * Z
    outcomes = tuple(product(*alphabet))
    d = Distribution(outcomes, pmf, base='linear')

    return d

def insert_frv(d, func, index=-1):
    """
    Returns a new distribution with an added random variable at index `idx`.

    The new random variable must be a function of the other random variables.
    By this, we mean that the entropy of the new random variable conditioned
    on the original random variables should be zero.

    Parameters
    ----------
    dist : Distribution
        The distribution used to construct the new distribution.
    func : callable | list of callable
        A function which takes a single argument---the value of the previous
        random variables---and returns a new random variable. Note, the return
        value will be added to the outcome using `__add__`, and so it should be
        a hashable, orderable sequence (as every outcome must be). If a list of
        callables is provided, then multiple random variables are added
        simultaneously and will appear in the same order as the list.
    idx : int
        The index at which to insert the random variable. A value of -1 is
        will append the random variable to the end.

    Returns
    -------
    d : Distribution
        The new distribution.

    Examples
    --------
    >>> d = dit.Distribution(['00', '01', '10', '11'], [1/4]*4)
    >>> def xor(outcome):
    ...    return str(int(outcome[0] != outcome[1]))
    ...
    >>> d2 = dit.insert_frv(d, xor)
    >>> d.outcomes
    ('000', '011', '101', '110')

    """
    try:
        func[0]
    except TypeError:
        funcs = [func]
    else:
        funcs = func

    partial_outcomes = [map(func, d.outcomes) for func in funcs]
    # Now "flatten" the new contributions.
    partial_outcomes = [d._outcome_ctor([o for o_list in outcome for o in o_list])
                        for outcome in zip(*partial_outcomes)]
    outcomes = [old + new for old, new in zip(d.outcomes, partial_outcomes)]
    d2 = Distribution(outcomes, d.pmf.copy(), base=d.get_base())
    return d2

class BooleanFunctions(object):
    """
    Helper class for building a new random variable.

    The new random variable is the output of a boolean function.

    """
    def __init__(self, d):
        """
        Initialize the boolean function creator.

        Parameters
        ----------
        d : Distribution
            The distribution used to create the new random variables. Outcomes
            are assumed to be strings of '0' and '1', or tuples of 0 and 1.
            The returned functions act appropriately.

        Examples
        --------
        >>> d = dit.Distribution(['00', '01', '10', '11'], [1/4]*4)
        >>> bf = dit.BooleanFunctions(d)
        >>> d = dit.insert_frv(d, bf.xor([0,1]))
        >>> d = dit.insert_frv(d, bf.xor([1,2]))
        >>> d.outcomes
        ('0000', '0110', '1011', '1101')

        """
        try:
            d.outcomes[0] + ''
        except TypeError:
            is_int = True
        else:
            is_int = False

        self.is_int = is_int
        self.L = d.outcome_length()

    def xor(self, indices):
        """
        Returns a callable which returns the logical XOR of the given indices.

        Parameters
        ----------
        indices : list
            A list of two indices used to take the XOR.

        Returns
        -------
        func : function
            A callable implementing the XOR function. It receives a single
            argument, the outcome and returns an outcome for the calculation.

        Examples
        --------
        >>> d = dit.Distribution(['00', '01', '10', '11'], [1/4]*4)
        >>> bf = dit.BooleanFunctions(d)
        >>> d = dit.insert_frv(d, bf.xor([0,1]))
        >>> d.outcomes
        ('000', '011', '101', '110')

        """
        if self.is_int:
            def func(outcome):
                result = outcome[indices[0]] != outcome[indices[1]]
                return (int(result),)
        else:
            def func(outcome):
                result = outcome[indices[0]] != outcome[indices[1]]
                return str(int(result))

        return func

    def from_hexes(self, hexes):
        """
        Returns a callable implementing a boolean function on up to 3-bits.

        The original outcomes are represented in base-16 as one of the letters
        in '0123456789ABCDEF' (not case sensitive). Then, each boolean function
        is a specification of the outcomes for which it be should true. The
        random variable will be false for the complement of this set---so this
        function assumes full support. For example, if the random variable is a
        function of 3-bits and should be true only for '010' or '111', then
        `hexes` should be: '27'. This nicely handles 1- and 2-bit inputs in a
        similar fashion.

        Parameters
        ----------
        hexes : str
            A string of base-16 characters representing the (up to) 4-bit
            outcomes for which the random variable should be true.

        Returns
        -------
        func : function
            A callable implementing the desired function. It receives a single
            argument, the outcome and returns an outcome for the calculation.

        Examples
        --------
        >>> outcomes = ['000', '001', '010', '011', '100', '101', '110', '111']
        >>> pmf = [1/8] * 8
        >>> d = dit.Distribution(outcomes, pmf)
        >>> bf = dit.BooleanFunctions(d)
        >>> d = dit.insert_frv(d, bf.from_hexes('27'))
        >>> d.outcomes
        ('0000', '0010', '0101', '0110', '1000', '1010', '1100', '1111')

        """
        base = 16
        template = "{0:0{1}b}"
        outcomes = [template.format(int(h, base), self.L) for h in hexes]
        if self.is_int:
            outcomes = [tuple(map(int, o)) for o in outcomes]
        outcomes = set(outcomes)

        if self.is_int:
            def func(outcome):
                result = outcome in outcomes
                return (int(result),)
        else:
            def func(outcome):
                result = outcome in outcomes
                return str(int(result))
        return func
