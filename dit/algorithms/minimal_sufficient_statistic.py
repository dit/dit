# -*- coding: utf-8 -*-
"""
Functions for computing minimal sufficient statistics.
"""

from collections import defaultdict

from .lattice import dist_from_induced_sigalg, insert_join, insert_rv
from .prune_expand import pruned_samplespace
from ..helpers import flatten, parse_rvs, normalize_rvs
from ..math import sigma_algebra
from ..samplespace import CartesianProduct

__all__ = ['info_trim',
           'insert_mss',
           'mss',
           'mss_sigalg',
          ]

def partial_match(first, second, places):
    """
    Returns whether `second` is a marginal outcome at `places` of `first`.

    Parameters
    ----------
    first : iterable
        The un-marginalized outcome.
    second : iterable
        The smaller, marginalized outcome.
    places : list
        The locations of `second` in `first`.

    Returns
    -------
    match : bool
        Whether `first` and `second` match or not.

    """
    return tuple([first[i] for i in places]) == tuple(second)

def mss_sigalg(dist, rvs, about=None, rv_mode=None):
    """
    Construct the sigma algebra for the minimal sufficient statistic of `rvs`
    about `about`.

    Parameters
    ----------
    dist : Distribution
        The distribution which defines the base sigma-algebra.
    rvs : list
        A list of random variables to be compressed into a minimal sufficient
        statistic.
    about : list
        A list of random variables for which the minimal sufficient static will
        retain all information about.
    rv_mode : str, None
        Specifies how to interpret the elements of `rvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `rvs` are interpreted as random variable indices. If equal to 'names',
        the the elements are interpreted as random variable names. If `None`,
        then the value of `dist._rv_mode` is consulted.

    Returns
    -------
    mss_sa : frozenset of frozensets
        The induced sigma-algebra of the minimal sufficient statistic.

    Examples
    --------
    >>> d = Xor()
    >>> mss_sigalg(d, [0], [1, 2])
    frozenset({frozenset(),
               frozenset({'000', '011'}),
               frozenset({'101', '110'}),
               frozenset({'000', '011', '101', '110'})})

    """
    mapping = parse_rvs(dist, rvs, rv_mode=rv_mode)[1]

    partition = defaultdict(list)

    md, cds = dist.condition_on(rvs=about, crvs=rvs, rv_mode=rv_mode)

    for marg, cd in zip(md.outcomes, cds):
        matches = [o for o in dist.outcomes if partial_match(o, marg, mapping)]
        for c in partition.keys():
            if c.is_approx_equal(cd):
                partition[c].extend(matches)
                break
        else:
            partition[cd].extend(matches)

    mss_sa = sigma_algebra(map(frozenset, partition.values()))

    return mss_sa

def insert_mss(dist, idx, rvs, about=None, rv_mode=None):
    """
    Inserts the minimal sufficient statistic of `rvs` about `about` into `dist`
    at index `idx`.

    Parameters
    ----------
    dist : Distribution
        The distribution which defines the base sigma-algebra.
    idx : int
        The location in the distribution to insert the minimal sufficient
        statistic.
    rvs : list
        A list of random variables to be compressed into a minimal sufficient
        statistic.
    about : list
        A list of random variables for which the minimal sufficient static will
        retain all information about.
    rv_mode : str, None
        Specifies how to interpret the elements of `rvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `rvs` are interpreted as random variable indices. If equal to 'names',
        the the elements are interpreted as random variable names. If `None`,
        then the value of `dist._rv_mode` is consulted.

    Returns
    -------
    d : Distribution
        The distribution `dist` modified to contain the minimal sufficient
        statistic.

    Examples
    --------
    >>> d = Xor()
    >>> print(insert_mss(d, -1, [0], [1, 2]))
    Class:          Distribution
    Alphabet:       ('0', '1') for all rvs
    Base:           linear
    Outcome Class:  str
    Outcome Length: 4
    RV Names:       None
    x      p(x)
    0000   0.25
    0110   0.25
    1011   0.25
    1101   0.25

    """
    mss_sa = mss_sigalg(dist, rvs, about, rv_mode)
    new_dist = insert_rv(dist, idx, mss_sa)
    return pruned_samplespace(new_dist)

def mss(dist, rvs, about=None, rv_mode=None, int_outcomes=True):
    """
    Parameters
    ----------
    dist : Distribution
        The distribution which defines the base sigma-algebra.
    rvs : list
        A list of random variables to be compressed into a minimal sufficient
        statistic.
    about : list
        A list of random variables for which the minimal sufficient static will
        retain all information about.
    rv_mode : str, None
        Specifies how to interpret the elements of `rvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of `rvs`
        are interpreted as random variable indices. If equal to 'names', the the
        elements are interpreted as random variable names. If `None`, then the
        value of `dist._rv_mode` is consulted.
    int_outcomes : bool
        If `True`, then the outcomes of the minimal sufficient statistic are
        relabeled as integers instead of as the atoms of the induced
        sigma-algebra.

    Returns
    -------
    d : ScalarDistribution
        The distribution of the minimal sufficient statistic.

    Examples
    --------
    >>> d = Xor()
    >>> print(mss(d, [0], [1, 2]))
    Class:    ScalarDistribution
    Alphabet: (0, 1)
    Base:     linear
    x   p(x)
    0   0.5
    1   0.5

    """
    mss_sa = mss_sigalg(dist, rvs, about, rv_mode)
    d = dist_from_induced_sigalg(dist, mss_sa, int_outcomes)
    return d

def insert_joint_mss(dist, idx, rvs=None, rv_mode=None):
    """
    Returns a new distribution with the join of the minimal sufficient statistic
    of each random variable in `rvs` about all the other variables.

    Parameters
    ----------
    dist : Distribution
        The distribution contiaining the random variables from which the joint
        minimal sufficent statistic will be computed.
    idx : int
        The location in the distribution to insert the joint minimal sufficient
        statistic.
    rvs : list
        A list of random variables to be compressed into a joint minimal
        sufficient statistic.
    rv_mode : str, None
        Specifies how to interpret the elements of `rvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `rvs` are interpreted as random variable indices. If equal to 'names',
        the the elements are interpreted as random variable names. If `None`,
        then the value of `dist._rv_mode` is consulted.

    """
    rvs, _, rv_mode = normalize_rvs(dist, rvs, None, rv_mode)

    d = dist.copy()
    l1 = d.outcome_length()

    rvs = set( tuple(rv) for rv in rvs )

    for rv in rvs:
        about = list(flatten(rvs-set([rv])))
        d = insert_mss(d, -1, rvs=list(rv), about=about, rv_mode=rv_mode)

    l2 = d.outcome_length()

    idx = -1 if idx > l1 else idx
    d = insert_join(d, idx, [[i] for i in range(l1, l2)])
    delta = 0 if idx == -1 else 1
    d = d.marginalize([i + delta for i in range(l1, l2)])
    d = pruned_samplespace(d)

    if isinstance(dist._sample_space, CartesianProduct):
        d._sample_space = CartesianProduct(d.alphabet)

    return d

def info_trim(dist, rvs=None, rv_mode=None):
    """
    Returns a new distribution with the minimal sufficient statistics
    of each random variable in `rvs` about all the other variables.

    Parameters
    ----------
    dist : Distribution
        The distribution contiaining the random variables from which the joint
        minimal sufficent statistic will be computed.
    rvs : list
        A list of random variables to be compressed into minimal sufficient
        statistics.
    rv_mode : str, None
        Specifies how to interpret the elements of `rvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `rvs` are interpreted as random variable indices. If equal to 'names',
        the the elements are interpreted as random variable names. If `None`,
        then the value of `dist._rv_mode` is consulted.

    """
    rvs, _, rv_mode = normalize_rvs(dist, rvs, None, rv_mode)

    d = dist.copy()

    rvs2 = set( tuple(rv) for rv in rvs )

    for rv in rvs:
        about = list(flatten(rvs2-{tuple(rv)}))
        d = insert_mss(d, -1, rvs=tuple(rv), about=about, rv_mode=rv_mode)

    d = pruned_samplespace(d.marginalize(list(flatten(rvs))))

    if isinstance(dist._sample_space, CartesianProduct):
        d._sample_space = CartesianProduct(d.alphabet)

    return d
