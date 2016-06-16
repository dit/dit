# -*- coding: utf-8 -*-

"""
Functions for pruning or expanding the sample space of a distribution.

This can be important when calculating meet and join random variables. It
is also important for the calculations of various PID quantities.

"""
from six.moves import map

from dit.samplespace import ScalarSampleSpace, SampleSpace, CartesianProduct

def pruned_samplespace(d, sample_space=None):
    """
    Returns a new distribution with pruned sample space.

    The pruning is such that zero probability outcomes are removed.

    Parameters
    ----------
    d : distribution
        The distribution used to create the pruned distribution.

    sample_space : set
        A list of outcomes with zero probability that should be kept in the
        sample space. If `None`, then all outcomes with zero probability
        will be removed.

    Returns
    -------
    pd : distribution
        The distribution with a pruned sample space.

    """
    if sample_space is None:
        sample_space = []

    keep = set(sample_space)
    outcomes = []
    pmf = []
    for o, p in d.zipped(mode='atoms'):
        if not d.ops.is_null_exact(p) or o in keep:
            outcomes.append(o)
            pmf.append(p)

    if d.is_joint():
        sample_space = SampleSpace(outcomes)
    else:
        sample_space = ScalarSampleSpace(outcomes)
    pd = d.__class__(outcomes, pmf,
                     sample_space=sample_space, base=d.get_base())
    return pd

def expanded_samplespace(d, alphabets=None, union=True):
    """
    Returns a new distribution with an expanded sample space.

    Expand the sample space so that it is the Cartesian product of the
    alphabets for each random variable. Note, only the effective alphabet of
    each random variable is used. So if one index in an outcome only has the
    value 1, then its alphabet is [1], and not [0,1] for example.

    Parameters
    ----------
    d : distribution
        The distribution used to create the pruned distribution.

    alphabets : list
        A list of alphabets, with length equal to the outcome length in `d`.
        Each alphabet specifies the alphabet to be used for a single index
        random variable. The sample space of the new distribution will be the
        Cartesian product of these alphabets.

    union : bool
        If True, then the alphabet for each random variable is unioned.
        The unioned alphabet is then used for each random variable.

    Returns
    -------
    ed : distribution
        The distribution with an expanded sample space.

    Notes
    -----
    The default constructor for Distribution will create a Cartesian product
    sample space if not sample space is provided.

    """
    joint = d.is_joint()

    if alphabets is None:
        # Note, we sort the alphabets now, so we are possibly changing the
        # order of the original sample space.
        alphabets = list(map(sorted, d.alphabet))
    elif joint and len(alphabets) != d.outcome_length():
        L = len(alphabets)
        raise Exception("You need to provide {0} alphabets".format(L))

    if joint and union:
        alphabet = set.union(*map(set, alphabets))
        alphabet = list(sorted(alphabet))
        alphabets = [alphabet] * len(alphabets)

    if joint:
        sample_space = CartesianProduct(alphabets, d._product)
    else:
        sample_space = ScalarSampleSpace(alphabets)

    ed = d.__class__(d.outcomes, d.pmf,
                     sample_space=sample_space, base=d.get_base())
    return ed
