"""
Utilities related to testing.
"""

from __future__ import division

from boltons.iterutils import pairwise

import numpy as np

from hypothesis import assume
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import composite, floats, integers, lists, tuples

from .. import Distribution

__all__ = ['distributions',
           'distribution_structures',
           'markov_chains',
          ]

@composite
def distributions(draw, alphabets=(2, 2, 2), nondegenerate=False):
    """
    Generate distributions for use with hypothesis.

    Parameters
    ----------
    draw : function
        A sampling function passed in by hypothesis.
    alphabets : int, tuple of ints, tuple of pairs of ints
        If an int, it is the length of the chain and each variable is assumed to be binary.
        If a tuple of ints, the ints are assumed to be the size of each variable. If a tuple
        of pairs of ints, each pair represents the min and max alphabet size of each variable.

    Returns
    -------
    dist : Distribution
        A distribution with variable sizes.
    """
    try:
        len(alphabets)
        try:
            len(alphabets[0])
        except TypeError:
            alphabets = tuple((alpha, alpha) for alpha in alphabets)
    except TypeError:
        alphabets = ((2, 2),)*alphabets

    alphabets = [draw(integers(*alpha)) for alpha in alphabets]

    pmf = draw(arrays(np.float, shape=alphabets, elements=floats(0, 1)))

    assume(pmf.sum() > 0)

    if nondegenerate:
        axes = set(range(len(alphabets)))
        for axis, _ in enumerate(alphabets):
            assume(np.sum(pmf.sum(axis=tuple(axes-set([axis]))) > 1e-6) > 1)

    pmf /= pmf.sum()

    dist = Distribution.from_ndarray(pmf)

    return dist


@composite
def distribution_structures(draw, size=(2, 4), alphabet=(2, 4), uniform=False, min_events=1):
    """
    A hypothesis strategy for generating distributions.

    Parameters
    ----------
    draw : function
        A sampling function passed in by hypothesis.
    size : int
        The size of outcomes desired. Defaults to a 3 or 4, randomly.
    alphabet : int
        The alphabet size for each variable. Defaults to 2, 3, or 4, randomly.
    uniform : bool
        Whether the probabilities should be uniform or random. Defaults to random.

    Returns
    -------
    dist : Distribution
        A random distribution.
    """
    try:
        len(size)
    except TypeError:
        size = (size, size)
    try:
        len(alphabet)
    except TypeError:
        alphabet = (alphabet, alphabet)

    size_ = draw(integers(*size))
    alphabet_ = draw(integers(*alphabet))

    events = draw(lists(tuples(*[integers(0, alphabet_ - 1)] * size_), min_size=min_events, unique=True))

    # make sure no marginal is a constant
    for var in zip(*events):
        assume(len(set(var)) > 1)

    if uniform:
        probs = [1 / len(events)] * len(events)
    else:
        probs = draw(tuples(*[floats(0, 1)] * len(events)))
        for prob in probs:
            assume(prob > 0)
        total = sum(probs)
        probs = [p / total for p in probs]

    return Distribution(events, probs)


colon = slice(None, None, None)

@composite
def markov_chains(draw, alphabets=((2, 4), (2, 4), (2, 4))):
    """
    Generate Markov chains for use with hypothesis.

    Parameters
    ----------
    draw : function
        A sampling function passed in by hypothesis.
    alphabets : int, tuple of ints, tuple of pairs of ints
        If an int, it is the length of the chain and each variable is assumed to be binary.
        If a tuple of ints, the ints are assumed to be the size of each variable. If a tuple
        of pairs of ints, each pair represents the min and max alphabet size of each variable.

    Returns
    -------
    dist : Distribution
        A Markov chain with variable sizes.
    """
    try:
        len(alphabets)
        try:
            len(alphabets[0])
        except TypeError:
            alphabets = tuple((alpha, alpha) for alpha in alphabets)
    except TypeError:
        alphabets = ((2, 2),)*alphabets

    alphabets = [draw(integers(*alpha)) for alpha in alphabets]

    px = draw(arrays(np.float, shape=alphabets[0], elements=floats(0, 1)))
    cds = [draw(arrays(np.float, shape=(a, b), elements=floats(0, 1))) for a, b in pairwise(alphabets)]

    # assume things
    assume(px.sum() > 0)
    for cd in cds:
        for row in cd:
            assume(row.sum() > 0)

    px /= px.sum()

    # construct dist
    for cd in cds:
        cd /= cd.sum(axis=1, keepdims=True)
        slc = (np.newaxis,)*(len(px.shape)-1) + (colon, colon)
        px = px[..., np.newaxis] * cd[slc]

    dist = Distribution.from_ndarray(px)

    return dist
