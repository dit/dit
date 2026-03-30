"""
Utilities related to testing.
"""

import numpy as np
from boltons.iterutils import pairwise
from hypothesis import assume
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import composite, floats, integers, lists, tuples

from ..distribution import Distribution
from .optimization import colon

__all__ = (
    "channel_pairs",
    "degraded_channel_pairs",
    "distributions",
    "distribution_structures",
    "markov_chains",
)


@composite
def channel_pairs(draw, input_size=(2, 4), output_y_size=(2, 4), output_z_size=(2, 4)):
    """
    Generate a pair of stochastic matrices (mu, kappa) with a common
    input alphabet.

    Parameters
    ----------
    draw : function
        Hypothesis sampling function.
    input_size : int or (int, int)
        Size (or range) of the input alphabet |S|.
    output_y_size : int or (int, int)
        Size (or range) of the kappa output alphabet |Y|.
    output_z_size : int or (int, int)
        Size (or range) of the mu output alphabet |Z|.

    Returns
    -------
    mu : ndarray, shape (|S|, |Z|)
        Channel P(Z|S).
    kappa : ndarray, shape (|S|, |Y|)
        Channel P(Y|S).
    """
    def _unpack(x):
        try:
            len(x)
            return x
        except TypeError:
            return (x, x)

    n_s = draw(integers(*_unpack(input_size)))
    n_y = draw(integers(*_unpack(output_y_size)))
    n_z = draw(integers(*_unpack(output_z_size)))

    mu_raw = draw(arrays(np.float64, shape=(n_s, n_z), elements=floats(0, 1)))
    kappa_raw = draw(arrays(np.float64, shape=(n_s, n_y), elements=floats(0, 1)))

    for row in mu_raw:
        assume(row.sum() > 0)
    for row in kappa_raw:
        assume(row.sum() > 0)

    mu = mu_raw / mu_raw.sum(axis=1, keepdims=True)
    kappa = kappa_raw / kappa_raw.sum(axis=1, keepdims=True)

    return mu, kappa


@composite
def degraded_channel_pairs(draw, input_size=(2, 4), output_z_size=(2, 4), output_y_size=(2, 4)):
    """
    Generate (mu, kappa) where kappa = lambda . mu for a random lambda,
    so that ``mu`` output-degrades to ``kappa`` (Blackwell order holds).

    Parameters
    ----------
    draw : function
        Hypothesis sampling function.
    input_size : int or (int, int)
        Size (or range) of the input alphabet |S|.
    output_z_size : int or (int, int)
        Size (or range) of the mu output alphabet |Z|.
    output_y_size : int or (int, int)
        Size (or range) of the kappa output alphabet |Y|.

    Returns
    -------
    mu : ndarray, shape (|S|, |Z|)
        The dominating channel.
    kappa : ndarray, shape (|S|, |Y|)
        The degraded channel, kappa = mu @ lam.
    """
    def _unpack(x):
        try:
            len(x)
            return x
        except TypeError:
            return (x, x)

    n_s = draw(integers(*_unpack(input_size)))
    n_z = draw(integers(*_unpack(output_z_size)))
    n_y = draw(integers(*_unpack(output_y_size)))

    mu_raw = draw(arrays(np.float64, shape=(n_s, n_z), elements=floats(0, 1)))
    lam_raw = draw(arrays(np.float64, shape=(n_z, n_y), elements=floats(0, 1)))

    for row in mu_raw:
        assume(row.sum() > 0)
    for row in lam_raw:
        assume(row.sum() > 0)

    mu = mu_raw / mu_raw.sum(axis=1, keepdims=True)
    lam = lam_raw / lam_raw.sum(axis=1, keepdims=True)
    kappa = mu @ lam

    return mu, kappa


@composite
def distributions(draw, alphabets=(2, 2, 2), nondegenerate=False, zeros=True):
    """
    Generate distributions for use with hypothesis.

    Parameters
    ----------
    draw : function
        A sampling function passed in by hypothesis.
    alphabets : int, tuple of ints, tuple of pairs of ints
        If an int, it is the length of the outcomes and each variable is assumed to be binary.
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
        alphabets = ((2, 2),) * alphabets

    alphabets = [int(draw(integers(*alpha))) for alpha in alphabets]

    lower = 1e-6 if not zeros else 0.0

    pmf = draw(arrays(np.float64, shape=alphabets, elements=floats(lower, 1)))

    assume(pmf.sum() > 0)

    if not zeros:
        assume(pmf.min() > 0)

    if nondegenerate:
        axes = set(range(len(alphabets)))
        for axis, _ in enumerate(alphabets):
            axes_to_sum = tuple(axes - {axis})
            if axes_to_sum:
                assume(np.all(pmf.sum(axis=tuple(axes - {axis})) > 1e-6))

    pmf /= pmf.sum()

    dist = Distribution.from_ndarray(pmf)
    dist.normalize()
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
    alphabet_ = int(draw(integers(*alphabet)))

    events = draw(lists(tuples(*[integers(0, alphabet_ - 1)] * size_), min_size=min_events, unique=True))

    # make sure no marginal is a constant
    for var in zip(*events, strict=True):
        assume(len(set(var)) > 1)

    if uniform:
        probs = [1 / len(events)] * len(events)
    else:
        probs = draw(tuples(*[floats(0, 1)] * len(events)))
        for prob in probs:
            assume(prob > 0)
        total = sum(probs)
        probs = [p / total for p in probs]

    dist = Distribution(events, probs)
    dist.normalize()
    return dist


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
        alphabets = ((2, 2),) * alphabets

    alphabets = [int(draw(integers(*alpha))) for alpha in alphabets]

    px = draw(arrays(np.float64, shape=alphabets[0], elements=floats(0, 1)))
    cds = [draw(arrays(np.float64, shape=(a, b), elements=floats(0, 1))) for a, b in pairwise(alphabets)]

    # assume things
    assume(px.sum() > 0)
    for cd in cds:
        for row in cd:
            assume(row.sum() > 0)

    px /= px.sum()

    # construct dist
    for cd in cds:
        cd /= cd.sum(axis=1, keepdims=True)
        slc = (np.newaxis,) * (len(px.shape) - 1) + (colon, colon)
        px = px[..., np.newaxis] * cd[slc]

    dist = Distribution.from_ndarray(px)
    dist.normalize()
    return dist
