"""
A lower bound on the two-way secret key agreement rate from iterated public
discarding.
"""

import numpy as np
from scipy.optimize import minimize

from ...utils import unitful

__all__ = ("iterated_discarding_skar",)


def _joint_array(dist, rvs, crvs):
    """
    The joint distribution of (X, Y, Z) as a dense array p[x, y, z].
    """
    d = dist.copy(base="linear").coalesce([rvs[0], rvs[1], crvs])
    alphabets = d.alphabet
    p = np.zeros([len(alphabet) for alphabet in alphabets])
    for outcome, prob in d.zipped():
        p[tuple(alphabet.index(o) for alphabet, o in zip(alphabets, outcome, strict=True))] += prob
    return p


def _mutual_information(pab):
    """
    I[A : B] in bits of a two-dimensional joint distribution.
    """
    pa = pab.sum(axis=1, keepdims=True)
    pb = pab.sum(axis=0, keepdims=True)
    mask = pab > 0
    return float((pab[mask] * np.log2(pab[mask] / (pa @ pb)[mask])).sum())


def _rate(p, keeps, alice_first):
    """
    The key rate of one iterated discarding protocol.

    Parameters
    ----------
    p : np.ndarray
        The joint distribution p[x, y, z].
    keeps : list of np.ndarray
        The keep probabilities of each round, indexed by the speaker's symbol.
    alice_first : bool
        Whether Alice speaks in the first round.

    Returns
    -------
    rate : float
        The largest suffix sum of the per-round terms, including the final key
        message.
    """
    terms = []
    mass = 1.0
    for i, keep in enumerate(keeps):
        alice = (i % 2 == 0) == alice_first
        kept = p * (keep[:, None, None] if alice else keep[None, :, None])
        q = np.stack([kept, p - kept])
        listener = q.sum(axis=(1, 3)) if alice else q.sum(axis=(2, 3))
        terms.append(mass * (_mutual_information(listener) - _mutual_information(q.sum(axis=(1, 2)))))
        survival = kept.sum()
        if survival <= 1e-15:
            mass = 0.0
            break
        mass *= survival
        p = kept / survival
    if mass > 0:
        alice_key = _mutual_information(p.sum(axis=2)) - _mutual_information(p.sum(axis=1))
        bob_key = _mutual_information(p.sum(axis=2)) - _mutual_information(p.sum(axis=0))
        terms.append(mass * max(alice_key, bob_key, 0.0))
    return max([0.0] + [sum(terms[i:]) for i in range(len(terms))])


@unitful
def iterated_discarding_skar(dist, rvs, crvs, rounds=8, niter=None, rng=None):
    """
    Compute a lower bound on the two-way secret key agreement rate achieved by
    iterated public discarding.

    Alice and Bob alternate rounds. In each round the speaker publicly
    announces, independently for each position, whether to keep it; the keep
    probability depends only on the speaker's own symbol. Discarded positions
    are abandoned, and after the final round whichever party fares better sends
    their (remaining) variable as a one-way key. This post-selection is the
    single-letter form of advantage distillation :cite:`maurer1993secret`, and
    is a special case of the interactive bound
    :py:func:`interactive_intrinsic_mutual_information` in which every
    auxiliary variable is a keep/discard flag. Because each round needs only
    one keep probability per symbol, many more rounds are tractable than for
    the general interactive bound.

    Parameters
    ----------
    dist : Distribution
        The distribution of interest.
    rvs : iterable of iterables, len(rvs) == 2
        The indices of the random variables agreeing upon a secret key.
    crvs : iterable
        The indices of the eavesdropper.
    rounds : int
        The number of discarding rounds, not counting the final key message.
        Defaults to 8.
    niter : int, None
        The number of random restarts per choice of first speaker. Defaults to
        10.
    rng : np.random.Generator, int, None
        The source of randomness for the restarts.

    Returns
    -------
    skar : float
        The lower bound on the two-way secret key agreement rate.
    """
    niter = 10 if niter is None else niter
    rng = np.random.default_rng(rng)
    p = _joint_array(dist, rvs, crvs)
    sizes = p.shape[:2]

    if rounds == 0:
        return _rate(p, [], True)

    best = 0.0
    for alice_first in (True, False):
        speakers = [sizes[(i % 2) != alice_first] for i in range(rounds)]
        splits = np.cumsum(speakers)[:-1]

        def objective(theta, alice_first=alice_first, splits=splits):
            keeps = [1 / (1 + np.exp(-np.clip(t, -40, 40))) for t in np.split(theta, splits)]
            return -_rate(p, keeps, alice_first)

        starts = [np.full(sum(speakers), 40.0)] + [rng.normal(0, 3, sum(speakers)) for _ in range(niter)]
        for start in starts:
            result = minimize(objective, start, method="L-BFGS-B")
            best = max(best, -result.fun)

    return best
