"""
Measures of multivariate information content from Finn & Lizier (2020),
"Generalised Measures of Multivariate Information Content",
Entropy 22(2), 216. https://doi.org/10.3390/e22020216

These measures decompose the joint entropy of a set of variables into
non-negative components using pointwise max/min of marginal surprisals.
"""

import numpy as np

from ..helpers import normalize_rvs
from ..shannon import entropy as shannon_entropy
from ..utils import flatten, unitful

__all__ = (
    "intersection_entropy",
    "synergistic_entropy",
    "union_entropy",
    "unique_entropy",
)


def _pointwise_surprisals(dist, rvs):
    """
    For each outcome and each rv group, compute -log2(p(marginal)).

    Parameters
    ----------
    dist : Distribution
    rvs : list of lists
        Random variable groups.

    Returns
    -------
    outcomes : list
        The outcomes with positive probability.
    probs : list of float
        The joint probability for each outcome.
    surprisals : list of list of float
        ``surprisals[i][j]`` is the marginal surprisal of rv group *j*
        for the *i*-th outcome.
    """
    marginals = [dist.marginal(list(flatten(rv))) for rv in rvs]

    outcomes = []
    probs = []
    surprisals = []
    for outcome in dist.outcomes:
        p = dist[outcome]
        if p <= 0:
            continue
        outcomes.append(outcome)
        probs.append(p)

        row = []
        for rv, marg in zip(rvs, marginals):
            idx = list(flatten(rv))
            marg_outcome = tuple(outcome[i] for i in idx)
            if len(marg_outcome) == 1:
                marg_outcome = marg_outcome[0]
            pm = marg[marg_outcome]
            row.append(-np.log2(pm) if pm > 0 else np.inf)
        surprisals.append(row)

    return outcomes, probs, surprisals


@unitful
def union_entropy(dist, rvs=None, crvs=None):
    """
    Compute the union entropy H(X_1 t X_2 t ... t X_n).

    The expected surprise of the most surprising marginal realisation:

        H(X_1 t ... t X_n) = E[ max(h(x_1), ..., h(x_n)) ]

    Parameters
    ----------
    dist : Distribution
        The distribution from which the union entropy is calculated.
    rvs : list, None
        The random variable groups. If None, each variable is its own group.
    crvs : list, None
        Variables to condition on (not supported; must be None or empty).

    Returns
    -------
    Hu : float
        The union entropy.
    """
    rvs, crvs = normalize_rvs(dist, rvs, crvs)
    if crvs:
        raise NotImplementedError("Conditioning is not supported for union_entropy")
    _, probs, surprisals = _pointwise_surprisals(dist, rvs)
    return sum(p * max(row) for p, row in zip(probs, surprisals))


@unitful
def intersection_entropy(dist, rvs=None, crvs=None):
    """
    Compute the intersection entropy H(X_1 u X_2 u ... u X_n).

    The expected surprise of the least surprising marginal realisation:

        H(X_1 u ... u X_n) = E[ min(h(x_1), ..., h(x_n)) ]

    Parameters
    ----------
    dist : Distribution
        The distribution from which the intersection entropy is calculated.
    rvs : list, None
        The random variable groups. If None, each variable is its own group.
    crvs : list, None
        Variables to condition on (not supported; must be None or empty).

    Returns
    -------
    Hi : float
        The intersection entropy.
    """
    rvs, crvs = normalize_rvs(dist, rvs, crvs)
    if crvs:
        raise NotImplementedError("Conditioning is not supported for intersection_entropy")
    _, probs, surprisals = _pointwise_surprisals(dist, rvs)
    return sum(p * min(row) for p, row in zip(probs, surprisals))


@unitful
def synergistic_entropy(dist, rvs=None, crvs=None):
    """
    Compute the synergistic entropy H(X_1 + X_2 + ... + X_n).

    How much more information the joint distribution provides beyond
    what the marginals can share:

        H(X_1 + ... + X_n) = H(X_1, ..., X_n) - H(X_1 t ... t X_n)

    Equivalently: E[ h(x_1, ..., x_n) - max(h(x_1), ..., h(x_n)) ]

    Parameters
    ----------
    dist : Distribution
        The distribution from which the synergistic entropy is calculated.
    rvs : list, None
        The random variable groups. If None, each variable is its own group.
    crvs : list, None
        Variables to condition on (not supported; must be None or empty).

    Returns
    -------
    Hs : float
        The synergistic entropy.
    """
    rvs, crvs = normalize_rvs(dist, rvs, crvs)
    if crvs:
        raise NotImplementedError("Conditioning is not supported for synergistic_entropy")

    all_idx = list(set(flatten(flatten(rvs))))
    H_joint = shannon_entropy(dist, all_idx)

    _, probs, surprisals = _pointwise_surprisals(dist, rvs)
    H_union = sum(p * max(row) for p, row in zip(probs, surprisals))

    return H_joint - H_union


@unitful
def unique_entropy(dist, rvs=None, crvs=None):
    """
    Compute the unique entropy H(X_1 \\ X_2).

    How much more information the first rv group provides relative to
    the second, on average:

        H(X \\ Y) = H(X t Y) - H(Y) = E[ max(h(x) - h(y), 0) ]

    Parameters
    ----------
    dist : Distribution
        The distribution from which the unique entropy is calculated.
    rvs : list, None
        Exactly two random variable groups ``[rv_a, rv_b]``.
        Returns ``H(rv_a \\ rv_b)``.
    crvs : list, None
        Variables to condition on (not supported; must be None or empty).

    Returns
    -------
    Hu : float
        The unique entropy.

    Raises
    ------
    ValueError
        If ``rvs`` does not contain exactly two groups.
    """
    rvs, crvs = normalize_rvs(dist, rvs, crvs)
    if crvs:
        raise NotImplementedError("Conditioning is not supported for unique_entropy")
    if len(rvs) != 2:
        raise ValueError(f"unique_entropy requires exactly 2 rv groups, got {len(rvs)}")

    _, probs, surprisals = _pointwise_surprisals(dist, rvs)
    return sum(p * max(row[0] - row[1], 0) for p, row in zip(probs, surprisals))
