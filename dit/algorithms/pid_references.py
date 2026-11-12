"""
Reference distributions of two sources and a target.

Several partial information decompositions of two sources :math:`X_0, X_1` and a
target :math:`Y` are fixed by the co-information of reference distributions that share
some of the marginals of :math:`p` :cite:`vansanten2026between`:

* :math:`Q_{\\mathrm{CI}} = p(y) p(x_0|y) p(x_1|y)` keeps the source-target marginals and
  makes the sources conditionally independent given the target, so its co-information is
  :math:`I_{Q_{\\mathrm{CI}}}(X_0 : X_1) \\geq 0`;
* :math:`P^{(2)}`, the maximum entropy distribution with the three pairwise marginals of
  :math:`p`, has the co-information :math:`C^{(2)} = C + I^{(3)}`, where
  :math:`I^{(3)} = H(P^{(2)}) - H(p)` is the third-order connected information.

Both lie among the distributions with the source-target marginals of :math:`p`, so their
co-informations also bound the redundancy of BROJA from below.
"""

from dataclasses import dataclass
from itertools import product

import numpy as np

from ..distribution import Distribution
from ..multivariate import coinformation
from ..shannon import entropy
from .distribution_optimizers import maxent_dist

__all__ = (
    "ReferenceCoinformations",
    "conditionally_independent",
    "pairwise_maxent",
    "reference_coinformations",
    "source_target_bounds",
)


def conditionally_independent(dist, sources, target):
    """
    The maximum entropy distribution with the source-target marginals of `dist`,
    q = p(t) prod_i p(x_i | t), over integer-coded sources and target.

    Parameters
    ----------
    dist : Distribution
        The distribution.
    sources : iterable of iterables
        The source variables.
    target : iterable
        The target variable.

    Returns
    -------
    q : Distribution
        The distribution of the sources (variables 0..n-1) and the target (variable n).
    """
    d = dist.coalesce(list(sources) + [target])
    d.make_dense()
    shape = [len(alphabet) for alphabet in d.alphabet]
    p = np.asarray(d.pmf, dtype=float).reshape(shape)
    n = len(shape) - 1
    p_t = p.sum(axis=tuple(range(n)))
    q = p_t.reshape([1] * n + [shape[-1]])
    for i in range(n):
        p_it = p.sum(axis=tuple(j for j in range(n) if j != i))
        conditional = np.divide(p_it, p_t, out=np.zeros_like(p_it), where=p_t > 0)
        q = q * conditional.reshape([shape[i] if j == i else 1 for j in range(n)] + [shape[-1]])
    outcomes = list(product(*(range(size) for size in shape)))
    return Distribution(outcomes, q.ravel())


def pairwise_maxent(dist, maxiter=None):
    """
    The maximum entropy distribution with the pairwise marginals of three variables.

    The fit uses :func:`~dit.algorithms.maxent_dist`, which falls back to a convex
    optimizer when iterative proportional fitting stalls. It stalls on cyclic structures
    with induced structural zeros, such as the AND gate, where plain iterative proportional
    fitting is still about 1e-5 bits off after 1e5 cycles.

    Parameters
    ----------
    dist : Distribution
        A distribution of three variables.
    maxiter : int, None
        The iteration limit of the convex optimizer.

    Returns
    -------
    p2 : Distribution
        The maximum entropy distribution with the pairwise marginals of `dist`.
    """
    return maxent_dist(dist, [[0, 1], [0, 2], [1, 2]], maxiter=maxiter)


@dataclass(frozen=True)
class ReferenceCoinformations:
    """
    Information values of two sources, a target, and their reference distributions.

    All values are in bits.

    Attributes
    ----------
    mutual_informations : tuple of float
        :math:`I(X_0 : Y)` and :math:`I(X_1 : Y)`.
    coinformation : float
        :math:`C`, the co-information of the distribution.
    pairwise : float
        :math:`C^{(2)}`, the co-information of the maximum entropy distribution with the
        pairwise marginals.
    connected : float
        :math:`I^{(3)} = H(P^{(2)}) - H(p) = C^{(2)} - C`, the third-order connected
        information.
    independent : float
        :math:`I_{Q_{\\mathrm{CI}}}(X_0 : X_1)`, the co-information of the distribution in
        which the sources are conditionally independent given the target.
    """

    mutual_informations: tuple[float, float]
    coinformation: float
    pairwise: float
    connected: float
    independent: float


def reference_coinformations(dist, sources, target, maxiter=None):
    """
    The co-informations of two sources and a target, and of their reference distributions.

    Parameters
    ----------
    dist : Distribution
        A distribution with linear probabilities.
    sources : pair of iterables
        The two source groups; each is coalesced into one variable.
    target : iterable
        The target variables, coalesced into one variable.
    maxiter : int, None
        The iteration limit of the maximum entropy fit.

    Returns
    -------
    references : ReferenceCoinformations
        The values, in bits.
    """
    triplet = dist.coalesce([*sources, target])
    pairwise = pairwise_maxent(triplet, maxiter=maxiter)
    independent = conditionally_independent(triplet, [[0], [1]], [2])
    return ReferenceCoinformations(
        mutual_informations=(
            float(coinformation(triplet, [[0], [2]])),
            float(coinformation(triplet, [[1], [2]])),
        ),
        coinformation=float(coinformation(triplet)),
        pairwise=float(coinformation(pairwise)),
        connected=float(entropy(pairwise) - entropy(triplet)),
        independent=float(coinformation(independent, [[0], [1]])),
    )


def source_target_bounds(dist, sources, target):
    """
    Bounds on the largest co-information over the distributions with the source-target
    marginals of `dist`, the redundancy of BROJA.

    The distribution itself and :math:`Q_{\\mathrm{CI}}` have those marginals, so the
    larger of their co-informations is a lower bound. Every such distribution has the
    mutual informations :math:`I(X_i : Y)` of `dist` and nonnegative conditional mutual
    informations, so its co-information is at most :math:`\\min_i I(X_i : Y)`.

    Parameters
    ----------
    dist : Distribution
        The distribution.
    sources : pair of iterables
        The two source groups; each is coalesced into one variable.
    target : iterable
        The target variables, coalesced into one variable.

    Returns
    -------
    lower : float
        The lower bound, in bits.
    upper : float
        The upper bound, in bits.
    mutual_informations : tuple of float
        :math:`I(X_0 : Y)` and :math:`I(X_1 : Y)`, in bits.
    """
    triplet = dist.coalesce([*sources, target])
    mutual_informations = (
        float(coinformation(triplet, [[0], [2]])),
        float(coinformation(triplet, [[1], [2]])),
    )
    independent = float(coinformation(conditionally_independent(triplet, [[0], [1]], [2]), [[0], [1]]))
    lower = max(float(coinformation(triplet)), independent)
    return lower, min(mutual_informations), mutual_informations
