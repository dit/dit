"""
Spectral diagnostics of the support of a pair of random variables.

Matveev & Romashchenko :cite:`matveev2026spectral` show that several
information-theoretic properties of a pair ``(X, Y)`` are visible in the
singular values of the 0/1 biadjacency matrix ``M`` of its support graph --
the bipartite graph on the alphabets of ``X`` and ``Y`` with an edge for each
outcome of positive probability. That graph carries no probabilities at all,
so these are properties of the *combinatorial* structure underlying the pair.

This lives in `algorithms` rather than alongside the measures because it is a
property of a support, not a functional of a distribution.

Relation to the entropy profile
-------------------------------
For a distribution uniform on a *biregular* support (every value of ``X``
consistent with the same number of values of ``Y``, and vice versa),

.. math::

    \\log \\lambda_1 = H[XY] - \\tfrac{1}{2}\\big(H[X] + H[Y]\\big),
    \\qquad
    \\frac{\\lambda_2}{\\lambda_1} = \\rho_m(X : Y),

where ``rho_m`` is the maximal correlation computed by
:func:`dit.divergences.maximum_correlation`. Both identities are exact in
that case, and both fail in general: a non-biregular support puts the
Perron vector out of alignment with the uniform marginals.

Because the ratio is the maximal correlation, the spectral bound below is
really a statement about maximal correlation, and ``-2 log rho_m`` remains
defined for pairs that are neither uniform nor biregular. Whether the bound
survives that generalization is open, and is *not* asserted here.
"""

import numpy as np

from ..exceptions import ditException
from ..helpers import normalize_rvs

__all__ = (
    "spectral_entanglement_bound",
    "support_biadjacency",
    "support_singular_values",
)


def support_biadjacency(dist, rvs=None, crvs=None):
    """
    The 0/1 biadjacency matrix of the support graph of a pair.

    Parameters
    ----------
    dist : Distribution
        The distribution of interest.
    rvs : list of lists, None
        The two variables. If None, all variables of `dist` are used.
    crvs : list, None
        Variables to condition on. Conditioning variables are folded into
        both sides, so the support graph is that of the coalesced pair.

    Returns
    -------
    matrix : np.ndarray
        The 0/1 matrix, with a one wherever the pair has positive
        probability.
    """
    rvs, crvs = normalize_rvs(dist, rvs, crvs)

    if len(rvs) != 2:
        msg = f"The support spectrum is defined for 2 variables, not {len(rvs)}."
        raise ditException(msg)

    dist = dist.copy().coalesce([rv + crvs for rv in rvs] if crvs else rvs)
    dist.make_dense()
    pmf = dist.pmf.reshape(list(map(len, dist.alphabet)))

    return (pmf > 0).astype(float)


def support_singular_values(dist, rvs=None, crvs=None):
    """
    The singular values of the support graph's biadjacency matrix.

    Parameters
    ----------
    dist : Distribution
        The distribution of interest.
    rvs : list of lists, None
        The two variables. If None, all variables of `dist` are used.
    crvs : list, None
        Variables to condition on.

    Returns
    -------
    sigmas : np.ndarray
        The singular values, in decreasing order.

    Examples
    --------
    The Fano plane's point-line incidence graph is biregular of degree three,
    so its top singular value is three:

    >>> import numpy as np
    >>> from dit import Distribution
    >>> lines = ['012', '034', '056', '136', '145', '235', '246']
    >>> outcomes = [(str(p), str(i)) for i, line in enumerate(lines) for p in line]
    >>> fano = Distribution(outcomes, [1 / 21] * 21)
    >>> float(np.round(support_singular_values(fano)[0], 10))
    3.0
    """
    return np.linalg.svd(support_biadjacency(dist, rvs, crvs), compute_uv=False)


def spectral_entanglement_bound(dist, rvs=None, crvs=None):
    """
    The nominal spectral lower bound on the entanglement.

    Matveev & Romashchenko :cite:`matveev2026spectral` bound the entanglement
    of a pair uniform on its support from below by

    .. math::

        E(X, Y) \\geq \\min\\{ I[X{:}Y],\\ 2\\log(\\lambda_1 / \\lambda_2) \\}
            - O(\\log H[XY]),

    which is what certifies that expander-like supports are maximally
    non-extractable. This function returns the *nominal* term
    ``min{I, 2 log(lambda_1 / lambda_2)}`` only.

    .. warning::

        The suppressed ``O(log H[XY])`` slop is not a formality; it dominates
        at small alphabets. The uniform distribution on an 8-cycle has
        entanglement about ``0.67`` bits against a nominal bound of ``1.00``.
        Use this to compare pairs or to reason asymptotically, never as a
        pointwise certificate.

    Parameters
    ----------
    dist : Distribution
        The distribution of interest.
    rvs : list of lists, None
        The two variables. If None, all variables of `dist` are used.
    crvs : list, None
        Variables to condition on.

    Returns
    -------
    bound : float
        The nominal bound, in bits. Infinite spectral gap (a support with
        only one nonzero singular value, i.e. a complete bipartite graph on
        each component) returns the mutual information.
    """
    from ..multivariate import total_correlation

    rvs, crvs = normalize_rvs(dist, rvs, crvs)
    sigmas = support_singular_values(dist, rvs, crvs)

    mutual_information = float(total_correlation(dist, rvs, crvs))

    if len(sigmas) < 2 or sigmas[1] <= 0:
        return mutual_information

    return min(mutual_information, 2 * np.log2(sigmas[0] / sigmas[1]))
