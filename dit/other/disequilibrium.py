"""
Disequilibrium, as measured by `Intensive entropic non-triviality measure` by
P.W. Lamberti, M.T. Martin, A. Plastino, O.A. Rosso.
"""

import numpy as np

from ..divergences.pmf import jensen_shannon_divergence as JSD
from ..shannon import entropy

__all__ = (
    "disequilibrium",
    "LMPR_complexity",
)


def disequilibrium(dist, rvs=None):
    """
    Compute the (normalized) disequilibrium as measured the Jensen-Shannon
    divergence from an equilibrium distribution.

    Parameters
    ----------
    dist : Distribution
        Distribution to compute the disequilibrium of.
    rvs : list, None
        The indexes of the random variable used to calculate the diseqilibrium.
        If None, then the disequilibrium is calculated over all random
        variables. This should remain `None` for scalar distributions.

    Returns
    -------
    D : float
        The disequilibrium.
    """
    d = dist.marginal(rvs) if rvs is not None else dist

    d = d.copy(base="linear")
    d.make_dense()
    pmf = d.pmf

    Pe = np.ones_like(pmf) / pmf.size
    Pu = np.zeros_like(pmf)
    Pu[0] = 1

    J = JSD(np.vstack([pmf, Pe]))
    Q = JSD(np.vstack([Pe, Pu]))
    D = J / Q

    return D


def LMPR_complexity(dist, rvs=None):
    """
    Compute the LMPR complexity.

    Parameters
    ----------
    dist : Distribution
        Distribution to compute the LMPR complexity of.
    rvs : list, None
        The indexes of the random variable used to calculate the LMPR
        complexity. If None, then the LMPR complexity is calculated over all
        variables. This should remain `None` for scalar distributions.

    Returns
    -------
    C : float
        The LMPR complexity.
    """
    d = dist.copy()
    d.make_dense()
    D = disequilibrium(d, rvs)
    H = entropy(d, rvs) / np.log2(len(d.outcomes))
    return D * H
