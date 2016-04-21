"""
Disequilibrium, as measured by `Intensive entropic non-triviality measure` by
P.W. Lamberti, M.T. Martin, A. Plastino, O.A. Rosso.
"""

from __future__ import division

import numpy as np

from ..divergences.pmf import jensen_shannon_divergence as JSD
from ..helpers import RV_MODES
from ..shannon import entropy

__all__ = ['disequilibrium',
           'LMPR_complexity',
          ]

def disequilibrium(dist, rvs=None, rv_mode=None):
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
        variables. This should remain `None` for ScalarDistributions.
    rv_mode : str, None
        Specifies how to interpret the elements of `rvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `rvs` are interpreted as random variable indices. If equal to 'names',
        the the elements are interpreted as random variable names. If `None`,
        then the value of `dist._rv_mode` is consulted.

    Returns
    -------
    D : float
        The disequilibrium.
    """
    if dist.is_joint():
        if rvs is None:
            # Set to entropy of entire distribution
            rvs = list(range(dist.outcome_length()))
            rv_mode = RV_MODES.INDICES

        d = dist.marginal(rvs, rv_mode=rv_mode)
    else:
        d = dist

    d = d.copy(base='linear')
    d.make_dense()
    pmf = d.pmf

    Pe = np.ones_like(pmf)/pmf.size
    Pu = np.zeros_like(pmf); Pu[0] = 1

    J = JSD(np.vstack([pmf, Pe]))
    Q = JSD(np.vstack([Pe, Pu]))
    D = J/Q

    return D

def LMPR_complexity(dist, rvs=None, rv_mode=None):
    """
    Compute the LMPR complexity.

    Parameters
    ----------
    dist : Distribution
        Distribution to compute the LMPR complexity of.
    rvs : list, None
        The indexes of the random variable used to calculate the LMPR
        complexity. If None, then the LMPR complexity is calculated over all
        random variables. This should remain `None` for ScalarDistributions.
    rv_mode : str, None
        Specifies how to interpret the elements of `rvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `rvs` are interpreted as random variable indices. If equal to 'names',
        the the elements are interpreted as random variable names. If `None`,
        then the value of `dist._rv_mode` is consulted.

    Returns
    -------
    C : float
        The LMPR complexity.
    """
    d = dist.copy()
    d.make_dense()
    D = disequilibrium(d, rvs, rv_mode)
    H = entropy(d, rvs, rv_mode)/np.log2(len(d.outcomes))
    return D*H
