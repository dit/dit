"""
Partial Entropy Decomposition using shared probability mass exclusions (sx).

Based on Varley, Pope, Puxeddu, Faskowitz & Sporns (2023), "Partial entropy
decomposition reveals higher-order information structures in human brain
activity", PNAS 120(30), e2300888120.
https://doi.org/10.1073/pnas.2300888120

The redundancy functional for a lattice node alpha = {A_1, ..., A_k} is:

    H_sx(alpha) = E[ -log P(A_1 = a_1 or ... or A_k = a_k) ]

the surprisal of the event that at least one source group takes its observed
value. It is the entropy analogue of the I_sx redundancy of Makkeh, Gutknecht
& Wibral.
"""

import numpy as np

from .ped import PointwisePED

__all__ = (
    "PED_SX",
    "h_sx",
    "h_sx_pw",
)


def h_sx_pw(d, sources, target=None):
    """
    Compute the per-outcome sx redundancy for a lattice node.

    For each joint outcome :math:`x`, returns
    :math:`-\\log_2 P\\left(\\bigcup_i \\{A_i = x_{A_i}\\}\\right)`.

    Parameters
    ----------
    d : Distribution
        The distribution to compute the measure for.
    sources : tuple of tuples
        The source groups forming the antichain node.
    target : None
        Unused. Present for API compatibility with BasePID.

    Returns
    -------
    pw : dict
        ``{outcome: float}`` for each outcome of positive probability.
    """
    outcomes = [o for o in d.outcomes if d[o] > 0]
    pmf = np.array([d[o] for o in outcomes])
    union = np.zeros((len(outcomes), len(outcomes)), dtype=bool)
    for src in sources:
        index = {}
        codes = np.array([index.setdefault(tuple(o[i] for i in src), len(index)) for o in outcomes])
        union |= codes[:, None] == codes[None, :]
    return {o: float(-np.log2(pmf[union[k]].sum())) for k, o in enumerate(outcomes)}


def h_sx(d, sources, target=None):
    """
    Compute the sx redundancy for a single lattice node.

    Parameters
    ----------
    d : Distribution
        The distribution to compute the measure for.
    sources : tuple of tuples
        The source groups forming the antichain node.
    target : None
        Unused. Present for API compatibility with BasePID.

    Returns
    -------
    hsx : float
        The expected sx redundancy.
    """
    pw = h_sx_pw(d, sources, target)
    return sum(d[outcome] * val for outcome, val in pw.items())


class PED_SX(PointwisePED):
    """
    The shared-exclusions partial entropy decomposition :math:`h_{sx}` of
    Varley et al. (2023).

    https://doi.org/10.1073/pnas.2300888120

    When ``pointwise=True``, per-outcome redundancies and PI atoms are also
    computed via Mobius inversion on the lattice.
    """

    _name = "H_sx"
    _measure = staticmethod(h_sx)
    _pw_measure = staticmethod(h_sx_pw)
