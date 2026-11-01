"""
Partial Entropy Decomposition using min-of-surprisals (MOS).

Based on Finn & Lizier (2020), "Generalised Measures of Multivariate
Information Content", Entropy 22(2), 216.
https://doi.org/10.3390/e22020216

The redundancy functional for a lattice node alpha = {A_1, ..., A_k}
(an antichain of source groups) is:

    H_mos(alpha) = E[ min_i  h(A_i) ]

where h(A_i) = -log p(A_i) is the joint surprisal of source group A_i.
"""

import numpy as np

from .ped import PointwisePED

__all__ = (
    "PED_MOS",
    "h_mos",
    "h_mos_pw",
)


def _compute_marginal_surprisals(d, sources):
    """
    For each source group and each outcome, compute the marginal surprisal.

    Returns a list of dicts, one per source group.  Each dict maps
    the original distribution's outcomes to -log2(p(source_group)).
    """
    marginals = [d.marginal(list(src)) for src in sources]
    result = []
    for src, marg in zip(sources, marginals, strict=True):
        src_surprisals = {}
        for outcome in d.outcomes:
            p_marg = marg[tuple(outcome[i] for i in src)]
            if p_marg > 0:
                src_surprisals[outcome] = -np.log2(p_marg)
            else:
                src_surprisals[outcome] = np.inf
        result.append(src_surprisals)
    return result


def h_mos_pw(d, sources, target=None):
    """
    Compute per-outcome min-of-surprisals redundancy for a lattice node.

    For each joint outcome, returns the minimum surprisal (-log p) across
    the marginal source groups.

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
        ``{outcome: float}`` mapping each joint outcome of the distribution
        to its pointwise min-of-surprisals value.
    """
    src_surprisals = _compute_marginal_surprisals(d, sources)

    pw = {}
    for outcome in d.outcomes:
        if d[outcome] <= 0:
            continue
        pw[outcome] = min(s[outcome] for s in src_surprisals)

    return pw


def h_mos(d, sources, target=None):
    """
    Compute the min-of-surprisals redundancy for a single lattice node.

    For each joint outcome, compute the minimum surprisal (-log p) across
    the marginal source groups, then take the expectation.

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
    hmos : float
        The expected min-of-surprisals value.
    """
    pw = h_mos_pw(d, sources, target)
    return sum(d[outcome] * val for outcome, val in pw.items())


class PED_MOS(PointwisePED):
    """
    The min-of-surprisals partial entropy decomposition.

    Based on Finn & Lizier (2020), "Generalised Measures of Multivariate
    Information Content", Entropy 22(2), 216.
    https://doi.org/10.3390/e22020216

    This is the :math:`h_{\\min}` decomposition. When ``pointwise=True``,
    per-outcome redundancies and PI atoms are also computed via Mobius
    inversion on the lattice.  Theorem 5 of the paper guarantees that every
    pointwise PI atom is non-negative.
    """

    _name = "H_mos"
    _measure = staticmethod(h_mos)
    _pw_measure = staticmethod(h_mos_pw)
