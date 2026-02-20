"""
Partial Entropy Decomposition with the Hcs measure from Ince (2017)

https://arxiv.org/abs/1702.01591
"""

from itertools import combinations

import numpy as np
from lattices.lattices import free_distributive_lattice

from .. import modify_outcomes
from ..algorithms import maxent_dist
from ..multivariate import entropy
from ..utils import flatten, powerset
from .pid import BasePID, _transform

__all__ = (
    "PED_CS",
    "h_cs",
)


def h_cs(d, sources, target=None):
    """
    Compute H_cs, the average of positive pointwise co-information values

    Parameters
    ----------
    d : Distribution
        The distribution to compute i_ccs for.
    sources : iterable of iterables
        The input variables.

    Returns
    -------
    hcs : float
        The value of H_cs.
    """
    rv_map = {rv: i for i, rv in enumerate(sources)}
    rvs = sorted(rv_map.values())
    d = d.coalesce(sources)
    n_variables = d.outcome_length()
    # pairwise marginal maxent
    if n_variables > 2:
        marginals = list(combinations(range(n_variables), 2))
        d = maxent_dist(d, marginals)
    d = modify_outcomes(d, lambda o: tuple(o))

    # calculate pointwise co-information
    sub_rvs = [rv for rv in powerset(rvs) if rv]
    sub_dists = {rv: d.marginal(rv) for rv in sub_rvs}
    coinfos = {}
    for e in d.outcomes:
        coinfos[e] = 0.0
        for sub_rv in sub_rvs:
            P = sub_dists[sub_rv][tuple(e[i] for i in flatten(sub_rv))]
            coinfos[e] = coinfos[e] + np.log2(P) * ((-1) ** (len(sub_rv)))

    # sum positive pointwise terms
    hcs = sum(d[e] * coinfos[e] for e in d.outcomes if coinfos[e] > 0.0)
    return hcs


class PED_CS(BasePID):
    """
    The change in surprisal partial entropy decomposition, as defined by Ince (2017).

    https://arxiv.org/abs/1702.01591
    """

    _name = "H_cs"
    _measure = staticmethod(h_cs)
    _red_string = "H_r"
    _pi_string = "H_d"

    def __init__(self, dist, sources=None, **kwargs):
        """
        Parameters
        ----------
        dist : Distribution
            The distribution to compute the decomposition on.
        sources : iter of iters, None
            The set of variables to include. If None, `dist.rvs` is used.
        """
        self._dist = dist

        if sources is None:
            sources = dist.rvs

        self._kwargs = kwargs
        self._sources = tuple(map(tuple, sources))
        self._target = None
        self._lattice = _transform(free_distributive_lattice(self._sources))
        self._total = entropy(self._dist, rvs=self._sources)
        self._reds = {}
        self._pis = {}
        self._compute()
