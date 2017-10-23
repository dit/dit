"""
Partial Entropy Decomposition with the Hcs measure from Ince (2017)

https://arxiv.org/abs/1702.01591
"""

from __future__ import division

import numpy as np
from itertools import combinations

import dit

from .pid import BasePID
from .lattice import pid_lattice

from .. import modify_outcomes
from ..algorithms import maxent_dist
from ..multivariate import entropy
from ..utils import flatten, powerset

def h_cs(d, inputs, output=None):
    """
    Compute H_cs, the average of positive pointwise co-information values

    Parameters
    ----------
    d : Distribution
        The distribution to compute i_ccs for.
    inputs : iterable of iterables
        The input variables.

    Returns
    -------
    hcs : float
        The value of H_cs.
    """

    var_map = {var: i for i, var in enumerate(inputs)}
    vars = list(sorted(var_map.values()))
    d = d.coalesce(inputs)
    n_variables = d.outcome_length()
    # pairwise marginal maxent
    if n_variables > 2:
        marginals = list(combinations(range(n_variables), 2))
        d = maxent_dist(d, marginals, 'indices')
    d = modify_outcomes(d, lambda o: tuple(o))

    # calculate pointwise co-information
    sub_vars = [var for var in powerset(vars) if var]
    sub_dists = {var: d.marginal(var) for var in sub_vars}
    coinfos = {}
    for e in d.outcomes:
        coinfos[e] = 0.0
        for sub_var in sub_vars:
            P = sub_dists[sub_var][tuple([e[i] for i in flatten(sub_var)])]
            coinfos[e] = coinfos[e] + np.log2(P)*((-1) ** (len(sub_var)))

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

    def __init__(self, dist, inputs=None):
        """
        Parameters
        ----------
        dist : Distribution
            The distribution to compute the decomposition on.
        inputs : iter of iters, None
            The set of variables to include. If None, `dist.rvs` is used.
        """
        self._dist = dist
        
        if inputs is None:
            inputs = dist.rvs

        self._red_string = "H_r"
        self._pi_string = "H_d"
        self._inputs = tuple(map(tuple, inputs))
        self._output = None
        self._lattice = pid_lattice(self._inputs)
        self._total = entropy(self._dist, rvs=self._inputs)
        self._compute()
