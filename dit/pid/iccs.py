"""
The I_ccs measure, as proposed by Ince.
"""

from __future__ import division

import numpy as np

from .pid import BasePID

from .. import modify_outcomes
from ..algorithms import maxent_dist
from ..utils import flatten, powerset


def i_ccs(d, inputs, output):
    """
    Compute I_ccs, the average pointwise coinformation, where the average is taken over
    events whose marginal pointwise input-output mutual information agree in sign with the
    pointwise coinformation.

    Parameters
    ----------
    d : Distribution
        The distribution to compute i_ccs for.
    inputs : iterable of iterables
        The input variables.
    output : iterable
        The output variable.

    Returns
    -------
    iccs : float
        The value of I_ccs.
    """
    var_map = {var: i for i, var in enumerate(inputs + (output,))}
    vars = list(sorted(var_map.values()))
    d = d.coalesce(inputs + (output,))
    marginals = [vars[:-1]] + [[i, vars[-1]] for i in vars[:-1]]
    d = maxent_dist(d, marginals)
    d = modify_outcomes(d, lambda o: tuple(o))
    sub_vars = [var for var in powerset(vars) if var]
    sub_dists = {var: d.marginal(var) for var in sub_vars}

    def pmi(input_, output):
        """
        Compute the pointwise mutual information.

        Parameters
        ----------
        input_ : iterable
            The input variable.
        output : iterable
            The output variable.

        Returns
        -------
        pmi : dict
            A dictionary mapping events to pointwise mutual information values.
        """
        jdist = sub_dists[(input_, output)]
        idist = sub_dists[(input_,)]
        odist = sub_dists[(output,)]
        return {e: np.log2(jdist[e] / (idist[(e[0],)] * odist[(e[1],)])) for e in jdist.outcomes}

    pmis = {tuple(marg): pmi(marg[0], marg[1]) for marg in marginals[1:]}

    inputs_dist = sub_dists[tuple(vars[:-1])]
    output_dist = sub_dists[(vars[-1],)]
    joint_pmis = {e: np.log2(d[e]/(inputs_dist[e[:-1]]*output_dist[(e[-1],)])) for e in d.outcomes}

    coinfos = {e: np.log2(np.prod(
        [sub_dists[sub_var][tuple(e[i] for i in flatten(sub_var))] ** ((-1) ** len(sub_var)) for sub_var in sub_vars]))
               for e in d.outcomes}

    # fix the sign of things close to zero
    for pmi in pmis.values():
        for e, val in pmi.items():
            if np.isclose(val, 0.0):
                pmi[e] = 0.0
    for e, val in coinfos.items():
        if np.isclose(val, 0.0):
            coinfos[e] = 0.0

    i = sum(d[e] * coinfos[e] for e in d.outcomes if
            all(np.sign(coinfos[e]) == np.sign(pmi[tuple(e[i] for i in marg)]) for marg, pmi in pmis.items()) \
            and np.sign(coinfos[e]) == np.sign(joint_pmis[e]))

    return i


class PID_CCS(BasePID):
    """
    The common change in surprisal partial information decomposition, as defined by Ince.
    """
    _name = "I_ccs"
    _measure = staticmethod(i_ccs)
