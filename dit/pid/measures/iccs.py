"""
The I_ccs measure, as proposed by Ince.
"""

import numpy as np

from ... import modify_outcomes
from ...algorithms import maxent_dist
from ...utils import flatten, powerset
from ..pid import BasePID

__all__ = ("PID_CCS",)


class PID_CCS(BasePID):
    """
    The common change in surprisal partial information decomposition, as defined by Ince.
    """

    _name = "I_ccs"

    @staticmethod
    def _measure(d, sources, target):
        """
        Compute I_ccs, the average pointwise coinformation, where the average is taken over
        events whose marginal pointwise source-target mutual information agree in sign with the
        pointwise coinformation.

        Parameters
        ----------
        d : Distribution
            The distribution to compute i_ccs for.
        sources : iterable of iterables
            The source variables.
        target : iterable
            The target variable.

        Returns
        -------
        iccs : float
            The value of I_ccs.
        """
        rv_map = {rv: i for i, rv in enumerate(sources + (target,))}
        rvs = sorted(rv_map.values())
        d = d.coalesce(sources + (target,))
        marginals = [rvs[:-1]] + [[i, rvs[-1]] for i in rvs[:-1]]
        d = maxent_dist(d, marginals)
        d = modify_outcomes(d, lambda o: tuple(o))
        sub_rvs = [rv for rv in powerset(rvs) if rv]
        sub_dists = {rv: d.marginal(rv) for rv in sub_rvs}

        def pmi(source, target):
            """
            Compute the pointwise mutual information.

            Parameters
            ----------
            source : iterable
                The source variable.
            target : iterable
                The target variable.

            Returns
            -------
            pmi : dict
                A dictionary mapping events to pointwise mutual information values.
            """
            jdist = sub_dists[(source, target)]
            idist = sub_dists[(source,)]
            odist = sub_dists[(target,)]
            return {e: np.log2(jdist[e] / (idist[(e[0],)] * odist[(e[1],)])) for e in jdist.outcomes}

        pmis = {tuple(marg): pmi(marg[0], marg[1]) for marg in marginals[1:]}

        sources_dist = sub_dists[tuple(rvs[:-1])]
        target_dist = sub_dists[(rvs[-1],)]
        joint_pmis = {e: np.log2(d[e] / (sources_dist[e[:-1]] * target_dist[(e[-1],)])) for e in d.outcomes}

        coinfos = {
            e: np.log2(
                np.prod(
                    [
                        sub_dists[sub_rv][tuple(e[i] for i in flatten(sub_rv))] ** ((-1) ** len(sub_rv))
                        for sub_rv in sub_rvs
                    ]
                )
            )
            for e in d.outcomes
        }

        # fix the sign of things close to zero
        for pmi in pmis.values():
            for e, val in pmi.items():
                if np.isclose(val, 0.0):  # pragma: no cover
                    pmi[e] = 0.0
        for e, val in coinfos.items():
            if np.isclose(val, 0.0):
                coinfos[e] = 0.0

        i = sum(
            d[e] * coinfos[e]
            for e in d.outcomes
            if all(np.sign(coinfos[e]) == np.sign(pmi[tuple(e[i] for i in marg)]) for marg, pmi in pmis.items())
            and np.sign(coinfos[e]) == np.sign(joint_pmis[e])
        )

        return i
