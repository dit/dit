# -*- coding: utf-8 -*-

"""
The I_rav measure, defining a 'redundancy' auxiliary variable to capture the redundancy information between sources
"""

from ..pid import BasePID

from ...multivariate import o_information
from ...utils import partitions, extended_partition
from ...distconst import RVFunctions, insert_rvf


def corex_o_information(dist, rvs, crvs):
    """
    """
    o_information(dist) - o_information(dist, rvs, crvs)


class PID_RAV(BasePID):
    """
    The maximum coinformation auxiliary random variable method.
    """
    _name = "I_RAV"

    @staticmethod
    def _measure(d, sources, target):
        """
        I_RAV is maximum coinformation between all sources, targets, and an
        arbitrary function of the sources.

        Parameters
        ----------
        d : Distribution
            The distribution to compute i_rav for.
        sources : iterable of iterables
            The source variables.
        target : iterable
            The target variable.

        Returns
        -------
        i_rav : float
            The value of I_RAV.
        """
        d = d.coalesce(sources + (target,))

        source_parts = partitions(d.marginal(sum(d.rvs[:-1], [])).outcomes)
        outcomes = d.outcomes
        ctor = d._outcome_ctor
        idxs = list(range(len(sources)))

        parts = [extended_partition(outcomes, idxs, source_part, ctor) for source_part in source_parts]

        bf = RVFunctions(d)
        extended_dists = [insert_rvf(d, bf.from_partition(part)) for part in parts]
        return max([corex_o_information(extended_dist, sources, target) for extended_dist in extended_dists])
