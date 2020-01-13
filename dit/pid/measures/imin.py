# -*- coding: utf-8 -*-

"""
The I_min measure as proposed by Williams & Beer.
"""

import numpy as np

from ..pid import BasePID


def s_i(d, source, target, target_value):
    """
    Compute the specific mutual information I(source : target=target_value)

    Parameters
    ----------
    d : Distribution
        The distribution from which this quantity is to be calculated.
    source : iterable
        The source aggregate variable.
    target : iterable
        The target aggregate variable.
    target_value : iterable
        The value of the target.

    Returns
    -------
    s : float
        The specific information
    """
    pp_s, pp_a_s = d.condition_on(target, rvs=source)
    p_s = pp_s[target_value]
    p_a_s = pp_a_s[pp_s.outcomes.index(target_value)]
    pp_a, pp_s_a = d.condition_on(source, rvs=target)
    p_s_a = {a: pp[target_value] for a, pp in zip(pp_a.outcomes, pp_s_a)}

    return np.nansum([p_a_s[a] * np.log2(psa / p_s) for a, psa in p_s_a.items()])


class PID_WB(BasePID):
    """
    The Williams & Beer partial information decomposition.
    """
    _name = "I_min"

    @staticmethod
    def _measure(d, sources, target):
        """
        Compute I_min(sources : target) =
            \\sum_{s \\in target} p(s) min_{source \\in sources} I(source : target=s)

        Parameters
        ----------
        d : Distribution
            The distribution to compute i_min for.
        sources : iterable of iterables
            The source variables.
        target : iterable
            The target variable.

        Returns
        -------
        imin : float
            The value of I_min.
        """
        p_s = d.marginal(target)
        return sum(p_s[s] * min(s_i(d, source, target, s) for source in sources) for s in p_s.outcomes)
