"""
The I_min measure as proposed by Williams & Beer.
"""

from collections import defaultdict

import numpy as np

from ..pid import BasePID

__all__ = ("PID_WB",)


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
    p_s_a = {a: pp[target_value] for a, pp in zip(pp_a.outcomes, pp_s_a, strict=True)}

    if d.is_symbolic():
        import sympy

        terms = []
        for a, psa in p_s_a.items():
            coeff = sympy.sympify(p_a_s[a])
            if coeff == 0:
                continue
            terms.append(coeff * sympy.log(psa / p_s, 2))
        return sympy.Add(*terms)

    return np.nansum([p_a_s[a] * np.log2(psa / p_s) for a, psa in p_s_a.items()])


def _specific_informations(d, source, target):
    """
    The specific informations I(source : target = t) and probabilities p(t) of every value
    t of the target, computed from the outcomes of `d`, a numeric distribution.

    This is what `s_i` computes, for every value of the target at once and without
    conditioning the distribution twice for each.

    Parameters
    ----------
    d : Distribution
        The distribution.
    source : iterable
        The source variables.
    target : iterable
        The target variables.

    Returns
    -------
    p_t : dict
        Maps each value of the target, a tuple, to its probability.
    specific : dict
        Maps each value of the target to I(source : target = t), in bits.
    """
    if d.is_log():
        d = d.copy(base="linear")
    dims = list(d.dims)
    source = [dims.index(name) for name in d._resolve_rv_names(list(source))]
    target = [dims.index(name) for name in d._resolve_rv_names(list(target))]
    p_at, p_a, p_t = defaultdict(float), defaultdict(float), defaultdict(float)
    for outcome, p in zip(d.outcomes, d.pmf, strict=True):
        if p > 0:
            a, t = tuple(outcome[i] for i in source), tuple(outcome[i] for i in target)
            p_at[a, t] += p
            p_a[a] += p
            p_t[t] += p
    specific = dict.fromkeys(p_t, 0.0)
    for (a, t), p in p_at.items():
        specific[t] += p / p_t[t] * np.log2(p / (p_a[a] * p_t[t]))
    return p_t, specific


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
        if d.is_symbolic():
            import sympy

            from ...symbolic import symbolic_min

            p_s = d.marginal(target)
            return sympy.Add(
                *[p_s[s] * symbolic_min([s_i(d, source, target, s) for source in sources]) for s in p_s.outcomes]
            )
        tables = [_specific_informations(d, source, target) for source in sources]
        p_t = tables[0][0]
        return float(sum(p * min(specific[t] for _, specific in tables) for t, p in p_t.items()))
