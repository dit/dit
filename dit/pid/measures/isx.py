"""
I^sx, Shared Exclusion Partial Information Decomposition by
Makkeh, A. & Gutknecht, A. & Wibral, M. (2021). Introducing a differentiable measure of pointwise shared information.
Phys. Rev. E 103, 032149
"""

import numpy as np

from ..pid import BasePointwisePID

__all__ = ("PID_SX",)


def _prob_union(outcomes, linear_pmf, outcome, var_sets):
    """
    Probability of the union of events where at least one var_set matches outcome.

    Parameters
    ----------
    outcomes : np.ndarray
        Object array of shape (n_outcomes, n_variables).
    linear_pmf : np.ndarray
        Linear probabilities aligned with `outcomes`.
    outcome : tuple
        The outcome whose events are matched.
    var_sets : iterable of iterables
        Variable index sets; each defines the event ``X_vs = outcome[vs]``.
    """
    mask = np.zeros(len(outcomes), dtype=bool)
    for var_set in var_sets:
        var_set = list(var_set)
        values = np.empty(len(var_set), dtype=object)
        values[:] = [outcome[i] for i in var_set]
        mask |= (outcomes[:, var_set] == values).all(axis=1)
    return linear_pmf[mask].sum()


def _pointwise_parts(dist, sources, target):
    """
    ``(plus, minus)`` dicts of i^sx+(s) = -log2 P(union of source events) and
    i^sx-(s, t) = -log2 P(union of (source, target) events) / P(t).
    """
    dist = dist.copy(base="linear") if dist.is_log() else dist
    outcomes = np.empty((len(dist.outcomes), dist.outcome_length()), dtype=object)
    outcomes[:] = [list(o) for o in dist.outcomes]
    lin_joint = np.asarray(dist.pmf, dtype=float)
    dist_target = dist.marginal(target)
    joint_sets = tuple(tuple(s) + tuple(target) for s in sources)

    plus, minus = {}, {}
    for outcome in dist.outcomes:
        p_t = dist_target[tuple(outcome[t] for t in target)]
        with np.errstate(divide="ignore", invalid="ignore"):
            plus[outcome] = -np.log2(_prob_union(outcomes, lin_joint, outcome, sources))
            minus[outcome] = -np.log2(_prob_union(outcomes, lin_joint, outcome, joint_sets) / p_t)
    return plus, minus


class PID_SX(BasePointwisePID):
    """
    Shared Exclusion Partial Information Decomposition (I^sx) as described in:

    Makkeh, A. & Gutknecht, A. & Wibral, M. (2021). Introducing a differentiable measure of pointwise shared information.
    Phys. Rev. E 103, 032149

    When constructed with ``pointwise=True``, also computes the per-outcome
    pointwise PID (PPID) and the informative / misinformative decomposition
    (i^sx+ and i^sx-).
    """

    _name = "I_sx"

    @staticmethod
    def _pointwise_measure(dist, sources, target):
        """
        Per joint-outcome pointwise shared information:
        i^sx(t : sources) = i^sx+(s) - i^sx-(s, t)

        Returns
        -------
        pw : dict
            ``{outcome: float}`` for every joint outcome in *dist*.
        """
        plus, minus = _pointwise_parts(dist, sources, target)
        return {outcome: plus[outcome] - minus[outcome] for outcome in plus}

    @staticmethod
    def _pointwise_measure_parts(dist, sources, target):
        """
        Informative (+) and misinformative (-) components of the pointwise
        shared information:

        * i^sx+(s) = -log2 P(union of source events)
        * i^sx-(s,t) = -log2 P(union of (source,target) events) / P(t)

        Returns
        -------
        plus_dict, minus_dict : dict, dict
        """
        return _pointwise_parts(dist, sources, target)
