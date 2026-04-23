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

from ..utils import build_table
from .ped import BasePED
from .pid import sort_key

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
            marg_outcome = tuple(outcome[i] for i in src)
            if len(marg_outcome) == 1:
                marg_outcome = marg_outcome[0]
            p_marg = marg[marg_outcome]
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


class PED_MOS(BasePED):
    """
    The min-of-surprisals partial entropy decomposition.

    Based on Finn & Lizier (2020), "Generalised Measures of Multivariate
    Information Content", Entropy 22(2), 216.
    https://doi.org/10.3390/e22020216

    When ``pointwise=True``, per-outcome redundancies and PI atoms are also
    computed via Mobius inversion on the lattice.  Theorem 5 of the paper
    guarantees that every pointwise PI atom is non-negative.
    """

    _name = "H_mos"
    _measure = staticmethod(h_mos)

    def __init__(self, dist, sources=None, pointwise=False, **kwargs):
        """
        Parameters
        ----------
        dist : Distribution
            The distribution to compute the decomposition on.
        sources : iter of iters, None
            The set of variables to include. If None, `dist.rvs` is used.
        pointwise : bool
            If True, also compute per-outcome redundancies and PI atoms.
        """
        self._pointwise = pointwise
        self._pw_reds = {}
        self._pw_pis = {}
        super().__init__(dist, sources=sources, **kwargs)

    def _compute(self):
        super()._compute()
        if self._pointwise:
            self._compute_pointwise()

    def _compute_pointwise(self):
        """Compute pointwise redundancies and PI atoms for all lattice nodes."""
        for node in self._lattice:
            self._pw_reds[node] = h_mos_pw(self._dist, node, self._target, **self._kwargs)

        any_node = next(iter(self._pw_reds))
        outcomes = list(self._pw_reds[any_node].keys())

        for node in reversed(list(self._lattice)):
            pw_pi = {}
            for o in outcomes:
                desc_sum = sum(self._pw_pis[n][o] for n in self._lattice.descendants(node))
                pw_pi[o] = self._pw_reds[node].get(o, 0.0) - desc_sum
            self._pw_pis[node] = pw_pi

    def get_pw_red(self, node):
        """
        Pointwise redundancy values for *node*.

        Returns
        -------
        pw_red : dict
            ``{outcome: float}``
        """
        return self._pw_reds[node]

    def get_pw_pi(self, node):
        """
        Pointwise PI atom values for *node*.

        Returns
        -------
        pw_pi : dict
            ``{outcome: float}``
        """
        return self._pw_pis[node]

    def pw_to_string(self, outcome=None, digits=4):
        """
        Create a table of pointwise PED values.

        Parameters
        ----------
        outcome : tuple, optional
            If given, show values for this single outcome. Otherwise show all.
        digits : int
            Precision for display.

        Returns
        -------
        table : str
        """
        if not self._pw_pis:
            return "(pointwise PED not computed -- pass pointwise=True)"

        any_node = next(iter(self._pw_pis))
        all_outcomes = list(self._pw_pis[any_node].keys())
        outcomes = [outcome] if outcome is not None else all_outcomes
        red_string = "h_r"
        pi_string = "h_d"

        columns = [self._name, "outcome", red_string, pi_string]
        dist_name = getattr(self._dist, "name", "")
        base_title = f"{self._name} (pointwise)"
        title = f"{base_title} | {dist_name}" if dist_name else base_title
        table = build_table(columns, title=title)
        table.float_format[red_string] = f"{digits + 2}.{digits}"
        table.float_format[pi_string] = f"{digits + 2}.{digits}"

        for node in sorted(self._lattice, key=sort_key(self._lattice)):
            node_label = "".join("{{{}}}".format(":".join(map(str, n))) for n in node)
            for o in outcomes:
                red_val = self._pw_reds[node].get(o, 0.0)
                pi_val = self._pw_pis[node].get(o, 0.0)
                table.add_row([node_label, str(o), red_val, pi_val])

        return table.get_string()
