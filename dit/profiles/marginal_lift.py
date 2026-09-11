"""
Profile from convex combinations of lifts of a joint's own marginals.

Sibling of :class:`~dit.profiles.MFlatConnectedInformations`: both approximate
:math:`P` by linear structure in the pmf, but here the building blocks are
*fixed* lifts of the data marginals :math:`P_S` rather than free ANOVA tables.
"""

import numpy as np

from ..algorithms.marginal_lifts import marginal_lift_dists
from .base_profile import BaseProfile, profile_docstring

__all__ = ("MarginalLiftProfile",)


class MarginalLiftProfile(BaseProfile):  # noqa: D101
    __doc__ = profile_docstring.format(
        name="MarginalLiftProfile",
        static_attributes="",
        attributes="",
        methods="",
    ) + (
        "\n\nNotes\n-----\n"
        "At order :math:`k`, fit nonnegative weights on lifts of all marginals\n"
        "with :math:`|S|\\le k` (plus uniform) by least squares. Profile atoms\n"
        "are consecutive drops in the :math:`L^2` residual to :math:`P`.\n"
        "Fitted coefficients ``alphas[k]`` show which marginals carry mass\n"
        "(e.g. Copy puts all weight on the copied pair at order 2)."
    )

    _name = "Marginal Lift Profile"
    xlabel = "marginal order k"

    def __init__(self, dist, k_max=None, mode="uniform", n_init=12, seed=0):
        """
        Parameters
        ----------
        dist : Distribution
        k_max : int or None
            Highest marginal order (default: number of variables).
        mode : {'uniform', 'product'}
            Lift of each marginal to the full joint.
        n_init, seed
            Convex-combination optimizer controls.
        """
        self._k_max = k_max
        self._mode = mode
        self._n_init = n_init
        self._seed = seed
        super().__init__(dist)

    def _compute(self):
        dists, metas = marginal_lift_dists(
            self.dist,
            k_max=self._k_max,
            mode=self._mode,
            n_init=self._n_init,
            seed=self._seed,
        )
        # L2 residual of each rung vs P (order 0 computed explicitly).
        from copy import deepcopy
        from itertools import product

        from ..algorithms.optutil import prepare_dist

        dense = prepare_dist(deepcopy(self.dist))
        alph = [tuple(sorted({o[i] for o in dense.outcomes})) for i in range(dense.outcome_length())]
        outs = list(product(*alph))
        pmf_map = {tuple(o): float(p) for o, p in zip(dense.outcomes, dense.pmf, strict=True)}
        p = np.array([pmf_map.get(o, 0.0) for o in outs], dtype=float)
        p /= p.sum()

        residuals = []
        for q in dists:
            qmap = {
                tuple(o) if not isinstance(o, tuple) else o: float(pr) for o, pr in zip(q.outcomes, q.pmf, strict=True)
            }
            qpmf = np.array([qmap.get(o, 0.0) for o in outs], dtype=float)
            qpmf /= qpmf.sum()
            residuals.append(float(np.linalg.norm(p - qpmf)))

        diffs = [residuals[i] - residuals[i + 1] for i in range(len(residuals) - 1)]
        self.profile = {i + 1: float(v) for i, v in enumerate(diffs)}
        self.widths = np.ones(len(self.profile))
        self.dists = dists
        self.metas = metas
        self.residuals = residuals
        self.alphas = [m["alpha"] for m in metas]
        self.labels = [m["labels"] for m in metas]
        self.mode = self._mode
