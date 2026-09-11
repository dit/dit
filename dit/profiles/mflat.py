"""
Amari m-flat connected informations: divergence gaps along the additive ANOVA ladder.

Distinct from :class:`~dit.profiles.ConnectedDualInformations`, which differences
dual total correlation along the *e-flat* MaxEnt (marginal) ladder, and from
:class:`~dit.profiles.MarginalLiftProfile`, which uses fixed lifts of the data's
own marginals rather than free ANOVA tables.
"""

import numpy as np

from ..algorithms import mflat_mprojection_dists
from ..algorithms.mprojection import _aligned_pmf, _alphabets, _divergence
from ..divergences import kullback_leibler_divergence as D
from .base_profile import BaseProfile, profile_docstring

__all__ = (
    "MFlatConnectedInformations",
    "AmariMFlatProfile",
)


class MFlatConnectedInformations(BaseProfile):  # noqa: D101
    __doc__ = profile_docstring.format(
        name="MFlatConnectedInformations",
        static_attributes="",
        attributes="",
        methods="",
    ) + (
        "\n\nNotes\n-----\n"
        "Gaps along Amari's m-flat ladder "
        "(:func:`~dit.algorithms.mflat_mprojection_dists`) under a chosen\n"
        "divergence ``criterion`` (default ``'jsd'``: Jensen–Shannon; also\n"
        "``'forward_kl'`` / ``'reverse_kl'``). Atoms are consecutive drops in\n"
        "the residual divergence to :math:`P`:\n"
        ":math:`C_k = D(P, Q^{(k-1)}) - D(P, Q^{(k)})`.\n"
        "For ``criterion='reverse_kl'`` this recovers the classical Amari\n"
        "m-projection gaps (Pythagorean under symmetric :math:`P_\\varepsilon`)."
    )

    _name = "M-Flat Connected Informations"

    def __init__(
        self,
        dist,
        nrestarts=16,
        maxiter=3000,
        criterion="jsd",
        eps=None,
        eps_schedule=(1e-4, 1e-6, 1e-8),
    ):
        """
        Parameters
        ----------
        dist : Distribution
            The distribution to profile.
        nrestarts, maxiter : int
            Optimizer controls for each projection.
        criterion : {'jsd', 'forward_kl', 'reverse_kl'}
            Projection / residual divergence. Default ``jsd`` is finite on
            sparse supports. Use ``reverse_kl`` for Amari's true m-projection
            with symmetric :math:`P_\\varepsilon`.
        eps : float or None
            Single smooth weight for ``reverse_kl`` (overrides ``eps_schedule``).
        eps_schedule : sequence of float or None
            Decreasing smooth weights for ``reverse_kl`` when ``eps`` is unset.
            Default ``(1e-4, 1e-6, 1e-8)``.
        """
        self._nrestarts = nrestarts
        self._maxiter = maxiter
        self._criterion = criterion
        self._eps = eps
        self._eps_schedule = None if eps_schedule is None else tuple(eps_schedule)
        super().__init__(dist)

    def _residual(self, p_pmf, q_dist, outcomes):
        q_pmf = _aligned_pmf(q_dist, outcomes)
        if self._criterion == "reverse_kl":
            return _divergence(p_pmf, q_pmf, "reverse_kl")
        return _divergence(p_pmf, q_pmf, self._criterion)

    def _compute(self):
        """
        Compute consecutive residual drops along the m-flat ladder.
        """
        kwargs = {
            "nrestarts": self._nrestarts,
            "maxiter": self._maxiter,
            "criterion": self._criterion,
        }
        if self._criterion == "reverse_kl":
            if self._eps is not None:
                kwargs["eps"] = self._eps
            else:
                kwargs["eps_schedule"] = self._eps_schedule
        dists = mflat_mprojection_dists(self.dist, **kwargs)
        alphabets = _alphabets(dists[0])
        from itertools import product

        outcomes = list(product(*alphabets))
        # Dense target pmf aligned to the ladder sample space.
        from copy import deepcopy

        from ..algorithms.optutil import prepare_dist

        dense = prepare_dist(deepcopy(self.dist))
        pmf_map = {
            tuple(o) if not isinstance(o, tuple) else o: float(p)
            for o, p in zip(dense.outcomes, dense.pmf, strict=True)
        }
        p_pmf = np.array([pmf_map.get(o, 0.0) for o in outcomes], dtype=float)
        p_pmf = p_pmf / p_pmf.sum()

        if self._criterion == "reverse_kl":
            # Preserve Pythagorean consecutive reverse-KL gaps between rungs.
            diffs = [D(dists[i], dists[i + 1]) for i in range(len(dists) - 1)]
        else:
            residuals = [self._residual(p_pmf, q, outcomes) for q in dists]
            diffs = [residuals[i] - residuals[i + 1] for i in range(len(residuals) - 1)]

        self.profile = {i + 1: float(v) for i, v in enumerate(diffs)}
        self.widths = np.ones(len(self.profile))
        self._dists = dists
        self.criterion = self._criterion


AmariMFlatProfile = MFlatConnectedInformations
