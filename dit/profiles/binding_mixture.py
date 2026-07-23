"""
Shared-randomness (binding) profile via mixtures of product distributions.

Rosas et al. (2019) split multivariate dependence into two faces:

* *Collective constraints* — total correlation :math:`T`, built from MaxEnt /
  e-flat marginal ladders (:class:`~dit.profiles.ConnectedInformations`,
  :class:`~dit.profiles.DependencyDecomposition`).
* *Shared randomness* — dual total correlation / binding entropy :math:`B`,
  which lives in shared latent structure.

This module implements the randomness face: MLE projections onto
:math:`k`-mixtures of fully factorized distributions
(:func:`~dit.algorithms.mixture_of_products.mixture_of_products_dists`), with
profile atoms equal to consecutive increments of :math:`B`.
"""

import numpy as np

from ..algorithms.mixture_of_products import mixture_of_products_dists
from ..multivariate import dual_total_correlation as B
from .base_profile import BaseProfile, profile_docstring
from .information_partitions import DependencyDecomposition

__all__ = (
    "BindingMixtureProfile",
    "SharedRandomnessDecomposition",
)


class BindingMixtureProfile(BaseProfile):  # noqa: D101
    __doc__ = profile_docstring.format(
        name="BindingMixtureProfile",
        static_attributes="",
        attributes="",
        methods="",
    ) + (
        "\n\nNotes\n-----\n"
        "Atoms are :math:`\\Delta B(Q^{(k)}) = B(Q^{(k)}) - B(Q^{(k-1)})` along\n"
        "the MLE mixture-of-products ladder "
        "(:func:`~dit.algorithms.mixture_of_products.mixture_of_products_dists`).\n"
        "They are nonnegative and sum to :math:`B(P)` once the fit saturates.\n"
        "Distinct from :class:`~dit.profiles.ConnectedDualInformations` (MaxEnt\n"
        "ladder with measure :math:`B`) and from "
        ":class:`~dit.profiles.MFlatConnectedInformations` (Amari additive\n"
        "m-flat / reverse KL)."
    )

    _name = "Binding Mixture Profile"
    xlabel = "mixture components k"

    def __init__(
        self,
        dist,
        k_max=None,
        *,
        n_init=12,
        max_iter=200,
        seed=0,
        early_stop=True,
    ):
        """
        Parameters
        ----------
        dist : Distribution
            The distribution to profile.
        k_max : int or None
            Maximum mixture cardinality (default ``min(8, |X|)``).
        n_init, max_iter, seed, early_stop
            Passed through to
            :func:`~dit.algorithms.mixture_of_products.mixture_of_products_dists`.
        """
        self._k_max = k_max
        self._n_init = n_init
        self._max_iter = max_iter
        self._seed = seed
        self._early_stop = early_stop
        super().__init__(dist)

    def _compute(self):
        """
        Compute :math:`\\Delta B` along the mixture-of-products ladder.
        """
        dists, meta = mixture_of_products_dists(
            self.dist,
            k_max=self._k_max,
            n_init=self._n_init,
            max_iter=self._max_iter,
            seed=self._seed,
            early_stop=self._early_stop,
        )
        bindings = [float(B(d)) for d in dists]
        # B(Q_0) := 0 with the empty / undefined mixture; first atom is B(Q_1).
        prev = 0.0
        profile = {}
        for k, b in enumerate(bindings, start=1):
            profile[k] = b - prev
            prev = b

        self.profile = profile
        self.widths = np.ones(len(self.profile))
        self.dists = dists
        self.meta = meta
        self.bindings = bindings
        self.forward_kl = [m["forward_kl"] for m in meta]
        self.I_xv = [m["I_xv"] for m in meta]


class SharedRandomnessDecomposition(DependencyDecomposition):
    """
    Dependency-lattice dual of collective-constraint decompositions, measuring
    shared randomness :math:`B`.

    At each antichain :math:`\\pi`, the reconstruction is the product of exact
    block marginals

    .. math::

        Q_\\pi = \\bigotimes_{b \\in \\pi} P_b,

    which is the *saturated* mixture-of-products model within each block
    (any discrete block marginal is a finite mixture of product components)
    with independence across blocks.  Default atom measure is dual total
    correlation :math:`B(Q_\\pi)`.

    This is the structural dual of :class:`DependencyDecomposition` on the
    same lattice: MaxEnt with within-block marginals yields the same
    :math:`Q_\\pi`, but here the reported quantity is binding / shared
    randomness rather than entropy or total correlation.

    For the *unsaturated* cardinality ladder (growing shared latent size),
    see :class:`BindingMixtureProfile`.
    """

    def __init__(self, dist, rvs=None, measures=None, cover=True, maxiter=None):
        if measures is None:
            measures = {"B": B}
        super().__init__(dist, rvs=rvs, measures=measures, cover=cover, maxiter=maxiter)
