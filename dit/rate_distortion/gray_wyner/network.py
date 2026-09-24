"""
The generalized Gray-Wyner network.

`GrayWynerNetwork` ties together the achievable rate region of the (possibly
lossy, n-source) Gray-Wyner system and the named common-information measures
that live at its corners.
"""

import numpy as np

from ...algorithms.optimization import parallel_sweep
from ...utils import flatten, unitful
from .optimizer import GrayWynerOptimizer
from .region import (
    rates_to_extension,
    rates_to_mutual_information,
    rates_to_tension,
)

__all__ = (
    "GrayWynerNetwork",
    "lossy_wyner_common_information",
)


class GrayWynerNetwork:
    """
    The generalized Gray-Wyner network for a source distribution.

    Parameters
    ----------
    dist : Distribution
        The source distribution.
    rvs : list of lists, None
        The source groups ``X_1, ..., X_n``. If None, each variable of `dist`
        is its own source.
    crvs : list, None
        Variables to condition the network on. If None, none.
    distortions : list, None
        Per-decoder distortion matrices, or None entries for lossless
        decoders. If None, every decoder is lossless.
    bounds : list, None
        Per-decoder distortion budgets ``D_i``. If None, all zero (lossless).
    bound : int, None
        Optional cap on the cardinality of the common auxiliary ``W``.
    """

    def __init__(self, dist, rvs=None, crvs=None, distortions=None, bounds=None, bound=None):
        self.dist = dist.copy()
        self.rvs = [[i] for i in flatten(dist.rvs)] if rvs is None else rvs
        self.crvs = crvs
        self.n = len(self.rvs)
        self.distortions = distortions
        self.bounds = bounds
        self.bound = bound

        self._lossless = bounds is None or all(b <= 0 for b in bounds)

        from ...multivariate import entropy

        self._marginal_entropies = tuple(float(entropy(dist, rv, crvs)) for rv in self.rvs)
        self._joint_entropy = float(entropy(dist, list(flatten(self.rvs)), crvs))

    def rate_point(self, lambdas, niter=None, maxiter=1000, polish=1e-6, rng=None, **kwargs):
        """
        Compute the Gray-Wyner rate point supporting a weight vector.

        Parameters
        ----------
        lambdas : iterable of float
            The weights ``(lambda_0, lambda_1, ..., lambda_n)``.
        niter : int, None
            Number of basin hops.
        maxiter : int
            Inner optimizer iterations.
        polish : float
            Polishing cutoff; if falsey, no polishing.
        kwargs : dict
            Additional keyword arguments for `GrayWynerOptimizer`, such as
            ``allow_signed`` or ``rate_equalities``.

        Returns
        -------
        point : GrayWynerPoint
            The supporting ``(common, private)`` rate point.
        """
        opt = GrayWynerOptimizer(
            self.dist,
            lambdas,
            rvs=self.rvs,
            crvs=self.crvs,
            distortions=self.distortions,
            bounds=self.bounds,
            bound=self.bound,
            **kwargs,
        )
        opt.optimize(niter=niter, maxiter=maxiter, polish=polish, rng=rng)
        return opt.rates()

    def region(self, num=20, niter=None, maxiter=1000, seed=None):
        """
        Sample the lower boundary of the achievable rate region.

        Weight vectors are drawn on the ``(n + 1)``-simplex (the vertices, plus
        random Dirichlet samples) and the supporting rate point of each is
        computed.

        Parameters
        ----------
        num : int
            The number of random weight vectors to sample (in addition to the
            ``n + 1`` simplex vertices).
        niter : int, None
            Number of basin hops per point.
        maxiter : int
            Inner optimizer iterations.
        seed : int, None
            Seed for the random weight sampler.

        Returns
        -------
        points : list of GrayWynerPoint
            The sampled boundary points.
        """
        rng = np.random.default_rng(seed)
        dim = self.n + 1

        weights = list(np.eye(dim))  # vertices: pure common, pure each private
        weights += list(rng.dirichlet(np.ones(dim), size=num))

        points = parallel_sweep(
            lambda w, task_rng: self.rate_point(w, niter=niter, maxiter=maxiter, rng=task_rng),
            weights,
        )
        return points

    def tension_point(self, lambdas, **kwargs):
        """
        Compute a point of the region of tension supporting a weight vector.

        Parameters
        ----------
        lambdas : iterable of float
            The Gray-Wyner weights ``(lambda_0, lambda_1, ..., lambda_n)``.
        kwargs : dict
            Additional keyword arguments for :meth:`rate_point`.

        Returns
        -------
        point : TensionPoint
            The supporting point, in tension coordinates.
        """
        rates = self.rate_point(lambdas, **kwargs)
        return rates_to_tension(rates, self._marginal_entropies, self._joint_entropy)

    def tension_region(self, num=20, niter=None, maxiter=1000, seed=None):
        """
        Sample the lower boundary of the region of tension.

        The region proper is the increasing hull of the attainable tension
        points :cite:`prabhakaran2014assisted`; the points returned here are
        attainable ones on its lower boundary.

        Parameters
        ----------
        num : int
            The number of random weight vectors to sample, in addition to the
            ``n + 1`` simplex vertices.
        niter : int, None
            Number of basin hops per point.
        maxiter : int
            Inner optimizer iterations.
        seed : int, None
            Seed for the random weight sampler.

        Returns
        -------
        points : list of TensionPoint
            The sampled boundary points.
        """
        rates = self.region(num=num, niter=niter, maxiter=maxiter, seed=seed)
        return [rates_to_tension(point, self._marginal_entropies, self._joint_entropy) for point in rates]

    def extension_profile(self, num=20, niter=None, maxiter=1000, seed=None):
        """
        Sample the lower boundary of the extension profile.

        The extension profile :cite:`matveev2026beyond` records
        ``(H[X_i | W], ..., T[X_{0:n} | W])``; for a pair this is the
        ``(H[X|W], H[Y|W], I[X:Y|W])`` body of that reference.

        Parameters
        ----------
        num : int
            The number of random weight vectors to sample, in addition to the
            ``n + 1`` simplex vertices.
        niter : int, None
            Number of basin hops per point.
        maxiter : int
            Inner optimizer iterations.
        seed : int, None
            Seed for the random weight sampler.

        Returns
        -------
        points : list of ExtensionPoint
            The sampled boundary points.
        """
        rates = self.region(num=num, niter=niter, maxiter=maxiter, seed=seed)
        return [rates_to_extension(point, self._joint_entropy) for point in rates]

    def mutual_information_region(self, num=20, niter=None, maxiter=1000, seed=None):
        """
        Sample the lower boundary of the mutual information region.

        The mutual information region :cite:`li2018extended` records
        ``(I[X_i : W], ..., I[X_{0:n} : W])``. Unlike the region of tension,
        which tensorizes exactly, this region is only superadditive.

        Parameters
        ----------
        num : int
            The number of random weight vectors to sample, in addition to the
            ``n + 1`` simplex vertices.
        niter : int, None
            Number of basin hops per point.
        maxiter : int
            Inner optimizer iterations.
        seed : int, None
            Seed for the random weight sampler.

        Returns
        -------
        points : list of MutualInformationPoint
            The sampled boundary points.
        """
        rates = self.region(num=num, niter=niter, maxiter=maxiter, seed=seed)
        return [rates_to_mutual_information(point, self._marginal_entropies) for point in rates]

    def _zero_tension_equalities(self, exclude=()):
        """
        Affine rate constraints pinning tension coordinates to zero.

        ``tau_i = R_0 + R_i - H[X_i]`` and
        ``tau_res = R_0 + sum_i R_i - H[X_{0:n}]``, so each vanishing
        coordinate is one affine equality in the rates.

        Parameters
        ----------
        exclude : iterable of {int, 'residual'}
            Coordinates to leave unconstrained. Source coordinates are given
            by index; the residual by the string ``'residual'``.

        Returns
        -------
        equalities : list of (list of float, float)
            Constraints in the ``(weights, constant)`` form taken by
            `GrayWynerOptimizer`.
        """
        exclude = set(exclude)
        equalities = []

        for i, h_i in enumerate(self._marginal_entropies):
            if i not in exclude:
                weights = [0.0] * (self.n + 1)
                weights[0] = 1.0
                weights[i + 1] = 1.0
                equalities.append((weights, h_i))

        if "residual" not in exclude:
            equalities.append(([1.0] * (self.n + 1), self._joint_entropy))

        return equalities

    def tension_intercepts(self, niter=None, maxiter=1000):
        """
        The axis intercepts of the region of tension.

        These are the ``T_i`` of :cite:`prabhakaran2014assisted` (their
        equation 1): the smallest value each coordinate can take when every
        other coordinate is zero. They are constrained optima rather than
        supporting-hyperplane queries.

        The residual intercept satisfies ``T_res = I[X:Y] - K[X:Y]``, which
        is the Gacs-Korner corollary of :cite:`prabhakaran2014assisted`
        (their Corollary 3.3); the source intercepts are the Wolf-Wullschleger
        monotones.

        Parameters
        ----------
        niter : int, None
            Number of basin hops per intercept.
        maxiter : int
            Inner optimizer iterations.

        Returns
        -------
        intercepts : dict
            Mapping from coordinate (source index, or ``'residual'``) to the
            intercept value.
        """
        intercepts = {}

        for i in range(self.n):
            # minimize tau_i = R_0 + R_i - H[X_i] with every other tension zero
            lambdas = [0.0] * (self.n + 1)
            lambdas[0] = 1.0
            lambdas[i + 1] = 1.0
            point = self.rate_point(
                lambdas,
                niter=niter,
                maxiter=maxiter,
                rate_equalities=self._zero_tension_equalities(exclude=[i]),
            )
            tension = rates_to_tension(point, self._marginal_entropies, self._joint_entropy)
            intercepts[i] = max(tension.tensions[i], 0.0)

        # minimize tau_res = R_0 + sum_i R_i - H[X_{0:n}] with every source tension zero
        point = self.rate_point(
            [1.0] * (self.n + 1),
            niter=niter,
            maxiter=maxiter,
            rate_equalities=self._zero_tension_equalities(exclude=["residual"]),
        )
        tension = rates_to_tension(point, self._marginal_entropies, self._joint_entropy)
        intercepts["residual"] = max(tension.residual, 0.0)

        return intercepts

    def corner_points(self, niter=None, maxiter=1000):
        """
        The named common-information operating points of the network.

        For a lossless network the corners are the standard common
        informations, computed by delegating to their canonical
        implementations so the values stay consistent across `dit`. The
        returned values are the common-rate (``R_0``) coordinates of those
        operating points.

        Parameters
        ----------
        niter : int, None
            Number of basin hops (forwarded to the optimization-based
            measures).
        maxiter : int
            Inner optimizer iterations (forwarded likewise).

        Returns
        -------
        corners : dict
            A mapping from measure name to its ``R_0`` value.
        """
        from ...multivariate import (
            exact_common_information,
            gk_common_information,
            kamath_common_information,
            wyner_common_information,
        )

        if not self._lossless:
            return {
                "lossy_wyner": lossy_wyner_common_information(
                    self.dist,
                    bounds=self.bounds,
                    distortions=self.distortions,
                    rvs=self.rvs,
                    crvs=self.crvs,
                    niter=niter,
                    maxiter=maxiter,
                ),
            }

        return {
            "gacs_korner": gk_common_information(self.dist, self.rvs, self.crvs),
            "wyner": wyner_common_information(self.dist, self.rvs, self.crvs, niter=niter, maxiter=maxiter),
            "exact": exact_common_information(self.dist, self.rvs, self.crvs, niter=niter, maxiter=maxiter),
            "kamath": kamath_common_information(self.dist, self.rvs, self.crvs),
        }


@unitful
def lossy_wyner_common_information(
    dist,
    bounds=None,
    distortions=None,
    rvs=None,
    crvs=None,
    niter=None,
    maxiter=1000,
    bound=None,
):
    """
    The lossy Wyner common information ``C(D_1, ..., D_n)``.

    This is the minimum common rate ``R_0 = I(X_{1:n} : W)`` over auxiliary
    variables ``W`` that place the network on its minimum sum-rate face while
    meeting every distortion budget (Viswanatha, Akyol, & Rose 2014). For
    ``D_i = 0`` (lossless) it coincides with the standard Wyner common
    information.

    Parameters
    ----------
    dist : Distribution
        The source distribution.
    bounds : list, None
        Per-decoder distortion budgets ``D_i``. If None (or all zero), the
        lossless Wyner common information is returned.
    distortions : list, None
        Per-decoder distortion matrices (None entries are lossless).
    rvs : list of lists, None
        The source groups. If None, each variable is its own source.
    crvs : list, None
        Variables to condition on.
    niter : int, None
        Number of basin hops.
    maxiter : int
        Inner optimizer iterations.
    bound : int, None
        Optional cap on the cardinality of ``W``.

    Returns
    -------
    C : float
        The lossy Wyner common information.
    """
    lossless = bounds is None or all(b <= 0 for b in bounds)

    if lossless:
        from ...multivariate import wyner_common_information

        return wyner_common_information(dist, rvs, crvs, niter=niter, maxiter=maxiter, bound=bound)

    rvs = [[i] for i in flatten(dist.rvs)] if rvs is None else rvs
    n = len(rvs)

    # The lossy common information is the smallest common rate R_0 on the
    # minimum sum-rate ("Pangloss") face of the region (Viswanatha, Akyol &
    # Rose 2014). Minimizing the total rate R_0 + sum_i R_i reaches that face;
    # the small extra weight on R_0 breaks ties toward the minimum common rate.
    # In the lossless limit this reduces to Wyner's common information.
    eps = 1e-3
    lambdas = [1.0 + eps] + [1.0] * n

    opt = GrayWynerOptimizer(
        dist,
        lambdas,
        rvs=rvs,
        crvs=crvs,
        distortions=distortions,
        bounds=bounds,
        bound=bound,
    )
    opt.optimize(niter=niter, maxiter=maxiter)
    point = opt.rates()
    return point.common
