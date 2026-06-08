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

    def rate_point(self, lambdas, niter=None, maxiter=1000, polish=1e-6, rng=None):
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
