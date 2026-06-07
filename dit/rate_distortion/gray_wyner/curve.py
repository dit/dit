"""
The Gray-Wyner trade-off curve.

`GrayWynerCurve` sweeps a scalar tension between the common rate ``R_0`` and
the total private rate ``sum_i R_i`` at a fixed distortion profile, tracing the
two-dimensional projection of the Gray-Wyner rate region most often drawn.
"""

import numpy as np

from ...multivariate import entropy
from ...utils import flatten
from .network import GrayWynerNetwork

__all__ = ("GrayWynerCurve",)


class GrayWynerCurve:
    """
    Compute the common-rate vs total-private-rate trade-off curve.
    """

    def __init__(
        self,
        dist,
        rvs=None,
        crvs=None,
        distortions=None,
        bounds=None,
        s_min=0.0,
        s_max=4.0,
        s_num=21,
        niter=None,
        maxiter=1000,
        bound=None,
    ):
        """
        Initialize the curve computer.

        Parameters
        ----------
        dist : Distribution
            The source distribution.
        rvs : list of lists, None
            The source groups. If None, each variable is its own source.
        crvs : list, None
            Variables to condition on.
        distortions : list, None
            Per-decoder distortion matrices (None entries are lossless).
        bounds : list, None
            Per-decoder distortion budgets. If None, lossless.
        s_min, s_max : float
            The range of the private-rate weight ``s``. The weight vector at
            each point is ``(1, s, s, ..., s)``.
        s_num : int
            The number of points along the curve.
        niter : int, None
            Number of basin hops per point.
        maxiter : int
            Inner optimizer iterations.
        bound : int, None
            Optional cap on the cardinality of ``W``.
        """
        self.dist = dist.copy()
        self.rvs = [[i] for i in flatten(dist.rvs)] if rvs is None else rvs
        self.crvs = crvs
        self.n = len(self.rvs)

        self._network = GrayWynerNetwork(
            dist,
            rvs=self.rvs,
            crvs=self.crvs,
            distortions=distortions,
            bounds=bounds,
            bound=bound,
        )

        rv = list(flatten(self.rvs))
        self._max_r0 = float(entropy(dist, rv, crvs))
        self._max_private = float(sum(entropy(dist, [r], crvs) for r in self.rvs))

        self.betas = np.linspace(s_min, s_max, s_num)
        self.label = getattr(dist, "name", "") or "Gray-Wyner"

        self.compute(niter=niter, maxiter=maxiter)

    def __add__(self, other):  # pragma: no cover
        """
        Combine two curves into a `GrayWynerPlotter`.

        Parameters
        ----------
        other : GrayWynerCurve
            The curve to aggregate with `self`.

        Returns
        -------
        plotter : GrayWynerPlotter
            A plotter holding both curves.
        """
        from .plotting import GrayWynerPlotter

        if isinstance(other, GrayWynerCurve):
            return GrayWynerPlotter(self, other)
        return NotImplemented

    def compute(self, niter=None, maxiter=1000):
        """
        Sweep the private-rate weight and compute the trade-off curve.

        Parameters
        ----------
        niter : int, None
            Number of basin hops per point.
        maxiter : int
            Inner optimizer iterations.
        """
        r0s = []
        private_totals = []

        for s in self.betas:
            lambdas = [1.0] + [float(s)] * self.n
            point = self._network.rate_point(lambdas, niter=niter, maxiter=maxiter)
            r0s.append(point.common)
            private_totals.append(sum(point.private))

        self.r0s = np.asarray(r0s)
        self.private_totals = np.asarray(private_totals)
        self.sum_rates = self.r0s + self.private_totals

    def plot(self, downsample=5):  # pragma: no cover
        """
        Plot the trade-off curve.

        Parameters
        ----------
        downsample : int
            How frequently to place markers along the curve.

        Returns
        -------
        fig : plt.Figure
            The resulting figure.
        """
        from .plotting import GrayWynerPlotter

        return GrayWynerPlotter(self).plot(downsample)
