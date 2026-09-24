"""
Plotting for Gray-Wyner trade-off curves.
"""

from operator import attrgetter

import matplotlib.pyplot as plt
import numpy as np

from ...exceptions import ditException
from ..plotting import Axis, BasePlotter
from .curve import GrayWynerCurve

__all__ = ("GrayWynerPlotter",)


class GrayWynerPlotter(BasePlotter):
    """
    A plotter for Gray-Wyner trade-off curves.
    """

    _curve_type = GrayWynerCurve

    _r0_axis = Axis(attrgetter("r0s"), attrgetter("_max_r0"), r"$R_0$  (common rate)")
    _private_axis = Axis(attrgetter("private_totals"), attrgetter("_max_private"), r"$\sum_i R_i$  (private rate)")
    _sum_axis = Axis(attrgetter("sum_rates"), attrgetter("_max_private"), r"$R_0 + \sum_i R_i$")

    def plot(self, downsample=5):
        """
        Plot the common-rate vs private-rate trade-off, and each rate against
        the sweep weight.

        Parameters
        ----------
        downsample : int
            How frequently to place markers along the curve.

        Returns
        -------
        fig : plt.Figure
            The resulting figure.
        """
        fig, axs = plt.subplots(2, 2, figsize=(16, 8))

        self._plot(axs[0, 0], self._r0_axis, self._private_axis, downsample)
        axs[0, 0].legend(loc="best")
        self._plot(axs[0, 1], self._beta_axis, self._r0_axis, downsample)
        self._plot(axs[1, 0], self._beta_axis, self._private_axis, downsample)
        self._plot(axs[1, 1], self._beta_axis, self._sum_axis, downsample)

        return fig

    @staticmethod
    def plot_tension(network, num=20, niter=None, maxiter=1000, seed=None, ax=None):
        """
        Plot the lower boundary of the region of tension of a pair.

        The region proper is the increasing hull of these points, so the
        surface drawn is the "floor" above which every attainable tension
        triple lies. The three axis intercepts are marked; the residual one
        is ``I[X:Y] - K[X:Y]``.

        Parameters
        ----------
        network : GrayWynerNetwork
            The network whose tension region is plotted. It must have
            exactly two sources, since the region is three-dimensional.
        num : int
            The number of directions to sample.
        niter : int, None
            Number of basin hops per point.
        maxiter : int
            Inner optimizer iterations.
        seed : int, None
            Seed for the random weight sampler.
        ax : mpl.Axis, None
            A 3D axis to draw on. If None, a new figure is created.

        Returns
        -------
        fig : plt.Figure
            The resulting figure.
        """
        if network.n != 2:
            msg = f"The tension region is plottable for 2 sources, not {network.n}."
            raise ditException(msg)

        points = network.tension_region(num=num, niter=niter, maxiter=maxiter, seed=seed)
        xs = np.array([p.tensions[0] for p in points])
        ys = np.array([p.tensions[1] for p in points])
        zs = np.array([p.residual for p in points])

        if ax is None:
            fig = plt.figure(figsize=(8, 7))
            ax = fig.add_subplot(projection="3d")
        else:
            fig = ax.get_figure()

        ax.plot_trisurf(xs, ys, zs, alpha=0.4, color="C0", linewidth=0.2, edgecolor="k")
        ax.scatter(xs, ys, zs, c="C0", s=12, depthshade=False)

        intercepts = network.tension_intercepts(niter=niter, maxiter=maxiter)
        corners = [(intercepts[0], 0, 0), (0, intercepts[1], 0), (0, 0, intercepts["residual"])]
        ax.scatter(*zip(*corners, strict=True), c="C3", s=60, marker="D", depthshade=False, label="intercepts")

        ax.set_xlabel(r"$\tau_1 = I[Y:W|X]$")
        ax.set_ylabel(r"$\tau_2 = I[X:W|Y]$")
        ax.set_zlabel(r"$\tau_3 = I[X:Y|W]$")
        ax.legend(loc="best")

        return fig

    @staticmethod
    def plot_shape(shape, ax=None):
        """
        Plot a shape function against its two entropy-forced envelopes.

        Parameters
        ----------
        shape : ShapeFunction
            The shape function to plot.
        ax : mpl.Axis, None
            An axis to draw on -- 3D for a pair, 2D otherwise. If None, a new
            figure is created.

        Returns
        -------
        fig : plt.Figure
            The resulting figure.
        """
        grid_shape = getattr(shape, "grid_shape", None)
        two_dimensional = shape.n == 2 and grid_shape is not None and len(grid_shape) == 2

        if ax is None:
            fig = plt.figure(figsize=(9, 7))
            ax = fig.add_subplot(projection="3d") if two_dimensional else fig.add_subplot()
        else:
            fig = ax.get_figure()

        if two_dimensional:
            alphas = shape.alphas[:, 0].reshape(grid_shape)
            betas = shape.alphas[:, 1].reshape(grid_shape)
            ax.plot_surface(alphas, betas, shape.values.reshape(grid_shape), alpha=0.8, cmap="viridis")
            ax.plot_wireframe(
                alphas, betas, shape.minima.reshape(grid_shape), color="C3", alpha=0.3, rstride=2, cstride=2
            )
            ax.plot_wireframe(
                alphas, betas, shape.maxima.reshape(grid_shape), color="C2", alpha=0.3, rstride=2, cstride=2
            )
            ax.set_xlabel(r"$\alpha$")
            ax.set_ylabel(r"$\beta$")
            ax.set_zlabel(r"$S(\alpha, \beta)$")
        else:
            # More than two sources: the default grid is the symmetric
            # diagonal, so a single sweep parameter suffices.
            sweep = shape.alphas[:, 0]
            ax.plot(sweep, shape.values, lw=2, label=r"$S$")
            ax.plot(sweep, shape.minima, ls=":", c="C3", label=r"$S_{\min}$")
            ax.plot(sweep, shape.maxima, ls=":", c="C2", label=r"$S_{\max}$")
            ax.set_xlabel(r"$\alpha$")
            ax.set_ylabel(r"$S(\alpha, \ldots, \alpha)$")
            ax.legend(loc="best")

        ax.set_title(f"shape function ({shape.label})")

        return fig
