"""
Routines for plotting rate-distortion and information bottleneck curves.
"""

from abc import ABCMeta, abstractmethod
from collections import namedtuple
from operator import attrgetter

import matplotlib.pyplot as plt
import numpy as np

from ..exceptions import ditException
from ..multivariate import entropy, total_correlation
from .curves import IBCurve, PFCurve, RDCurve

__all__ = (
    "BottleneckRegionPlotter",
    "IBPlotter",
    "RDPlotter",
)


Axis = namedtuple("Axis", ["data", "limit", "label"])


def _rescale_axes(ax, xmin=None, xmax=None, ymin=None, ymax=None):
    """
    Given a matplotlib axis, set the xmin and ymin to zero, and the xmax and
    ymax accordingly.

    Parameters
    ----------
    ax : plt.axis
        The axis to adjust the limits of.
    xmax : float
        The xmax value.
    ymax : float
        Tye ymax value.
    """
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    if xmin is not None:
        x_min = xmin
    if xmax is not None and not np.isnan(xmax):
        x_max = 1.05 * xmax
    if ymin is not None:
        y_min = ymin
    if ymax is not None and not np.isnan(ymax):
        y_max = 1.05 * ymax

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)


class BasePlotter(metaclass=ABCMeta):
    """
    A plotter of rate-distortion-like curves.
    """

    _beta_axis = Axis(attrgetter("betas"), lambda _: None, r"$\beta$")
    _rank_axis = Axis(attrgetter("ranks"), attrgetter("_max_rank"), r"rank")
    _alphabet_axis = Axis(attrgetter("alphabets"), attrgetter("_max_rank"), r"$|\mathcal{A}|$")
    _distortion_axis = Axis(
        attrgetter("distortions"), attrgetter("_max_distortion"), r"$\langle d(x, \hat{x}) \rangle$"
    )

    def __init__(self, *curves):
        """
        Initialize the plotter.

        Parameters
        ----------
        curves : *{RDCurve, IBCurve}
            The curves to plot.
        """
        self.curves = curves

    def __add__(self, other):
        """
        Add a new curve to the plot, or combine two plotters.

        Parameters
        ----------
        other : RDCurve, IBCurve, BasePlotter
            If a curve, add it to self.curves. If a plotter, add all its
            curves to self.curves.
        """
        if isinstance(other, self._curve_type):
            self.curves += (other,)
            return self
        elif isinstance(other, type(self)):
            self.curves += other.curves
            return self
        else:
            return NotImplemented

    def _plot(self, ax, axis_1, axis_2, downsample):
        """
        Plot two arrays relative to one another.

        Parameters
        ----------
        ax : mpl.Axis
            The axis to plot on.
        axis_1 : Axis
            The axis to put along the x axis.
        axis_2 : Axis
            The axis to put along the y axis.
        downsample : int
            The rate at which to put markers along the curve.

        Returns
        -------
        ax : mpl.Axis
            The modified axis.
        """
        for curve in self.curves:
            x = axis_1.data(curve)
            y = axis_2.data(curve)
            line = ax.plot(x, y, lw=2, label=curve.label)[0]
            ax.scatter(x[::downsample], y[::downsample], c=curve.betas[::downsample])

            # if there is an upper bound, plot it.
            lim_1 = axis_1.limit(curve)
            if lim_1 is not None:
                ax.axvline(lim_1, ls=":", c=line.get_c())
            lim_2 = axis_2.limit(curve)
            if lim_2 is not None:
                ax.axhline(lim_2, ls=":", c=line.get_c())

        ax.set_xlabel(axis_1.label)
        ax.set_ylabel(axis_2.label)

        # determine bounds so that axes can be scaled.
        ax_min_1 = 0
        ax_min_2 = 0

        try:
            ax_max_1 = max(axis_1.limit(c) for c in self.curves)
            if ax_max_1 is None:
                raise TypeError
        except TypeError:
            ax_min_1 = min(c.betas[0] for c in self.curves)
            ax_max_1 = max(c.betas[-1] for c in self.curves)

        ax_max_2 = max(axis_2.limit(c) for c in self.curves)
        _rescale_axes(ax, xmin=ax_min_1, xmax=ax_max_1, ymin=ax_min_2, ymax=ax_max_2)

        return ax

    @abstractmethod
    def plot(self, downsample=5):
        """
        Plot several rate-distortion-like curves.

        Parameters
        ----------
        downsample : int
            Show markers every `downsample` points.

        Returns
        -------
        fig : mpl.Figure
            The figure with several subplots.
        """
        pass


class RDPlotter(BasePlotter):
    """
    A plotter for rate-distortion curves.
    """

    _rate_axis = Axis(attrgetter("rates"), attrgetter("_max_rate"), r"$I[X:\hat{X}]$")

    _curve_type = RDCurve

    def plot(self, downsample=5):
        """
        Plot various views of the rate-distortion curve.

        Parameters
        ----------
        downsample : int
            Show markers every `downsample` points.
        """
        fig, axs = plt.subplots(2, 2, figsize=(16, 8))

        self._plot(axs[0, 0], self._beta_axis, self._rate_axis, downsample)
        axs[0, 0].legend(loc="best")
        self._plot(axs[0, 1], self._distortion_axis, self._rate_axis, downsample)
        self._plot(axs[1, 0], self._beta_axis, self._distortion_axis, downsample)
        axs[1, 1].axis("off")

        return fig


class IBPlotter(BasePlotter):
    """
    A plotter for information bottleneck curves.
    """

    _complexity_axis = Axis(attrgetter("complexities"), attrgetter("_max_complexity"), r"$I[X:\hat{X}]$")
    _entropy_axis = Axis(attrgetter("entropies"), attrgetter("_max_complexity"), r"$H[\hat{X}]$")
    _relevance_axis = Axis(attrgetter("relevances"), attrgetter("_max_relevance"), r"$I[Y:\hat{X}]$")
    _error_axis = Axis(attrgetter("errors"), attrgetter("_max_relevance"), r"$I[X:Y|\hat{X}]$")

    _curve_type = IBCurve

    def _plot(self, ax, axis_1, axis_2, downsample):
        """
        Plot various views of the information bottleneck curve.

        Parameters
        ----------
        downsample : int
            Show markers every `downsample` points.
        """
        ax = super()._plot(ax, axis_1, axis_2, downsample)

        if axis_1 is self._beta_axis:
            for curve in self.curves:
                for kink in curve.find_kinks():
                    ax.axvline(kink, ls=":", c="k")

        return ax

    def plot(self, downsample=5):
        """
        Plot various views of the information bottleneck curve.

        Parameters
        ----------
        downsample : int
            Show markers every `downsample` points.
        """
        fig, axs = plt.subplots(3, 2, figsize=(16, 16))

        self._plot(axs[0, 0], self._beta_axis, self._complexity_axis, downsample)
        axs[0, 0].legend(loc="best")
        self._plot(axs[1, 0], self._beta_axis, self._relevance_axis, downsample)
        self._plot(axs[2, 0], self._beta_axis, self._rank_axis, downsample)
        self._plot(axs[0, 1], self._complexity_axis, self._relevance_axis, downsample)
        self._plot(axs[1, 1], self._error_axis, self._complexity_axis, downsample)
        self._plot(axs[2, 1], self._distortion_axis, self._complexity_axis, downsample)

        return fig


def _boundary(xs, ys, anchors, upper):
    """
    Sort a boundary curve by x, adding the anchor points and keeping the
    extreme y among repeated x values.
    """
    points = {}
    for x, y in [*zip(xs, ys, strict=True), *anchors]:
        x = round(float(x), 12)
        points[x] = max(points.get(x, y), y) if upper else min(points.get(x, y), y)
    x = np.array(sorted(points))
    return x, np.array([points[k] for k in x])


def _region_corner(ib_curve, pf_curve):
    """
    The point ``(H[X | Z], I[S : X | Z])`` achieved by releasing T = X.
    """
    if pf_curve is not None:
        return pf_curve._max_utility, pf_curve._max_leakage
    if ib_curve is None:
        msg = "At least one curve is required."
        raise ditException(msg)
    z = list(ib_curve._z)
    return entropy(ib_curve.dist, ib_curve._x, z), total_correlation(ib_curve.dist, [ib_curve._x, ib_curve._y], z)


def _region_envelopes(ib_curve=None, pf_curve=None, num=201):
    """
    Evaluate the boundaries of the region of achievable
    ``(I[X : T | Z], I[S : T | Z])`` pairs on a shared grid.

    Both boundaries pass through ``(0, 0)`` (T constant) and
    ``(H[X | Z], I[S : X | Z])`` (T = X); the information bottleneck curve is the
    upper boundary and the privacy funnel curve the lower
    :cite:`asoodeh2020bottleneck`.

    Parameters
    ----------
    ib_curve : IBCurve, None
        The information bottleneck curve of X with respect to S.
    pf_curve : PFCurve, None
        The privacy funnel curve of S and X.
    num : int
        The number of grid points.

    Returns
    -------
    grid : np.ndarray
        Utilities ``I[X : T | Z]`` from 0 to ``H[X | Z]``.
    upper : np.ndarray, None
        The information bottleneck boundary on the grid, if given.
    lower : np.ndarray, None
        The privacy funnel boundary on the grid, if given.
    """
    max_x, max_y = _region_corner(ib_curve, pf_curve)
    anchors = [(0.0, 0.0), (max_x, max_y)]
    grid = np.linspace(0.0, max_x, num)

    upper = lower = None
    if ib_curve is not None:
        upper = np.interp(grid, *_boundary(ib_curve.complexities, ib_curve.relevances, anchors, upper=True))
    if pf_curve is not None:
        lower = np.interp(grid, *_boundary(pf_curve.utilities, pf_curve.leakages, anchors, upper=False))
    return grid, upper, lower


def _validate_region_curves(ib_curve, pf_curve):
    """
    Ensure the two curves bound the same region.
    """
    if ib_curve is not None and ib_curve._variant != "ib":
        msg = "The region is bounded by the standard information bottleneck (variant 'ib')."
        raise ditException(msg)
    if ib_curve is None or pf_curve is None:
        return
    same_vars = (
        list(ib_curve._x) == list(pf_curve._x)
        and list(ib_curve._y) == list(pf_curve._s)
        and list(ib_curve._z) == list(pf_curve._z)
    )
    if not same_vars:
        msg = "The IB curve must be of X with respect to S: IBCurve(rvs=[X, S]) and PFCurve(rvs=[S, X])."
        raise ditException(msg)
    if not ib_curve.dist.is_approx_equal(pf_curve.dist):
        msg = "The curves must be computed from the same distribution."
        raise ditException(msg)


class BottleneckRegionPlotter:
    """
    Plot the information bottleneck and privacy funnel curves together. For a
    fixed p(s, x), every release T of X gives a point
    ``(I[X : T | Z], I[S : T | Z])``; these points form a convex region whose
    upper boundary is the information bottleneck curve and whose lower boundary
    is the privacy funnel curve :cite:`asoodeh2020bottleneck`.
    """

    def __init__(self, ib_curve=None, pf_curve=None):
        """
        Initialize the plotter.

        Parameters
        ----------
        ib_curve : IBCurve, None
            The information bottleneck curve of X with respect to S, i.e.
            ``IBCurve(dist, rvs=[X, S])``.
        pf_curve : PFCurve, None
            The privacy funnel curve, i.e. ``PFCurve(dist, rvs=[S, X])``.

        Raises
        ------
        ditException
            If neither curve is given, the IB curve is not the standard
            variant, or the curves are of different distributions or
            variables.
        """
        if ib_curve is None and pf_curve is None:
            msg = "At least one curve is required."
            raise ditException(msg)
        _validate_region_curves(ib_curve, pf_curve)
        self.ib_curve = ib_curve
        self.pf_curve = pf_curve

    @classmethod
    def from_dist(cls, dist, rvs=None, crvs=None, ib_kwargs=None, pf_kwargs=None):
        """
        Compute both curves of a distribution and build a plotter for them.

        Parameters
        ----------
        dist : Distribution
            The distribution of interest.
        rvs : iterable, None
            The private variables S and the public variables X. If None, use
            [0], [1].
        crvs : iterable, None
            The random variables to condition on.
        ib_kwargs : dict, None
            Extra keyword arguments for :class:`IBCurve`.
        pf_kwargs : dict, None
            Extra keyword arguments for :class:`PFCurve`.

        Returns
        -------
        plotter : BottleneckRegionPlotter
            The plotter.
        """
        s, x = rvs if rvs is not None else ([0], [1])
        ib_curve = IBCurve(dist, rvs=[x, s], crvs=crvs, **(ib_kwargs or {}))
        pf_curve = PFCurve(dist, rvs=[s, x], crvs=crvs, **(pf_kwargs or {}))
        return cls(ib_curve=ib_curve, pf_curve=pf_curve)

    def plot(self, ax=None, num=201):  # pragma: no cover
        """
        Plot the region.

        Parameters
        ----------
        ax : mpl.Axis, None
            The axis to plot on. If None, make a new figure.
        num : int
            The number of points used to shade the region.

        Returns
        -------
        fig : mpl.Figure
            The figure containing the plot.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure

        grid, upper, lower = _region_envelopes(self.ib_curve, self.pf_curve, num=num)
        max_x, max_y = _region_corner(self.ib_curve, self.pf_curve)

        if upper is not None and lower is not None:
            ax.fill_between(grid, lower, upper, alpha=0.2, label="achievable")
        if upper is not None:
            ax.plot(self.ib_curve.complexities, self.ib_curve.relevances, "o", ms=3, c="C0")
            ax.plot(grid, upper, lw=2, c="C0", label="IB")
        if lower is not None:
            ax.plot(self.pf_curve.utilities, self.pf_curve.leakages, "o", ms=3, c="C1")
            ax.plot(grid, lower, lw=2, c="C1", label="PF")
            g0 = self.pf_curve.perfect_utility
            if g0:
                ax.plot([g0], [0.0], "*", ms=12, c="C3", label=r"perfect privacy $g_0$")

        ax.plot([0, max_y], [0, max_y], ls="--", c="k", lw=1, label="data processing")
        ax.axhline(max_y, ls=":", c="k", lw=1)
        ax.axvline(max_x, ls=":", c="k", lw=1)

        ax.set_xlabel(r"$I[X:T]$")
        ax.set_ylabel(r"$I[S:T]$")
        ax.legend(loc="best")
        _rescale_axes(ax, xmin=0, xmax=max_x, ymin=0, ymax=max_y)

        return fig
