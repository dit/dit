"""
Routines for plotting rate-distortion and information bottleneck curves.
"""

from abc import ABCMeta, abstractmethod
from collections import namedtuple
from operator import attrgetter

import matplotlib.pyplot as plt
import numpy as np

from .curves import IBCurve, RDCurve


__all__ = (
    'IBPlotter',
    'RDPlotter',
)


Axis = namedtuple('Axis', ['data', 'limit', 'label'])


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

    _beta_axis = Axis(attrgetter('betas'), lambda _: None, r"$\beta$")
    _rank_axis = Axis(attrgetter('ranks'), attrgetter('_max_rank'), r"rank")
    _alphabet_axis = Axis(attrgetter('alphabets'), attrgetter('_max_rank'), r"$|\mathcal{A}|$")
    _distortion_axis = Axis(attrgetter('distortions'), attrgetter('_max_distortion'), r"$\langle d(x, \hat{x}) \rangle$")

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
            ax.scatter(x[::downsample],
                       y[::downsample],
                       c=curve.betas[::downsample])

            # if there is an upper bound, plot it.
            lim_1 = axis_1.limit(curve)
            if lim_1 is not None:
                ax.axvline(lim_1, ls=':', c=line.get_c())
            lim_2 = axis_2.limit(curve)
            if lim_2 is not None:
                ax.axhline(lim_2, ls=':', c=line.get_c())

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

    _rate_axis = Axis(attrgetter('rates'), attrgetter('_max_rate'), "$I[X:\hat{X}]$")

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
        axs[0, 0].legend(loc='best')
        self._plot(axs[0, 1], self._distortion_axis, self._rate_axis, downsample)
        self._plot(axs[1, 0], self._beta_axis, self._distortion_axis, downsample)
        axs[1, 1].axis('off')

        return fig


class IBPlotter(BasePlotter):
    """
    A plotter for information bottleneck curves.
    """

    _complexity_axis = Axis(attrgetter('complexities'), attrgetter('_max_complexity'), "$I[X:\hat{X}]$")
    _entropy_axis = Axis(attrgetter('entropies'), attrgetter('_max_complexity'), r"$H[\hat{X}]$")
    _relevance_axis = Axis(attrgetter('relevances'), attrgetter('_max_relevance'), r"$I[Y:\hat{X}]$")
    _error_axis = Axis(attrgetter('errors'), attrgetter('_max_relevance'), r"$I[X:Y|\hat{X}]$")

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
                    ax.axvline(kink, ls=':', c='k')

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
        axs[0, 0].legend(loc='best')
        self._plot(axs[1, 0], self._beta_axis, self._relevance_axis, downsample)
        self._plot(axs[2, 0], self._beta_axis, self._rank_axis, downsample)
        self._plot(axs[0, 1], self._complexity_axis, self._relevance_axis, downsample)
        self._plot(axs[1, 1], self._error_axis, self._complexity_axis, downsample)
        self._plot(axs[2, 1], self._distortion_axis, self._complexity_axis, downsample)

        return fig
