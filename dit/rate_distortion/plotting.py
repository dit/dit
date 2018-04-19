"""
Routines for plotting rate-distortion and information bottleneck curves.
"""

from abc import ABCMeta, abstractmethod
from collections import namedtuple
from operator import attrgetter

from six import with_metaclass

import numpy as np
import matplotlib.pyplot as plt

from .curves import RDCurve, IBCurve


__all__ = (
    'RDPlotter',
    'IBPlotter',
)

Axis = namedtuple('Axis', ['data', 'limit', 'label'])


def _rescale_axes(ax, xmax, ymax):
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
    xmax = ax.get_xlim()[1] if np.isnan(xmax) else 1.05*xmax
    ymax = ax.get_ylim()[1] if np.isnan(ymax) else 1.05*ymax

    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)


class BasePlotter(with_metaclass(ABCMeta, object)):
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
        try:
            ax_lim_1 = max([axis_1.limit(c) for c in self.curves])
            if ax_lim_1 is None:
                raise TypeError
        except TypeError:
            ax_lim_1 = max([c.betas[-1] for c in self.curves])
        ax_lim_2 = max([axis_2.limit(c) for c in self.curves])
        _rescale_axes(ax, ax_lim_1, ax_lim_2)

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
        ax = super(IBPlotter, self)._plot(ax, axis_1, axis_2, downsample)

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
