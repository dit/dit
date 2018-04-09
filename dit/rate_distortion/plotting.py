"""
"""

from abc import ABCMeta, abstractmethod
from collections import namedtuple
from operator import attrgetter

from six import with_metaclass

import matplotlib.pyplot as plt

from .curves import RDCurve, IBCurve


__all__ = (
    'RDPlotter',
)

Axis = namedtuple('Axis', ['data' ,'limit', 'label'])


def _rescale_axes(ax, xmax, ymax):
    """
    """
    ax.set_xlim(0, 1.05*xmax)
    ax.set_ylim(0, 1.05*ymax)


class BasePlotter(with_metaclass(ABCMeta, object)):
    """
    """
    _beta_axis = Axis(attrgetter('betas'), lambda _: None, r"$\beta$")
    _rank_axis = Axis(attrgetter('ranks'), attrgetter('_max_rank'), r"rank")
    _alphabet_axis = Axis(attrgetter('alphabets'), attrgetter('_max_rank'), r"$|\mathcal{A}|$")

    def __init__(self, *curves):
        """
        """
        self.curves = curves

    def __add__(self, other):
        """
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
        """
        for curve in self.curves:
            x = axis_1.data(curve)
            y = axis_2.data(curve)
            line = ax.plot(x, y, lw=2, label=curve.label)[0]
            ax.scatter(x[::downsample],
                       y[::downsample],
                       c=curve.betas[::downsample])

            lim_1 = axis_1.limit(curve)
            if lim_1 is not None:
                ax.axvline(lim_1, ls=':', c=line.get_c())
            lim_2 = axis_2.limit(curve)
            if lim_2 is not None:
                ax.axhline(lim_2, ls=':', c=line.get_c())

        ax.set_xlabel(axis_1.label)
        ax.set_ylabel(axis_2.label)

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
        """
        pass


class RDPlotter(BasePlotter):
    """
    """
    _rate_axis = Axis(attrgetter('rates'), attrgetter('_max_rate'), "$I[X:\hat{X}]$")
    _distortion_axis = Axis(attrgetter('distortions'), attrgetter('_max_distortion'), r"$\langle d(x, \hat{x}) \rangle$")

    _curve_type = RDCurve

    def plot(self, downsample=5):
        """
        """
        fig, axs = plt.subplots(2, 2, figsize=(16, 8))

        self._plot(axs[0, 0], self._beta_axis, self._rate_axis, downsample)
        axs[0, 0].legend(loc='best')
        self._plot(axs[0, 1], self._distortion_axis, self._rate_axis, downsample)
        self._plot(axs[1, 0], self._beta_axis, self._distortion_axis, downsample)

        return fig


class IBPlotter(BasePlotter):
    """
    """
    _complexity_axis = Axis(attrgetter('complexities'), attrgetter('_max_complexity'), "$I[X:T]$")
    _entropy_axis = Axis(attrgetter('entropies'), attrgetter('_max_complexity'), r"$H[T]$")
    _relevance_axis = Axis(attrgetter('relevances'), attrgetter('_max_relevance'), r"$I[Y:T]$")
    _error_axis = Axis(attrgetter('errors'), attrgetter('_max_relevance'), r"$I[X:Y|T]$")

    _curve_type = IBCurve

    def _plot(self, ax, axis_1, axis_2, downsample):
        """
        """
        ax = super(IBPlotter, self)._plot(ax, axis_1, axis_2, downsample)

        if axis_1 is self._beta_axis:
            for curve in self.curves:
                for kink in curve.find_kinks():
                    ax.axvline(kink, ls=':', c='k')

        return ax

    def plot(self, downsample=5):
        """
        """
        fig, axs = plt.subplots(3, 2, figsize=(16, 16))

        self._plot(axs[0, 0], self._beta_axis, self._complexity_axis, downsample)
        axs[0, 0].legend(loc='best')
        self._plot(axs[1, 0], self._beta_axis, self._relevance_axis, downsample)
        self._plot(axs[2, 0], self._beta_axis, self._rank_axis, downsample)
        self._plot(axs[0, 1], self._complexity_axis, self._relevance_axis, downsample)
        self._plot(axs[1, 1], self._entropy_axis, self._relevance_axis, downsample)
        self._plot(axs[2, 1], self._error_axis, self._complexity_axis, downsample)

        return fig
