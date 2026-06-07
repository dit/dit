"""
Plotting for Gray-Wyner trade-off curves.
"""

from operator import attrgetter

import matplotlib.pyplot as plt

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
