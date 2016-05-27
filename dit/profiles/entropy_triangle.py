"""
The entropy triangle, from [Valverde-Albacete, Francisco Jose, and Carmen
Pelaez-Moreno. "The Multivariate Entropy Triangle and Applications." Hybrid
Artificial Intelligent Systems. Springer International Publishing, 2016.
647-658].
"""
from abc import ABCMeta, abstractmethod

from ..distribution import BaseDistribution
from ..distconst import product_distribution, uniform_like
from ..multivariate import (entropy, residual_entropy, dual_total_correlation,
                            total_correlation)

__all__ = ['EntropyTriangle',
           'EntropyTriangle2',
]

class BaseEntropyTriangle(object):
    """
    BaseEntropyTriangle

    Static Attributes
    -----------------
    left_label : str
        The label for the bottom axis when plotting.
    right_label : str
        The label for the right axis when plotting.
    bottom_label : str
        The label for the bottom axis when plotting.

    Attributes
    ----------
    dists : [Distribution]
    points : list of tuples

    Methods
    -------
    draw
        Plot the entropy triangle.
    """
    __metaclass__ = ABCMeta

    left_label = r"$\operatorname{R}[\mathrm{dist}]$"
    right_label = r"$\operatorname{T}[\mathrm{dist}] + \operatorname{B}[\mathrm{dist}]$"
    bottom_label = r"$\Delta \operatorname{H}_{\Pi_\overline{X}}$"

    def __init__(self, dists):
        """
        Initialize the entropy triangle.

        Parameters
        ----------
        dists : [Distribution] or Distribution
            The list of distributions to plot on the entropy triangle. If a
            single distribution is provided, it alone will be computed.
        """
        if isinstance(dists, BaseDistribution):
            self.dists = [dists]
        else:
            self.dists = dists
        self.points = [self._compute_point(dist) for dist in self.dists]

    @staticmethod
    @abstractmethod
    def _compute_point(dist): # pragma: no cover
        """
        Compute the three normalized axis.

        Parameters
        ----------
        dist : Distribution
            The distribution to compute values for.
        """
        pass

    def draw(self, ax=None, setup=True, marker='o', color='k'): # pragma: no cover
        """
        Plot the entropy triangle.

        Parameters
        ----------
        ax : Axis or None
            The matplotlib axis to plot on. If none is provided, one will be
            constructed.
        setup : bool
            If true, labels, tick marks, gridlines, and a boundary will be added
            to the plot. Defaults to True.
        marker : str
            The matplotlib marker shape to use.
        color : str
            The color of marker to use.
        """
        import ternary

        if ax is None:
            fig, ax = ternary.figure()
            fig.set_size_inches(10, 8)
        else:
            ax = ternary.TernaryAxesSubplot(ax=ax)

        if setup:
            ax.boundary()
            ax.gridlines(multiple=0.1)

            fontsize = 20
            ax.set_title("Entropy Triangle", fontsize=fontsize)
            ax.left_axis_label(self.left_label, fontsize=fontsize)
            ax.right_axis_label(self.right_label, fontsize=fontsize)
            ax.bottom_axis_label(self.bottom_label, fontsize=fontsize)

            ax.ticks(axis='lbr', multiple=0.1, linewidth=1)
            ax.clear_matplotlib_ticks()

        ax.scatter(self.points, marker=marker, color=color)
        ax._redraw_labels()

        return ax


class EntropyTriangle(BaseEntropyTriangle):
    """
    Construct the Multivariate Entropy Triangle, as defined in
    [Valverde-Albacete, Francisco Jose, and Carmen Pelaez-Moreno. "The
    Multivariate Entropy Triangle and Applications." Hybrid Artificial
    Intelligent Systems. Springer International Publishing, 2016. 647-658]
    """

    left_label = r"$\operatorname{R}[\mathrm{dist}]$"
    right_label = r"$\operatorname{T}[\mathrm{dist}] + \operatorname{B}[\mathrm{dist}]$"
    bottom_label = r"$\Delta \operatorname{H}_{\Pi_\overline{X}}$"

    @staticmethod
    def _compute_point(dist):
        """
        Compute the deviation from uniformity, dependence, and independence of a
        distribution.

        Parameters
        ----------
        dist : Distribution
            The distribution to compute values for.
        """
        H_U = entropy(uniform_like(dist))
        H_P = entropy(product_distribution(dist))

        Delta = H_U - H_P
        VI = residual_entropy(dist)
        M = H_P - VI

        return (Delta/H_U, M/H_U, VI/H_U)


class EntropyTriangle2(BaseEntropyTriangle):
    """
    Construct a variation on the Entropy Triangle, comparing the amount of
    independence in the distribution (residual entropy) to two types of
    dependence (total correlation and dual total correlation).
    """

    left_label = r"$\operatorname{B}[\mathrm{dist}]$"
    right_label = r"$\operatorname{T}[\mathrm{dist}]$"
    bottom_label = r"$\operatorname{R}[\mathrm{dist}]$"

    @staticmethod
    def _compute_point(dist):
        """
        Compute the residual entropy, total correlation, and dual total
        correlation for the distribution, and normalize them.

        Parameters
        ----------
        dist : Distribution
            The distribution to compute values for.
        """

        R = residual_entropy(dist)
        B = dual_total_correlation(dist)
        T = total_correlation(dist)
        total = R + B + T

        return (R/total, T/total, B/total)
