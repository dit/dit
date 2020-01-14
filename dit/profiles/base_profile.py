"""
The base information profile.
"""

from abc import ABCMeta, abstractmethod

import numpy as np

from .. import Distribution


profile_docstring = """
{name}

Static Attributes
-----------------
xlabel : str
    The label for the x-axis when plotting.
ylabel : str
    The label for the y-axis when plotting.
{static_attributes}

Attributes
----------
dist : Distribution
profile : dict
widths : [float]
{attributes}

Methods
-------
draw
    Plot the profile
{methods}

Private Methods
---------------
_compute
    Compute the profile
"""


class BaseProfile(metaclass=ABCMeta):
    """
    BaseProfile

    Static Attributes
    -----------------
    xlabel : str
        The label for the x-axis when plotting.
    ylabel : str
        The label for the y-axis when plotting.

    Attributes
    ----------
    dist : Distribution
    profile : dict
    widths : [float]

    Methods
    -------
    draw
        Plot the profile.

    Abstract Methods
    ----------------
    _compute
        Compute the profile.
    """

    xlabel = 'scale'
    ylabel = 'information [bits]'
    align = 'center'

    def __init__(self, dist):
        """
        Initialize the profile.

        Parameters
        ----------
        dist : Distribution
            The distribution to compute the profile for.
        """
        super().__init__()
        outcomes, pmf = zip(*dist.zipped(mode='atoms'))
        self.dist = Distribution(outcomes, pmf)
        self._compute()

    @abstractmethod
    def _compute(self):
        """
        Abstract method to compute the profile.
        """
        pass

    def draw(self, ax=None):  # pragma: no cover
        """
        Draw the profile using matplotlib.

        Parameters
        ----------
        ax : axis
            The axis to draw the profile on. If None, a new axis is created.

        Returns
        -------
        ax : axis
            The axis with profile.
        """
        if ax is None:
            import matplotlib.pyplot as plt
            ax = plt.figure().gca()

        # pylint: disable=no-member
        left, height = zip(*sorted(self.profile.items()))
        ax.bar(left, height, width=self.widths, align=self.align)

        ax.set_xticks(sorted(self.profile.keys()))

        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)

        low, high = ax.get_ylim()
        if np.isclose(low, 0, atol=1e-5):
            low = -0.1
        if np.isclose(high, 0, atol=1e-5):
            high = 0.1
        ax.set_ylim((low, high))

        return ax
