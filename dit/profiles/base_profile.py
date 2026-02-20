"""
The base information profile.
"""

from abc import ABCMeta, abstractmethod

import numpy as np

from .. import Distribution
from ..params import ditParams
from ..utils import build_table

__all__ = (
    'BaseProfile',
)


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
    unit = 'bits'

    def __init__(self, dist):
        """
        Initialize the profile.

        Parameters
        ----------
        dist : Distribution
            The distribution to compute the profile for.
        """
        super().__init__()
        outcomes, pmf = zip(*dist.zipped(mode='atoms'), strict=True)
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
        left, height = zip(*sorted(self.profile.items()), strict=True)
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

    def __repr__(self):
        """
        Represent using the str().
        """
        if ditParams['repr.print']:
            return self.to_string()
        else:
            return super().__repr__()

    def __str__(self):
        """
        Use PrettyTable to create a nice table.
        """
        return self.to_string()

    def to_string(self, digits=3):
        """
        Use PrettyTable to create a nice table.
        """
        table = build_table(field_names=['measure', self.unit], title=self._name)
        table.float_format[self.unit] = f' 5.{digits}'  # pylint: disable=no-member
        for level, value in sorted(self.profile.items(), reverse=True):
            # gets rid of pesky -0.0 display values
            if np.isclose(value, 0.0):
                value = 0.0
            table.add_row([level, value])
        return table.get_string()
