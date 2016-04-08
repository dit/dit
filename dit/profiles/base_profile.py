"""
"""

from abc import ABCMeta, abstractmethod

class BaseProfile(object):
    """
    """
    __metaclass__ = ABCMeta

    xlabel = 'scale'
    ylabel = 'information [bits]'

    def __init__(self, dist):
        """
        """
        super(BaseProfile, self).__init__()
        self.dist = dist
        self._compute()

    @abstractmethod
    def _compute(self):
        """
        """
        pass

    def draw(self, ax=None):
        """
        """
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.figure().gca()

        left, height = zip(*self.profile.items())
        ax.bar(left, height, width=1)

        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)

        return ax
