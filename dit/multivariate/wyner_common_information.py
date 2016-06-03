"""
"""

import numpy as np

from .base_markov_optimizer import MinimizingMarkovVarOptimizer

__all__ = ['wyner_common_information']


class WynerCommonInformation(MinimizingMarkovVarOptimizer):
    """
    Compute the Wyner common information, min I[X:V], taken over all V which render the X_i conditionally independent.
    """

    def compute_bound(self):
        """
        From the Caratheodory-Fenchel theorem.

        Returns
        -------
        bound : int
            The upper bound on the alphabet size of the auxiliary variable.
        """
        bound = np.prod([ sum(s) for s in self._rv_sizes ]) + 1

        return bound


    def objective(self, x):
        """
        Parameters
        ----------
        """
        return self.mutual_information(x)


def wyner_common_information(dist, rvs=None, crvs=None, rv_mode=None):
    """
    """
    wci = WynerCommonInformation(dist, rvs, crvs, rv_mode)
    wci.optimize()
    return wci.objective(wci._res.x)
