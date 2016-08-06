"""
The Wyner common information.
"""

import numpy as np

from .base_markov_optimizer import MinimizingMarkovVarOptimizer

__all__ = ['wyner_common_information']


class WynerCommonInformation(MinimizingMarkovVarOptimizer):
    """
    Compute the Wyner common information, min I[X:V], taken over all V which
    render the X_i conditionally independent.
    """
    name = 'wyner'
    description = 'min I[X:V] such that V renders all X_i independent'

    def compute_bound(self):
        """
        An upper bound on the cardinality of the auxiliary random varaible, from
        the Caratheodory-Fenchel theorem.

        Returns
        -------
        bound : int
            The upper bound on the alphabet size of the auxiliary variable.
        """
        bound = np.prod([ sum(s) for s in self._rv_sizes ]) + 1

        return bound


    def objective(self, x):
        """
        The mutual information between the auxiliary random variable and `rvs`.

        Parameters
        ----------
        x : ndarray
            An optimization vector.

        Returns
        -------
        mi : float
            The mutual information.
        """
        return self.mutual_information(x)


wyner_common_information = WynerCommonInformation.functional()
