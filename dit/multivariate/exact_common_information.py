"""
The exact common information.
"""

from itertools import combinations

import numpy as np

from .base_markov_optimizer import MarkovVarOptimizer

__all__ = ['exact_common_information']

class ExactCommonInformation(MarkovVarOptimizer):
    """
    Compute the Exact common information, min H[V], taken over all V which
    render the X_i conditionally independent.
    """
    name = 'exact'
    description = 'min H[V] where V renders all `rvs` independent'

    def compute_bound(self):
        """
        Compute the upper bound on the cardinality of the auxiliary random
        variable. The bound is the minimum of one from the Caratheodory-Fenchel
        theorem, and the other from a pidgenhole argument.

        Returns
        -------
        bound : int
            The bound.
        """
        sizes = [ sum(s) for s in self._rv_sizes ]
        # from the Caratheodory-Fenchel theorem
        bound_1 = np.prod(sizes)

        # from number of support-distinct conditional distributions
        combos = combinations(sizes, len(sizes)-1)
        bound_2 = 2**min(np.prod(combo) for combo in combos) - 1

        return min(bound_1, bound_2)

    def objective(self, x):
        """
        The entropy of the auxiliary random variable.

        Parameters
        ----------
        x : ndarray
            An optimization vector.

        Returns
        -------
        h : float
            The entropy of the auxiliar random variable.
        """
        return self.entropy(x)


exact_common_information = ExactCommonInformation.functional()
