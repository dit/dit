"""
"""

from itertools import combinations

import numpy as np

from .base_markov_optimizer import MarkovVarOptimizer

__all__ = ['exact_common_information']

class ExactCommonInformation(MarkovVarOptimizer):
    """
    Compute the Exact common information, min H[V], taken over all V which render the X_i conditionally independent.
    """

    def compute_bound(self):
        """
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
        Parameters
        ----------
        """
        return self.entropy(x)


def exact_common_information(dist, rvs=None, crvs=None, rv_mode=None):
    """
    """
    eci = ExactCommonInformation(dist, rvs, crvs, rv_mode)
    eci.optimize()
    return eci.objective(eci._res.x)
