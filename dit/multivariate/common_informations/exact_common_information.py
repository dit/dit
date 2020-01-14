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
        # from the Caratheodory-Fenchel theorem
        bound_1 = np.prod(self._shape[:-1])

        # from number of support-distinct conditional distributions
        combos = combinations(self._shape[:-1], len(self._shape[:-1])-1)
        bound_2 = 2**min(np.prod(combo) for combo in combos) - 1

        return min([bound_1, bound_2])

    def _objective(self):
        """
        The entropy of the auxiliary random variable.

        Returns
        -------
        obj : func
            The objective function.
        """
        entropy = self._entropy(self._W, self._crvs)

        def objective(self, x):
            """
            Compute H[W | crvs]

            Parameters
            ----------
            x : np.ndarray
                An optimization vector.

            Returns
            -------
            obj : float
                The value of the objective.
            """
            pmf = self.construct_joint(x)
            return entropy(pmf)

        return objective


exact_common_information = ExactCommonInformation.functional()
