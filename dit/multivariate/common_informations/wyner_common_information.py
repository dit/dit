"""
The Wyner common information.
"""

import numpy as np

from .base_markov_optimizer import MarkovVarOptimizer

__all__ = ['wyner_common_information']


class WynerCommonInformation(MarkovVarOptimizer):
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
        bound = np.prod(self._shape) + 1

        return bound

    def _objective(self):
        """
        The mutual information between the auxiliary random variable and `rvs`.

        Returns
        -------
        obj : func
            The objective function.
        """
        conditional_mutual_information = self._conditional_mutual_information(self._rvs, self._W, self._crvs)

        def objective(self, x):
            """
            Compute I[rvs : W | crvs]

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
            return conditional_mutual_information(pmf)

        return objective


wyner_common_information = WynerCommonInformation.functional()
