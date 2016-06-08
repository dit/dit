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


def exact_common_information(dist, rvs=None, crvs=None, rv_mode=None, nhops=5):
    """
    Computes the exact common information, min H[V] where V renders all `rvs`
    independent.

    Parameters
    ----------
    dist : Distribution
        The distribution for which the exact common information will be
        computed.
    rvs : list, None
        A list of lists. Each inner list specifies the indexes of the random
        variables used to calculate the total correlation. If None, then the
        total correlation is calculated over all random variables, which is
        equivalent to passing `rvs=dist.rvs`.
    crvs : list, None
        A single list of indexes specifying the random variables to condition
        on. If None, then no variables are conditioned on.
    rv_mode : str, None
        Specifies how to interpret `rvs` and `crvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `crvs` and `rvs` are interpreted as random variable indices. If equal
        to 'names', the the elements are interpreted as random variable names.
        If `None`, then the value of `dist._rv_mode` is consulted, which
        defaults to 'indices'.
    nhops : int > 0
        Number of basin hoppings to perform during the optimization.

    Returns
    -------
    G : float
        The exact common information.
    """
    eci = ExactCommonInformation(dist, rvs, crvs, rv_mode)
    eci.optimize(nhops=nhops)
    return eci.objective(eci._res.x)
