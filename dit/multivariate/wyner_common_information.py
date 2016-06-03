"""
The Wyner common information.
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


def wyner_common_information(dist, rvs=None, crvs=None, rv_mode=None):
    """
    Compute the Wyner common information: the smallest I[X:V] such that V
    renders all X_i independent.

    Parameters
    ----------
    dist : Distribution
        The distribution for which the Wyner common informatino will be
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

    Returns
    -------
    C : float
        The Wyner common information.
    """
    wci = WynerCommonInformation(dist, rvs, crvs, rv_mode)
    wci.optimize()
    return wci.objective(wci._res.x)
