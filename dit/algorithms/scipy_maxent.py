"""
"""

from itertools import combinations

import numpy as np

from scipy.optimize import minimize

from .. import Distribution, product_distribution
from ..helpers import RV_MODES
from .maxentropy import marginal_constraints_generic
from .optutil import prepare_dist

__all__ = [
    'MaxEntOptimizer',
    'maxent_dist',
    'marginal_maxent_dists',
]

class MaxEntOptimizer(object):
    """
    Calculate maximum entropy distributions consistant with the given marginal constraints.
    """

    def __init__(self, dist, rvs, rv_mode=None):
        """
        Initialize the optimizer.

        Parameters
        ----------
        dist : Distribution
            The distribution from which the corresponding maximum entropy
            distribution will be calculated.
        rvs : list, None
            The list of sets of variables whose marginals will be constrained to
            match the given distribution.
        rv_mode : str, None
            Specifies how to interpret `rvs` and `crvs`. Valid options are:
            {'indices', 'names'}. If equal to 'indices', then the elements of
            `crvs` and `rvs` are interpreted as random variable indices. If
            equal to 'names', the the elements are interpreted as random
            variable names. If `None`, then the value of `dist._rv_mode` is
            consulted, which defaults to 'indices'.
        """
        self.dist = prepare_dist(dist)
        self._A, self._b = marginal_constraints_generic(self.dist, rvs, rv_mode)

    def constraint_match_marginals(self, x):
        """
        Ensure that the joint distribution represented by the optimization
        vector matches that of the distribution.

        Parameters
        ----------
        x : ndarray
            An optimization vector.

        Returns
        -------
        d : float
            The deviation from the constraint.
        """
        return sum((np.dot(self._A, x) - self._b)**2)

    def objective(self, x):
        """
        The entropy of optimization vector `x`.

        Parameters
        ----------
        x : ndarray
            An optimization vector.

        Returns
        -------
        H : float
            The (neg)entropy of `x`.
        """
        return np.nansum(x * np.log2(x))

    def optimize(self):
        """
        Perform the optimization.

        Notes
        -----
        This is a convex optimization, and so we use scipy's minimize optimizer
        frontend and the SLSQP algorithm because it is one of the few generic
        optimizers which can work with both bounds and constraints.
        """
        x0 = np.ones_like(self.dist.pmf)/len(self.dist.pmf)

        constraints = [{'type': 'eq',
                        'fun': self.constraint_match_marginals,
                       },
                      ]

        kwargs = {'method': 'SLSQP',
                  'bounds': [(0, 1)]*len(x0),
                  'constraints': constraints,
                  'tol': None,
                  'callback': None,
                  'options': {'maxiter': 1000,
                              'ftol': 15e-11,
                              'eps': 1.4901161193847656e-12,
                             },
                 }

        res = minimize(fun=self.objective,
                       x0=x0,
                       **kwargs
                      )

        self._optima = res.x

    def construct_dist(self, x=None, cutoff=1e-6):
        """
        Construct the maximum entropy distribution.

        Parameters
        ----------
        x : ndarray
            An optimization vector.

        Returns
        -------
        d : distribution
            The maximum entropy distribution.
        """
        if x is None:
            x = self._optima.copy()

        x[x < cutoff] = 0
        x /= x.sum()
        
        new_dist = self.dist.copy()
        new_dist.pmf = x
        new_dist.make_sparse()

        new_dist.set_rv_names(self.dist.get_rv_names())

        return new_dist




def maxent_dist(dist, rvs, rv_mode=None):
    """
    Return the maximum entropy distribution consistant with the marginals from
    `dist` specified in `rvs`.

    Parameters
    ----------
    dist : Distribution
        The distributions whose marginals should be matched.
    rvs : list of lists
        The marginals from `dist` to constrain.
    rv_mode : str, None
        Specifies how to interpret `rvs` and `crvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `crvs` and `rvs` are interpreted as random variable indices. If
        equal to 'names', the the elements are interpreted as random
        variable names. If `None`, then the value of `dist._rv_mode` is
        consulted, which defaults to 'indices'.
    """
    meo = MaxEntOptimizer(dist, rvs, rv_mode)
    meo.optimize()
    return meo.construct_dist()



def marginal_maxent_dists(dist, k_max=None, verbose=False):
    """
    Return the marginal-constrained maximum entropy distributions.

    Parameters
    ----------
    dist : distribution
        The distribution used to constrain the maxent distributions.
    k_max : int
        The maximum order to calculate.


    """
    dist = prepare_dist(dist)

    n_variables = dist.outcome_length()
    symbols = dist.alphabet[0]

    if k_max is None:
        k_max = n_variables

    outcomes = list(dist._sample_space)

    # Optimization for the k=0 and k=1 cases are slow since you have to optimize
    # the full space. We also know the answer in these cases.

    # This is safe since the distribution must be dense.
    k0 = Distribution(outcomes, [1]*len(outcomes), base='linear', validate=False)
    k0.normalize()

    k1 = product_distribution(dist)

    dists = [k0, k1]
    for k in range(k_max + 1):
        if verbose:
            print("Constraining maxent dist to match {0}-way marginals.".format(k))

        if k in [0, 1, n_variables]:
            continue

        rv_mode = dist._rv_mode

        if rv_mode in [RV_MODES.NAMES, 'names']:
            vars = dist.get_rv_names()
            rvs = list(combinations(vars, k))
        else:
            rvs = list(combinations(range(n_variables), k))

        dists.append(maxent_dist(dist, rvs, rv_mode))

    # To match the all-way marginal is to match itself. Again, this is a time
    # savings decision, even though the optimization should be fast.
    if k_max == n_variables:
        dists.append(dist)

    return dists
