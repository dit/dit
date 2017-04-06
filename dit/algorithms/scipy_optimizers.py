"""
"""
from __future__ import division

from abc import ABCMeta, abstractmethod

from collections import namedtuple

from itertools import combinations
from iterutils import powerset

import numpy as np

from scipy.optimize import basinhopping, minimize

from .. import Distribution, product_distribution
from ..exceptions import ditException
from ..helpers import RV_MODES
from ..math import close
from .maxentropy import marginal_constraints_generic
from ..multivariate import coinformation as I
from .optutil import prepare_dist
from .pid_broja import (extra_constraints as broja_extra_constraints,
                        prepare_dist as broja_prepare_dist)
from ..utils.optimization import BasinHoppingCallBack, accept_test, basinhop_status

__all__ = [
    'MaxCoInfoOptimizer',
    'MaxEntOptimizer',
    'maxent_dist',
    'marginal_maxent_dists',
    'pid_broja',
]

class BaseOptimizer(object):
    """
    Calculate an optimized distribution consistant with the given marginal constraints.
    """
    __metaclass__ = ABCMeta

    def __init__(self, dist, rvs, rv_mode=None):
        """
        Initialize the optimizer.

        Parameters
        ----------
        dist : Distribution
            The distribution from which the corresponding optimal distribution
            will be calculated.
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
        self._pmf = self.dist.pmf.copy()
        self._A, self._b = marginal_constraints_generic(self.dist, rvs, rv_mode)
        self._shape = list(map(len, self.dist.alphabet))
        self._subvars = list(powerset(range(len(self._shape))))[:-1]
        self._free = [ i for i, n in enumerate(self._A[self._b==0, :].sum(axis=0)) if n == 0]

    def _expand(self, x):
        """
        Expand the `x` argument to the full pmf.

        Parameters
        ----------
        x : np.array
            optimization vector

        Returns
        -------
        pmf : np.array
            full pmf
        """
        self._pmf[self._free] = x
        return self._pmf

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
        pmf = self._expand(x)
        return sum((np.dot(self._A, pmf) - self._b)**2)

    @abstractmethod
    def objective(self, x):
        """
        The objective of optimization vector `x`.

        Parameters
        ----------
        x : ndarray
            An optimization vector.

        Returns
        -------
        O : float
            The objective to be minimized.
        """
        pass

    def entropy(self, x):
        """
        The entropy of optimization vector `x`.

        Parameters
        ----------
        x : ndarray
            An optimization vector.

        Returns
        -------
        H : float
            The entropy of `x`.
        """
        pmf = self._expand(x)
        return -np.nansum(pmf * np.log2(pmf))

    def co_information(self, x):
        """
        The co-information of optimization vector `x`.

        Parameters
        ----------
        x : ndarray
            An optimization vector.

        Returns
        -------
        I : float
            The co-information of `x`.
        """
        n = len(self._shape)
        pmf = self._expand(x).reshape(self._shape)
        spmf = [ pmf.sum(axis=subset, keepdims=True)**((-1)**(n - len(subset))) for subset in self._subvars ]
        return np.nansum(pmf * np.log2(np.prod(spmf)))

    def optimize_convex(self, x0=None, bounds=None):
        """
        Perform the optimization.

        Notes
        -----
        This is a convex optimization, and so we use scipy's minimize optimizer
        frontend and the SLSQP algorithm because it is one of the few generic
        optimizers which can work with both bounds and constraints.
        """
        if x0 is None:
            x0 = np.ones_like(self._free)/len(self._free)

        if bounds is None:
            bounds = [(0, 1)]*x0.size

        constraints = [{'type': 'eq',
                        'fun': self.constraint_match_marginals,
                       },
                      ]

        kwargs = {'method': 'SLSQP',
                  'bounds': bounds,
                  'constraints': constraints,
                  'tol': None,
                  'callback': None,
                  'options': {'maxiter': 1000,
                              'ftol': 1e-7,
                              'eps': 1.4901161193847656e-08,
                             },
                 }

        res = minimize(fun=self.objective,
                       x0=x0,
                       **kwargs
                      )

        if not res.success:
            msg = "Optimization failed: {}".format(res.message)
            raise ditException(msg)

        self._optima = res.x

    def optimize_nonconvex(self, x0=None, bounds=None, nhops=10):
        """
        Perform the optimization. This is a non-convex optimization, and utilizes
        basin hopping.

        Notes
        -----
        This is a convex optimization, and so we use scipy's minimize optimizer
        frontend and the SLSQP algorithm because it is one of the few generic
        optimizers which can work with both bounds and constraints.
        """
        if x0 is None:
            x0 = np.ones_like(self._free)/len(self._free)

        if bounds is None:
            bounds = [(0, 1)]*x0.size

        constraints = [{'type': 'eq',
                        'fun': self.constraint_match_marginals,
                       },
                      ]

        kwargs = {'method': 'SLSQP',
                  'bounds': bounds,
                  'constraints': constraints,
                  'tol': None,
                  'callback': None,
                  'options': {'maxiter': 1000,
                              'ftol': 1e-7,
                              'eps': 1.4901161193847656e-08,
                             },
                 }

        self._callback = BasinHoppingCallBack(kwargs['constraints'], None)

        res = basinhopping(func=self.objective,
                           x0=x0,
                           minimizer_kwargs=kwargs,
                           niter=nhops,
                           callback=self._callback,
                           accept_test=accept_test,
                          )


        success, msg = basinhop_status(res)
        if success:
            self._optima = res.x
        else: # pragma: no cover
            minimum = self._callback.minimum()
            if minimum is not None:
                self._optima = minimum
            else:
                raise ditException("Optima not found")


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

        pmf = self._expand(x)

        pmf[pmf < cutoff] = 0
        pmf /= pmf.sum()

        new_dist = self.dist.copy()
        new_dist.pmf = pmf
        new_dist.make_sparse()

        new_dist.set_rv_names(self.dist.get_rv_names())

        return new_dist


class MaxEntOptimizer(BaseOptimizer):
    """
    """

    def objective(self, x):
        """
        """
        return -self.entropy(x)

    optimize = BaseOptimizer.optimize_convex


class MinEntOptimizer(BaseOptimizer):
    """
    """

    def objective(self, x):
        """
        """
        return self.entropy(x)

    optimize = BaseOptimizer.optimize_nonconvex


class MaxCoInfoOptimizer(BaseOptimizer):
    """
    """

    def objective(self, x):
        """
        """
        return -self.co_information(x)

    optimize = BaseOptimizer.optimize_nonconvex


class MinCoInfoOptimizer(BaseOptimizer):
    """
    """

    def objective(self, x):
        """
        """
        return self.co_information(x)

    optimize = BaseOptimizer.optimize_nonconvex


class BROJAOptimizer(MaxCoInfoOptimizer):
    """
    """

    def __init__(self, dist, sources, target, rv_mode=None):
        """
        Initialize the optimizer. It is assumed tha

        Parameters
        ----------
        dist : Distribution
            The distribution from which the corresponding optimal distribution
            will be calculated.
        sources : list, len = 2
            List of two source sets of variables.
        target : list
            The target variables.
        rv_mode : str, None
            Specifies how to interpret `rvs` and `crvs`. Valid options are:
            {'indices', 'names'}. If equal to 'indices', then the elements of
            `crvs` and `rvs` are interpreted as random variable indices. If
            equal to 'names', the the elements are interpreted as random
            variable names. If `None`, then the value of `dist._rv_mode` is
            consulted, which defaults to 'indices'.
        """
        dist = broja_prepare_dist(dist, sources, target, rv_mode)
        super(BROJAOptimizer, self).__init__(dist, [[0, 2], [1, 2]])

        extra_free = broja_extra_constraints(self.dist, 2).free
        self._free = list(sorted(set(self._free) & set(extra_free)))

    def optimize(self, x0=None):
        """
        """
        if len(self._free) == 0:
            self._optima = self._pmf
        else:
            super(BROJAOptimizer, self).optimize_convex(x0=x0)


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


PID = namedtuple('PID', ['R', 'U0', 'U1', 'S'])


def pid_broja(dist, sources, target, rv_mode=None):
    """
    """
    broja = BROJAOptimizer(dist, sources, target, rv_mode)
    broja.optimize()
    opt_dist = broja.construct_dist()
    r = -broja.objective(broja._optima)
    u0 = I(opt_dist, [[0], [2]], [1])
    u1 = I(opt_dist, [[1], [2]], [0])
    r = 0.0 if close(r, 0) else r
    u0 = 0.0 if close(u0, 0) else u0
    u1 = 0.0 if close(u1, 0) else u1
    s = I(dist, [[0, 1], [2]]) - r - u0 - u1
    return PID(R=r, U0=u0, U1=u1, S=s)
