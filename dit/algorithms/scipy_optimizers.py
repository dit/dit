"""
A variety of distribution optimizers using scipy.optimize's methods.
"""
from __future__ import division

from abc import ABCMeta, abstractmethod

from collections import namedtuple

from itertools import combinations

from debtcollector import removals

import numpy as np

from scipy.optimize import basinhopping, minimize

from .maxentropy import marginal_constraints_generic
from .optutil import prepare_dist
from .pid_broja import (extra_constraints as broja_extra_constraints,
                        prepare_dist as broja_prepare_dist)
from .. import Distribution, product_distribution
from ..exceptions import ditException
from ..helpers import RV_MODES
from ..math import close
from ..multivariate import coinformation as I
from ..utils import flatten, powerset
from ..utils.optimization import BasinHoppingCallBack, accept_test, basinhop_status

__all__ = [
    'maxent_dist',
    'marginal_maxent_dists',
    'pid_broja',
]

def infer_free_values(A, b):
    """
    Infer the indices of fixed values in an optimization vector.

    Parameters
    ----------
    A : np.ndarray
        The constraint matrix.
    b : np.ndarray
        The constraint values.

    Returns
    -------
    fixed : list
        The list of fixed indices.
    """
    # find locations of b==0, since pmf values are non-negative, this means they are identically zero.
    free = [i for i, n in enumerate(A[b==0, :].sum(axis=0)) if n == 0]
    while True:
        # now find rows of A with only a single free value in them. those values must also be fixed.
        fixed = A[:, free].sum(axis=1) == 1
        new_fixed = [[i for i, n in enumerate(row) if n and (i in free)][0] for i, row in enumerate(A) if fixed[i]]
        free = list(sorted(set(free) - set(new_fixed)))
        if not new_fixed:
            break
    return free


class BaseOptimizer(object):
    """
    Calculate an optimized distribution consistent with the given marginal constraints.
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
        self._free = infer_free_values(self._A, self._b)
        self.constraints = [{'type': 'eq',
                             'fun': self.constraint_match_marginals,
                            },
                           ]


    def _expand(self, x):
        """
        Expand the `x` argument to the full pmf.

        Parameters
        ----------
        x : np.ndarray
            optimization vector

        Returns
        -------
        pmf : np.array
            full pmf
        """
        if self._free:
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

    @abstractmethod # pragma: no cover
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
        spmf = [pmf.sum(axis=subset, keepdims=True)**((-1)**(n - len(subset))) for subset in self._subvars]
        coi = np.nansum(pmf * np.log2(np.prod(spmf)))
        return coi

    def dual_total_correlation(self, x):
        """
        The dual total correlation of optimization vector `x`.

        Parameters
        ----------
        x : ndarray
            An optimization vector.

        Returns
        -------
        B : float
            The dual total correlation of `x`.
        """
        pmf = self._expand(x).reshape(self._shape)
        entropy = -np.nansum(pmf * np.log2(pmf))
        spmfs = [pmf.sum(axis=i) for i, _ in enumerate(self._shape)]
        ind_entropy = sum(entropy - -np.nansum(spmf * np.log2(spmf)) for spmf in spmfs)
        dtc = entropy - ind_entropy
        return dtc

    def optimize(self, x0=None, bounds=None, nhops=10, polish=1e-10, maxiters=1000):
        """
        Perform the optimization. Dispatches to the appropriate backend.

        Parameters
        ----------
        x0 : np.ndarray, None
            An initial optimization vector. If None, a uniform vector is used.
        bounds : [tuple], None
            A list of (lower, upper) bounds on each element of the optimization vector.
            If None, (0,1) is assumed.
        polish : float, False
            Attempt to improve the solution. Set probabilities lower than this to zero,
            reducing the total optimization dimension.

        Raises
        ------
        ditException
            Raised if the optimization failed for any reason.
        """
        if len(self._free) == 0:
            self._optima = self._pmf
            return

        if x0 is None:
            x0 = np.ones_like(self._free)/len(self._free)

        if bounds is None:
            bounds = [(0, 1)]*x0.size

        kwargs = {'method': 'SLSQP',
                  'bounds': bounds,
                  'constraints': self.constraints,
                  'tol': None,
                  'callback': None,
                  'options': {'maxiter': maxiters,
                              'ftol': 1e-7,
                              'eps': 1.4901161193847656e-08,
                             },
                 }

        try:
            kwargs['jac'] = self._jacobian
        except AttributeError:
            pass

        self._optimization_backend(x0, kwargs, nhops)

        if polish:
            self._polish(cutoff=polish)

    def _polish(self, cutoff):
        """
        Improve the solution found by the optimizer.

        Parameters
        ----------
        cutoff : float
            Set probabilities lower than this to zero, reducing the total
            optimization dimension.
        """
        x0 = self._optima
        count = (x0 < cutoff).sum()
        x0[x0 < cutoff] = 0

        kwargs = {'method': 'SLSQP',
                  'bounds': [(0, 0) if close(x, 0) else (0, 1) for x in x0],
                  'constraints': self.constraints,
                  'tol': None,
                  'callback': None,
                  'options': {'maxiter': 1000,
                              'ftol': 15e-12,
                              'eps': 1.4901161193847656e-14,
                             },
                 }

        res = minimize(fun=self.objective,
                       x0=x0,
                       **kwargs
                      )

        self._optima = res.x

        if count < (res.x < cutoff).sum():
            self._polish(cutoff=cutoff)

    @abstractmethod # pragma: no cover
    def _optimization_backend(self, x0, kwargs, nhops):
        """
        Abstract method for performing an optimization.

        Parameters
        ----------
        x0 : np.ndarray
            An initial optimization vector.
        kwargs : dict
            A dictionary of keyword arguments to pass to the optimizer.
        nhops : int
            If applicable, the number of iterations to make.
        """
        pass


    def construct_dist(self, x=None, cutoff=1e-6, sparse=True):
        """
        Construct the optimal distribution.

        Parameters
        ----------
        x : np.ndarray
            An optimization vector.
        cutoff : float
            A probability cutoff. Any joint event with probability below
            this will be set to zero.

        Returns
        -------
        d : distribution
            The optimized distribution.
        """
        if x is None:
            x = self._optima.copy()

        pmf = self._expand(x)

        pmf[pmf < cutoff] = 0
        pmf /= pmf.sum()

        new_dist = self.dist.copy()
        new_dist.pmf = pmf
        if sparse:
            new_dist.make_sparse()

        new_dist.set_rv_names(self.dist.get_rv_names())

        return new_dist


class BaseConvexOptimizer(BaseOptimizer):
    """
    Base class to optimize distributions according to convex objectives.
    """

    def _optimization_backend(self, x0, kwargs, nhops):
        """
        Perform the optimization.

        Parameters
        ----------
        x0 : np.ndarray
            An initial optimization vector.
        kwargs : dict
            A dictionary of keyword arguments to pass to the optimizer.
        nhops : int
            If applicable, the number of iterations to make.

        Raises
        ------
        ditException
            Raised if the optimization failed for any reason.

        Notes
        -----
        This is a convex optimization, and so we use scipy's minimize optimizer
        frontend and the SLSQP algorithm because it is one of the few generic
        optimizers which can work with both bounds and constraints.
        """
        self.res = minimize(fun=self.objective,
                            x0=x0,
                            **kwargs
                           )

        if not self.res.success: # pragma: no cover
            msg = "Optimization failed: {}".format(self.res.message)
            raise ditException(msg)

        self._optima = self.res.x


class BaseNonConvexOptimizer(BaseOptimizer):
    """
    Base class to optimize distributions according to non-convex objectives.
    """

    def _optimization_backend(self, x0, kwargs, nhops):
        """
        Perform the optimization. This is a non-convex optimization, and utilizes
        basin hopping.

        Parameters
        ----------
        x0 : np.ndarray
            An initial optimization vector.
        kwargs : dict
            A dictionary of keyword arguments to pass to the optimizer.
        nhops : int
            If applicable, the number of iterations to make.

        Raises
        ------
        ditException
            Raised if the optimization failed for any reason.

        Notes
        -----
        This is a nonconvex optimization, and so we use scipy's basinhopping meta-optimizer
        frontend and the SLSQP algorithm backend because it is one of the few generic
        optimizers which can work with both bounds and constraints.
        """
        self._callback = BasinHoppingCallBack(kwargs['constraints'], None)

        self.res = basinhopping(func=self.objective,
                                x0=x0,
                                minimizer_kwargs=kwargs,
                                niter=nhops,
                                callback=self._callback,
                                accept_test=accept_test,
                               )

        success, msg = basinhop_status(self.res)
        if success:
            self._optima = self.res.x
        else: # pragma: no cover
            minimum = self._callback.minimum()
            if minimum is not None:
                self._optima = minimum
            else:
                raise ditException("Optima not found")


class MaxEntOptimizer(BaseConvexOptimizer):
    """
    Compute maximum entropy distributions.
    """

    def objective(self, x):
        """
        Compute the negative entropy.

        Parameters
        ----------
        x : np.ndarray
            An optimization vector.
        """
        return -self.entropy(x)


class MinEntOptimizer(BaseNonConvexOptimizer):
    """
    Compute minimum entropy distributions.
    """

    def objective(self, x):
        """
        Compute the entropy.

        Parameters
        ----------
        x : np.ndarray
            An optimization vector.
        """
        return self.entropy(x)


class MaxCoInfoOptimizer(BaseNonConvexOptimizer):
    """
    Compute maximum co-information distributions.
    """

    def objective(self, x):
        """
        Compute the negative co-information.

        Parameters
        ----------
        x : np.ndarray
            An optimization vector.
        """
        return -self.co_information(x)


class MinCoInfoOptimizer(BaseNonConvexOptimizer):
    """
    Compute minimum co-information distributions.
    """

    def objective(self, x):
        """
        Compute the co-information.

        Parameters
        ----------
        x : np.ndarray
            An optimization vector.
        """
        return self.co_information(x)

class MaxDualTotalCorrelationOptimizer(BaseNonConvexOptimizer):
    """
    Compute maximum dual total correlation distributions.
    """

    def objective(self, x):
        """
        Compute the negative dual total correlation.

        Parameters
        ----------
        x : np.ndarray
            An optimization vector.
        """
        return -self.dual_total_correlation(x)

class MinDualTotalCorrelationOptimizer(BaseNonConvexOptimizer):
    """
    Compute minimum dual total correlation distributions.
    """

    def objective(self, x):
        """
        Compute the dual total correlation.

        Parameters
        ----------
        x : np.ndarray
            An optimization vector.
        """
        return self.dual_total_correlation(x)

class BROJABivariateOptimizer(MaxCoInfoOptimizer):
    """
    An optimizer for constructing the maximum co-information distribution
    consistent with (source, target) marginals of the given distribution.

    Notes
    -----
    Though maximizing co-information is generically a non-convex optimization,
    with the specific constraints involved in this calculation the problem is
    convex.
    """

    def __init__(self, dist, sources, target, rv_mode=None):
        """
        Initialize the optimizer.

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
        super(BROJABivariateOptimizer, self).__init__(dist, [[0, 2], [1, 2]])

        extra_free = broja_extra_constraints(self.dist, 2).free
        self._free = list(sorted(set(self._free) & set(extra_free)))


def maxent_dist(dist, rvs, rv_mode=None, x0=None, sparse=True, maxiters=1000):
    """
    Return the maximum entropy distribution consistent with the marginals from
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
    x0 : np.ndarray
        Initial condition for the optimizer.
    dense : bool
        Whether the returned distribution should be dense or sparse.

    Returns
    -------
    me : Distribution
        The maximum entropy distribution.
    """
    meo = MaxEntOptimizer(dist, rvs, rv_mode)
    meo.optimize(x0=x0, maxiters=maxiters)
    dist = meo.construct_dist(sparse=sparse)
    return dist


def marginal_maxent_dists(dist, k_max=None):
    """
    Return the marginal-constrained maximum entropy distributions.

    Parameters
    ----------
    dist : distribution
        The distribution used to constrain the maxent distributions.
    k_max : int
        The maximum order to calculate.
    verbose : bool
        If True, print more information.

    Returns
    -------
    dists : list
        A list of distributions, where the `i`th element is the maxent
        distribution with the i-size marginals fixed.
    """
    dist = prepare_dist(dist)

    n_variables = dist.outcome_length()

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

@removals.remove(message="Please see dit.pid.PID_BROJA.", version='1.0.0.dev8')
def pid_broja(dist, sources, target, rv_mode=None, return_opt=False):
    """
    Compute the BROJA partial information decomposition.

    Parameters
    ----------
    dist : Distribution
        The distribution to compute the partial information decomposition of.
    sources : iterable
        The source variables of the distribution.
    target : iterable
        The target variable of the distribution.
    rv_mode : str, None
        Specifies how to interpret `sources` and `target`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `crvs` and `rvs` are interpreted as random variable indices. If
        equal to 'names', the the elements are interpreted as random
        variable names. If `None`, then the value of `dist._rv_mode` is
        consulted, which defaults to 'indices'.
    return_opt : bool
        If True, return the distribution resulting from the optimization.
        Defaults to False.

    Returns
    -------
    pid : PID namedtuple
        The partial information decomposition.
    opt_dist : Distribution
        The distribution resulting from the optimizaiton. Note that var [0]
        is sources[0], [1] is sources[1] and [2] is target.
    """
    broja = BROJABivariateOptimizer(dist, sources, target, rv_mode)
    broja.optimize()
    opt_dist = broja.construct_dist()
    r = -broja.objective(broja._optima)
    # in opt_dist, source[0] is [0], sources[1] is [1], and target is [2]
    #   see broja_prepare_dist() for details
    u0 = I(opt_dist, [[0], [2]], [1])
    u1 = I(opt_dist, [[1], [2]], [0])
    # r = 0.0 if close(r, 0, rtol=1e-6, atol=1e-6) else r
    # u0 = 0.0 if close(u0, 0, rtol=1e-6, atol=1e-6) else u0
    # u1 = 0.0 if close(u1, 0, rtol=1e-6, atol=1e-6) else u1
    s = I(dist, [list(flatten(sources)), target]) - r - u0 - u1

    pid = PID(R=r, U0=u0, U1=u1, S=s)

    if return_opt:
        return pid, opt_dist
    else:
        return pid
