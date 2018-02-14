"""
A variety of distribution optimizers using scipy.optimize's methods.
"""
from __future__ import division

from collections import namedtuple

from itertools import combinations

from debtcollector import removals

import numpy as np

from .maxentropy import marginal_constraints_generic
from .optimization import BaseOptimizer, BaseConvexOptimizer, BaseNonConvexOptimizer
from .optutil import prepare_dist
from .pid_broja import (extra_constraints as broja_extra_constraints,
                        prepare_dist as broja_prepare_dist)
from .. import Distribution, product_distribution
from ..helpers import RV_MODES
from ..multivariate import coinformation as I
from ..utils import flatten

__all__ = [
    'maxent_dist',
    'marginal_maxent_dists',
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
    # find locations of b == 0, since pmf values are non-negative, this means they are identically zero.
    free = [i for i, n in enumerate(A[b == 0, :].sum(axis=0)) if n == 0]
    while True:
        # now find rows of A with only a single free value in them. those values must also be fixed.
        fixed = A[:, free].sum(axis=1) == 1
        new_fixed = [[i for i, n in enumerate(row) if n and (i in free)][0] for i, row in enumerate(A) if fixed[i]]
        free = list(sorted(set(free) - set(new_fixed)))
        if not new_fixed:
            break
    return free


class BaseDistOptimizer(BaseOptimizer):
    """
    Calculate an optimized distribution consistent with the given marginal constraints.
    """

    construct_initial = BaseOptimizer.construct_uniform_initial

    def __init__(self, dist, marginals, rv_mode=None):
        """
        Initialize the optimizer.

        Parameters
        ----------
        dist : Distribution
            The distribution from which the corresponding optimal distribution
            will be calculated.
        marginals : list, None
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
        super(BaseDistOptimizer, self).__init__(dist, dist.rvs, crvs=[], rv_mode='indices')

        # todo: actually make this class support crvs?
        self._all_vars = self._rvs

        self.dist = prepare_dist(dist)
        self._vpmf = self.dist.pmf.copy()
        self._A, self._b = marginal_constraints_generic(self.dist, marginals, rv_mode)
        self._shape = list(map(len, self.dist.alphabet))
        self._free = infer_free_values(self._A, self._b)
        self.constraints = [{'type': 'eq',
                             'fun': self.constraint_match_marginals,
                             },
                            ]
        self._optvec_size = len(self._free)
        self._default_hops = 50

        self._additional_options = {'options': {'maxiter': 1000,
                                                'ftol': 1e-7,
                                                'eps': 1.4901161193847656e-08,
                                                }
                                    }

    def optimize(self, x0=None, niter=None, maxiter=None, polish=1e-8, callback=False):
        """
        Optimize this distribution w.r.t the objective.

        Parameters
        ----------
        x0 : np.ndarray
            An initial optimization vector.
        niter : int
            The number of optimization iterations to perform.
        maxiter : int
            The number of steps for an optimization subroutine to perform.
        polish : float
            The threshold for valid optimization elements. If 0, no polishing is
            performed.
        callback : bool
            Whether to use a callback to track the performance of the optimization.
            Generally, this should be False as it adds some significant time to the
            optimization.

        Returns
        -------
        result : OptimizeResult, None
            Return the optimization result, or None if no optimization was needed.
        """
        if len(self._free) == 0:
            self._optima = self._vpmf
        else:
            if x0 is not None and len(x0) == len(self._vpmf):
                # if a full pmf vector was passed in, restrict it to the free
                # indices:
                x0 = x0[self._free]
            result = super(BaseDistOptimizer, self).optimize(x0=x0,
                                                             niter=niter,
                                                             maxiter=maxiter,
                                                             polish=polish,
                                                             callback=callback)
            return result

    def construct_vector(self, x):
        """
        Expand the `x` argument to the full pmf.

        Parameters
        ----------
        x : np.ndarray
            An optimization vector.

        Returns
        -------
        vpmf : np.array
            The full pmf as a vector.
        """
        if self._free:
            self._vpmf[self._free] = x
        return self._vpmf

    def construct_joint(self, x):
        """
        Construct the joint distribution.

        Parameters
        ----------
        x : np.ndarray
            An optimization vector.

        Returns
        -------
        pmf : np.ndarray
            The joint distribution.
        """
        vec = self.construct_vector(x)
        pmf = vec.reshape(self._shape)
        return pmf

    def constraint_match_marginals(self, x):
        """
        Ensure that the joint distribution represented by the optimization
        vector matches that of the distribution.

        Parameters
        ----------
        x : np.ndarray
            An optimization vector.

        Returns
        -------
        d : float
            The deviation from the constraint.
        """
        pmf = self.construct_vector(x)
        return sum((np.dot(self._A, pmf) - self._b)**2)

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
        sparse : bool
            Whether to make the distribution sparse or not. Defaults to True.

        Returns
        -------
        d : distribution
            The optimized distribution.
        """
        if x is None:
            x = self._optima.copy()

        pmf = self.construct_vector(x)

        pmf[pmf < cutoff] = 0
        pmf /= pmf.sum()

        new_dist = self.dist.copy()
        new_dist.pmf = pmf.ravel()
        if sparse:
            new_dist.make_sparse()

        new_dist.set_rv_names(self.dist.get_rv_names())

        return new_dist


class MaxEntOptimizer(BaseDistOptimizer, BaseConvexOptimizer):
    """
    Compute maximum entropy distributions.
    """

    def _objective(self):
        """
        Compute the negative entropy.

        Returns
        -------
        objective : func
            The objective function.
        """
        entropy = self._entropy(self._rvs)

        def objective(self, x):
            """
            Compute -H[rvs]

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
            return -entropy(pmf)

        return objective


class MinEntOptimizer(BaseDistOptimizer, BaseNonConvexOptimizer):
    """
    Compute minimum entropy distributions.
    """

    def _objective(self):
        """
        Compute the entropy.

        Returns
        -------
        objective : func
            The objective function.
        """
        entropy = self._entropy(self._rvs)

        def objective(self, x):
            """
            Compute H[rvs]

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


class MaxCoInfoOptimizer(BaseDistOptimizer, BaseNonConvexOptimizer):
    """
    Compute maximum co-information distributions.
    """

    def _objective(self):
        """
        Compute the negative co-information.

        Returns
        -------
        objective : func
            The objective function.
        """
        coinformation = self._coinformation(self._rvs)

        def objective(self, x):
            """
            Compute -I[rvs]

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
            return -coinformation(pmf)

        return objective


class MinCoInfoOptimizer(BaseDistOptimizer, BaseNonConvexOptimizer):
    """
    Compute minimum co-information distributions.
    """

    def _objective(self):
        """
        Compute the co-information.

        Returns
        -------
        objective : func
            The objective function.
        """
        coinformation = self._coinformation(self._rvs)

        def objective(self, x):
            """
            Compute I[rvs]

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
            return coinformation(pmf)

        return objective


class MaxDualTotalCorrelationOptimizer(BaseDistOptimizer, BaseNonConvexOptimizer):
    """
    Compute maximum dual total correlation distributions.
    """

    def _objective(self):
        """
        Compute the negative dual total correlation.

        Returns
        -------
        objective : func
            The objective function.
        """
        dual_total_correlation = self._dual_total_correlation(self._rvs)

        def objective(self, x):
            """
            Compute -B[rvs]

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
            return -dual_total_correlation(pmf)

        return objective


class MinDualTotalCorrelationOptimizer(BaseDistOptimizer, BaseNonConvexOptimizer):
    """
    Compute minimum dual total correlation distributions.
    """

    def _objective(self):
        """
        Compute the dual total correlation.

        Returns
        -------
        objective : func
            The objective function.
        """
        dual_total_correlation = self._dual_total_correlation(self._rvs)

        def objective(self, x):
            """
            Compute B[rvs]

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
            return dual_total_correlation(pmf)

        return objective


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
        self._optvec_size = len(self._free)


def maxent_dist(dist, rvs, x0=None, maxiter=1000, sparse=True, rv_mode=None):
    """
    Return the maximum entropy distribution consistent with the marginals from
    `dist` specified in `rvs`.

    Parameters
    ----------
    dist : Distribution
        The distributions whose marginals should be matched.
    rvs : list of lists
        The marginals from `dist` to constrain.
    x0 : np.ndarray
        Initial condition for the optimizer.
    maxiter : int
        The number of optimization iterations to perform.
    sparse : bool
        Whether the returned distribution should be sparse or dense.
    rv_mode : str, None
        Specifies how to interpret `rvs` and `crvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `crvs` and `rvs` are interpreted as random variable indices. If
        equal to 'names', the the elements are interpreted as random
        variable names. If `None`, then the value of `dist._rv_mode` is
        consulted, which defaults to 'indices'.

    Returns
    -------
    me : Distribution
        The maximum entropy distribution.
    """
    meo = MaxEntOptimizer(dist, rvs, rv_mode)
    meo.optimize(x0=x0, maxiter=maxiter)
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

        dists.append(maxent_dist(dist, rvs, rv_mode=rv_mode))

    # To match the all-way marginal is to match itself. Again, this is a time
    # savings decision, even though the optimization should be fast.
    if k_max == n_variables:
        dists.append(dist)

    return dists


PID = namedtuple('PID', ['R', 'U0', 'U1', 'S'])


@removals.remove(message="Please see dit.pid.PID_BROJA.",
                 version='1.0.0.dev8')
def pid_broja(dist, sources, target, niter=10, return_opt=False, rv_mode=None):
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
    niter : int
        The number of optimization steps to perform.
    return_opt : bool
        If True, return the distribution resulting from the optimization.
        Defaults to False.
    rv_mode : str, None
        Specifies how to interpret `sources` and `target`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `crvs` and `rvs` are interpreted as random variable indices. If
        equal to 'names', the the elements are interpreted as random
        variable names. If `None`, then the value of `dist._rv_mode` is
        consulted, which defaults to 'indices'.

    Returns
    -------
    pid : PID namedtuple
        The partial information decomposition.
    opt_dist : Distribution
        The distribution resulting from the optimization. Note that var [0]
        is sources[0], [1] is sources[1] and [2] is target.
    """
    broja = BROJABivariateOptimizer(dist, sources, target, rv_mode)
    broja.optimize(niter=niter)
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
