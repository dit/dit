"""
Base class for optimization.
"""

from __future__ import division

from abc import ABCMeta, abstractmethod

from collections import namedtuple

from copy import deepcopy

from functools import reduce

from six import with_metaclass

from string import ascii_letters, digits

from types import MethodType

from boltons.iterutils import pairwise

import numpy as np
from scipy.optimize import basinhopping, differential_evolution, minimize

from .. import Distribution, insert_rvf, modify_outcomes
from ..algorithms.channelcapacity import channel_capacity
from ..exceptions import OptimizationException
from ..helpers import flatten, normalize_rvs, parse_rvs
from ..math import prod
from ..utils import partitions, powerset
from ..utils.optimization import (BasinHoppingCallBack,
                                  BasinHoppingInnerCallBack,
                                  Uniquifier,
                                  accept_test,
                                  basinhop_status,
                                  colon,
                                  )

__all__ = [
    'BaseOptimizer',
    'BaseConvexOptimizer',
    'BaseNonConvexOptimizer',
    'BaseAuxVarOptimizer',
]


class BaseOptimizer(with_metaclass(ABCMeta, object)):
    """
    Base class for performing optimizations.
    """

    def __init__(self, dist, rvs=None, crvs=None, rv_mode=None):
        """
        Initialize the optimizer.

        Parameters
        ----------
        dist : Distribution
            The distribution to compute the intrinsic mutual information of.
        rvs : iterable of iterables
            The variables of interest.
        crvs : iterable
            The variables to be conditioned on.
        rv_mode : str, None
            Specifies how to interpret `rvs` and `crvs`. Valid options are:
            {'indices', 'names'}. If equal to 'indices', then the elements of
            `crvs` and `rvs` are interpreted as random variable indices. If
            equal to 'names', the the elements are interpreted as random
            variable names. If `None`, then the value of `dist._rv_mode` is
            consulted, which defaults to 'indices'.
        """
        rvs, crvs, rv_mode = normalize_rvs(dist, rvs, crvs, rv_mode)
        self._dist = dist.copy(base='linear')

        self._alphabet = self._dist.alphabet
        self._original_shape = list(map(len, self._dist.alphabet))

        self._true_rvs = [parse_rvs(self._dist, rv, rv_mode=rv_mode)[1] for rv in rvs]
        self._true_crvs = parse_rvs(self._dist, crvs, rv_mode=rv_mode)[1]
        self._dist = modify_outcomes(self._dist, tuple)

        # compress all random variables down to single vars
        self._unqs = []
        for var in self._true_rvs + [self._true_crvs]:
            unq = Uniquifier()
            self._dist = insert_rvf(self._dist, lambda x: (unq(tuple(x[i] for i in var)),))
            self._unqs.append(unq)

        self._dist.make_dense()

        self._full_shape = list(map(len, self._dist.alphabet))
        self._full_pmf = self._dist.pmf.reshape(self._full_shape)

        self._n = dist.outcome_length()
        self._pmf = self._full_pmf.sum(axis=tuple(range(self._n)))
        self._shape = self._pmf.shape

        self._full_vars = set(range(len(self._full_shape)))
        self._all_vars = set(range(len(rvs)+1))
        self._rvs = set(range(len(rvs)))
        self._crvs = {len(rvs)}

        self._proxy_vars = tuple(range(self._n, self._n+len(rvs)+1))

        self._additional_options = {}

        self.constraints = []

    ###########################################################################
    # Required methods in subclasses

    @abstractmethod
    def construct_initial(self):
        """
        Select the default method of generating an initial condition.

        Returns
        -------
        x : np.ndarray
            An optimization vector.
        """
        pass

    @abstractmethod
    def _optimization_backend(self, x0, minimizer_kwargs, niter):
        """
        Abstract method for performing an optimization.

        Parameters
        ----------
        x0 : np.ndarray
            An initial optimization vector.
        minimizer_kwargs : dict
            A dictionary of keyword arguments to pass to the optimizer.
        niter : int
            If applicable, the number of iterations to make.

        Returns
        -------
        result : OptimizeResult, None
            Returns the result of the optimization, or None if it failed.
        """
        pass

    @abstractmethod
    def _objective(self):
        """
        Construct the objective function.

        Returns
        -------
        obj : func
            The objective function.
        """
        pass

    ###########################################################################
    # Various initial conditions

    def construct_random_initial(self):
        """
        Construct a random optimization vector.

        Returns
        -------
        x : np.ndarray
            A random optimization vector.
        """
        vec = np.random.random(size=self._optvec_size)
        return vec

    def construct_uniform_initial(self):
        """
        Construct a uniform optimization vector.

        Returns
        -------
        x : np.ndarray
            A random optimization vector.
        """
        vec = np.ones(self._optvec_size) / self._optvec_size
        return vec

    ###########################################################################
    # Convenience functions for constructing objectives.

    @staticmethod
    def _h(p):
        """
        Compute the entropy of `p`.

        Parameters
        ----------
        p : np.ndarray
            A vector of probabilities.

        Returns
        -------
        h : float
            The entropy.
        """
        return -np.nansum(p*np.log2(p))

    def _entropy(self, rvs, crvs=None):
        """
        Compute the conditional entropy, H[X|Y]

        Parameters
        ----------
        rvs : collection
            The indices to consider as the X variable.
        crvs : collection
            The indices to consider as the Y variable.

        Returns
        -------
        h : func
            The conditional entropy.
        """
        if crvs is None:
            crvs = set()
        idx_joint = tuple(self._all_vars - (rvs|crvs))
        idx_crvs = tuple(self._all_vars - crvs)

        def entropy(pmf):
            """
            Compute the specified entropy.

            Parameters
            ----------
            pmf : np.ndarray
                The joint probability distribution.

            Returns
            -------
            h : float
                The entropy.
            """
            pmf_joint = pmf.sum(axis=idx_joint, keepdims=True)
            pmf_crvs = pmf_joint.sum(axis=idx_crvs, keepdims=True)

            h_joint = self._h(pmf_joint)
            h_crvs = self._h(pmf_crvs)

            ch = h_joint - h_crvs

            return ch

        return entropy

    def _mutual_information(self, rv_x, rv_y):
        """
        Compute the mutual information, I[X:Y].

        Parameters
        ----------
        rv_x : collection
            The indices to consider as the X variable.
        rv_y : collection
            The indices to consider as the Y variable.

        Returns
        -------
        mi : func
            The mutual information.
        """
        idx_xy = tuple(self._all_vars - (rv_x | rv_y))
        idx_x = tuple(self._all_vars - rv_x)
        idx_y = tuple(self._all_vars - rv_y)

        def mutual_information(pmf):
            """
            Compute the specified mutual information.

            Parameters
            ----------
            pmf : np.ndarray
                The joint probability distribution.

            Returns
            -------
            mi : float
                The mutual information.
            """
            pmf_xy = pmf.sum(axis=idx_xy, keepdims=True)
            pmf_x = pmf_xy.sum(axis=idx_x, keepdims=True)
            pmf_y = pmf_xy.sum(axis=idx_y, keepdims=True)

            mi = np.nansum(pmf_xy * np.log2(pmf_xy / (pmf_x * pmf_y)))

            return mi

        return mutual_information

    def _conditional_mutual_information(self, rv_x, rv_y, rv_z):
        """
        Compute the conditional mutual information, I[X:Y|Z].

        Parameters
        ----------
        rv_x : collection
            The indices to consider as the X variable.
        rv_y : collection
            The indices to consider as the Y variable.
        rv_z : collection
            The indices to consider as the Z variable.

        Returns
        -------
        cmi : func
            The conditional mutual information.
        """
        idx_xyz = tuple(self._all_vars - (rv_x | rv_y | rv_z))
        idx_xz = tuple(self._all_vars - (rv_x | rv_z))
        idx_yz = tuple(self._all_vars - (rv_y | rv_z))
        idx_z = tuple(self._all_vars - rv_z)

        def conditional_mutual_information(pmf):
            """
            Compute the specified conditional mutual information.

            Parameters
            ----------
            pmf : np.ndarray
                The joint probability distribution.

            Returns
            -------
            cmi : float
                The conditional mutual information.
            """
            pmf_xyz = pmf.sum(axis=idx_xyz, keepdims=True)
            pmf_xz = pmf_xyz.sum(axis=idx_xz, keepdims=True)
            pmf_yz = pmf_xyz.sum(axis=idx_yz, keepdims=True)
            pmf_z = pmf_xz.sum(axis=idx_z, keepdims=True)

            cmi = np.nansum(pmf_xyz * np.log2(pmf_z * pmf_xyz / pmf_xz / pmf_yz))

            return cmi

        return conditional_mutual_information

    def _coinformation(self, rvs, crvs=None):
        """
        Compute the coinformation.

        Parameters
        ----------
        rvs : set
            The random variables to compute the coinformation of.
        crvs : set
            The random variables to condition on.

        Returns
        -------
        ci : func
            The coinformation.
        """
        if crvs is None:
            crvs = set()
        idx_joint = tuple(self._all_vars - (rvs | crvs))
        idx_crvs = tuple(self._all_vars - crvs)
        idx_subrvs = [tuple(self._all_vars - set(ss)) for ss in sorted(powerset(rvs), key=len)[1:-1]]
        power = [(-1)**len(ss) for ss in sorted(powerset(rvs), key=len)[1:-1]]
        power += [(-1)**len(rvs)]
        power += [-sum(power)]

        def coinformation(pmf):
            """
            Compute the specified co-information.

            Parameters
            ----------
            pmf : np.ndarray
                The joint probability distribution.

            Returns
            -------
            ci : float
                The co-information.
            """
            pmf_joint = pmf.sum(axis=idx_joint, keepdims=True)
            pmf_crvs = pmf_joint.sum(axis=idx_crvs, keepdims=True)
            pmf_subrvs = [pmf_joint.sum(axis=idx, keepdims=True) for idx in idx_subrvs] + [pmf_joint, pmf_crvs]

            pmf_ci = reduce(np.multiply, [pmf**p for pmf, p in zip(pmf_subrvs, power)])

            ci = np.nansum(pmf_joint * np.log2(pmf_ci))

            return ci

        return coinformation

    def _total_correlation(self, rvs, crvs=None):
        """
        Compute the total correlation.

        Parameters
        ----------
        rvs : set
            The random variables to compute the total correlation of.
        crvs : set
            The random variables to condition on.

        Returns
        -------
        tc : func
            The total correlation.
        """
        if crvs is None:
            crvs = set()
        idx_joint = tuple(self._all_vars - (rvs | crvs))
        idx_margs = [tuple(self._all_vars - ({rv} | crvs)) for rv in rvs]
        idx_crvs = tuple(self._all_vars - crvs)
        n = len(rvs) - 1

        def total_correlation(pmf):
            """
            Compute the specified total correlation.

            Parameters
            ----------
            pmf : np.ndarray
                The joint probability distribution.

            Returns
            -------
            ci : float
                The total correlation.
            """
            pmf_joint = pmf.sum(axis=idx_joint, keepdims=True)
            pmf_margs = [pmf_joint.sum(axis=marg, keepdims=True) for marg in idx_margs]
            pmf_crvs = pmf_margs[0].sum(axis=idx_crvs, keepdims=True)

            h_crvs = self._h(pmf_crvs.ravel())
            h_margs = sum(self._h(p.ravel()) for p in pmf_margs)
            h_joint = self._h(pmf_joint.ravel())

            tc = h_margs - h_joint - n*h_crvs

            return tc

        return total_correlation

    def _dual_total_correlation(self, rvs, crvs=None):
        """
        Compute the dual total correlation.

        Parameters
        ----------
        rvs : set
            The random variables to compute the dual total correlation of.
        crvs : set
            The random variables to condition on.

        Returns
        -------
        dtc : func
            The dual total correlation.
        """
        if crvs is None:
            crvs = set()
        idx_joint = tuple(self._all_vars - (rvs | crvs))
        idx_margs = [tuple(self._all_vars - ((rvs - {rv}) | crvs)) for rv in rvs]
        idx_crvs = tuple(self._all_vars - crvs)
        n = len(rvs) - 1

        def dual_total_correlation(pmf):
            """
            Compute the specified dual total correlation.

            Parameters
            ----------
            pmf : np.ndarray
                The joint probability distribution.

            Returns
            -------
            ci : float
                The dual total correlation.
            """
            pmf_joint = pmf.sum(axis=idx_joint, keepdims=True)
            pmf_margs = [pmf_joint.sum(axis=marg, keepdims=True) for marg in idx_margs]
            pmf_crvs = pmf_joint.sum(axis=idx_crvs, keepdims=True)

            h_crvs = self._h(pmf_crvs)
            h_joint = self._h(pmf_joint) - h_crvs
            h_margs = [self._h(marg) - h_crvs for marg in pmf_margs]

            dtc = sum(h_margs) - n*h_joint

            return dtc

        return dual_total_correlation

    def _caekl_mutual_information(self, rvs, crvs=None):
        """
        Compute the CAEKL mutual information.

        Parameters
        ----------
        rvs : set
            The random variables to compute the CAEKL mutual information of.
        crvs : set
            The random variables to condition on.

        Returns
        -------
        caekl : func
            The CAEKL mutual information.
        """
        if crvs is None:
            crvs = set()
        parts = [p for p in partitions(rvs) if len(p) > 1]
        idx_parts = {}
        for part in parts:
            for p in part:
                if p not in idx_parts:
                    idx_parts[p] = tuple(self._all_vars - (p|crvs))
        part_norms = [len(part) - 1 for part in parts]
        idx_joint = tuple(self._all_vars - (rvs|crvs))
        idx_crvs = tuple(self._all_vars - crvs)

        def caekl_mutual_information(pmf):
            """
            Compute the specified CAEKL mutual information.

            Parameters
            ----------
            pmf : np.ndarray
                The joint probability distribution.

            Returns
            -------
            caekl : float
                The CAEKL mutual information.
            """
            pmf_joint = pmf.sum(axis=idx_joint, keepdims=True)
            pmf_parts = {p: pmf_joint.sum(axis=idx, keepdims=True) for p, idx in idx_parts.items()}
            pmf_crvs = pmf_joint.sum(axis=idx_crvs, keepdims=True)

            h_crvs = self._h(pmf_crvs)
            h_joint = self._h(pmf_joint) - h_crvs

            pairs = zip(parts, part_norms)
            candidates = [(sum(self._h(pmf_parts[p]) - h_crvs for p in part)-h_joint)/norm for part, norm in pairs]

            caekl = min(candidates)

            return caekl

        return caekl_mutual_information

    ###########################################################################
    # Optimization methods.

    def optimize(self, x0=None, niter=None, maxiter=None, polish=1e-6, callback=False):
        """
        Perform the optimization.

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
        result : OptimizeResult
            The result of the optimization.
        """
        try:
            callable(self.objective)
        except AttributeError:
            self.objective = MethodType(self._objective(), self)

        x0 = x0.copy() if x0 is not None else self.construct_initial()

        icb = BasinHoppingInnerCallBack() if callback else None

        minimizer_kwargs = {'bounds': [(0, 1)] * x0.size,
                            'callback': icb,
                            'constraints': self.constraints,
                            'options': {},
                            }

        try:  # pragma: no cover
            if callable(self._jacobian):
                minimizer_kwargs['jac'] = self._jacobian
            else:  # compute jacobians for objective, constraints using numdifftools
                import numdifftools as ndt
                minimizer_kwargs['jac'] = ndt.Jacobian(self.objective)
                for const in minimizer_kwargs['constraints']:
                    const['jac'] = ndt.Jacobian(const['fun'])
        except AttributeError:
            pass

        additions = deepcopy(self._additional_options)
        options = additions.pop('options', {})
        minimizer_kwargs['options'].update(options)
        minimizer_kwargs.update(additions)

        if maxiter:
            minimizer_kwargs['options']['maxiter'] = maxiter

        self._callback = BasinHoppingCallBack(minimizer_kwargs.get('constraints', {}), icb)

        result = self._optimization_backend(x0, minimizer_kwargs, niter)

        if result:
            self._optima = result.x
        else:  # pragma: no cover
            msg = "No optima found."
            raise OptimizationException(msg)

        if polish:
            self._polish(cutoff=polish)

        return result

    def _optimize_shotgun(self, x0, minimizer_kwargs, niter):
        """
        Perform a non-convex optimization. This uses a "shotgun" approach, minimizing
        several random initial conditions and selecting the minimal result.

        Parameters
        ----------
        x0 : ndarray, None
            Initial optimization vector. If None, use a random vector.
        minimizer_kwargs : dict
            A dictionary of keyword arguments to pass to the optimizer.
        niter : int
            If applicable, the number of iterations to make.

        Returns
        -------
        result : OptimizeResult, None
            The result of the optimization. Returns None if the optimization failed.

        TODO
        ----
         * Rather than random initial conditions, use latin hypercube sampling.
         * Make parallel in some fashion (threads, processes).
        """
        if niter is None:
            niter = self._default_hops

        results = []

        if x0 is not None:
            res = minimize(fun=self.objective,
                           x0=x0,
                           **minimizer_kwargs
                           )
            if res.success:
                results.append(res)
            niter -= 1

        ics = (self.construct_random_initial() for _ in range(niter))
        for initial in ics:
            res = minimize(fun=self.objective,
                           x0=initial,
                           **minimizer_kwargs
                           )
            if res.success:
                results.append(res)

        try:
            result = min(results, key=lambda r: self.objective(r.x))
        except ValueError:  # pragma: no cover
            result = None

        return result

    def _polish(self, cutoff=1e-6):
        """
        Improve the solution found by the optimizer.

        Parameters
        ----------
        cutoff : float
            Set probabilities lower than this to zero, reducing the total
            optimization dimension.
        """
        x0 = self._optima.copy()
        count = (x0 < cutoff).sum()
        x0[x0 < cutoff] = 0

        minimizer_kwargs = {'bounds': [(0, 0) if np.isclose(x, 0) else (0, 1) for x in x0],
                            'tol': None,
                            'callback': None,
                            }

        try:
            minimizer_kwargs['constraints'] = self.constraints
        except AttributeError:
            pass

        try:  # pragma: no cover
            if callable(self._jacobian):
                minimizer_kwargs['jac'] = self._jacobian
            else:  # compute jacobians for objective, constraints using numdifftools
                import numdifftools as ndt
                minimizer_kwargs['jac'] = ndt.Jacobian(self.objective)
                for const in minimizer_kwargs['constraints']:
                    const['jac'] = ndt.Jacobian(const['fun'])
        except AttributeError:
            pass

        res = minimize(fun=self.objective,
                       x0=x0,
                       **minimizer_kwargs
                       )

        if res.success:
            self._optima = res.x.copy()

        if count < (res.x < cutoff).sum():
            self._polish(cutoff=cutoff)


class BaseConvexOptimizer(BaseOptimizer):
    """
    Implement convex optimization.
    """

    def _optimization_backend(self, x0, minimizer_kwargs, niter):
        """
        Perform a convex optimization.

        Parameters
        ----------
        x0 : ndarray, None
            Initial optimization vector. If None, use a random vector.
        minimizer_kwargs : dict
            A dictionary of keyword arguments to pass to the optimizer.
        niter : int
            If applicable, the number of iterations to make.

        Returns
        -------
        result : OptimizeResult, None
            The result of the optimization. Returns None if the optimization failed.
        """
        if niter is None:
            # even though this is convex, there might still be some optimization issues,
            # so we use niter > 1.
            niter = 2
        return self._optimize_shotgun(x0, minimizer_kwargs, niter=niter)


class BaseNonConvexOptimizer(BaseOptimizer):
    """
    Implement non-convex optimization.
    """

    _shotgun = False

    def _optimization_basinhopping(self, x0, minimizer_kwargs, niter):
        """
        Perform a non-convex optimization. This uses scipy.optimize.basinhopping,
        a simulated annealing-like algorithm.

        Parameters
        ----------
        x0 : ndarray, None
            Initial optimization vector. If None, use a random vector.
        minimizer_kwargs : dict
            A dictionary of keyword arguments to pass to the optimizer.
        niter : int
            If applicable, the number of iterations to make.

        Returns
        -------
        result : OptimizeResult, None
            The result of the optimization. Returns None if the optimization failed.
        """
        if niter is None:
            niter = self._default_hops

        if self._shotgun:
            res_shotgun = self._optimize_shotgun(x0.copy(), minimizer_kwargs, 5)
            if res_shotgun:
                x0 = res_shotgun.x.copy()
        else:
            res_shotgun = None

        result = basinhopping(func=self.objective,
                              x0=x0,
                              minimizer_kwargs=minimizer_kwargs,
                              niter=niter,
                              accept_test=accept_test,
                              callback=self._callback,
                              )

        success, _ = basinhop_status(result)
        if not success:  # pragma: no cover
            result = self._callback.minimum() or res_shotgun

        return result

    def _optimization_diffevo(self, x0, minimizer_kwargs, niter):  # pragma: no cover
        """

        Parameters
        ----------
        x0 : np.ndarray
            An optimization vector.
        minimizer_kwargs : dict
            A dictionary of keyword arguments to pass to the optimizer.
        niter : int
            If applicable, the number of iterations to make.

        Returns
        -------
        result : OptimizeResult, None
            The result of the optimization. Returns None if the optimization failed.
        """
        if 'constraints' in minimizer_kwargs:
            msg = "Differential Evolution can only be used in unconstrained optimization."
            raise OptimizationException(msg)

        if niter is None:
            niter = self._default_hops

        result = differential_evolution(func=self.objective,
                                        bounds=minimizer_kwargs['bounds'],
                                        maxiter=minimizer_kwargs['options']['maxiter'],
                                        popsize=niter,
                                        tol=minimizer_kwargs['options']['ftol'],
                                        )

        if result.success:
            return result

    _optimization_backend = _optimization_basinhopping


AuxVar = namedtuple('AuxVar', ['bases', 'bound', 'shape', 'mask', 'size'])


class BaseAuxVarOptimizer(BaseNonConvexOptimizer):
    """
    Base class that performs many methods related to optimizing auxiliary variables.
    """

    ###########################################################################
    # Register the auxiliary variables.

    def _construct_auxvars(self, auxvars):
        """
        Register the auxiliary variables.

        Parameters
        ----------
        auxvars : [(tuple, int)]
            The bases and bounds for each auxiliary variable.
        """
        self._aux_vars = []

        for bases, bound in auxvars:
            shape = [self._shape[i] for i in bases] + [bound]
            mask = np.ones(shape) / bound
            self._aux_vars.append(AuxVar(bases, bound, shape, mask, prod(shape)))
            self._shape += (bound,)
            self._full_shape += (bound,)
            self._all_vars |= {len(self._all_vars)}

        self._arvs = self._all_vars - (self._rvs | self._crvs)
        self._aux_bounds = [av.bound for av in self._aux_vars]
        self._optvec_size = sum([av.size for av in self._aux_vars])
        self._default_hops = prod(self._aux_bounds)
        self._parts = list(pairwise(np.cumsum([0] + [av.size for av in self._aux_vars])))
        self._construct_slices()
        if len(self._aux_vars) == 1:
            self.construct_joint = self._construct_joint_single

    def _construct_slices(self):
        """
        Construct the slices used to construct the joint pmf.
        """
        arvs = sorted(self._arvs)

        self._full_slices = []
        for i, (auxvar, var) in enumerate(zip(self._aux_vars, arvs)):
            relevant_vars = {self._n + b for b in auxvar.bases}
            index = sorted(self._full_vars) + [self._n + a for a in arvs[:i + 1]]
            var += self._n
            self._full_slices.append([colon if i in relevant_vars | {var} else np.newaxis for i in index])

        self._slices = []
        for i, (auxvar, var) in enumerate(zip(self._aux_vars, arvs)):
            relevant_vars = auxvar.bases
            index = sorted(self._rvs | self._crvs | set(arvs[:i + 1]))
            self._slices.append([colon if i in relevant_vars | {var} else np.newaxis for i in index])

    ###########################################################################
    # Constructing the joint distribution.

    def _construct_channels(self, x):
        """
        Construct the conditional distributions which produce the
        auxiliary variables.

        Parameters
        ----------
        x : np.ndarray
            An optimization vector

        Yields
        ------
        channel : np.ndarray
            A conditional distribution.
        """
        parts = [x[a:b] for a, b in self._parts]

        for part, auxvar in zip(parts, self._aux_vars):
            channel = part.reshape(auxvar.shape)
            channel /= channel.sum(axis=(-1,), keepdims=True)
            channel[np.isnan(channel)] = auxvar.mask[np.isnan(channel)]

            yield channel

    def construct_joint(self, x):
        """
        Construct the joint distribution.

        Parameters
        ----------
        x : np.ndarray
            An optimization vector.

        Returns
        -------
        joint : np.ndarray
            The joint distribution resulting from the distribution passed
            in and the optimization vector.
        """
        joint = self._pmf

        channels = self._construct_channels(x)

        for channel, slc in zip(channels, self._slices):
            joint = joint[..., np.newaxis] * channel[slc]

        return joint

    def _construct_joint_single(self, x):
        """
        Construct the joint distribution.

        Parameters
        ----------
        x : np.ndarray
            An optimization vector.

        Returns
        -------
        joint : np.ndarray
            The joint distribution resulting from the distribution passed
            in and the optimization vector.
        """
        _, _, shape, mask, _ = self._aux_vars[0]
        channel = x.reshape(shape)
        channel /= channel.sum(axis=-1, keepdims=True)
        channel[np.isnan(channel)] = mask[np.isnan(channel)]

        joint = self._pmf[..., np.newaxis] * channel[self._slices[0]]

        return joint

    def construct_full_joint(self, x):
        """
        Construct the joint distribution.

        Parameters
        ----------
        x : np.ndarray
            An optimization vector.

        Returns
        -------
        joint : np.ndarray
            The joint distribution resulting from the distribution passed
            in and the optimization vector.
        """
        joint = self._full_pmf

        channels = self._construct_channels(x)

        for channel, slc in zip(channels, self._full_slices):
            joint = joint[..., np.newaxis] * channel[slc]

        return joint

    ###########################################################################
    # Various initial conditions

    def construct_random_initial(self):
        """
        Construct a random optimization vector.

        Returns
        -------
        x : np.ndarray
            A random optimization vector.
        """
        vecs = []
        for av in self._aux_vars:
            vec = np.random.random(av.shape) / av.bound
            vecs.append(vec.ravel())
        return np.concatenate(vecs, axis=0)

    def construct_uniform_initial(self):
        """
        Construct a uniform optimization vector.

        Returns
        -------
        x : np.ndarray
            A uniform optimization vector.
        """
        vecs = []
        for av in self._aux_vars:
            vec = np.ones(av.shape) / av.bound
            vecs.append(vec.ravel())
        return np.concatenate(vecs, axis=0)

    def construct_copy_initial(self):
        """
        Construct a copy optimization vector.

        Returns
        -------
        x : np.ndarray
            A copy optimization vector.
        """
        vecs = []
        for av in self._aux_vars:
            shape = [prod(av.shape[:-1]), av.shape[-1]]
            vec = np.eye(*shape) / av.bound
            vecs.append(vec.ravel())
        return np.concatenate(vecs, axis=0)

    def construct_constant_initial(self):
        """
        Construct a constant optimization vector.

        Returns
        -------
        x : np.ndarray
            A constant optimization vector.
        """
        vecs = []
        for av in self._aux_vars:
            vec = np.zeros(av.shape)
            vec[..., 0] = 1
            vecs.append(vec.ravel())
        return np.concatenate(vecs, axis=0)

    ###########################################################################
    # Construct the optimized distribution.

    def construct_distribution(self, x=None, cutoff=1e-5):
        """
        Construct the distribution.

        Parameters
        ----------
        x : np.ndarray, None
            An optimization vector. If None, use `self._optima`.
        cutoff : float
            Ignore probabilities smaller than this.

        Returns
        -------
        d : Distribution
            The original distribution, plus its auxiliary variables.
        """
        if x is None:
            x = self._optima

        alphabets = list(self._alphabet)

        try:
            # if all outcomes are strings, make new variable strings too.
            ''.join(flatten(alphabets))
            for bound in self._aux_bounds:
                alphabets += [(digits + ascii_letters)[:bound]]
            string = True
        except TypeError:
            for bound in self._aux_bounds:
                alphabets += [list(range(bound))]
            string = False

        joint = self.construct_full_joint(x)
        outcomes, pmf = zip(*[(o, p) for o, p in np.ndenumerate(joint) if p > cutoff])

        # normalize, in case cutoffs removed a significant amount of pmf
        pmf = np.asarray(pmf)
        pmf /= pmf.sum()

        d = Distribution(outcomes, pmf)

        mapping = {}
        for i, unq in zip(sorted(self._n + i for i in self._rvs | self._crvs), self._unqs):
            if len(unq.inverse) > 1:
                n = d.outcome_length()
                d = insert_rvf(d, lambda o: unq.inverse[o[i]])
                mapping[i] = tuple(range(n, n+len(unq.inverse[0])))

        new_map = {}
        for rv, rvs in zip(sorted(self._rvs), self._true_rvs):
            i = rv + self._n
            for a, b in zip(rvs, mapping[i]):
                new_map[a] = b

        mapping = [[(new_map[i] if i in new_map else i) for i in range(len(self._full_shape))
                                                                 if i not in self._proxy_vars]]

        d = d.coalesce(mapping, extract=True)

        if string:
            d = modify_outcomes(d, lambda o: ''.join(map(str, o)))

        return d

    ###########################################################################
    # Constraints which may or may not be useful in certain contexts.

    def _constraint_deterministic(self):
        """
        Constructor for a constraint which specifies that auxiliary variables
        must be a deterministic function of the random variables.

        Returns
        -------
        constraint : func
            The constraint function.
        """
        entropy = self._entropy(self._arvs, self._rvs | self._crvs)

        def constraint(x):
            """
            Constraint which ensure that the auxiliary variables are a function
            of the random variables.

            Parameters
            ----------
            x : np.ndarray
                An optimization vector.

            Returns
            -------

            """
            pmf = self.construct_joint(x)
            return entropy(pmf) ** 2

        return constraint

    ###########################################################################
    # todo

    def _channel_capacity(self, x):  # pragma: no cover
        """
        Compute the channel capacity of the mapping z -> z_bar.

        Parameters
        ----------
        x : np.ndarray
            An optimization vector.

        Returns
        -------
        ccd : [float]
            The channel capacity of each auxiliary variable.

        TODO
        ----
        Make this compute the channel capacity of any/all auxvar(s)
        """
        ccs = []
        for channel in self._construct_channels(x):
            ccs.append(channel_capacity(channel)[0])
        return ccs

    def _post_process(self, style='entropy', minmax='min', niter=10, maxiter=None):  # pragma: no cover
        """
        Find a solution to the minimization with a secondary property.

        Parameters
        ----------
        style : 'entropy', 'channel'
            The measure to perform the secondary optimization on. If 'entropy',
            the entropy of z_bar is optimized. If 'channel', the channel capacity
            of p(z_bar|z) is optimized.
        minmax : 'min', 'max'
            Whether to minimize or maximize the objective.
        niter : int
            The number of basin hops to perform.
        maxiter : int
            The number of minimization steps to perform.

        Notes
        -----
        This seems to not work well. Presumably the channel space, once
        restricted to matching the correct objective, is very fragmented.

        TODO
        ----
        This should really use some sort of multiobjective optimization.
        """
        entropy = self._entropy(self._arvs)

        def objective_entropy(x):
            """
            Post-process the entropy.

            Parameters
            ----------
            x : np.ndarray
                An optimization vector.

            Returns
            -------
            ent : float
                The entropy.
            """
            ent = entropy(self.construct_joint(x))
            return sign * ent

        def objective_channelcapacity(x):
            """
            Post-process the channel capacity.

            Parameters
            ----------
            x : np.ndarray
                An optimization vector.

            Returns
            -------
            cc : float
                The sum of the channel capacities.
            """
            cc = sum(self._channel_capacity(x))
            return sign * cc

        sign = +1 if minmax == 'min' else -1

        if style == 'channel':
            objective = objective_channelcapacity
        elif style == 'entropy':
            objective = objective_entropy
        else:
            msg = "Style {} is not understood.".format(style)
            raise OptimizationException(msg)

        true_objective = self.objective(self._optima)

        def constraint_match_objective(x):
            """
            Constraint to ensure that the new solution is not worse than that
            found before.

            Parameters
            ----------
            x : np.ndarray
                An optimization vector.

            Returns
            -------
            obj : float
                The l2 deviation of the current objective from the true.
            """
            obj = (self.objective(x) - true_objective)**2
            return obj

        constraint = [{'type': 'eq',
                       'fun': constraint_match_objective,
                       }]

        # set up required attributes
        try:
            self.constraints += constraint
        except AttributeError:
            self.constraints = constraint

        self.__old_objective, self.objective = self.objective, objective

        self.optimize(x0=self._optima.copy(), niter=niter, maxiter=maxiter)

        # and remove them again.
        self.constraints = self.constraints[:-1]
        if not self.constraints:
            del self.constraints

        self.objective = self.__old_objective
        del self.__old_objective
