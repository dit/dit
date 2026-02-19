"""
Base class for optimization using JAX for automatic differentiation.

This module provides the same functionality as optimization.py but leverages
JAX for:
- Automatic differentiation (gradients and jacobians)
- JIT compilation for improved performance
- GPU/TPU acceleration (when available)

Notes
-----
Requires JAX to be installed: pip install jax jaxlib
"""

from abc import ABCMeta
from abc import abstractmethod
from collections import namedtuple
from copy import deepcopy
from string import ascii_letters
from string import digits
from types import MethodType

import numpy as np

from loguru import logger

try:
    import jax
    import jax.numpy as jnp

    from jax import grad
    from jax import jacobian
    from jax import jit
    from jax import vmap
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    # Provide fallback so module can be imported even without JAX
    jnp = np
    def jit(f, **kwargs):
        return f
    def grad(f, **kwargs):
        raise ImportError("JAX is required for automatic differentiation")
    def jacobian(f, **kwargs):
        raise ImportError("JAX is required for automatic differentiation")
    def vmap(f, **kwargs):
        raise ImportError("JAX is required for vmap")

try:
    import optimistix as optx

    from slsqp_jax import SLSQP as JaxSLSQP
    SLSQP_JAX_AVAILABLE = True
except ImportError:
    SLSQP_JAX_AVAILABLE = False

from boltons.iterutils import pairwise
from scipy.optimize import basinhopping
from scipy.optimize import Bounds
from scipy.optimize import differential_evolution
from scipy.optimize import minimize
from scipy.optimize import shgo

from .. import Distribution
from .. import insert_rvf
from .. import modify_outcomes
from ..algorithms.channelcapacity import channel_capacity
from ..exceptions import ditException
from ..exceptions import OptimizationException
from ..helpers import flatten
from ..helpers import normalize_rvs
from ..helpers import parse_rvs
from ..math import prod
from ..math import sample_simplex
from ..utils import partitions
from ..utils import powerset
from ..utils.optimization import accept_test
from ..utils.optimization import basinhop_status
from ..utils.optimization import BasinHoppingCallBack
from ..utils.optimization import BasinHoppingInnerCallBack
from ..utils.optimization import colon
from ..utils.optimization import Uniquifier

__all__ = (
    'BaseJaxOptimizer',
    'BaseConvexJaxOptimizer',
    'BaseNonConvexJaxOptimizer',
    'BaseAuxVarJaxOptimizer',
)


def _check_jax():
    """
    Raise an error if JAX is not available.  Also enables 64-bit precision
    which is required for scipy's SLSQP solver.

    Raises
    ------
    ImportError
        If JAX is not installed.
    """
    if not JAX_AVAILABLE:
        raise ImportError(
            "JAX is required for this module. Install with: pip install jax jaxlib"
        )
    jax.config.update('jax_enable_x64', True)


# SVD singular values using JAX
@jit
def _svdvals(m):
    """
    Compute singular values of a matrix using JAX.

    Parameters
    ----------
    m : array_like
        The matrix to compute singular values of.

    Returns
    -------
    s : jnp.ndarray
        The singular values.
    """
    return jnp.linalg.svd(m, compute_uv=False)


class BaseJaxOptimizer(metaclass=ABCMeta):
    """
    Base class for performing optimizations using JAX for automatic differentiation.

    This class mirrors BaseOptimizer but uses JAX for:
    - Computing gradients via autodiff
    - JIT-compiling objective functions
    - Computing jacobians for constraints
    """

    # Whether to use JIT compilation (can be disabled for debugging)
    _use_jit = True

    # Whether to use autodiff for jacobians
    _use_autodiff = True

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
        _check_jax()

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
        self._full_pmf = jnp.array(self._dist.pmf.reshape(self._full_shape))

        self._n = dist.outcome_length()
        self._pmf = self._full_pmf.sum(axis=tuple(range(self._n)))
        self._shape = self._pmf.shape

        self._full_vars = set(range(len(self._full_shape)))
        self._all_vars = set(range(len(rvs) + 1))
        self._rvs = set(range(len(rvs)))
        self._crvs = {len(rvs)}

        self._proxy_vars = tuple(range(self._n, self._n + len(rvs) + 1))

        self._additional_options = {}

        self.constraints = []

        # Cache for JIT-compiled functions
        self._jit_cache = {}

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
    # JAX-specific helper methods

    def _maybe_jit(self, func, static_argnums=None):
        """
        Optionally JIT-compile a function.

        Parameters
        ----------
        func : callable
            The function to potentially JIT-compile.
        static_argnums : tuple, optional
            Arguments to treat as static for JIT compilation.

        Returns
        -------
        func : callable
            The (potentially JIT-compiled) function.
        """
        if self._use_jit and JAX_AVAILABLE:
            if static_argnums is not None:
                return jit(func, static_argnums=static_argnums)
            return jit(func)
        return func

    def _compute_jacobian(self, func):
        """
        Compute the Jacobian of a function using JAX autodiff.

        Parameters
        ----------
        func : callable
            The function to differentiate.

        Returns
        -------
        jac : callable
            The Jacobian function.
        """
        if self._use_autodiff and JAX_AVAILABLE:
            return jit(jacobian(func))
        return None

    def _compute_gradient(self, func):
        """
        Compute the gradient of a scalar function using JAX autodiff.

        Parameters
        ----------
        func : callable
            The scalar function to differentiate.

        Returns
        -------
        grad_func : callable
            The gradient function.
        """
        if self._use_autodiff and JAX_AVAILABLE:
            return jit(grad(func))
        return None

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
        vec = sample_simplex(self._optvec_size)
        return vec

    def construct_uniform_initial(self):
        """
        Construct a uniform optimization vector.

        Returns
        -------
        x : np.ndarray
            A uniform optimization vector.
        """
        vec = np.ones(self._optvec_size) / self._optvec_size
        return vec

    ###########################################################################
    # Convenience functions for constructing objectives using JAX.

    @staticmethod
    def _h(p):
        """
        Compute the entropy of `p` using JAX.

        Parameters
        ----------
        p : jnp.ndarray
            A vector of probabilities.

        Returns
        -------
        h : float
            The entropy.
        """
        # Safe for both forward (0*log(0)=0) and backward (no NaN gradients).
        # Using the "safe where" pattern: replace the operand in the masked
        # branch with a value that produces a finite log, so that jax.grad
        # never sees -inf or NaN in either branch.
        safe_p = jnp.where(p > 0, p, 1.0)
        return -jnp.nansum(jnp.where(p > 0, p * jnp.log2(safe_p), 0.0))

    def _entropy(self, rvs, crvs=None):
        """
        Compute the conditional entropy, H[X|Y], using JAX.

        Parameters
        ----------
        rvs : collection
            The indices to consider as the X variable.
        crvs : collection
            The indices to consider as the Y variable.

        Returns
        -------
        h : func
            The conditional entropy function.
        """
        if crvs is None:
            crvs = set()
        idx_joint = tuple(self._all_vars - (rvs | crvs))
        idx_crvs = tuple(self._all_vars - crvs)

        @self._maybe_jit
        def entropy(pmf):
            """
            Compute the specified entropy.

            Parameters
            ----------
            pmf : jnp.ndarray
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
        Compute the mutual information, I[X:Y], using JAX.

        Parameters
        ----------
        rv_x : collection
            The indices to consider as the X variable.
        rv_y : collection
            The indices to consider as the Y variable.

        Returns
        -------
        mi : func
            The mutual information function.
        """
        idx_xy = tuple(self._all_vars - (rv_x | rv_y))
        idx_x = tuple(self._all_vars - rv_x)
        idx_y = tuple(self._all_vars - rv_y)

        @self._maybe_jit
        def mutual_information(pmf):
            """
            Compute the specified mutual information.

            Parameters
            ----------
            pmf : jnp.ndarray
                The joint probability distribution.

            Returns
            -------
            mi : float
                The mutual information.
            """
            pmf_xy = pmf.sum(axis=idx_xy, keepdims=True)
            pmf_x = pmf_xy.sum(axis=idx_x, keepdims=True)
            pmf_y = pmf_xy.sum(axis=idx_y, keepdims=True)

            # Safe division and log -- use "safe where" for both forward
            # and backward: replace denominators/arguments in masked branches
            # with values that keep log2 finite.
            safe_denom = jnp.where(
                (pmf_x > 0) & (pmf_y > 0), pmf_x * pmf_y, 1.0)
            safe_xy = jnp.where(pmf_xy > 0, pmf_xy, 1.0)
            ratio = safe_xy / safe_denom
            mi = jnp.nansum(jnp.where(pmf_xy > 0, pmf_xy * jnp.log2(ratio), 0.0))

            return mi

        return mutual_information

    def _conditional_mutual_information(self, rv_x, rv_y, rv_z):
        """
        Compute the conditional mutual information, I[X:Y|Z], using JAX.

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
            The conditional mutual information function.
        """
        if not rv_z:
            return self._mutual_information(rv_x=rv_x, rv_y=rv_y)

        idx_xyz = tuple(self._all_vars - (rv_x | rv_y | rv_z))
        idx_xz = tuple(self._all_vars - (rv_x | rv_z))
        idx_yz = tuple(self._all_vars - (rv_y | rv_z))
        idx_z = tuple(self._all_vars - rv_z)

        @self._maybe_jit
        def conditional_mutual_information(pmf):
            """
            Compute the specified conditional mutual information.

            Parameters
            ----------
            pmf : jnp.ndarray
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

            # Safe computation: use "safe where" to avoid NaN gradients
            safe_denom = jnp.where(
                (pmf_xz > 0) & (pmf_yz > 0), pmf_xz * pmf_yz, 1.0)
            safe_numer = jnp.where(
                (pmf_xyz > 0) & (pmf_z > 0), pmf_z * pmf_xyz, 1.0)
            ratio = safe_numer / safe_denom
            cmi = jnp.nansum(jnp.where(pmf_xyz > 0, pmf_xyz * jnp.log2(ratio), 0.0))

            return cmi

        return conditional_mutual_information

    def _coinformation(self, rvs, crvs=None):
        """
        Compute the coinformation using JAX.

        Parameters
        ----------
        rvs : set
            The random variables to compute the coinformation of.
        crvs : set
            The random variables to condition on.

        Returns
        -------
        ci : func
            The coinformation function.
        """
        from functools import reduce

        if crvs is None:
            crvs = set()
        idx_joint = tuple(self._all_vars - (rvs | crvs))
        idx_crvs = tuple(self._all_vars - crvs)
        idx_subrvs = [tuple(self._all_vars - set(ss)) for ss in sorted(powerset(rvs), key=len)[1:-1]]
        power = [(-1)**len(ss) for ss in sorted(powerset(rvs), key=len)[1:-1]]
        power += [(-1)**len(rvs)]
        power += [-sum(power)]

        @self._maybe_jit
        def coinformation(pmf):
            """
            Compute the specified co-information.

            Parameters
            ----------
            pmf : jnp.ndarray
                The joint probability distribution.

            Returns
            -------
            ci : float
                The co-information.
            """
            pmf_joint = pmf.sum(axis=idx_joint, keepdims=True)
            pmf_crvs = pmf_joint.sum(axis=idx_crvs, keepdims=True)
            pmf_subrvs = [pmf_joint.sum(axis=idx, keepdims=True) for idx in idx_subrvs] + [pmf_joint, pmf_crvs]

            pmf_ci = reduce(jnp.multiply, [p**pw for p, pw in zip(pmf_subrvs, power)])
            safe_pmf_ci = jnp.where(pmf_joint > 0, pmf_ci, 1.0)

            ci = jnp.nansum(jnp.where(pmf_joint > 0, pmf_joint * jnp.log2(safe_pmf_ci), 0.0))

            return ci

        return coinformation

    def _total_correlation(self, rvs, crvs=None):
        """
        Compute the total correlation using JAX.

        Parameters
        ----------
        rvs : set
            The random variables to compute the total correlation of.
        crvs : set
            The random variables to condition on.

        Returns
        -------
        tc : func
            The total correlation function.
        """
        if crvs is None:
            crvs = set()
        idx_joint = tuple(self._all_vars - (rvs | crvs))
        idx_margs = [tuple(self._all_vars - ({rv} | crvs)) for rv in rvs]
        idx_crvs = tuple(self._all_vars - crvs)
        n = len(rvs) - 1

        @self._maybe_jit
        def total_correlation(pmf):
            """
            Compute the specified total correlation.

            Parameters
            ----------
            pmf : jnp.ndarray
                The joint probability distribution.

            Returns
            -------
            tc : float
                The total correlation.
            """
            pmf_joint = pmf.sum(axis=idx_joint, keepdims=True)
            pmf_margs = [pmf_joint.sum(axis=marg, keepdims=True) for marg in idx_margs]
            pmf_crvs = pmf_margs[0].sum(axis=idx_crvs, keepdims=True)

            h_crvs = self._h(pmf_crvs.ravel())
            h_margs = sum(self._h(p.ravel()) for p in pmf_margs)
            h_joint = self._h(pmf_joint.ravel())

            tc = h_margs - h_joint - n * h_crvs

            return tc

        return total_correlation

    def _dual_total_correlation(self, rvs, crvs=None):
        """
        Compute the dual total correlation using JAX.

        Parameters
        ----------
        rvs : set
            The random variables to compute the dual total correlation of.
        crvs : set
            The random variables to condition on.

        Returns
        -------
        dtc : func
            The dual total correlation function.
        """
        if crvs is None:
            crvs = set()
        idx_joint = tuple(self._all_vars - (rvs | crvs))
        idx_margs = [tuple(self._all_vars - ((rvs - {rv}) | crvs)) for rv in rvs]
        idx_crvs = tuple(self._all_vars - crvs)
        n = len(rvs) - 1

        @self._maybe_jit
        def dual_total_correlation(pmf):
            """
            Compute the specified dual total correlation.

            Parameters
            ----------
            pmf : jnp.ndarray
                The joint probability distribution.

            Returns
            -------
            dtc : float
                The dual total correlation.
            """
            pmf_joint = pmf.sum(axis=idx_joint, keepdims=True)
            pmf_margs = [pmf_joint.sum(axis=marg, keepdims=True) for marg in idx_margs]
            pmf_crvs = pmf_joint.sum(axis=idx_crvs, keepdims=True)

            h_crvs = self._h(pmf_crvs)
            h_joint = self._h(pmf_joint) - h_crvs
            h_margs = [self._h(marg) - h_crvs for marg in pmf_margs]

            dtc = sum(h_margs) - n * h_joint

            return dtc

        return dual_total_correlation

    def _caekl_mutual_information(self, rvs, crvs=None):
        """
        Compute the CAEKL mutual information using JAX.

        Parameters
        ----------
        rvs : set
            The random variables to compute the CAEKL mutual information of.
        crvs : set
            The random variables to condition on.

        Returns
        -------
        caekl : func
            The CAEKL mutual information function.
        """
        if crvs is None:
            crvs = set()
        parts = [p for p in partitions(rvs) if len(p) > 1]
        idx_parts = {}
        for part in parts:
            for p in part:
                if p not in idx_parts:
                    idx_parts[p] = tuple(self._all_vars - (p | crvs))
        part_norms = [len(part) - 1 for part in parts]
        idx_joint = tuple(self._all_vars - (rvs | crvs))
        idx_crvs = tuple(self._all_vars - crvs)

        # Convert to tuples for JAX tracing
        parts_tuple = tuple(tuple(p) for p in parts)
        idx_parts_items = tuple((k, v) for k, v in idx_parts.items())

        @self._maybe_jit
        def caekl_mutual_information(pmf):
            """
            Compute the specified CAEKL mutual information.

            Parameters
            ----------
            pmf : jnp.ndarray
                The joint probability distribution.

            Returns
            -------
            caekl : float
                The CAEKL mutual information.
            """
            pmf_joint = pmf.sum(axis=idx_joint, keepdims=True)
            pmf_parts = {p: pmf_joint.sum(axis=idx, keepdims=True) for p, idx in idx_parts_items}
            pmf_crvs = pmf_joint.sum(axis=idx_crvs, keepdims=True)

            h_crvs = self._h(pmf_crvs)
            h_joint = self._h(pmf_joint) - h_crvs

            candidates = []
            for part, norm in zip(parts_tuple, part_norms):
                h_parts = sum(self._h(pmf_parts[p]) - h_crvs for p in part)
                candidates.append((h_parts - h_joint) / norm)

            caekl = jnp.min(jnp.stack(candidates))

            return caekl

        return caekl_mutual_information

    def _maximum_correlation(self, rv_x, rv_y):
        """
        Compute the maximum correlation using JAX.

        Parameters
        ----------
        rv_x : collection
            The index to consider as the X variable.
        rv_y : collection
            The index to consider as the Y variable.

        Returns
        -------
        mc : func
            The maximum correlation function.
        """
        idx_xy = tuple(self._all_vars - (rv_x | rv_y))
        idx_x = tuple(self._all_vars - rv_x)
        idx_y = tuple(self._all_vars - rv_y)

        @self._maybe_jit
        def maximum_correlation(pmf):
            """
            Compute the specified maximum correlation.

            Parameters
            ----------
            pmf : jnp.ndarray
                The joint probability distribution.

            Returns
            -------
            mc : float
                The maximum correlation.
            """
            pmf_xy = pmf.sum(axis=idx_xy)
            pmf_x = pmf.sum(axis=idx_x)[:, jnp.newaxis]
            pmf_y = pmf.sum(axis=idx_y)[jnp.newaxis, :]

            Q = pmf_xy / (jnp.sqrt(pmf_x) * jnp.sqrt(pmf_y))
            Q = jnp.where(jnp.isnan(Q), 0.0, Q)

            mc = _svdvals(Q)[1]

            return mc

        return maximum_correlation

    def _conditional_maximum_correlation(self, rv_x, rv_y, rv_z):
        """
        Compute the conditional maximum correlation using JAX.

        Parameters
        ----------
        rv_x : collection
            The index to consider as the X variable.
        rv_y : collection
            The index to consider as the Y variable.
        rv_z : collection
            The index to consider as the Z variable.

        Returns
        -------
        cmc : func
            The conditional maximum correlation function.
        """
        idx_xyz = tuple(self._all_vars - (rv_x | rv_y | rv_z))
        idx_xz = tuple(self._all_vars - (rv_x | rv_z))
        idx_yz = tuple(self._all_vars - (rv_y | rv_z))

        @self._maybe_jit
        def conditional_maximum_correlation(pmf):
            """
            Compute the specified conditional maximum correlation.

            Parameters
            ----------
            pmf : jnp.ndarray
                The joint probability distribution.

            Returns
            -------
            cmc : float
                The conditional maximum correlation.
            """
            p_xyz = pmf.sum(axis=idx_xyz)
            p_xz = pmf.sum(axis=idx_xz)[:, jnp.newaxis, :]
            p_yz = pmf.sum(axis=idx_yz)[jnp.newaxis, :, :]

            Q = jnp.where(p_xyz > 0, p_xyz / (jnp.sqrt(p_xz * p_yz)), 0.0)

            # Compute SVD for each z slice
            def svd_slice(m):
                return _svdvals(jnp.squeeze(m))[1]

            # Use vmap for vectorized SVD computation if possible
            slices = jnp.dsplit(Q, Q.shape[2])
            cmc = max(svd_slice(m) for m in slices)

            return cmc

        return conditional_maximum_correlation

    def _total_variation(self, rv_x, rv_y):
        """
        Compute the total variation, TV[X||Y], using JAX.

        Parameters
        ----------
        rv_x : collection
            The indices to consider as the X variable.
        rv_y : collection
            The indices to consider as the Y variable.

        Returns
        -------
        tv : func
            The total variation function.

        Note
        ----
        The pmfs are assumed to be over the same alphabet.
        """
        idx_xy = tuple(self._all_vars - (rv_x | rv_y))
        idx_x = tuple(self._all_vars - rv_x)
        idx_y = tuple(self._all_vars - rv_y)

        @self._maybe_jit
        def total_variation(pmf):
            """
            Compute the specified total variation.

            Parameters
            ----------
            pmf : jnp.ndarray
                The joint probability distribution.

            Returns
            -------
            tv : float
                The total variation.
            """
            pmf_xy = pmf.sum(axis=idx_xy, keepdims=True)
            pmf_x = pmf_xy.sum(axis=idx_x)
            pmf_y = pmf_xy.sum(axis=idx_y)

            tv = jnp.abs(pmf_x - pmf_y).sum() / 2

            return tv

        return total_variation

    ###########################################################################
    # Optimization methods.

    def _build_jacobian(self):
        """
        Build the Jacobian function for the objective using JAX autodiff.

        Returns
        -------
        jac : callable or None
            The Jacobian function, or None if autodiff is disabled.
        """
        if not self._use_autodiff or not JAX_AVAILABLE:
            return None

        try:
            obj = self.objective
        except AttributeError:
            return None

        def jac_wrapper(x):
            x_jax = jnp.array(x)
            g = grad(lambda x: float(obj(x)))(x_jax)
            return np.array(g)

        return jac_wrapper

    ###########################################################################
    # slsqp-jax native optimization helpers

    def _build_slsqp_jax_solver(self, x0_size, maxiter=None):
        """
        Build a slsqp-jax SLSQP solver and the adapted objective/constraint
        functions for use with optimistix.minimise.

        Parameters
        ----------
        x0_size : int
            The size of the optimization vector (needed for inequality
            constraint count).
        maxiter : int, optional
            Maximum number of SLSQP iterations.

        Returns
        -------
        solver : JaxSLSQP
            The configured SLSQP solver.
        obj_fn : callable
            Objective with signature ``(x, args) -> (scalar, None)``.
        """
        raw_objective = self._jax_raw_objective
        raw_constraints = self._jax_raw_constraints

        # The information-theoretic functions (_h, MI, CMI, etc.) now use
        # the "safe where" pattern: arguments in masked branches are replaced
        # with values that keep log2 finite, so both the forward pass and
        # jax.grad are free of NaN/Inf even when some probabilities are zero
        # or slightly negative during SLSQP line search.
        def obj_fn(x, args):
            return raw_objective(x), None

        eq_funs = [c['fun'] for c in raw_constraints if c['type'] == 'eq']
        ineq_funs = [c['fun'] for c in raw_constraints if c['type'] == 'ineq']

        n_eq = 0
        eq_constraint_fn = None
        if eq_funs:
            def eq_constraint_fn(x, args):
                vals = [jnp.atleast_1d(f(x)) for f in eq_funs]
                return jnp.concatenate(vals)
            # compute count from a dummy call
            _dummy = jnp.zeros(x0_size)
            n_eq = int(eq_constraint_fn(_dummy, None).shape[0])

        # box bounds [0, 1] encoded as inequality constraints (>= 0)
        def _box_ineq(x, args):
            return jnp.concatenate([x, 1.0 - x])

        n_ineq_box = 2 * x0_size

        n_ineq_extra = 0
        if ineq_funs:
            def _extra_ineq(x, args):
                vals = [jnp.atleast_1d(f(x)) for f in ineq_funs]
                return jnp.concatenate(vals)
            _dummy = jnp.zeros(x0_size)
            n_ineq_extra = int(_extra_ineq(_dummy, None).shape[0])

            def ineq_constraint_fn(x, args):
                return jnp.concatenate([_box_ineq(x, args),
                                        _extra_ineq(x, args)])
        else:
            def ineq_constraint_fn(x, args):
                return _box_ineq(x, args)

        n_ineq = n_ineq_box + n_ineq_extra

        solver_kwargs = {
            'rtol': 1e-10,
            'atol': 1e-10,
        }
        if eq_constraint_fn is not None:
            solver_kwargs['eq_constraint_fn'] = eq_constraint_fn
            solver_kwargs['n_eq_constraints'] = n_eq
        solver_kwargs['ineq_constraint_fn'] = ineq_constraint_fn
        solver_kwargs['n_ineq_constraints'] = n_ineq

        solver = JaxSLSQP(**solver_kwargs)

        max_steps = maxiter or 1000

        return solver, obj_fn, max_steps

    def _slsqp_jax_minimize(self, x0, maxiter=None):
        """
        Run a single slsqp-jax optimization from *x0*.

        Parameters
        ----------
        x0 : jnp.ndarray
            Initial point (JAX array).
        maxiter : int, optional
            Maximum SLSQP iterations.

        Returns
        -------
        result : OptimizeResult
            A scipy-compatible OptimizeResult.
        """
        from scipy.optimize import OptimizeResult

        solver, obj_fn, max_steps = self._build_slsqp_jax_solver(
            x0.size, maxiter=maxiter)

        if maxiter:
            max_steps = maxiter

        sol = optx.minimise(
            obj_fn, solver, jnp.array(x0), has_aux=True,
            max_steps=max_steps, throw=False,
        )

        x_opt = np.asarray(sol.value, dtype=np.float64)
        fun_val = float(np.asarray(self._jax_raw_objective(sol.value)))
        success = str(sol.result) == 'RESULTS.successful'

        return OptimizeResult(
            x=x_opt, fun=fun_val, success=success,
            message=str(sol.result),
            nit=int(sol.stats.get('num_steps', 0)) if hasattr(sol.stats, 'get') else 0,
        )

    ###########################################################################
    # scipy-based fallback optimization helpers

    def _scipy_minimize_kwargs(self, x0, callback=False, maxiter=None):
        """
        Build ``minimizer_kwargs`` for scipy-based optimization (fallback path).

        Returns
        -------
        minimizer_kwargs : dict
        icb : BasinHoppingInnerCallBack or None
        """
        icb = BasinHoppingInnerCallBack() if callback else None

        def _np_objective(x):
            val = self._jax_raw_objective(x)
            return float(np.asarray(val))

        wrapped_constraints = []
        for const in self._jax_raw_constraints:
            _raw_fun = const['fun']

            def _np_const_fun(x, _f=_raw_fun):
                val = _f(x)
                return float(np.asarray(val))

            wrapped_constraints.append({**const, 'fun': _np_const_fun})

        minimizer_kwargs = {
            'bounds': [(0, 1)] * x0.size,
            'callback': icb,
            'constraints': wrapped_constraints,
            'options': {},
        }

        if not (self._use_autodiff and JAX_AVAILABLE):
            try:  # pragma: no cover
                import numdifftools as ndt
                minimizer_kwargs['jac'] = ndt.Jacobian(_np_objective)
                for const in minimizer_kwargs['constraints']:
                    const['jac'] = ndt.Jacobian(const['fun'])
            except ImportError:
                pass

        additions = deepcopy(self._additional_options)
        options = additions.pop('options', {})
        minimizer_kwargs['options'].update(options)
        minimizer_kwargs.update(additions)

        if maxiter:
            minimizer_kwargs['options']['maxiter'] = maxiter

        # Store the numpy-safe objective for use by scipy callers
        self.objective = _np_objective

        return minimizer_kwargs, icb

    ###########################################################################
    # Main optimization entry point

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

        x0 = x0.copy().flatten() if x0 is not None else self.construct_initial()
        x0 = np.asarray(x0, dtype=np.float64)

        logger.info("Starting JAX optimization: dim={dim}, niter={niter}", dim=x0.size, niter=niter)

        # Store raw (JAX-native) objective and constraints for both native
        # and fallback paths.
        self._jax_raw_objective = self.objective
        self._jax_raw_constraints = list(self.constraints)

        # For very large problems, slsqp-jax JIT compilation is
        # prohibitively slow. Fall back to scipy for those cases.
        _MAX_NATIVE_DIM = 200

        use_native = (SLSQP_JAX_AVAILABLE
                      and self._use_autodiff
                      and x0.size <= _MAX_NATIVE_DIM)

        if use_native:
            # ---- native slsqp-jax path (autodiff gradients) ----
            logger.info("Using slsqp-jax native optimizer with autodiff (dim={dim})", dim=x0.size)
            result = self._optimization_backend_native(x0, niter, maxiter)

            # Validate the native result: check that equality constraints
            # are approximately satisfied.  The slsqp-jax solver can
            # struggle with certain constraint formulations (e.g.
            # sum-of-squares equality constraints have near-zero Jacobians
            # at the solution).  If the result is clearly infeasible, fall
            # back to scipy SLSQP.
            if result is not None and self._jax_raw_constraints:
                max_eq_viol = 0.0
                for c in self._jax_raw_constraints:
                    if c['type'] == 'eq':
                        viol = abs(float(np.asarray(c['fun'](result.x))))
                        max_eq_viol = max(max_eq_viol, viol)
                if max_eq_viol > 0.1:
                    logger.info("slsqp-jax result has high constraint violation "
                                "({viol:.2e}); falling back to scipy SLSQP",
                                viol=max_eq_viol)
                    result = None  # force scipy fallback

            if result is None:
                minimizer_kwargs, icb = self._scipy_minimize_kwargs(
                    x0, callback=callback, maxiter=maxiter)
                self._callback = BasinHoppingCallBack(
                    minimizer_kwargs.get('constraints', {}), icb)
                result = self._optimization_backend(
                    x0, minimizer_kwargs, niter)
            else:
                # Set self.objective for _polish and later use, then do
                # a quick scipy refinement from the native result to
                # improve precision (slsqp-jax sometimes converges to
                # a slightly suboptimal point).
                minimizer_kwargs, _ = self._scipy_minimize_kwargs(
                    x0, callback=False, maxiter=maxiter)
                res_refine = minimize(
                    fun=self.objective,
                    x0=result.x.flatten(),
                    **minimizer_kwargs,
                )
                if res_refine.success and res_refine.fun < result.fun:
                    logger.debug("scipy refinement improved native result: "
                                 "{old:.6f} -> {new:.6f}",
                                 old=result.fun, new=res_refine.fun)
                    result = res_refine
        else:
            # ---- scipy fallback path (finite-difference gradients) ----
            if x0.size > _MAX_NATIVE_DIM:
                logger.info("Problem dimension {dim} exceeds native limit {lim}; "
                            "falling back to scipy SLSQP", dim=x0.size, lim=_MAX_NATIVE_DIM)
            else:
                logger.info("Using scipy SLSQP fallback (slsqp-jax not available)")
            minimizer_kwargs, icb = self._scipy_minimize_kwargs(
                x0, callback=callback, maxiter=maxiter)
            self._callback = BasinHoppingCallBack(
                minimizer_kwargs.get('constraints', {}), icb)
            result = self._optimization_backend(x0, minimizer_kwargs, niter)

        if result:
            self._optima = np.array(result.x)
        else:  # pragma: no cover
            msg = "No optima found."
            raise OptimizationException(msg)

        if polish:
            self._polish(cutoff=polish)

        obj_val = float(np.asarray(self._jax_raw_objective(self._optima)))
        logger.info("JAX optimization complete: objective={obj}", obj=obj_val)

        return result

    ###########################################################################
    # slsqp-jax native shotgun / basinhopping

    def _optimization_backend_native(self, x0, niter, maxiter):
        """
        Dispatch to the native shotgun or basinhopping backend.

        Subclasses set ``_optimization_backend`` for the scipy path;
        this method mirrors that dispatch for the slsqp-jax path.
        """
        # Delegate to the same strategy as the scipy path would use.
        # BaseConvexJaxOptimizer -> shotgun; BaseNonConvexJaxOptimizer -> basinhopping
        return self._optimize_shotgun_native(x0, niter, maxiter)

    def _optimize_shotgun_native(self, x0, niter, maxiter):
        """
        Multi-start optimization using slsqp-jax.

        Builds the solver once, JIT-compiles a single-start solve function,
        and reuses it across all starting points for efficient execution.

        Parameters
        ----------
        x0 : ndarray
            Initial optimization vector.
        niter : int or None
            Number of random restarts.
        maxiter : int or None
            Maximum SLSQP iterations per start.

        Returns
        -------
        result : OptimizeResult or None
        """
        if niter is None:
            niter = self._default_hops

        from scipy.optimize import OptimizeResult

        # Collect all starting points
        starts = []
        if x0 is not None:
            starts.append(jnp.array(x0.flatten(), dtype=jnp.float64))
            niter -= 1
        for _ in range(max(niter, 0)):
            ic = self.construct_random_initial()
            starts.append(jnp.array(ic.flatten(), dtype=jnp.float64))

        if not starts:  # pragma: no cover
            return None

        # Build solver and objective ONCE; the JIT-compiled function
        # captures these as closed-over constants so JAX can cache the
        # compilation across calls with different x0 values.
        solver, obj_fn, max_steps = self._build_slsqp_jax_solver(
            starts[0].size, maxiter=maxiter)

        if maxiter:
            max_steps = maxiter

        @jit
        def _solve_one(x0_i):
            sol = optx.minimise(
                obj_fn, solver, x0_i,
                has_aux=True, max_steps=max_steps, throw=False,
            )
            return sol.value, obj_fn(sol.value, None)[0]

        # Run each start through the JIT-compiled solver.
        # First call triggers XLA compilation; subsequent calls reuse it.
        results = []
        for i, s in enumerate(starts):
            logger.debug("Shotgun (slsqp-jax): start {i}/{n}", i=i + 1, n=len(starts))
            try:
                x_opt, fun_val = _solve_one(s)
                x_opt = np.asarray(x_opt, dtype=np.float64)
                fun_val = float(np.asarray(fun_val))
                results.append(OptimizeResult(
                    x=x_opt, fun=fun_val, success=True,
                    message='slsqp-jax sequential',
                ))
            except Exception as e:
                logger.debug("slsqp-jax start {i} failed: {e}", i=i, e=e)

        if not results:  # pragma: no cover
            return None

        raw_obj = self._jax_raw_objective
        return min(results, key=lambda r: float(np.asarray(raw_obj(r.x))))

    ###########################################################################
    # scipy-based shotgun (fallback path)

    def _optimize_shotgun(self, x0, minimizer_kwargs, niter):
        """
        Perform a non-convex optimization. This uses a "shotgun" approach,
        minimizing several random initial conditions and selecting the minimal
        result.

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
            The result of the optimization. Returns None if the optimization
            failed.
        """
        if niter is None:
            niter = self._default_hops

        results = []

        if x0 is not None:
            logger.debug("Shotgun: trying provided initial condition")
            res = minimize(fun=self.objective,
                           x0=x0.flatten(),
                           **minimizer_kwargs
                           )
            if res.success:
                results.append(res)
            niter -= 1

        ics = (self.construct_random_initial() for _ in range(niter))
        for i, initial in enumerate(ics):
            logger.debug("Shotgun: random initial condition {i}/{niter}", i=i + 1, niter=niter)
            res = minimize(fun=self.objective,
                           x0=initial.flatten(),
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
        x0[x0 > 1 - cutoff] = 1

        logger.debug("Polishing: zeroed {count} variables below cutoff={cutoff}", count=int(count), cutoff=cutoff)

        lb = np.array([1.0 if np.isclose(x, 1) else 0.0 for x in x0])
        ub = np.array([0.0 if np.isclose(x, 0) else 1.0 for x in x0])
        feasible = np.array([True for _ in x0])

        # Wrap constraints for scipy (return numpy scalars)
        raw_constraints = getattr(self, '_jax_raw_constraints', self.constraints)
        wrapped_constraints = []
        for const in raw_constraints:
            _raw_fun = const['fun']

            def _np_const_fun(x, _f=_raw_fun):
                val = _f(x)
                return float(np.asarray(val))

            wrapped_constraints.append({**const, 'fun': _np_const_fun})

        minimizer_kwargs = {
            'bounds': Bounds(lb, ub, feasible),
            'tol': None,
            'callback': None,
            'constraints': wrapped_constraints,
        }

        if not (self._use_autodiff and JAX_AVAILABLE):
            try:  # pragma: no cover
                import numdifftools as ndt
                minimizer_kwargs['jac'] = ndt.Jacobian(self.objective)
                for const in minimizer_kwargs['constraints']:
                    const['jac'] = ndt.Jacobian(const['fun'])
            except ImportError:
                pass

        if np.allclose(lb, ub):
            self._optima = x0
            return

        res = minimize(
            fun=self.objective,
            x0=x0,
            **minimizer_kwargs
        )

        if res.success:
            logger.debug("Polishing successful: objective={obj}", obj=res.fun)
            self._optima = res.x.copy()

            if count < (res.x < cutoff).sum():
                self._polish(cutoff=cutoff)


class BaseConvexJaxOptimizer(BaseJaxOptimizer):
    """
    Implement convex optimization using JAX.
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
            # even though this is convex, there might still be some optimization
            # issues, so we use niter > 1.
            niter = 2
        return self._optimize_shotgun(x0, minimizer_kwargs, niter=niter)


class BaseNonConvexJaxOptimizer(BaseJaxOptimizer):
    """
    Implement non-convex optimization using JAX.
    """

    _shotgun = False

    def _optimization_backend_native(self, x0, niter, maxiter):
        """
        Native slsqp-jax backend for non-convex optimization.

        Uses a multi-start (shotgun) strategy with ``2 * niter`` random
        starting points plus the supplied ``x0``.  The JIT-compiled SLSQP
        solver makes each start fast, so we can afford many restarts instead
        of a perturbation-based walk.  For the scipy fallback path, scipy's
        ``basinhopping`` is still used (see ``_optimization_basinhopping``).
        """
        if niter is None:
            niter = self._default_hops

        return self._optimize_shotgun_native(x0, niter, maxiter)

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

        logger.debug("Basin hopping (JAX): starting with niter={niter}", niter=niter)

        if self._shotgun:
            res_shotgun = self._optimize_shotgun(x0.copy(), minimizer_kwargs, self._shotgun)
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

        success, msg = basinhop_status(result)
        logger.info("Basin hopping (JAX) result: success={success}, message={msg}", success=success, msg=msg)
        if not success:  # pragma: no cover
            result = self._callback.minimum() or res_shotgun

        return result

    def _optimization_diffevo(self, x0, minimizer_kwargs, niter):  # pragma: no cover
        """
        Perform optimization using differential evolution.

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
            The result of the optimization. Returns None if the optimization
            failed.
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

    def _optimization_shgo(self, x0, minimizer_kwargs, niter):
        """
        Perform a non-convex optimization using scipy.optimize.shgo.

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

        result = shgo(func=self.objective,
                      bounds=minimizer_kwargs['bounds'],
                      constraints=minimizer_kwargs['constraints'],
                      iters=niter,
                      )

        if result.success:  # pragma: no cover
            return result

    _optimization_backend = _optimization_basinhopping


AuxVar = namedtuple('AuxVar', ['bases', 'bound', 'shape', 'mask', 'size'])


class BaseAuxVarJaxOptimizer(BaseNonConvexJaxOptimizer):
    """
    Base class that performs many methods related to optimizing auxiliary variables,
    using JAX for automatic differentiation.
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
            mask = jnp.ones(shape) / bound
            self._aux_vars.append(AuxVar(bases, bound, tuple(shape), mask, prod(shape)))
            self._shape += (bound,)
            self._full_shape += (bound,)
            self._all_vars |= {len(self._all_vars)}

        self._arvs = self._all_vars - (self._rvs | self._crvs)
        self._aux_bounds = [av.bound for av in self._aux_vars]
        self._optvec_size = sum(av.size for av in self._aux_vars)
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
            self._full_slices.append(tuple(colon if i in relevant_vars | {var} else jnp.newaxis for i in index))

        self._slices = []
        for i, (auxvar, var) in enumerate(zip(self._aux_vars, arvs)):
            relevant_vars = auxvar.bases
            index = sorted(self._rvs | self._crvs | set(arvs[:i + 1]))
            self._slices.append(tuple(colon if i in relevant_vars | {var} else jnp.newaxis for i in index))

    ###########################################################################
    # Constructing the joint distribution using JAX.

    def _construct_channels(self, x):
        """
        Construct the conditional distributions which produce the
        auxiliary variables.

        Parameters
        ----------
        x : jnp.ndarray
            An optimization vector

        Yields
        ------
        channel : jnp.ndarray
            A conditional distribution.
        """
        parts = [x[a:b] for a, b in self._parts]

        for part, auxvar in zip(parts, self._aux_vars):
            channel = part.reshape(auxvar.shape)
            channel = channel / channel.sum(axis=(-1,), keepdims=True)
            channel = jnp.where(jnp.isnan(channel), auxvar.mask, channel)

            yield channel

    def construct_joint(self, x):
        """
        Construct the joint distribution using JAX.

        Parameters
        ----------
        x : jnp.ndarray
            An optimization vector.

        Returns
        -------
        joint : jnp.ndarray
            The joint distribution resulting from the distribution passed
            in and the optimization vector.
        """
        x = jnp.array(x)
        joint = self._pmf

        channels = self._construct_channels(x.copy())

        for channel, slc in zip(channels, self._slices):
            joint = joint[..., jnp.newaxis] * channel[slc]

        return joint

    def _construct_joint_single(self, x):
        """
        Construct the joint distribution (optimized for single auxiliary variable).

        Parameters
        ----------
        x : jnp.ndarray
            An optimization vector.

        Returns
        -------
        joint : jnp.ndarray
            The joint distribution resulting from the distribution passed
            in and the optimization vector.
        """
        x = jnp.array(x)
        _, _, shape, mask, _ = self._aux_vars[0]
        channel = x.copy().reshape(shape)
        channel = channel / channel.sum(axis=-1, keepdims=True)
        channel = jnp.where(jnp.isnan(channel), mask, channel)

        joint = self._pmf[..., jnp.newaxis] * channel[self._slices[0]]

        return joint

    def construct_full_joint(self, x):
        """
        Construct the full joint distribution using JAX.

        Parameters
        ----------
        x : jnp.ndarray
            An optimization vector.

        Returns
        -------
        joint : jnp.ndarray
            The joint distribution resulting from the distribution passed
            in and the optimization vector.
        """
        x = jnp.array(x)
        joint = self._full_pmf

        channels = self._construct_channels(x.copy())

        for channel, slc in zip(channels, self._full_slices):
            joint = joint[..., jnp.newaxis] * channel[slc]

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
            vec = sample_simplex(av.shape[-1], prod(av.shape[:-1]))
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

    # Default to using a random initial condition:
    construct_initial = construct_random_initial

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

        joint = np.array(self.construct_full_joint(x))
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
                mapping[i] = tuple(range(n, n + len(unq.inverse[0])))

        new_map = {}
        for rv, rvs in zip(sorted(self._rvs), self._true_rvs):
            i = rv + self._n
            for a, b in zip(rvs, mapping[i]):
                new_map[a] = b

        mapping = [[(new_map[i] if i in new_map else i) for i in range(len(self._full_shape))
                                                                 if i not in self._proxy_vars]]

        d = d.coalesce(mapping, extract=True)

        if string:
            try:
                d = modify_outcomes(d, lambda o: ''.join(map(str, o)))
            except ditException:
                pass

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

        @self._maybe_jit
        def constraint(x):
            """
            Constraint which ensures that the auxiliary variables are a function
            of the random variables.

            Parameters
            ----------
            x : jnp.ndarray
                An optimization vector.

            Returns
            -------
            val : float
                The squared entropy (should be 0 for deterministic).
            """
            pmf = self.construct_joint(x)
            return entropy(pmf) ** 2

        return constraint

    ###########################################################################
    # Channel capacity methods

    def _channel_capacity(self, x):  # pragma: no cover
        """
        Compute the channel capacity of the mapping z -> z_bar.

        Parameters
        ----------
        x : array_like
            An optimization vector.

        Returns
        -------
        ccs : list of float
            The channel capacity of each auxiliary variable.
        """
        ccs = []
        for channel in self._construct_channels(jnp.array(x)):
            ccs.append(channel_capacity(np.array(channel))[0])
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
        """
        logger.debug("Post-processing (JAX): style={style}, minmax={minmax}", style=style, minmax=minmax)
        entropy = self._entropy(self._arvs)

        sign = +1 if minmax == 'min' else -1

        @self._maybe_jit
        def objective_entropy(x):
            """
            Post-process the entropy.

            Parameters
            ----------
            x : array_like
                An optimization vector.

            Returns
            -------
            ent : float
                The entropy scaled by sign (minimize or maximize).
            """
            ent = entropy(self.construct_joint(x))
            return sign * ent

        def objective_channelcapacity(x):
            """
            Post-process the channel capacity.

            Parameters
            ----------
            x : array_like
                An optimization vector.

            Returns
            -------
            cc : float
                The sum of channel capacities scaled by sign.
            """
            cc = sum(self._channel_capacity(x))
            return sign * cc

        if style == 'channel':
            objective = objective_channelcapacity
        elif style == 'entropy':
            objective = objective_entropy
        else:
            msg = "Style {} is not understood.".format(style)
            raise OptimizationException(msg)

        true_objective = self.objective(self._optima)

        @self._maybe_jit
        def constraint_match_objective(x):
            """
            Constraint to ensure that the new solution is not worse than that
            found before.

            Parameters
            ----------
            x : array_like
                An optimization vector.

            Returns
            -------
            obj : float
                The squared deviation of the current objective from the
                previously found optimum.
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


# Convenience function to check JAX availability
def is_jax_available():
    """
    Check if JAX is available for use.

    Returns
    -------
    available : bool
        True if JAX is installed and can be imported.
    """
    return JAX_AVAILABLE


