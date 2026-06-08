"""
Base class for optimization.
"""

import contextlib
import os
import threading
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from functools import reduce
from string import ascii_letters, digits
from types import MethodType

import numpy as np
from boltons.iterutils import pairwise
from loguru import logger
from scipy.optimize import Bounds, basinhopping, brute, differential_evolution, dual_annealing, minimize, shgo

from ..algorithms.channelcapacity import channel_capacity
from ..distconst import insert_rvf, modify_outcomes
from ..distribution import Distribution
from ..exceptions import OptimizationException, ditException
from ..helpers import flatten, normalize_rvs, parse_rvs
from ..math import prod, sample_simplex
from ..utils import partitions, powerset
from ..utils.optimization import (
    BasinHoppingCallBack,
    BasinHoppingInnerCallBack,
    Uniquifier,
    accept_test,
    basinhop_status,
    colon,
    make_bound_callback,
)

__all__ = (
    "BaseOptimizer",
    "BaseConvexOptimizer",
    "BaseNonConvexOptimizer",
    "BaseAuxVarOptimizer",
    "parallel_sweep",
)


svdvals = lambda m: np.linalg.svd(m, compute_uv=False)


# Thread-local flag marking that the current thread is executing inside a
# ``parallel_sweep`` worker. Used as a nesting guard so that only the outermost
# sweep fans out across threads; any inner parallelism (e.g. the shotgun
# restarts) runs serially within the worker to avoid CPU oversubscription.
_sweep_state = threading.local()


def _in_sweep():
    """Whether the current thread is already inside a ``parallel_sweep`` worker."""
    return getattr(_sweep_state, "active", False)


def _resolve_n_jobs(n_tasks):
    """
    Determine how many worker threads to use for an embarrassingly-parallel
    batch of *n_tasks* independent local minimizations.

    Controlled by the ``DIT_OPT_JOBS`` environment variable:

    * unset or ``0`` → automatic (``min(n_tasks, os.cpu_count())``)
    * ``1``          → serial (disables parallelism)
    * ``k > 1``      → use exactly ``k`` workers (capped at ``n_tasks``)

    Always returns ``1`` when called from within a ``parallel_sweep`` worker, so
    nested parallelism collapses to serial execution.

    Returns
    -------
    n_jobs : int
        The number of worker threads (``1`` means run serially).
    """
    if n_tasks <= 1:
        return 1
    if _in_sweep():
        return 1
    raw = os.environ.get("DIT_OPT_JOBS", "0")
    try:
        requested = int(raw)
    except ValueError:  # pragma: no cover
        requested = 0
    if requested == 1:
        return 1
    if requested > 1:
        return min(requested, n_tasks)
    return min(n_tasks, os.cpu_count() or 1)


def parallel_sweep(func, items, *, base_seed=None):
    """
    Run ``func(item, rng)`` for each item in *items*, in parallel when safe.

    Each task gets its own ``numpy.random.Generator`` seeded deterministically
    (from *base_seed*, or a single draw from the global ``np.random`` state when
    *base_seed* is None) so that workers never race the global RNG and results
    are reproducible with respect to a seeded ``np.random``. The serial and
    parallel paths consume the *same* per-task generators, so a sweep produces
    identical results regardless of ``DIT_OPT_JOBS``.

    Results are returned in input order, so ``max`` / ``min`` / ``append`` over
    them stay deterministic.

    Parameters
    ----------
    func : callable
        Called as ``func(item, rng)`` where ``rng`` is a ``Generator``.
    items : iterable
        The independent work items.
    base_seed : int, None
        Seed offset for the per-task generators. If None, a seed is drawn once
        (serially) from the global ``np.random`` state.

    Returns
    -------
    results : list
        ``[func(item, rng) for item in items]`` in input order.
    """
    items = list(items)
    n = len(items)
    if n == 0:
        return []

    if base_seed is None:
        base_seed = int(np.random.randint(0, 2**31 - 1))
    rngs = [np.random.default_rng(base_seed + i) for i in range(n)]

    n_jobs = _resolve_n_jobs(n)

    if n_jobs <= 1:
        return [func(item, rng) for item, rng in zip(items, rngs)]

    def _run(args):
        item, rng = args
        _sweep_state.active = True
        try:
            return func(item, rng)
        finally:
            _sweep_state.active = False

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        return list(executor.map(_run, zip(items, rngs)))


class BaseOptimizer(metaclass=ABCMeta):
    """
    Base class for performing optimizations.
    """

    def __init__(self, dist, rvs=None, crvs=None):
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
        """
        rvs, crvs = normalize_rvs(dist, rvs, crvs)
        self._dist = dist.copy(base="linear")

        self._alphabet = self._dist.alphabet
        self._original_shape = list(map(len, self._dist.alphabet))

        self._true_rvs = [parse_rvs(self._dist, rv)[1] for rv in rvs]
        self._true_crvs = parse_rvs(self._dist, crvs)[1]
        self._dist = modify_outcomes(self._dist, tuple)

        # compress all random variables down to single vars
        self._unqs = []
        for var in self._true_rvs + [self._true_crvs]:
            unq = Uniquifier()
            self._dist = insert_rvf(self._dist, lambda x, unq=unq, var=var: (unq(tuple(x[i] for i in var)),))
            self._unqs.append(unq)

        self._dist.make_dense()

        self._full_shape = list(map(len, self._dist.alphabet))
        self._full_pmf = self._dist.pmf.reshape(self._full_shape)

        self._n = dist.outcome_length()
        self._pmf = self._full_pmf.sum(axis=tuple(range(self._n)))
        self._shape = self._pmf.shape

        self._full_vars = set(range(len(self._full_shape)))
        self._all_vars = set(range(len(rvs) + 1))
        self._rvs = set(range(len(rvs)))
        self._crvs = {len(rvs)}

        self._proxy_vars = tuple(range(self._n, self._n + len(rvs) + 1))

        self._additional_options = {}
        if not hasattr(self, "_objective_bound"):
            self._objective_bound = None

        # Optional per-call random generator for RNG isolation in parallel
        # outer sweeps. None means use the global ``np.random`` state.
        self._rng = None

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
        vec = sample_simplex(self._optvec_size, rng=self._rng)
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
        return -np.nansum(p * np.log2(p))

    def _entropy(self, rvs, crvs=None):
        """
        Compute the conditional entropy, :math`H[X|Y]`.

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
        idx_joint = tuple(self._all_vars - (rvs | crvs))
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
        Compute the mutual information, :math:`I[X:Y]`.

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

            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = pmf_xy / (pmf_x * pmf_y)
                log_ratio = np.where(pmf_xy > 0, np.log2(np.maximum(ratio, 1e-300)), 0.0)
            mi = np.nansum(pmf_xy * log_ratio)

            return mi

        return mutual_information

    def _conditional_mutual_information(self, rv_x, rv_y, rv_z):
        """
        Compute the conditional mutual information, :math:`I[X:Y|Z]`.

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
        if not rv_z:
            return self._mutual_information(rv_x=rv_x, rv_y=rv_y)

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

            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = pmf_z * pmf_xyz / pmf_xz / pmf_yz
                log_ratio = np.where(pmf_xyz > 0, np.log2(np.maximum(ratio, 1e-300)), 0.0)
            cmi = np.nansum(pmf_xyz * log_ratio)

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
        power = [(-1) ** len(ss) for ss in sorted(powerset(rvs), key=len)[1:-1]]
        power += [(-1) ** len(rvs)]
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

            with np.errstate(divide="ignore", invalid="ignore"):
                pmf_ci = reduce(np.multiply, [pmf**p for pmf, p in zip(pmf_subrvs, power, strict=True)])
                log_ci = np.where(pmf_joint > 0, np.log2(np.maximum(pmf_ci, 1e-300)), 0.0)
            ci = np.nansum(pmf_joint * log_ci)

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

            tc = h_margs - h_joint - n * h_crvs

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

            dtc = sum(h_margs) - n * h_joint

            return dtc

        return dual_total_correlation

    ###########################################################################
    # Analytic gradients of the objective building blocks.
    #
    # Every measure above is a signed sum of marginal entropies
    # ``sum_k s_k H(M_k(pmf))``. The exact pmf-gradient of one marginal entropy
    # ``H(M(pmf))`` (where ``M`` sums ``pmf`` over the axes ``idx``) is
    #
    #     d H(M(pmf)) / d pmf_i = -(log2(M(pmf))[cell(i)] + 1/ln2),
    #
    # i.e. ``-(log2(marginal) + 1/ln2)`` broadcast back to the full shape (each
    # full cell takes the value of its marginal cell). So each measure's
    # gradient builder mirrors the corresponding factory's ``idx_*``/sign setup
    # and returns the matching signed sum. These are exact (validated against
    # SciPy finite differences); the ``1/ln2`` constants are real.

    _objective_gradient = None

    def _marginal_entropy_grad(self, pmf, idx):
        """
        Full-shape gradient of ``H(M(pmf))`` w.r.t. ``pmf``, where ``M`` sums
        ``pmf`` over the axes ``idx``.

        Parameters
        ----------
        pmf : np.ndarray
            The joint probability distribution.
        idx : tuple
            The axes summed out to form the marginal.

        Returns
        -------
        grad : np.ndarray
            ``-(log2(marginal) + 1/ln2)``, broadcastable against ``pmf``.
        """
        marginal = pmf.sum(axis=idx, keepdims=True)
        with np.errstate(divide="ignore"):
            return -(np.log2(np.maximum(marginal, 1e-300)) + 1.0 / np.log(2))

    @staticmethod
    def _full_grad(g, pmf):
        """Broadcast a (possibly reduced) gradient to a contiguous full-shape array."""
        return np.ascontiguousarray(np.broadcast_to(g, pmf.shape))

    def _entropy_grad(self, rvs, crvs=None):
        """Gradient builder for :meth:`_entropy`."""
        if crvs is None:
            crvs = set()
        idx_joint = tuple(self._all_vars - (rvs | crvs))
        idx_crvs = tuple(self._all_vars - crvs)

        def grad(pmf):
            g = self._marginal_entropy_grad(pmf, idx_joint) - self._marginal_entropy_grad(pmf, idx_crvs)
            return self._full_grad(g, pmf)

        return grad

    def _mutual_information_grad(self, rv_x, rv_y):
        """Gradient builder for :meth:`_mutual_information`."""
        idx_xy = tuple(self._all_vars - (rv_x | rv_y))
        idx_x = tuple(self._all_vars - rv_x)
        idx_y = tuple(self._all_vars - rv_y)

        def grad(pmf):
            g = (
                self._marginal_entropy_grad(pmf, idx_x)
                + self._marginal_entropy_grad(pmf, idx_y)
                - self._marginal_entropy_grad(pmf, idx_xy)
            )
            return self._full_grad(g, pmf)

        return grad

    def _conditional_mutual_information_grad(self, rv_x, rv_y, rv_z):
        """Gradient builder for :meth:`_conditional_mutual_information`."""
        if not rv_z:
            return self._mutual_information_grad(rv_x=rv_x, rv_y=rv_y)

        idx_xyz = tuple(self._all_vars - (rv_x | rv_y | rv_z))
        idx_xz = tuple(self._all_vars - (rv_x | rv_z))
        idx_yz = tuple(self._all_vars - (rv_y | rv_z))
        idx_z = tuple(self._all_vars - rv_z)

        def grad(pmf):
            g = (
                self._marginal_entropy_grad(pmf, idx_xz)
                + self._marginal_entropy_grad(pmf, idx_yz)
                - self._marginal_entropy_grad(pmf, idx_xyz)
                - self._marginal_entropy_grad(pmf, idx_z)
            )
            return self._full_grad(g, pmf)

        return grad

    def _coinformation_grad(self, rvs, crvs=None):
        """Gradient builder for :meth:`_coinformation`."""
        if crvs is None:
            crvs = set()
        idx_joint = tuple(self._all_vars - (rvs | crvs))
        idx_crvs = tuple(self._all_vars - crvs)
        idx_subrvs = [tuple(self._all_vars - set(ss)) for ss in sorted(powerset(rvs), key=len)[1:-1]]
        power = [(-1) ** len(ss) for ss in sorted(powerset(rvs), key=len)[1:-1]]
        power += [(-1) ** len(rvs)]
        power += [-sum(power)]
        idxs = idx_subrvs + [idx_joint, idx_crvs]

        def grad(pmf):
            # coinformation == -sum_k power_k H(marginal_k); differentiate term by term.
            g = -sum(p * self._marginal_entropy_grad(pmf, idx) for p, idx in zip(power, idxs, strict=True))
            return self._full_grad(g, pmf)

        return grad

    def _total_correlation_grad(self, rvs, crvs=None):
        """Gradient builder for :meth:`_total_correlation`."""
        if crvs is None:
            crvs = set()
        idx_joint = tuple(self._all_vars - (rvs | crvs))
        idx_margs = [tuple(self._all_vars - ({rv} | crvs)) for rv in rvs]
        idx_crvs = tuple(self._all_vars - crvs)
        n = len(rvs) - 1

        def grad(pmf):
            g = sum(self._marginal_entropy_grad(pmf, marg) for marg in idx_margs)
            g = g - self._marginal_entropy_grad(pmf, idx_joint) - n * self._marginal_entropy_grad(pmf, idx_crvs)
            return self._full_grad(g, pmf)

        return grad

    def _dual_total_correlation_grad(self, rvs, crvs=None):
        """Gradient builder for :meth:`_dual_total_correlation`."""
        if crvs is None:
            crvs = set()
        idx_joint = tuple(self._all_vars - (rvs | crvs))
        idx_margs = [tuple(self._all_vars - ((rvs - {rv}) | crvs)) for rv in rvs]
        idx_crvs = tuple(self._all_vars - crvs)
        n = len(rvs) - 1

        def grad(pmf):
            # dtc == sum_i H(marg_i) - n H(joint) - H(crvs).
            g = sum(self._marginal_entropy_grad(pmf, marg) for marg in idx_margs)
            g = g - n * self._marginal_entropy_grad(pmf, idx_joint) - self._marginal_entropy_grad(pmf, idx_crvs)
            return self._full_grad(g, pmf)

        return grad

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
                    idx_parts[p] = tuple(self._all_vars - (p | crvs))
        part_norms = [len(part) - 1 for part in parts]
        idx_joint = tuple(self._all_vars - (rvs | crvs))
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

            pairs = zip(parts, part_norms, strict=True)
            candidates = [(sum(self._h(pmf_parts[p]) - h_crvs for p in part) - h_joint) / norm for part, norm in pairs]

            caekl = min(candidates)

            return caekl

        return caekl_mutual_information

    def _maximum_correlation(self, rv_x, rv_y):
        """
        Compute the maximum correlation.

        Parameters
        ----------
        rv_x : collection
            The index to consider as the X variable.
        rv_y : collection
            The index to consider as the Y variable.

        Returns
        -------
        mc : func
            The maximum correlation.
        """
        idx_xy = tuple(self._all_vars - (rv_x | rv_y))
        idx_x = tuple(self._all_vars - rv_x)
        idx_y = tuple(self._all_vars - rv_y)

        def maximum_correlation(pmf):
            """
            Compute the specified maximum correlation.

            Parameters
            ----------
            pmf : np.ndarray
                The joint probability distribution.

            Returns
            -------
            mi : float
                The mutual information.
            """
            pmf_xy = pmf.sum(axis=idx_xy)
            pmf_x = pmf.sum(axis=idx_x)[:, np.newaxis]
            pmf_y = pmf.sum(axis=idx_y)[np.newaxis, :]

            Q = pmf_xy / (np.sqrt(pmf_x) * np.sqrt(pmf_y))
            Q[np.isnan(Q)] = 0

            mc = svdvals(Q)[1]

            return mc

        return maximum_correlation

    def _conditional_maximum_correlation(self, rv_x, rv_y, rv_z):
        """
        Compute the conditional maximum correlation.

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
            The conditional maximum correlation.
        """
        idx_xyz = tuple(self._all_vars - (rv_x | rv_y | rv_z))
        idx_xz = tuple(self._all_vars - (rv_x | rv_z))
        idx_yz = tuple(self._all_vars - (rv_y | rv_z))

        def conditional_maximum_correlation(pmf):
            """
            Compute the specified maximum correlation.

            Parameters
            ----------
            pmf : np.ndarray
                The joint probability distribution.

            Returns
            -------
            mi : float
                The mutual information.
            """
            p_xyz = pmf.sum(axis=idx_xyz)
            p_xz = pmf.sum(axis=idx_xz)[:, np.newaxis, :]
            p_yz = pmf.sum(axis=idx_yz)[np.newaxis, :, :]

            Q = np.where(p_xyz, p_xyz / (np.sqrt(p_xz * p_yz)), 0)

            cmc = max(svdvals(np.squeeze(m))[1] for m in np.dsplit(Q, Q.shape[2]))

            return cmc

        return conditional_maximum_correlation

    def _total_variation(self, rv_x, rv_y):
        """
        Compute the total variation, :math:`TV[X||Y]`.

        Parameters
        ----------
        rv_x : collection
            The indices to consider as the X variable.
        rv_y : collection
            The indices to consider as the Y variable.

        Returns
        -------
        tv : func
            The total variation.

        Note
        ----
        The pmfs are assumed to be over the same alphabet.
        """
        idx_xy = tuple(self._all_vars - (rv_x | rv_y))
        idx_x = tuple(self._all_vars - rv_x)
        idx_y = tuple(self._all_vars - rv_y)

        def total_variation(pmf):
            """
            Compute the specified total variation.

            Parameters
            ----------
            pmf : np.ndarray
                The joint probability distribution.

            Returns
            -------
            tv : float
                The total variation.
            """
            pmf_xy = pmf.sum(axis=idx_xy, keepdims=True)
            pmf_x = pmf_xy.sum(axis=idx_x)
            pmf_y = pmf_xy.sum(axis=idx_y)

            tv = abs(pmf_x - pmf_y).sum() / 2

            return tv

        return total_variation

    ###########################################################################
    # Optimization methods.

    def _apply_analytic_jacobians(self, minimizer_kwargs):
        """
        Wire an analytic objective gradient into *minimizer_kwargs* when the
        subclass provides one.

        A subclass opts in to exact gradients by defining an
        ``_objective_gradient()`` factory (returning ``grad(pmf)``); the
        matching ``_jacobian(x)`` defined on the optimizer base then composes it
        with the parametrization Jacobian. Analytic *constraint* jacobians are
        supplied directly by the subclass via a ``'jac'`` entry in each
        constraint dict, which SciPy picks up automatically. Anything not
        provided falls back to SciPy's internal finite differences, so this is
        always safe.
        """
        if getattr(self, "_objective_gradient", None) is not None:
            jac = getattr(self, "_jacobian", None)
            if callable(jac):
                minimizer_kwargs["jac"] = jac

    def optimize(self, x0=None, niter=None, maxiter=None, polish=1e-6, callback=False, rng=None):
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
        rng : np.random.Generator, None
            An optional random number generator used for all randomness in this
            optimization (random initial conditions and the basin-hopping step).
            When None (default), the global ``np.random`` state is used,
            preserving backwards-compatible behavior. Supplying a per-task
            generator isolates parallel outer sweeps from one another.

        Returns
        -------
        result : OptimizeResult
            The result of the optimization.
        """
        self._rng = rng

        try:
            callable(self.objective)
        except AttributeError:
            self.objective = MethodType(self._objective(), self)

        x0 = x0.copy().flatten() if x0 is not None else self.construct_initial()

        logger.info("Starting optimization: dim={dim}, niter={niter}", dim=x0.size, niter=niter)

        icb = BasinHoppingInnerCallBack() if callback else None

        minimizer_kwargs = {
            "bounds": [(0, 1)] * x0.size,
            "callback": icb,
            "constraints": self.constraints,
            "options": {},
        }

        self._apply_analytic_jacobians(minimizer_kwargs)

        additions = deepcopy(self._additional_options)
        options = additions.pop("options", {})
        minimizer_kwargs["options"].update(options)
        minimizer_kwargs.update(additions)

        if maxiter:
            minimizer_kwargs["options"]["maxiter"] = maxiter

        self._callback = BasinHoppingCallBack(
            minimizer_kwargs.get("constraints", {}),
            icb,
            objective_bound=self._objective_bound,
        )

        result = self._optimization_backend(x0, minimizer_kwargs, niter)

        if result:
            self._optima = result.x
        else:  # pragma: no cover
            msg = "No optima found."
            raise OptimizationException(msg)

        if polish:
            self._polish(cutoff=polish)

        logger.info("Optimization complete: objective={obj}", obj=self.objective(self._optima))

        return result

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

        TODO
        ----
         * Rather than random initial conditions, use latin hypercube sampling.
        """
        if niter is None:
            niter = self._default_hops

        bound = self._objective_bound
        atol = 1e-8

        results = []

        if x0 is not None:
            logger.debug("Shotgun: trying provided initial condition")
            res = minimize(fun=self.objective, x0=x0.flatten(), **minimizer_kwargs)
            if res.success:
                results.append(res)
                if bound is not None and res.fun <= bound + atol:
                    logger.debug("Shotgun early stop: objective {f} reached bound {b}", f=res.fun, b=bound)
                    return res
            niter -= 1

        # Generate all random initial conditions up front (serially) so the
        # global RNG state is consumed deterministically regardless of whether
        # the subsequent local minimizations run serially or in parallel.
        ics = [self.construct_random_initial() for _ in range(max(niter, 0))]

        # When a lower bound on the objective is known we keep the serial path
        # so we can stop hopping the instant the bound is reached. Otherwise the
        # starts are independent and run concurrently; numpy releases the GIL
        # for the heavy array work inside each ``minimize`` call.
        n_jobs = 1 if bound is not None else _resolve_n_jobs(len(ics))

        if n_jobs > 1:
            logger.debug("Shotgun: minimizing {n} starts across {j} threads", n=len(ics), j=n_jobs)

            def _run(initial):
                return minimize(fun=self.objective, x0=initial.flatten(), **minimizer_kwargs)

            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                for res in executor.map(_run, ics):
                    if res.success:
                        results.append(res)
        else:
            for i, initial in enumerate(ics):
                logger.debug("Shotgun: random initial condition {i}/{niter}", i=i + 1, niter=len(ics))
                res = minimize(fun=self.objective, x0=initial.flatten(), **minimizer_kwargs)
                if res.success:
                    results.append(res)
                    if bound is not None and res.fun <= bound + atol:
                        logger.debug("Shotgun early stop: objective {f} reached bound {b}", f=res.fun, b=bound)
                        return res

        try:
            result = min(results, key=lambda r: r.fun)
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

        minimizer_kwargs = {
            "bounds": Bounds(lb, ub, feasible),
            "tol": None,
            "callback": None,
            "constraints": self.constraints,
        }

        self._apply_analytic_jacobians(minimizer_kwargs)

        if np.allclose(lb, ub):
            self._optima = x0
            return

        res = minimize(fun=self.objective, x0=x0, **minimizer_kwargs)

        if res.success:
            logger.debug("Polishing successful: objective={obj}", obj=res.fun)
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
            # even though this is convex, there might still be some optimization
            # issues, so we use niter > 1.
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

        logger.debug("Basin hopping: starting with niter={niter}", niter=niter)

        if self._shotgun:
            res_shotgun = self._optimize_shotgun(x0.copy(), minimizer_kwargs, self._shotgun)
            if res_shotgun:
                x0 = res_shotgun.x.copy()
        else:
            res_shotgun = None

        bh_kwargs = {}
        if self._rng is not None:
            bh_kwargs["rng"] = self._rng

        result = basinhopping(
            func=self.objective,
            x0=x0,
            minimizer_kwargs=minimizer_kwargs,
            niter=niter,
            accept_test=accept_test,
            callback=self._callback,
            **bh_kwargs,
        )

        success, msg = basinhop_status(result)
        logger.info("Basin hopping result: success={success}, message={msg}", success=success, msg=msg)
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
            The result of the optimization. Returns None if the optimization
            failed.
        """
        if "constraints" in minimizer_kwargs:
            msg = "Differential Evolution can only be used in unconstrained optimization."
            raise OptimizationException(msg)

        if niter is None:
            niter = self._default_hops

        bound = self._objective_bound
        atol = 1e-8

        def callback(xk, convergence=0):
            f = self.objective(xk)
            if bound is not None and f <= bound + atol:
                logger.debug("DiffEvo early stop: objective {f} reached bound {b}", f=f, b=bound)
                return True
            return False

        de_kwargs = {}
        if self._rng is not None:
            de_kwargs["rng"] = self._rng

        result = differential_evolution(
            func=self.objective,
            bounds=minimizer_kwargs["bounds"],
            maxiter=minimizer_kwargs["options"]["maxiter"],
            popsize=niter,
            tol=minimizer_kwargs["options"]["ftol"],
            callback=callback,
            **de_kwargs,
        )

        if result.success:
            return result

    def _optimization_shgo(self, x0, minimizer_kwargs, niter):
        """
        Perform a non-convex optimization. This uses the relatively new
        scipy.optimize.shgo.

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

        result = shgo(
            func=self.objective,
            bounds=minimizer_kwargs["bounds"],
            constraints=minimizer_kwargs["constraints"],
            iters=niter,
        )

        if result.success:  # pragma: no cover
            return result

    def _optimization_dual_annealing(self, x0, minimizer_kwargs, niter):
        """
        Perform a non-convex optimization using scipy.optimize.dual_annealing.

        Parameters
        ----------
        x0 : ndarray, None
            Initial optimization vector. If None, use a random vector.
        minimizer_kwargs : dict
            A dictionary of keyword arguments to pass to the local minimizer.
        niter : int
            The maximum number of global iterations.

        Returns
        -------
        result : OptimizeResult, None
            The result of the optimization. Returns None if the optimization failed.
        """
        if niter is None:
            niter = self._default_hops

        callback = None
        if self._objective_bound is not None:
            callback = make_bound_callback(self._objective_bound)

        da_kwargs = {}
        if self._rng is not None:
            da_kwargs["rng"] = self._rng

        result = dual_annealing(
            func=self.objective,
            bounds=minimizer_kwargs["bounds"],
            minimizer_kwargs=minimizer_kwargs,
            maxiter=niter,
            x0=x0,
            callback=callback,
            **da_kwargs,
        )

        if result.success:
            return result

    def _optimization_brute(self, x0, minimizer_kwargs, niter):
        """
        Perform a non-convex optimization using scipy.optimize.brute,
        a brute-force grid search over the parameter space.

        Parameters
        ----------
        x0 : ndarray, None
            Initial optimization vector (unused; kept for interface consistency).
        minimizer_kwargs : dict
            A dictionary of keyword arguments to pass to the optimizer.
        niter : int
            The number of grid points per dimension.

        Returns
        -------
        result : OptimizeResult, None
            The result of the optimization. Returns None if the optimization failed.
        """
        from scipy.optimize import OptimizeResult

        if niter is None:
            niter = self._default_hops

        ranges = [slice(0, 1, 1 / niter) for _ in minimizer_kwargs["bounds"]]

        x0_result = brute(
            func=self.objective,
            ranges=ranges,
            finish=minimize,
        )

        return OptimizeResult(
            {
                "x": x0_result,
                "fun": self.objective(x0_result),
                "success": True,
            }
        )

    _optimization_backend = _optimization_basinhopping


AuxVar = namedtuple("AuxVar", ["bases", "bound", "shape", "mask", "size"])


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
        for i, (auxvar, var) in enumerate(zip(self._aux_vars, arvs, strict=True)):
            relevant_vars = {self._n + b for b in auxvar.bases}
            index = sorted(self._full_vars) + [self._n + a for a in arvs[: i + 1]]
            var += self._n
            self._full_slices.append(tuple(colon if i in relevant_vars | {var} else np.newaxis for i in index))

        self._slices = []
        for i, (auxvar, var) in enumerate(zip(self._aux_vars, arvs, strict=True)):
            relevant_vars = auxvar.bases
            index = sorted(self._rvs | self._crvs | set(arvs[: i + 1]))
            self._slices.append(tuple(colon if i in relevant_vars | {var} else np.newaxis for i in index))

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

        for part, auxvar in zip(parts, self._aux_vars, strict=True):
            channel = part.reshape(auxvar.shape)
            channel /= channel.sum(axis=(-1,), keepdims=True)
            channel = np.where(np.isnan(channel), auxvar.mask, channel)
            # channel[np.isnan(channel)] = auxvar.mask[np.isnan(channel)]

            yield channel

    # Maximum number of distinct optimization vectors to memoize the joint for.
    # During a single SLSQP iteration scipy evaluates the objective and every
    # constraint (plus their finite-difference perturbations) at the *same*
    # points; memoizing lets the objective and constraints share one build of
    # the joint instead of recomputing it. Keyed on the exact bytes of ``x`` so
    # the cache can only ever return an identical result (correctness is never
    # affected; only the hit rate depends on caller behaviour).
    _joint_cache_size = 64

    def _joint_cache_dict(self):
        """
        Return this thread's joint cache.

        The cache is thread-local so that the parallel shotgun
        (:meth:`_optimize_shotgun`) — where independent minimizations run on
        separate threads sharing one optimizer instance — never mutates a dict
        concurrently. Sharing only needs to happen *within* a single
        minimization (objective and constraints at the same point), which all
        executes on one thread.
        """
        store = self.__dict__.get("_joint_cache")
        if store is None:
            store = self._joint_cache = threading.local()
        cache = getattr(store, "cache", None)
        if cache is None:
            cache = store.cache = {}
        return cache

    def _joint_cache_lookup(self, x):
        """
        Return ``(key, cached_joint_or_None)`` for optimization vector *x*.

        The returned ``key`` should be passed to :meth:`_joint_cache_store`
        once the joint has been built (on a cache miss).
        """
        cache = self._joint_cache_dict()
        key = x.tobytes()
        return key, cache.get(key)

    def _joint_cache_store(self, key, joint):
        """Store *joint* under *key*, evicting the oldest entry when full."""
        cache = self._joint_cache_dict()
        if len(cache) >= self._joint_cache_size:
            # FIFO eviction: drop the oldest inserted key.
            cache.pop(next(iter(cache)))
        cache[key] = joint

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
        key, joint = self._joint_cache_lookup(x)
        if joint is not None:
            return joint

        joint = self._pmf

        channels = self._construct_channels(x.copy())

        for channel, slc in zip(channels, self._slices, strict=True):
            joint = joint[..., np.newaxis] * channel[slc]

        self._joint_cache_store(key, joint)
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
        key, joint = self._joint_cache_lookup(x)
        if joint is not None:
            return joint

        _, _, shape, mask, _ = self._aux_vars[0]
        channel = x.copy().reshape(shape)
        channel /= channel.sum(axis=-1, keepdims=True)
        # channel[np.isnan(channel)] = mask[np.isnan(channel)]
        channel = np.where(np.isnan(channel), mask, channel)

        joint = self._pmf[..., np.newaxis] * channel[self._slices[0]]

        self._joint_cache_store(key, joint)
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

        channels = self._construct_channels(x.copy())

        for channel, slc in zip(channels, self._full_slices, strict=True):
            joint = joint[..., np.newaxis] * channel[slc]

        return joint

    ###########################################################################
    # Analytic objective gradient (reverse-mode through construct_joint).

    def _construct_joint_vjp(self, x, g):
        """
        Vector-Jacobian product through :meth:`construct_joint`.

        Given ``g = d(objective)/d(joint)`` (same shape as the joint), return
        ``d(objective)/dx`` by back-propagating through the broadcast channel
        products and the per-channel normalization. This is exact reverse-mode
        differentiation of the forward construction (validated against finite
        differences).

        Parameters
        ----------
        x : np.ndarray
            An optimization vector.
        g : np.ndarray
            The gradient of the objective w.r.t. the joint, same shape as the
            output of :meth:`construct_joint`.

        Returns
        -------
        grad : np.ndarray
            The objective gradient w.r.t. ``x``, of length ``len(x)``.
        """
        x = x.copy()
        channels = list(self._construct_channels(x))

        # Forward pass, recording the running joint before each channel multiply.
        prev_joints = []
        joint = self._pmf
        for channel, slc in zip(channels, self._slices, strict=True):
            prev_joints.append(joint)
            joint = joint[..., np.newaxis] * channel[slc]

        # Reverse pass.
        grads = [None] * len(channels)
        for k in reversed(range(len(channels))):
            channel = channels[k]
            slc = self._slices[k]
            channel_b = channel[slc]

            # d/d(channel[slc]); sum the broadcast (newaxis) axes back to channel shape.
            g_channel_b = g * prev_joints[k][..., np.newaxis]
            newaxis_axes = tuple(ax for ax, s in enumerate(slc) if s is np.newaxis)
            g_channel = g_channel_b.sum(axis=newaxis_axes).reshape(channel.shape)

            # propagate to the previous joint (sum out the newest auxiliary axis).
            g = (g * channel_b).sum(axis=-1)

            # back-prop the channel through normalization + nan-mask to its slice of x.
            a, b = self._parts[k]
            grads[k] = self._channel_vjp(x[a:b], channel, g_channel)

        return np.concatenate(grads) if grads else np.zeros_like(x)

    @staticmethod
    def _channel_vjp(part, channel, g_channel):
        """
        Back-propagate a channel gradient through ``c = normalize(reshape(part))``.

        For an unnormalized row ``r`` with sum ``s`` and normalized row
        ``c = r/s``, ``d c_j / d r_i = (delta_ij - c_j)/s``, so
        ``dL/dr_i = (g_i - sum_j g_j c_j)/s``. Rows whose sum is zero were
        replaced by the constant uniform mask in the forward pass and so receive
        zero gradient.

        Parameters
        ----------
        part : np.ndarray
            The slice of ``x`` parametrizing this channel.
        channel : np.ndarray
            The normalized channel (post nan-mask).
        g_channel : np.ndarray
            The gradient w.r.t. the channel, same shape as ``channel``.

        Returns
        -------
        grad : np.ndarray
            The gradient w.r.t. ``part``, of length ``len(part)``.
        """
        raw = part.reshape(channel.shape)
        s = raw.sum(axis=-1, keepdims=True)
        inner = (g_channel * channel).sum(axis=-1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            g_raw = np.where(s > 0, (g_channel - inner) / np.where(s > 0, s, 1.0), 0.0)
        return g_raw.ravel()

    def _jacobian(self, x):
        """
        Exact objective gradient w.r.t. ``x`` for auxiliary-variable optimizers.

        Composes the measure pmf-gradient (:meth:`_objective_gradient`) with the
        :meth:`_construct_joint_vjp` back-propagation through the channel
        parametrization. Wired into SciPy only when the subclass defines
        ``_objective_gradient``; otherwise finite differences are used.

        Parameters
        ----------
        x : np.ndarray
            An optimization vector.

        Returns
        -------
        jac : np.ndarray
            The objective gradient, of length ``len(x)``.
        """
        g = self._objective_gradient()(self.construct_joint(x))
        return self._construct_joint_vjp(x, np.asarray(g, dtype=float))

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
            vec = sample_simplex(av.shape[-1], prod(av.shape[:-1]), rng=self._rng)
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
            "".join(flatten(alphabets))
            for bound in self._aux_bounds:
                alphabets += [(digits + ascii_letters)[:bound]]
            string = True
        except TypeError:
            for bound in self._aux_bounds:
                alphabets += [list(range(bound))]
            string = False

        joint = self.construct_full_joint(x)
        outcomes, pmf = zip(*[(o, p) for o, p in np.ndenumerate(joint) if p > cutoff], strict=True)

        # normalize, in case cutoffs removed a significant amount of pmf
        pmf = np.asarray(pmf)
        pmf /= pmf.sum()

        d = Distribution(outcomes, pmf)

        mapping = {}
        for i, unq in zip(sorted(self._n + i for i in self._rvs | self._crvs), self._unqs, strict=True):
            if len(unq.inverse) > 1:
                n = d.outcome_length()
                d = insert_rvf(d, lambda o, unq=unq, i=i: unq.inverse[o[i]])
                mapping[i] = tuple(range(n, n + len(unq.inverse[0])))

        new_map = {}
        for rv, rvs in zip(sorted(self._rvs), self._true_rvs, strict=True):
            i = rv + self._n
            for a, b in zip(rvs, mapping[i], strict=True):
                new_map[a] = b

        mapping = [[new_map.get(i, i) for i in range(len(self._full_shape)) if i not in self._proxy_vars]]

        d = d.coalesce(mapping, extract=True)

        if string:
            with contextlib.suppress(ditException):
                d = modify_outcomes(d, lambda o: "".join(map(str, o)))

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
    # TODO: make these works

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

    def _post_process(self, style="entropy", minmax="min", niter=10, maxiter=None):  # pragma: no cover
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
        logger.debug("Post-processing: style={style}, minmax={minmax}", style=style, minmax=minmax)
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

        sign = +1 if minmax == "min" else -1

        # The post-process optimizes a *different* objective than the main one,
        # so the main ``_objective_gradient`` (if any) does not apply. Install
        # the gradient that matches the active post-process objective: the exact
        # entropy gradient for the entropy style, or finite differences (None)
        # for the channel-capacity style.
        if style == "channel":
            objective = objective_channelcapacity
            pp_objective_gradient = None
        elif style == "entropy":
            objective = objective_entropy
            entropy_grad = self._entropy_grad(self._arvs)

            def pp_objective_gradient():
                def grad(pmf):
                    return sign * entropy_grad(pmf)

                return grad
        else:
            msg = f"Style {style} is not understood."
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
            obj = (self.objective(x) - true_objective) ** 2
            return obj

        constraint = [
            {
                "type": "eq",
                "fun": constraint_match_objective,
            }
        ]

        # set up required attributes
        try:
            self.constraints += constraint
        except AttributeError:
            self.constraints = constraint

        self.__old_objective, self.objective = self.objective, objective

        # Temporarily swap in the post-process gradient (instance attribute
        # shadows the class-level ``_objective_gradient``), restoring afterward.
        _missing = object()
        saved_grad = self.__dict__.get("_objective_gradient", _missing)
        self._objective_gradient = pp_objective_gradient

        try:
            self.optimize(x0=self._optima.copy(), niter=niter, maxiter=maxiter)
        finally:
            if saved_grad is _missing:
                del self._objective_gradient
            else:
                self._objective_gradient = saved_grad

        # and remove them again.
        self.constraints = self.constraints[:-1]
        if not self.constraints:
            del self.constraints

        self.objective = self.__old_objective
        del self.__old_objective
