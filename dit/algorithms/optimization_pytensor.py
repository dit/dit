"""
Base class for optimization using PyTensor for automatic differentiation.

This module provides the same functionality as :mod:`dit.algorithms.optimization`
but leverages `PyTensor <https://pytensor.readthedocs.io>`_ (the maintained
successor to the now-archived Aesara, itself a fork of Theano) for:

- Symbolic graph construction and compilation of the objective.
- Exact gradients / Jacobians via ``pytensor.grad`` (reverse-mode autodiff).
- Optional Numba compilation of the compiled functions.

Unlike JAX (``slsqp-jax``/``optimistix``) and PyTorch (``torch.optim``),
PyTensor ships no native constrained optimizer.  This backend therefore
compiles the objective, its gradient, and each constraint Jacobian *once* and
hands them to :func:`scipy.optimize.minimize` (SLSQP), with an additional
native *augmented-Lagrangian* path (compiled value+gradient + ``L-BFGS-B``
inner solves) for moderate problem sizes.

Because PyTensor is *define-then-run* (a single static graph is compiled and
then evaluated), a handful of measures whose objective re-enters a nested
Python optimizer mid-evaluation (e.g. the reduced / two-part intrinsic mutual
informations) cannot be expressed as one symbolic graph.  To support these too,
the information-theoretic helpers and :meth:`construct_joint` are *polymorphic*:
given a symbolic input they build a graph (compiled, autodiff gradients); given
a concrete NumPy array they evaluate eagerly.  When the symbolic compile fails,
:meth:`optimize` transparently falls back to an eager NumPy objective with
finite-difference gradients (matching the NumPy backend's behaviour).

Notes
-----
Requires PyTensor to be installed: ``pip install pytensor``.
"""

import contextlib
import operator
import os
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from copy import deepcopy
from functools import reduce
from string import ascii_letters, digits
from types import MethodType

import numpy as np
from loguru import logger

try:
    import pytensor
    import pytensor.tensor as pt
    from pytensor.graph.basic import Variable as _PtVariable

    PYTENSOR_AVAILABLE = True
except ImportError:
    PYTENSOR_AVAILABLE = False
    pt = None
    # ``isinstance(x, ())`` is always False, so ``_is_pt`` degrades gracefully.
    _PtVariable = ()

from boltons.iterutils import pairwise
from scipy.optimize import Bounds, basinhopping, brute, differential_evolution, dual_annealing, minimize, shgo

from ..algorithms.caekl_psp import caekl_from_partition_pmf, caekl_partition_indices
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
)

__all__ = (
    "BasePytensorOptimizer",
    "BaseConvexPytensorOptimizer",
    "BaseNonConvexPytensorOptimizer",
    "BaseAuxVarPytensorOptimizer",
    "is_pytensor_available",
)


_PYTENSOR_CONFIGURED = False


def _check_pytensor():
    """
    Raise an error if PyTensor is not available, and configure it for use.

    Sets ``floatX = float64`` (required for scipy's SLSQP solver) and, if the
    ``DIT_PYTENSOR_COMPILEDIR`` environment variable is set, points PyTensor's
    persistent compilation cache there so repeated process launches don't pay
    C-compilation time on every invocation.

    Raises
    ------
    ImportError
        If PyTensor is not installed.
    """
    global _PYTENSOR_CONFIGURED  # noqa: PLW0603

    if not PYTENSOR_AVAILABLE:
        raise ImportError("PyTensor is required for this module. Install with: pip install pytensor")

    if _PYTENSOR_CONFIGURED:
        return

    # scipy's SLSQP (and our L-BFGS-B inner solves) require 64-bit precision.
    pytensor.config.floatX = "float64"

    compiledir = os.environ.get("DIT_PYTENSOR_COMPILEDIR")
    if compiledir:
        with contextlib.suppress(Exception):
            pytensor.config.compiledir = compiledir

    _PYTENSOR_CONFIGURED = True


def is_pytensor_available():
    """
    Check if PyTensor is available for use.

    Returns
    -------
    available : bool
        True if PyTensor is installed and can be imported.
    """
    return PYTENSOR_AVAILABLE


# ── NumPy / PyTensor dispatch helpers ────────────────────────────────────
#
# The information-theoretic helpers below are written *once* against these
# dispatchers so that they operate symbolically on PyTensor variables (for the
# compiled fast path) and eagerly on NumPy arrays (for ``construct_distribution``
# and the nested-objective measures).  Most array operations (``+``, ``*``,
# ``/``, ``**``, ``.sum``, ``.reshape``, indexing) share the same spelling on
# both libraries; only the functions that differ are wrapped here.


def _is_pt(x):
    """Return True if *x* is a PyTensor symbolic variable."""
    return PYTENSOR_AVAILABLE and isinstance(x, _PtVariable)


def _any_pt(*xs):
    """Return True if any of *xs* is a PyTensor symbolic variable."""
    return any(_is_pt(x) for x in xs)


def _where(cond, a, b):
    """Backend-agnostic ``where`` / ``switch``."""
    if _any_pt(cond, a, b):
        return pt.switch(cond, a, b)
    return np.where(cond, a, b)


def _log2(x):
    """Backend-agnostic base-2 logarithm."""
    return pt.log2(x) if _is_pt(x) else np.log2(x)


def _sqrt(x):
    """Backend-agnostic square root."""
    return pt.sqrt(x) if _is_pt(x) else np.sqrt(x)


def _abs(x):
    """Backend-agnostic absolute value."""
    return pt.abs(x) if _is_pt(x) else np.abs(x)


def _isnan(x):
    """Backend-agnostic ``isnan``."""
    return pt.isnan(x) if _is_pt(x) else np.isnan(x)


def _total_sum(x):
    """Backend-agnostic total (all-elements) sum."""
    return pt.sum(x) if _is_pt(x) else np.sum(x)


def _stack(xs):
    """Backend-agnostic ``stack``."""
    return pt.stack(xs) if _any_pt(*xs) else np.stack(xs)


def _amin(x):
    """Backend-agnostic ``min`` reduction."""
    return pt.min(x) if _is_pt(x) else np.min(x)


def _svdvals(m):
    """Compute the singular values of a matrix (NumPy or PyTensor)."""
    if _is_pt(m):
        return pt.linalg.svd(m, compute_uv=False)
    return np.linalg.svd(m, compute_uv=False)


class BasePytensorOptimizer(metaclass=ABCMeta):
    """
    Base class for performing optimizations using PyTensor.

    This class mirrors :class:`~dit.algorithms.optimization.BaseOptimizer` but
    builds a symbolic graph for the objective and compiles it (together with its
    exact gradient and the constraint Jacobians) once per ``optimize`` call.
    """

    # Whether to use autodiff (symbolic compilation) for gradients.  When the
    # symbolic build fails, ``optimize`` falls back to eager evaluation.
    _use_autodiff = True

    # Whether to attempt the native augmented-Lagrangian path.
    _use_native = True

    def __init__(self, dist, rvs=None, crvs=None):
        """
        Initialize the optimizer.

        Parameters
        ----------
        dist : Distribution
            The distribution to optimize over.
        rvs : iterable of iterables
            The variables of interest.
        crvs : iterable
            The variables to be conditioned on.
        """
        _check_pytensor()

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
        # Stored as NumPy arrays: PyTensor ops combine NumPy constants with the
        # symbolic optimization vector to produce the objective graph.
        self._full_pmf = np.asarray(self._dist.pmf.reshape(self._full_shape), dtype=np.float64)

        self._n = dist.outcome_length()
        self._pmf = self._full_pmf.sum(axis=tuple(range(self._n)))
        self._shape = self._pmf.shape

        self._full_vars = set(range(len(self._full_shape)))
        self._all_vars = set(range(len(rvs) + 1))
        self._rvs = set(range(len(rvs)))
        self._crvs = {len(rvs)}

        self._proxy_vars = tuple(range(self._n, self._n + len(rvs) + 1))

        self._additional_options = {}
        self._rng = None

        self.constraints = []

        # Compiled-graph state, (re)populated by ``optimize``.
        self._pt_compiled = False
        self._pt_obj_fn = None
        self._pt_grad_fn = None
        self._pt_constraint_fns = []
        self._pt_al_fn = None

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
    # PyTensor-specific helpers

    @property
    def _pt_mode(self):
        """
        The PyTensor compilation mode.

        Defaults to PyTensor's configured default (the C backend, ``FAST_RUN``).
        Set the ``DIT_PYTENSOR_MODE`` environment variable (e.g. ``NUMBA`` or
        ``FAST_COMPILE``) to override.
        """
        return os.environ.get("DIT_PYTENSOR_MODE") or None

    def _maybe_jit(self, func, **kwargs):
        """
        No-op decorator kept for structural parity with the JAX/Torch backends.

        PyTensor compiles whole graphs in :meth:`optimize`; the per-primitive
        information-theoretic helpers merely *build* graph fragments, so there is
        nothing to compile here.
        """
        return func

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
        return sample_simplex(self._optvec_size, rng=self._rng)

    def construct_uniform_initial(self):
        """
        Construct a uniform optimization vector.

        Returns
        -------
        x : np.ndarray
            A uniform optimization vector.
        """
        return np.ones(self._optvec_size) / self._optvec_size

    ###########################################################################
    # Convenience functions for constructing objectives.

    @staticmethod
    def _h(p):
        """
        Compute the entropy of `p`.

        Uses the "safe where" pattern so that both the forward value and the
        symbolic gradient are finite even where ``p == 0``.

        Parameters
        ----------
        p : array_like
            A vector of probabilities (NumPy or PyTensor).

        Returns
        -------
        h : scalar
            The entropy.
        """
        safe_p = _where(p > 0, p, 1.0)
        return -_total_sum(_where(p > 0, p * _log2(safe_p), 0.0))

    def _entropy(self, rvs, crvs=None):
        """
        Construct the conditional entropy, H[X|Y].

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
            pmf_joint = pmf.sum(axis=idx_joint, keepdims=True)
            pmf_crvs = pmf_joint.sum(axis=idx_crvs, keepdims=True)

            return self._h(pmf_joint) - self._h(pmf_crvs)

        return entropy

    def _mutual_information(self, rv_x, rv_y):
        """
        Construct the mutual information, I[X:Y].

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
            pmf_xy = pmf.sum(axis=idx_xy, keepdims=True)
            pmf_x = pmf_xy.sum(axis=idx_x, keepdims=True)
            pmf_y = pmf_xy.sum(axis=idx_y, keepdims=True)

            safe_denom = _where((pmf_x > 0) & (pmf_y > 0), pmf_x * pmf_y, 1.0)
            safe_xy = _where(pmf_xy > 0, pmf_xy, 1.0)
            ratio = safe_xy / safe_denom
            return _total_sum(_where(pmf_xy > 0, pmf_xy * _log2(ratio), 0.0))

        return mutual_information

    def _conditional_mutual_information(self, rv_x, rv_y, rv_z):
        """
        Construct the conditional mutual information, I[X:Y|Z].

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
            pmf_xyz = pmf.sum(axis=idx_xyz, keepdims=True)
            pmf_xz = pmf_xyz.sum(axis=idx_xz, keepdims=True)
            pmf_yz = pmf_xyz.sum(axis=idx_yz, keepdims=True)
            pmf_z = pmf_xz.sum(axis=idx_z, keepdims=True)

            safe_denom = _where((pmf_xz > 0) & (pmf_yz > 0), pmf_xz * pmf_yz, 1.0)
            safe_numer = _where((pmf_xyz > 0) & (pmf_z > 0), pmf_z * pmf_xyz, 1.0)
            ratio = safe_numer / safe_denom
            return _total_sum(_where(pmf_xyz > 0, pmf_xyz * _log2(ratio), 0.0))

        return conditional_mutual_information

    def _coinformation(self, rvs, crvs=None):
        """
        Construct the coinformation.

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
        if crvs is None:
            crvs = set()
        idx_joint = tuple(self._all_vars - (rvs | crvs))
        idx_crvs = tuple(self._all_vars - crvs)
        idx_subrvs = [tuple(self._all_vars - set(ss)) for ss in sorted(powerset(rvs), key=len)[1:-1]]
        power = [(-1) ** len(ss) for ss in sorted(powerset(rvs), key=len)[1:-1]]
        power += [(-1) ** len(rvs)]
        power += [-sum(power)]

        @self._maybe_jit
        def coinformation(pmf):
            pmf_joint = pmf.sum(axis=idx_joint, keepdims=True)
            pmf_crvs = pmf_joint.sum(axis=idx_crvs, keepdims=True)
            pmf_subrvs = [pmf_joint.sum(axis=idx, keepdims=True) for idx in idx_subrvs] + [pmf_joint, pmf_crvs]

            pmf_ci = reduce(operator.mul, [p**pw for p, pw in zip(pmf_subrvs, power, strict=True)])
            safe_pmf_ci = _where(pmf_joint > 0, pmf_ci, 1.0)

            return _total_sum(_where(pmf_joint > 0, pmf_joint * _log2(safe_pmf_ci), 0.0))

        return coinformation

    def _total_correlation(self, rvs, crvs=None):
        """
        Construct the total correlation.

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
            pmf_joint = pmf.sum(axis=idx_joint, keepdims=True)
            pmf_margs = [pmf_joint.sum(axis=marg, keepdims=True) for marg in idx_margs]
            pmf_crvs = pmf_margs[0].sum(axis=idx_crvs, keepdims=True)

            h_crvs = self._h(pmf_crvs.ravel())
            h_margs = sum(self._h(p.ravel()) for p in pmf_margs)
            h_joint = self._h(pmf_joint.ravel())

            return h_margs - h_joint - n * h_crvs

        return total_correlation

    def _dual_total_correlation(self, rvs, crvs=None):
        """
        Construct the dual total correlation.

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
            pmf_joint = pmf.sum(axis=idx_joint, keepdims=True)
            pmf_margs = [pmf_joint.sum(axis=marg, keepdims=True) for marg in idx_margs]
            pmf_crvs = pmf_joint.sum(axis=idx_crvs, keepdims=True)

            h_crvs = self._h(pmf_crvs)
            h_joint = self._h(pmf_joint) - h_crvs
            h_margs = [self._h(marg) - h_crvs for marg in pmf_margs]

            return sum(h_margs) - n * h_joint

        return dual_total_correlation

    def _caekl_mutual_information(self, rvs, crvs=None):
        """
        Construct the CAEKL mutual information.

        The symbolic (PyTensor) path still enumerates partitions so the
        objective can be compiled as a single graph.  Eager NumPy evaluation
        uses PSP for scalability.

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

        parts_tuple = tuple(tuple(p) for p in parts)
        idx_parts_items = tuple((k, v) for k, v in idx_parts.items())
        all_vars = self._all_vars

        @self._maybe_jit
        def caekl_mutual_information_symbolic(pmf):
            pmf_joint = pmf.sum(axis=idx_joint, keepdims=True)
            pmf_parts = {p: pmf_joint.sum(axis=idx, keepdims=True) for p, idx in idx_parts_items}
            pmf_crvs = pmf_joint.sum(axis=idx_crvs, keepdims=True)

            h_crvs = self._h(pmf_crvs)
            h_joint = self._h(pmf_joint) - h_crvs

            candidates = []
            for part, norm in zip(parts_tuple, part_norms, strict=True):
                h_parts = sum(self._h(pmf_parts[p]) - h_crvs for p in part)
                candidates.append((h_parts - h_joint) / norm)

            return _amin(_stack(candidates))

        def caekl_mutual_information_eager(pmf):
            pmf_np = np.asarray(pmf)
            partition, idx_joint_, idx_crvs_, idx_parts_sel = caekl_partition_indices(
                pmf_np,
                all_vars=all_vars,
                rvs=rvs,
                crvs=crvs,
                h=self._h,
                sum_axes=lambda p, idx: np.asarray(p).sum(axis=idx, keepdims=True),
            )
            return caekl_from_partition_pmf(
                pmf_np,
                partition,
                idx_joint=idx_joint_,
                idx_crvs=idx_crvs_,
                idx_parts=idx_parts_sel,
                h=self._h,
                sum_axes=lambda p, idx: np.asarray(p).sum(axis=idx, keepdims=True),
            )

        def caekl_mutual_information(pmf):
            if _is_pt(pmf):
                return caekl_mutual_information_symbolic(pmf)
            return caekl_mutual_information_eager(pmf)

        return caekl_mutual_information

    def _maximum_correlation(self, rv_x, rv_y):
        """
        Construct the maximum correlation.

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
            pmf_xy = pmf.sum(axis=idx_xy)
            pmf_x = pmf.sum(axis=idx_x)[:, None]
            pmf_y = pmf.sum(axis=idx_y)[None, :]

            Q = pmf_xy / (_sqrt(pmf_x) * _sqrt(pmf_y))
            Q = _where(_isnan(Q), 0.0, Q)

            return _svdvals(Q)[1]

        return maximum_correlation

    def _conditional_maximum_correlation(self, rv_x, rv_y, rv_z):
        """
        Construct the conditional maximum correlation.

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
            p_xyz = pmf.sum(axis=idx_xyz)
            p_xz = pmf.sum(axis=idx_xz)[:, None, :]
            p_yz = pmf.sum(axis=idx_yz)[None, :, :]

            Q = _where(p_xyz > 0, p_xyz / _sqrt(p_xz * p_yz), 0.0)

            cmc = max(_svdvals(Q[:, :, k])[1] for k in range(Q.shape[2]))

            return cmc

        return conditional_maximum_correlation

    def _total_variation(self, rv_x, rv_y):
        """
        Construct the total variation, TV[X||Y].

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
            pmf_xy = pmf.sum(axis=idx_xy, keepdims=True)
            pmf_x = pmf_xy.sum(axis=idx_x)
            pmf_y = pmf_xy.sum(axis=idx_y)

            return _abs(pmf_x - pmf_y).sum() / 2

        return total_variation

    ###########################################################################
    # Symbolic compilation helpers

    def _compile(self, out, x_sym):
        """Compile a PyTensor expression ``out`` of the input ``x_sym``."""
        out = pt.as_tensor_variable(out)
        return pytensor.function([x_sym], out, on_unused_input="ignore", mode=self._pt_mode)

    def _compile_grad(self, out, x_sym):
        """Compile the gradient of a scalar expression ``out`` w.r.t. ``x_sym``."""
        out = pt.as_tensor_variable(out)
        g = pytensor.gradient.grad(out, x_sym, disconnected_inputs="ignore")
        return pytensor.function([x_sym], g, on_unused_input="ignore", mode=self._pt_mode)

    def _build_symbolic(self, x0):
        """
        Attempt to build and compile the symbolic objective, gradient, and
        constraint functions.

        Returns
        -------
        compiled : bool
            True if compilation succeeded, False if the objective could not be
            expressed as a single symbolic graph (in which case ``optimize``
            falls back to eager evaluation).
        """
        if not (self._use_autodiff and PYTENSOR_AVAILABLE):
            return False

        try:
            x_sym = pt.dvector("x")
            cost = self._pt_raw_objective(x_sym)
            self._pt_obj_fn = self._compile(cost, x_sym)
            self._pt_grad_fn = self._compile_grad(cost, x_sym)

            constraint_fns = []
            for c in self._pt_raw_constraints:
                cexpr = c["fun"](x_sym)
                constraint_fns.append(
                    {
                        "type": c["type"],
                        "fun": self._compile(cexpr, x_sym),
                        "jac": self._compile_grad(cexpr, x_sym),
                    }
                )
            self._pt_constraint_fns = constraint_fns
            logger.info("PyTensor objective compiled (dim={dim})", dim=x0.size)
            return True
        except Exception as e:  # pragma: no cover - measure-specific fallbacks
            logger.info("PyTensor symbolic compile unavailable ({e}); using eager numpy objective", e=repr(e))
            self._pt_obj_fn = None
            self._pt_grad_fn = None
            self._pt_constraint_fns = []
            return False

    def _eval_objective(self, x):
        """Evaluate the objective at *x* (compiled fast path or eager)."""
        x = np.asarray(x, dtype=np.float64)
        if self._pt_compiled:
            return float(np.asarray(self._pt_obj_fn(x)))
        return float(np.asarray(self._pt_raw_objective(x)))

    ###########################################################################
    # scipy-based optimization helpers

    def _scipy_minimize_kwargs(self, x0, callback=False, maxiter=None):
        """
        Build ``minimizer_kwargs`` for scipy-based optimization.

        When the symbolic objective compiled, exact gradients / constraint
        Jacobians are wired in (avoiding scipy's ``O(n)`` finite-difference
        probes per SLSQP step).  Otherwise the eager objective is used with
        ``numdifftools`` Jacobians when available.

        Returns
        -------
        minimizer_kwargs : dict
        icb : BasinHoppingInnerCallBack or None
        """
        icb = BasinHoppingInnerCallBack() if callback else None

        if self._pt_compiled:
            obj_fn = self._pt_obj_fn
            grad_fn = self._pt_grad_fn

            def _np_objective(x):
                return float(np.asarray(obj_fn(np.asarray(x, dtype=np.float64))))

            def _np_grad(x):
                return np.asarray(grad_fn(np.asarray(x, dtype=np.float64)), dtype=np.float64)

            wrapped_constraints = []
            for c in self._pt_constraint_fns:
                _f, _j = c["fun"], c["jac"]

                def _cf(x, _f=_f):
                    return float(np.asarray(_f(np.asarray(x, dtype=np.float64))))

                def _cj(x, _j=_j):
                    return np.asarray(_j(np.asarray(x, dtype=np.float64)), dtype=np.float64)

                wrapped_constraints.append({"type": c["type"], "fun": _cf, "jac": _cj})

            minimizer_kwargs = {
                "bounds": [(0, 1)] * x0.size,
                "callback": icb,
                "constraints": wrapped_constraints,
                "options": {},
                "jac": _np_grad,
            }
        else:
            raw_obj = self._pt_raw_objective

            def _np_objective(x):
                return float(np.asarray(raw_obj(np.asarray(x, dtype=np.float64))))

            wrapped_constraints = []
            for c in self._pt_raw_constraints:
                _f = c["fun"]

                def _cf(x, _f=_f):
                    return float(np.asarray(_f(np.asarray(x, dtype=np.float64))))

                wrapped_constraints.append({"type": c["type"], "fun": _cf})

            minimizer_kwargs = {
                "bounds": [(0, 1)] * x0.size,
                "callback": icb,
                "constraints": wrapped_constraints,
                "options": {},
            }

            try:  # pragma: no cover
                import numdifftools as ndt

                minimizer_kwargs["jac"] = ndt.Jacobian(_np_objective)
                for const in minimizer_kwargs["constraints"]:
                    const["jac"] = ndt.Jacobian(const["fun"])
            except ImportError:
                pass

        additions = deepcopy(self._additional_options)
        options = additions.pop("options", {})
        minimizer_kwargs["options"].update(options)
        minimizer_kwargs.update(additions)

        if maxiter:
            minimizer_kwargs["options"]["maxiter"] = maxiter

        self.objective = _np_objective

        return minimizer_kwargs, icb

    ###########################################################################
    # Main optimization entry point

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
            Whether to use a callback to track the performance of the
            optimization.

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
        x0 = np.asarray(x0, dtype=np.float64)

        logger.info("Starting PyTensor optimization: dim={dim}, niter={niter}", dim=x0.size, niter=niter)

        # Keep the raw (graph-builder / eager-capable) callables for both paths.
        self._pt_raw_objective = self.objective
        self._pt_raw_constraints = list(self.constraints)
        self._pt_al_fn = None

        self._pt_compiled = self._build_symbolic(x0)

        _MAX_NATIVE_DIM = 200
        use_native = self._pt_compiled and self._use_native and x0.size <= _MAX_NATIVE_DIM

        if use_native:
            logger.info("Using PyTensor native augmented-Lagrangian optimizer (dim={dim})", dim=x0.size)
            result = self._optimization_backend_native(x0, niter, maxiter)

            # Validate equality-constraint satisfaction; fall back to scipy SLSQP
            # if the augmented-Lagrangian result is clearly infeasible.
            if result is not None and self._pt_constraint_fns:
                max_eq_viol = 0.0
                for c in self._pt_constraint_fns:
                    if c["type"] == "eq":
                        max_eq_viol = max(max_eq_viol, abs(float(np.asarray(c["fun"](result.x)))))
                if max_eq_viol > 0.1:
                    logger.info(
                        "AL result has high constraint violation ({viol:.2e}); falling back to scipy SLSQP",
                        viol=max_eq_viol,
                    )
                    result = None

            if result is None:
                minimizer_kwargs, icb = self._scipy_minimize_kwargs(x0, callback=callback, maxiter=maxiter)
                self._callback = BasinHoppingCallBack(minimizer_kwargs.get("constraints", {}), icb)
                result = self._optimization_backend(x0, minimizer_kwargs, niter)
            else:
                # Refine the native result with a short scipy polish.
                minimizer_kwargs, _ = self._scipy_minimize_kwargs(x0, callback=False, maxiter=maxiter)
                res_refine = minimize(fun=self.objective, x0=result.x.flatten(), **minimizer_kwargs)
                if res_refine.success and res_refine.fun < result.fun:
                    logger.debug(
                        "scipy refinement improved native result: {old:.6f} -> {new:.6f}",
                        old=result.fun,
                        new=res_refine.fun,
                    )
                    result = res_refine
        else:
            if x0.size > _MAX_NATIVE_DIM:
                logger.info(
                    "Problem dimension {dim} exceeds native limit {lim}; using scipy SLSQP",
                    dim=x0.size,
                    lim=_MAX_NATIVE_DIM,
                )
            minimizer_kwargs, icb = self._scipy_minimize_kwargs(x0, callback=callback, maxiter=maxiter)
            self._callback = BasinHoppingCallBack(minimizer_kwargs.get("constraints", {}), icb)
            result = self._optimization_backend(x0, minimizer_kwargs, niter)

        if result:
            self._optima = np.array(result.x)
        else:  # pragma: no cover
            msg = "No optima found."
            raise OptimizationException(msg)

        if polish:
            self._polish(cutoff=polish)

        logger.info("PyTensor optimization complete: objective={obj}", obj=self._eval_objective(self._optima))

        return result

    ###########################################################################
    # Native augmented-Lagrangian optimization

    def _build_al_fn(self):
        """
        Compile (once) the augmented-Lagrangian value and gradient.

        Returns a compiled function with signature
        ``(x, lam_eq, lam_ineq, mu) -> (L, dL/dx)`` and the equality /
        inequality constraint counts.
        """
        if self._pt_al_fn is not None:
            return self._pt_al_fn, self._n_eq_al, self._n_ineq_al

        x = pt.dvector("x")
        lam_eq = pt.dvector("lam_eq")
        lam_ineq = pt.dvector("lam_ineq")
        mu = pt.dscalar("mu")

        loss = pt.as_tensor_variable(self._pt_raw_objective(x))

        eq_funs = [c["fun"] for c in self._pt_raw_constraints if c["type"] == "eq"]
        ineq_funs = [c["fun"] for c in self._pt_raw_constraints if c["type"] == "ineq"]

        for i, f in enumerate(eq_funs):
            ci = pt.as_tensor_variable(f(x))
            loss = loss + lam_eq[i] * ci + (mu / 2.0) * ci**2

        for j, f in enumerate(ineq_funs):
            hj = pt.as_tensor_variable(f(x))
            slack = pt.minimum(0.0, hj - lam_ineq[j] / mu)
            loss = loss + lam_ineq[j] * slack + (mu / 2.0) * slack**2

        g = pytensor.gradient.grad(loss, x, disconnected_inputs="ignore")
        fn = pytensor.function(
            [x, lam_eq, lam_ineq, mu],
            [loss, g],
            on_unused_input="ignore",
            mode=self._pt_mode,
        )

        self._pt_al_fn = fn
        self._n_eq_al = len(eq_funs)
        self._n_ineq_al = len(ineq_funs)
        return fn, self._n_eq_al, self._n_ineq_al

    def _al_minimize(
        self,
        x0,
        maxiter=None,
        al_outer_iters=50,
        al_inner_iters=100,
        mu_init=1.0,
        mu_factor=2.0,
        al_tol=1e-7,
    ):
        """
        Augmented-Lagrangian minimization using scipy ``L-BFGS-B`` inner solves.

        Solves ``min f(x)`` subject to ``g_i(x) = 0``, ``h_j(x) >= 0`` and
        ``0 <= x <= 1`` by iteratively minimizing the augmented Lagrangian (box
        bounds are handled natively by L-BFGS-B) and updating the dual variables
        and penalty parameter.  Mirrors the PyTorch backend's AL solver, with a
        PyTensor-compiled value+gradient and scipy as the unconstrained inner
        solver.

        Parameters
        ----------
        x0 : array_like
            Initial point.
        maxiter : int, optional
            Max L-BFGS-B iterations per outer step.
        al_outer_iters : int
            Number of outer AL iterations.
        al_inner_iters : int
            Max inner L-BFGS-B iterations when ``maxiter`` is not given.
        mu_init : float
            Initial penalty parameter.
        mu_factor : float
            Multiplicative increase of mu each outer iteration.
        al_tol : float
            Convergence tolerance on the max constraint violation.

        Returns
        -------
        result : OptimizeResult
            Scipy-compatible result object.
        """
        from scipy.optimize import OptimizeResult

        al_fn, n_eq, n_ineq = self._build_al_fn()

        eq_cfuns = [c["fun"] for c in self._pt_constraint_fns if c["type"] == "eq"]
        ineq_cfuns = [c["fun"] for c in self._pt_constraint_fns if c["type"] == "ineq"]

        n = x0.size
        bounds = [(0.0, 1.0)] * n
        x = np.clip(np.asarray(x0, dtype=np.float64).flatten(), 0.0, 1.0)

        lam_eq = np.zeros(n_eq, dtype=np.float64)
        lam_ineq = np.zeros(n_ineq, dtype=np.float64)
        mu = float(mu_init)

        inner_maxiter = maxiter or al_inner_iters

        best_x = x.copy()
        best_obj = float("inf")
        max_viol = float("inf")

        for outer in range(al_outer_iters):
            _lam_eq = lam_eq.copy()
            _lam_ineq = lam_ineq.copy()
            _mu = mu

            def fun(z, _le=_lam_eq, _li=_lam_ineq, _m=_mu):
                val, grad = al_fn(np.asarray(z, dtype=np.float64), _le, _li, np.float64(_m))
                return float(np.asarray(val)), np.asarray(grad, dtype=np.float64)

            res = minimize(fun, x, method="L-BFGS-B", jac=True, bounds=bounds, options={"maxiter": inner_maxiter})
            x = np.clip(res.x, 0.0, 1.0)

            max_eq_viol = 0.0
            for i, f in enumerate(eq_cfuns):
                ci = float(np.asarray(f(x)))
                lam_eq[i] = lam_eq[i] + mu * ci
                max_eq_viol = max(max_eq_viol, abs(ci))

            max_ineq_viol = 0.0
            for j, f in enumerate(ineq_cfuns):
                hj = float(np.asarray(f(x)))
                max_ineq_viol = max(max_ineq_viol, max(0.0, -hj))
                lam_ineq[j] = max(0.0, lam_ineq[j] - mu * hj)

            max_viol = max(max_eq_viol, max_ineq_viol)

            obj_val = float(np.asarray(self._pt_obj_fn(x)))
            if obj_val < best_obj and max_viol < al_tol * 100:
                best_obj = obj_val
                best_x = x.copy()

            logger.debug(
                "AL outer={outer}: obj={obj:.6f}, max_viol={viol:.2e}, mu={mu:.1f}",
                outer=outer,
                obj=obj_val,
                viol=max_viol,
                mu=mu,
            )

            if max_viol < al_tol:
                logger.debug("AL converged at outer iteration {outer}", outer=outer)
                break

            mu *= mu_factor

        if not np.isfinite(best_obj):
            best_x = x.copy()
            best_obj = float(np.asarray(self._pt_obj_fn(x)))

        success = max_viol < al_tol * 10

        return OptimizeResult(
            x=best_x,
            fun=best_obj,
            success=success,
            message=f"AL: max_viol={max_viol:.2e}, mu={mu:.1f}",
        )

    def _optimization_backend_native(self, x0, niter, maxiter):
        """
        Dispatch to the native augmented-Lagrangian shotgun backend.
        """
        return self._optimize_shotgun_native(x0, niter, maxiter)

    def _optimize_shotgun_native(self, x0, niter, maxiter):
        """
        Multi-start optimization using the augmented-Lagrangian method.

        Parameters
        ----------
        x0 : ndarray
            Initial optimization vector.
        niter : int or None
            Number of random restarts.
        maxiter : int or None
            Maximum inner iterations per start.

        Returns
        -------
        result : OptimizeResult or None
        """
        if niter is None:
            niter = self._default_hops

        starts = []
        if x0 is not None:
            starts.append(np.asarray(x0).flatten())
            niter -= 1
        for _ in range(max(niter, 0)):
            starts.append(np.asarray(self.construct_random_initial()).flatten())

        if not starts:  # pragma: no cover
            return None

        results = [self._al_minimize(s, maxiter=maxiter) for s in starts]

        if not results:  # pragma: no cover
            return None

        return min(results, key=lambda r: float(np.asarray(self._pt_obj_fn(r.x))))

    ###########################################################################
    # scipy-based shotgun (also used by the fallback path)

    def _optimize_shotgun(self, x0, minimizer_kwargs, niter):
        """
        Perform a non-convex optimization using a multi-start "shotgun" approach.

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
            The result of the optimization. Returns None if it failed.
        """
        if niter is None:
            niter = self._default_hops

        results = []

        if x0 is not None:
            logger.debug("Shotgun: trying provided initial condition")
            res = minimize(fun=self.objective, x0=x0.flatten(), **minimizer_kwargs)
            if res.success:
                results.append(res)
            niter -= 1

        ics = (self.construct_random_initial() for _ in range(niter))
        for i, initial in enumerate(ics):
            logger.debug("Shotgun: random initial condition {i}/{niter}", i=i + 1, niter=niter)
            res = minimize(fun=self.objective, x0=initial.flatten(), **minimizer_kwargs)
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

        if self._pt_compiled:
            grad_fn = self._pt_grad_fn

            def _np_grad(x):
                return np.asarray(grad_fn(np.asarray(x, dtype=np.float64)), dtype=np.float64)

            wrapped_constraints = []
            for c in self._pt_constraint_fns:
                _f, _j = c["fun"], c["jac"]

                def _cf(x, _f=_f):
                    return float(np.asarray(_f(np.asarray(x, dtype=np.float64))))

                def _cj(x, _j=_j):
                    return np.asarray(_j(np.asarray(x, dtype=np.float64)), dtype=np.float64)

                wrapped_constraints.append({"type": c["type"], "fun": _cf, "jac": _cj})
        else:
            _np_grad = None
            wrapped_constraints = []
            for c in self._pt_raw_constraints:
                _f = c["fun"]

                def _cf(x, _f=_f):
                    return float(np.asarray(_f(np.asarray(x, dtype=np.float64))))

                wrapped_constraints.append({"type": c["type"], "fun": _cf})

        minimizer_kwargs = {
            "bounds": Bounds(lb, ub, feasible),
            "tol": None,
            "callback": None,
            "constraints": wrapped_constraints,
        }

        if _np_grad is not None:
            minimizer_kwargs["jac"] = _np_grad
        else:
            try:  # pragma: no cover
                import numdifftools as ndt

                minimizer_kwargs["jac"] = ndt.Jacobian(self.objective)
                for const in minimizer_kwargs["constraints"]:
                    const["jac"] = ndt.Jacobian(const["fun"])
            except ImportError:
                pass

        if np.allclose(lb, ub):
            self._optima = x0
            return

        res = minimize(fun=self.objective, x0=x0, **minimizer_kwargs)

        if res.success:
            logger.debug("Polishing successful: objective={obj}", obj=res.fun)
            self._optima = res.x.copy()

            if count < (res.x < cutoff).sum():
                self._polish(cutoff=cutoff)


class BaseConvexPytensorOptimizer(BasePytensorOptimizer):
    """
    Implement convex optimization using PyTensor.
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
            The result of the optimization. Returns None if it failed.
        """
        if niter is None:
            # even though this is convex, there might still be some optimization
            # issues, so we use niter > 1.
            niter = 2
        return self._optimize_shotgun(x0, minimizer_kwargs, niter=niter)


class BaseNonConvexPytensorOptimizer(BasePytensorOptimizer):
    """
    Implement non-convex optimization using PyTensor.
    """

    _shotgun = False

    def _optimization_backend_native(self, x0, niter, maxiter):
        """
        Native augmented-Lagrangian backend for non-convex optimization.

        Uses a multi-start (shotgun) strategy over ``x0`` plus random starts.
        """
        if niter is None:
            niter = self._default_hops
        return self._optimize_shotgun_native(x0, niter, maxiter)

    def _optimization_basinhopping(self, x0, minimizer_kwargs, niter):
        """
        Perform a non-convex optimization using scipy.optimize.basinhopping.

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
            The result of the optimization. Returns None if it failed.
        """
        if niter is None:
            niter = self._default_hops

        logger.debug("Basin hopping (PyTensor): starting with niter={niter}", niter=niter)

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
        logger.info("Basin hopping (PyTensor) result: success={success}, message={msg}", success=success, msg=msg)
        if not success:  # pragma: no cover
            result = self._callback.minimum() or res_shotgun

        return result

    def _optimization_diffevo(self, x0, minimizer_kwargs, niter):  # pragma: no cover
        """
        Perform optimization using differential evolution.
        """
        if "constraints" in minimizer_kwargs:
            msg = "Differential Evolution can only be used in unconstrained optimization."
            raise OptimizationException(msg)

        if niter is None:
            niter = self._default_hops

        result = differential_evolution(
            func=self.objective,
            bounds=minimizer_kwargs["bounds"],
            maxiter=minimizer_kwargs["options"]["maxiter"],
            popsize=niter,
            tol=minimizer_kwargs["options"]["ftol"],
        )

        if result.success:
            return result

    def _optimization_shgo(self, x0, minimizer_kwargs, niter):
        """
        Perform a non-convex optimization using scipy.optimize.shgo.
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
        """
        if niter is None:
            niter = self._default_hops

        da_kwargs = {}
        if self._rng is not None:
            da_kwargs["rng"] = self._rng

        result = dual_annealing(
            func=self.objective,
            bounds=minimizer_kwargs["bounds"],
            minimizer_kwargs=minimizer_kwargs,
            maxiter=niter,
            x0=x0,
            **da_kwargs,
        )

        if result.success:
            return result

    def _optimization_brute(self, x0, minimizer_kwargs, niter):
        """
        Perform a non-convex optimization using scipy.optimize.brute.
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


class BaseAuxVarPytensorOptimizer(BaseNonConvexPytensorOptimizer):
    """
    Base class that performs many methods related to optimizing auxiliary
    variables, using PyTensor for automatic differentiation.
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
        for i, (auxvar, var) in enumerate(zip(self._aux_vars, arvs, strict=True)):
            relevant_vars = {self._n + b for b in auxvar.bases}
            index = sorted(self._full_vars) + [self._n + a for a in arvs[: i + 1]]
            var += self._n
            self._full_slices.append(tuple(colon if i in relevant_vars | {var} else None for i in index))

        self._slices = []
        for i, (auxvar, var) in enumerate(zip(self._aux_vars, arvs, strict=True)):
            relevant_vars = auxvar.bases
            index = sorted(self._rvs | self._crvs | set(arvs[: i + 1]))
            self._slices.append(tuple(colon if i in relevant_vars | {var} else None for i in index))

    ###########################################################################
    # Constructing the joint distribution.

    def _construct_channels(self, x):
        """
        Construct the conditional distributions which produce the auxiliary
        variables.

        Parameters
        ----------
        x : array_like
            An optimization vector (NumPy or PyTensor).

        Yields
        ------
        channel : array_like
            A conditional distribution.
        """
        parts = [x[a:b] for a, b in self._parts]

        for part, auxvar in zip(parts, self._aux_vars, strict=True):
            channel = part.reshape(auxvar.shape)
            channel = channel / channel.sum(axis=(-1,), keepdims=True)
            channel = _where(_isnan(channel), auxvar.mask, channel)

            yield channel

    def construct_joint(self, x):
        """
        Construct the joint distribution.

        Parameters
        ----------
        x : array_like
            An optimization vector.

        Returns
        -------
        joint : array_like
            The joint distribution resulting from the distribution passed in and
            the optimization vector.
        """
        joint = self._pmf

        for channel, slc in zip(self._construct_channels(x), self._slices, strict=True):
            joint = joint[..., None] * channel[slc]

        return joint

    def _construct_joint_single(self, x):
        """
        Construct the joint distribution (optimized for a single auxiliary
        variable).

        Parameters
        ----------
        x : array_like
            An optimization vector.

        Returns
        -------
        joint : array_like
            The joint distribution.
        """
        _, _, shape, mask, _ = self._aux_vars[0]
        channel = x.reshape(shape)
        channel = channel / channel.sum(axis=-1, keepdims=True)
        channel = _where(_isnan(channel), mask, channel)

        return self._pmf[..., None] * channel[self._slices[0]]

    def construct_full_joint(self, x):
        """
        Construct the full joint distribution.

        Parameters
        ----------
        x : array_like
            An optimization vector.

        Returns
        -------
        joint : array_like
            The joint distribution.
        """
        joint = self._full_pmf

        for channel, slc in zip(self._construct_channels(x), self._full_slices, strict=True):
            joint = joint[..., None] * channel[slc]

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

        # ``x`` is a concrete NumPy vector, so the polymorphic helpers evaluate
        # ``construct_full_joint`` eagerly and return a NumPy array.
        joint = np.asarray(self.construct_full_joint(np.asarray(x, dtype=np.float64)))
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

        @self._maybe_jit
        def constraint(x):
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
        for channel in self._construct_channels(np.asarray(x, dtype=np.float64)):
            ccs.append(channel_capacity(np.asarray(channel))[0])
        return ccs

    def _post_process(self, style="entropy", minmax="min", niter=10, maxiter=None):  # pragma: no cover
        """
        Find a solution to the minimization with a secondary property.

        Parameters
        ----------
        style : 'entropy', 'channel'
            The measure to perform the secondary optimization on.
        minmax : 'min', 'max'
            Whether to minimize or maximize the objective.
        niter : int
            The number of basin hops to perform.
        maxiter : int
            The number of minimization steps to perform.
        """
        logger.debug("Post-processing (PyTensor): style={style}, minmax={minmax}", style=style, minmax=minmax)
        entropy = self._entropy(self._arvs)

        sign = +1 if minmax == "min" else -1

        def objective_entropy(x):
            return sign * entropy(self.construct_joint(x))

        def objective_channelcapacity(x):
            return sign * sum(self._channel_capacity(x))

        if style == "channel":
            objective = objective_channelcapacity
        elif style == "entropy":
            objective = objective_entropy
        else:
            msg = f"Style {style} is not understood."
            raise OptimizationException(msg)

        true_objective = self.objective(self._optima)

        def constraint_match_objective(x):
            return (self.objective(x) - true_objective) ** 2

        constraint = [
            {
                "type": "eq",
                "fun": constraint_match_objective,
            }
        ]

        try:
            self.constraints += constraint
        except AttributeError:
            self.constraints = constraint

        self.__old_objective, self.objective = self.objective, objective

        self.optimize(x0=self._optima.copy(), niter=niter, maxiter=maxiter)

        self.constraints = self.constraints[:-1]
        if not self.constraints:
            del self.constraints

        self.objective = self.__old_objective
        del self.__old_objective
