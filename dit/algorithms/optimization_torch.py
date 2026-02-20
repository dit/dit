"""
Base class for optimization using PyTorch for automatic differentiation.

This module provides the same functionality as optimization.py but leverages
PyTorch for:
- Automatic differentiation (gradients and jacobians)
- GPU acceleration (when available)
- torch.compile for improved performance (PyTorch 2.0+)

Notes
-----
PyTorch is required. Install with: pip install torch
"""

import contextlib
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from copy import deepcopy
from functools import reduce
from string import ascii_letters, digits
from types import MethodType

import numpy as np
from loguru import logger

try:
    import torch
    from torch import Tensor

    TORCH_AVAILABLE = True

    # Check for torch.compile (PyTorch 2.0+)
    TORCH_COMPILE_AVAILABLE = hasattr(torch, "compile")
except ImportError:
    TORCH_AVAILABLE = False
    TORCH_COMPILE_AVAILABLE = False
    torch = None
    Tensor = None

from boltons.iterutils import pairwise
from scipy.optimize import Bounds, basinhopping, differential_evolution, minimize, shgo

from ..algorithms.channelcapacity import channel_capacity
from ..distconst import insert_rvf, modify_outcomes
from ..exceptions import OptimizationException, ditException
from ..helpers import flatten, normalize_rvs, parse_rvs
from ..math import prod, sample_simplex
from ..npdist import Distribution
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
    "BaseTorchOptimizer",
    "BaseConvexTorchOptimizer",
    "BaseNonConvexTorchOptimizer",
    "BaseAuxVarTorchOptimizer",
)


def _check_torch():
    """Raise an error if PyTorch is not available."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for this module. Install with: pip install torch")


def _to_tensor(x, requires_grad=False, dtype=None):
    """
    Convert input to PyTorch tensor.

    Parameters
    ----------
    x : array-like
        Input to convert.
    requires_grad : bool
        Whether to track gradients.
    dtype : torch.dtype, optional
        Data type for the tensor.

    Returns
    -------
    tensor : torch.Tensor
        The converted tensor.
    """
    if dtype is None:
        dtype = torch.float64
    if isinstance(x, Tensor):
        if x.dtype != dtype:
            x = x.to(dtype)
        if x.requires_grad != requires_grad:
            x = x.detach().requires_grad_(requires_grad)
        return x
    return torch.tensor(x, dtype=dtype, requires_grad=requires_grad)


def _to_numpy(x):
    """
    Convert PyTorch tensor to numpy array.

    Parameters
    ----------
    x : torch.Tensor or array-like
        Input to convert.

    Returns
    -------
    arr : np.ndarray
        The numpy array.
    """
    if isinstance(x, Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _marginalize(tensor, dim, keepdim=True):
    """
    Sum *tensor* over *dim*, returning the tensor unchanged when *dim* is empty.

    PyTorch's ``Tensor.sum(dim=())`` reduces **all** dimensions (unlike NumPy's
    ``ndarray.sum(axis=())`` which is a no-op).  This helper normalises that
    behaviour so the information-theoretic functions work correctly when the
    set of dimensions to marginalise out is empty.

    Parameters
    ----------
    tensor : torch.Tensor
    dim : tuple of int
        Dimensions to sum over.  May be an empty tuple.
    keepdim : bool
        Whether to retain the summed dimensions as size-1.

    Returns
    -------
    result : torch.Tensor
    """
    if not dim:
        return tensor
    return tensor.sum(dim=dim, keepdim=keepdim)


class BaseTorchOptimizer(metaclass=ABCMeta):
    """
    Base class for performing optimizations using PyTorch for automatic differentiation.

    This class mirrors BaseOptimizer but uses PyTorch for:
    - Computing gradients via autograd
    - GPU acceleration (when available)
    - torch.compile for performance (PyTorch 2.0+)
    """

    # Whether to use torch.compile (PyTorch 2.0+)
    _use_compile = False  # Disabled by default as it can cause issues with some operations

    # Whether to use autodiff for jacobians
    _use_autodiff = True

    # Device to use for computations ('cpu', 'cuda', 'mps', etc.)
    _device = "cpu"

    # Data type for tensors
    _dtype = torch.float64 if TORCH_AVAILABLE else None

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
        _check_torch()

        rvs, crvs, rv_mode = normalize_rvs(dist, rvs, crvs, rv_mode)
        self._dist = dist.copy(base="linear")

        self._alphabet = self._dist.alphabet
        self._original_shape = list(map(len, self._dist.alphabet))

        self._true_rvs = [parse_rvs(self._dist, rv, rv_mode=rv_mode)[1] for rv in rvs]
        self._true_crvs = parse_rvs(self._dist, crvs, rv_mode=rv_mode)[1]
        self._dist = modify_outcomes(self._dist, tuple)

        # compress all random variables down to single vars
        self._unqs = []
        for var in self._true_rvs + [self._true_crvs]:
            unq = Uniquifier()
            self._dist = insert_rvf(self._dist, lambda x, unq=unq, var=var: (unq(tuple(x[i] for i in var)),))
            self._unqs.append(unq)

        self._dist.make_dense()

        self._full_shape = list(map(len, self._dist.alphabet))
        # Convert to PyTorch tensor
        self._full_pmf = _to_tensor(self._dist.pmf.reshape(self._full_shape), dtype=self._dtype).to(self._device)

        self._n = dist.outcome_length()
        self._pmf = self._full_pmf.sum(dim=tuple(range(self._n)))
        self._shape = tuple(self._pmf.shape)

        self._full_vars = set(range(len(self._full_shape)))
        self._all_vars = set(range(len(rvs) + 1))
        self._rvs = set(range(len(rvs)))
        self._crvs = {len(rvs)}

        self._proxy_vars = tuple(range(self._n, self._n + len(rvs) + 1))

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
    # PyTorch-specific helper methods

    def _maybe_compile(self, func):
        """
        Optionally compile a function with torch.compile (PyTorch 2.0+).

        Parameters
        ----------
        func : callable
            The function to potentially compile.

        Returns
        -------
        func : callable
            The (potentially compiled) function.
        """
        if self._use_compile and TORCH_COMPILE_AVAILABLE:
            return torch.compile(func)
        return func

    def _compute_gradient(self, func):
        """
        Create a gradient function using PyTorch autograd.

        Parameters
        ----------
        func : callable
            The scalar function to differentiate.

        Returns
        -------
        grad_func : callable or None
            The gradient function, or None if autodiff is unavailable.
        """
        if not self._use_autodiff or not TORCH_AVAILABLE:
            return None

        def grad_func(x):
            """Compute gradient using PyTorch autograd."""
            x_tensor = _to_tensor(x, requires_grad=True, dtype=self._dtype).to(self._device)
            y = func(x_tensor)
            if isinstance(y, Tensor):
                y.backward()
                grad = x_tensor.grad
                return _to_numpy(grad)
            return None

        return grad_func

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
    # Convenience functions for constructing objectives using PyTorch.

    @staticmethod
    def _h(p):
        """
        Compute the entropy of `p` using PyTorch.

        Parameters
        ----------
        p : torch.Tensor
            A vector of probabilities.

        Returns
        -------
        h : torch.Tensor
            The entropy.
        """
        # "Safe where" pattern: replace p with 1.0 before log so both branches
        # of torch.where produce finite gradients (autograd traces both paths).
        safe_p = torch.where(p > 0, p, torch.ones_like(p))
        return -torch.sum(torch.where(p > 0, p * torch.log2(safe_p), torch.zeros_like(p)))

    def _entropy(self, rvs, crvs=None):
        """
        Compute the conditional entropy, H[X|Y], using PyTorch.

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

        def entropy(pmf):
            """
            Compute the specified entropy.

            Parameters
            ----------
            pmf : torch.Tensor
                The joint probability distribution.

            Returns
            -------
            h : torch.Tensor
                The entropy.
            """
            pmf_joint = _marginalize(pmf, idx_joint)
            pmf_crvs = _marginalize(pmf_joint, idx_crvs)

            h_joint = self._h(pmf_joint)
            h_crvs = self._h(pmf_crvs)

            ch = h_joint - h_crvs

            return ch

        return self._maybe_compile(entropy)

    def _mutual_information(self, rv_x, rv_y):
        """
        Compute the mutual information, I[X:Y], using PyTorch.

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

        def mutual_information(pmf):
            """
            Compute the specified mutual information.

            Parameters
            ----------
            pmf : torch.Tensor
                The joint probability distribution.

            Returns
            -------
            mi : torch.Tensor
                The mutual information.
            """
            pmf_xy = _marginalize(pmf, idx_xy)
            pmf_x = _marginalize(pmf_xy, idx_x)
            pmf_y = _marginalize(pmf_xy, idx_y)

            # "Safe where" pattern: use safe denominators so autograd never
            # traces through 0/0 or log(0), even in the unused branch.
            product = pmf_x * pmf_y
            safe_product = torch.where(product > 0, product, torch.ones_like(product))
            safe_pmf_xy = torch.where(pmf_xy > 0, pmf_xy, torch.ones_like(pmf_xy))
            ratio = safe_pmf_xy / safe_product
            mi = torch.sum(torch.where(pmf_xy > 0, pmf_xy * torch.log2(ratio), torch.zeros_like(pmf_xy)))

            return mi

        return self._maybe_compile(mutual_information)

    def _conditional_mutual_information(self, rv_x, rv_y, rv_z):
        """
        Compute the conditional mutual information, I[X:Y|Z], using PyTorch.

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

        def conditional_mutual_information(pmf):
            """
            Compute the specified conditional mutual information.

            Parameters
            ----------
            pmf : torch.Tensor
                The joint probability distribution.

            Returns
            -------
            cmi : torch.Tensor
                The conditional mutual information.
            """
            pmf_xyz = _marginalize(pmf, idx_xyz)
            pmf_xz = _marginalize(pmf_xyz, idx_xz)
            pmf_yz = _marginalize(pmf_xyz, idx_yz)
            pmf_z = _marginalize(pmf_xz, idx_z)

            # "Safe where" pattern: use safe values so autograd never traces
            # through 0/0 or log(0), even in the unused branch.
            numer = pmf_z * pmf_xyz
            denom = pmf_xz * pmf_yz
            safe_denom = torch.where(denom > 0, denom, torch.ones_like(denom))
            safe_numer = torch.where(pmf_xyz > 0, numer, torch.ones_like(numer))
            ratio = safe_numer / safe_denom
            cmi = torch.sum(torch.where(pmf_xyz > 0, pmf_xyz * torch.log2(ratio), torch.zeros_like(pmf_xyz)))

            return cmi

        return self._maybe_compile(conditional_mutual_information)

    def _coinformation(self, rvs, crvs=None):
        """
        Compute the coinformation using PyTorch.

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

        def coinformation(pmf):
            """
            Compute the specified co-information.

            Parameters
            ----------
            pmf : torch.Tensor
                The joint probability distribution.

            Returns
            -------
            ci : torch.Tensor
                The co-information.
            """
            pmf_joint = _marginalize(pmf, idx_joint)
            pmf_crvs = _marginalize(pmf_joint, idx_crvs)
            pmf_subrvs = [_marginalize(pmf_joint, idx) for idx in idx_subrvs] + [pmf_joint, pmf_crvs]

            pmf_ci = reduce(torch.mul, [p.pow(pw) for p, pw in zip(pmf_subrvs, power, strict=True)])

            # "Safe where" pattern: ensure log argument is always positive
            safe_pmf_ci = torch.where(pmf_joint > 0, pmf_ci, torch.ones_like(pmf_ci))
            ci = torch.sum(torch.where(pmf_joint > 0, pmf_joint * torch.log2(safe_pmf_ci), torch.zeros_like(pmf_joint)))

            return ci

        return self._maybe_compile(coinformation)

    def _total_correlation(self, rvs, crvs=None):
        """
        Compute the total correlation using PyTorch.

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

        def total_correlation(pmf):
            """
            Compute the specified total correlation.

            Parameters
            ----------
            pmf : torch.Tensor
                The joint probability distribution.

            Returns
            -------
            tc : torch.Tensor
                The total correlation.
            """
            pmf_joint = _marginalize(pmf, idx_joint)
            pmf_margs = [_marginalize(pmf_joint, marg) for marg in idx_margs]
            pmf_crvs = _marginalize(pmf_margs[0], idx_crvs)

            h_crvs = self._h(pmf_crvs.flatten())
            h_margs = sum(self._h(p.flatten()) for p in pmf_margs)
            h_joint = self._h(pmf_joint.flatten())

            tc = h_margs - h_joint - n * h_crvs

            return tc

        return self._maybe_compile(total_correlation)

    def _dual_total_correlation(self, rvs, crvs=None):
        """
        Compute the dual total correlation using PyTorch.

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

        def dual_total_correlation(pmf):
            """
            Compute the specified dual total correlation.

            Parameters
            ----------
            pmf : torch.Tensor
                The joint probability distribution.

            Returns
            -------
            dtc : torch.Tensor
                The dual total correlation.
            """
            pmf_joint = _marginalize(pmf, idx_joint)
            pmf_margs = [_marginalize(pmf_joint, marg) for marg in idx_margs]
            pmf_crvs = _marginalize(pmf_joint, idx_crvs)

            h_crvs = self._h(pmf_crvs)
            h_joint = self._h(pmf_joint) - h_crvs
            h_margs = [self._h(marg) - h_crvs for marg in pmf_margs]

            dtc = sum(h_margs) - n * h_joint

            return dtc

        return self._maybe_compile(dual_total_correlation)

    def _caekl_mutual_information(self, rvs, crvs=None):
        """
        Compute the CAEKL mutual information using PyTorch.

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

        def caekl_mutual_information(pmf):
            """
            Compute the specified CAEKL mutual information.

            Parameters
            ----------
            pmf : torch.Tensor
                The joint probability distribution.

            Returns
            -------
            caekl : torch.Tensor
                The CAEKL mutual information.
            """
            pmf_joint = _marginalize(pmf, idx_joint)
            pmf_parts = {p: _marginalize(pmf_joint, idx) for p, idx in idx_parts.items()}
            pmf_crvs = _marginalize(pmf_joint, idx_crvs)

            h_crvs = self._h(pmf_crvs)
            h_joint = self._h(pmf_joint) - h_crvs

            candidates = []
            for part, norm in zip(parts, part_norms, strict=True):
                h_parts = sum(self._h(pmf_parts[p]) - h_crvs for p in part)
                candidates.append((h_parts - h_joint) / norm)

            caekl = torch.min(torch.stack(candidates))

            return caekl

        return self._maybe_compile(caekl_mutual_information)

    def _maximum_correlation(self, rv_x, rv_y):
        """
        Compute the maximum correlation using PyTorch.

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

        def maximum_correlation(pmf):
            """
            Compute the specified maximum correlation.

            Parameters
            ----------
            pmf : torch.Tensor
                The joint probability distribution.

            Returns
            -------
            mc : torch.Tensor
                The maximum correlation.
            """
            pmf_xy = _marginalize(pmf, idx_xy, keepdim=False)
            pmf_x = _marginalize(pmf, idx_x, keepdim=False).unsqueeze(-1)
            pmf_y = _marginalize(pmf, idx_y, keepdim=False).unsqueeze(0)

            Q = pmf_xy / (torch.sqrt(pmf_x) * torch.sqrt(pmf_y))
            Q = torch.where(torch.isnan(Q), torch.zeros_like(Q), Q)

            # SVD to get singular values
            svd_result = torch.linalg.svd(Q)
            mc = svd_result.S[1]

            return mc

        return self._maybe_compile(maximum_correlation)

    def _conditional_maximum_correlation(self, rv_x, rv_y, rv_z):
        """
        Compute the conditional maximum correlation using PyTorch.

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

        def conditional_maximum_correlation(pmf):
            """
            Compute the specified conditional maximum correlation.

            Parameters
            ----------
            pmf : torch.Tensor
                The joint probability distribution.

            Returns
            -------
            cmc : torch.Tensor
                The conditional maximum correlation.
            """
            p_xyz = _marginalize(pmf, idx_xyz, keepdim=False)
            p_xz = _marginalize(pmf, idx_xz, keepdim=False).unsqueeze(1)
            p_yz = _marginalize(pmf, idx_yz, keepdim=False).unsqueeze(0)

            Q = torch.where(p_xyz > 0, p_xyz / torch.sqrt(p_xz * p_yz), torch.zeros_like(p_xyz))

            # Compute SVD for each z slice
            max_corr = torch.tensor(0.0, dtype=self._dtype, device=self._device)
            for i in range(Q.shape[-1]):
                m = Q[..., i]
                svd_result = torch.linalg.svd(m)
                if len(svd_result.S) > 1:
                    max_corr = torch.max(max_corr, svd_result.S[1])

            return max_corr

        return self._maybe_compile(conditional_maximum_correlation)

    def _total_variation(self, rv_x, rv_y):
        """
        Compute the total variation, TV[X||Y], using PyTorch.

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

        def total_variation(pmf):
            """
            Compute the specified total variation.

            Parameters
            ----------
            pmf : torch.Tensor
                The joint probability distribution.

            Returns
            -------
            tv : torch.Tensor
                The total variation.
            """
            pmf_xy = _marginalize(pmf, idx_xy)
            pmf_x = _marginalize(pmf_xy, idx_x, keepdim=False)
            pmf_y = _marginalize(pmf_xy, idx_y, keepdim=False)

            tv = torch.abs(pmf_x - pmf_y).sum() / 2

            return tv

        return self._maybe_compile(total_variation)

    ###########################################################################
    # Optimization methods.

    ###########################################################################
    # Augmented Lagrangian native optimization helpers

    def _al_minimize(
        self,
        x0,
        constraints=None,
        maxiter=None,
        al_outer_iters=50,
        al_inner_iters=100,
        mu_init=1.0,
        mu_factor=2.0,
        al_tol=1e-7,
    ):
        """
        Augmented Lagrangian minimization using torch.optim.LBFGS.

        Solves::

            min  f(x)
            s.t. g_i(x) = 0   (equality constraints)
                 h_j(x) >= 0  (inequality constraints)
                 0 <= x <= 1   (box bounds)

        by iteratively minimizing the augmented Lagrangian::

            L(x, lam, mu) = f(x)
                + sum_i lam_eq_i * g_i(x)   + (mu/2) * sum_i g_i(x)^2
                + sum_j lam_ineq_j * s_j(x) + (mu/2) * sum_j s_j(x)^2

        where s_j(x) = min(0, h_j(x) - lam_ineq_j / mu) for inequality
        constraints.

        Each outer iteration performs a single LBFGS solve (with up to
        ``al_inner_iters`` quasi-Newton steps) to approximately minimize the
        augmented Lagrangian, then updates dual variables and increases the
        penalty parameter.

        Parameters
        ----------
        x0 : array_like
            Initial point (numpy array, will be converted to tensor).
        constraints : list of dict, optional
            Constraints in scipy format. Defaults to ``self._torch_raw_constraints``.
        maxiter : int, optional
            Max LBFGS iterations per outer step (overrides ``al_inner_iters``).
        al_outer_iters : int
            Number of outer AL iterations.
        al_inner_iters : int
            Max LBFGS iterations within each outer step.
        mu_init : float
            Initial penalty parameter.
        mu_factor : float
            Multiplicative increase of mu each outer iteration.
        al_tol : float
            Convergence tolerance on max constraint violation.

        Returns
        -------
        result : OptimizeResult
            Scipy-compatible result object.
        """
        from scipy.optimize import OptimizeResult

        if constraints is None:
            constraints = self._torch_raw_constraints

        raw_objective = self._torch_raw_objective
        eq_funs = [c["fun"] for c in constraints if c["type"] == "eq"]
        ineq_funs = [c["fun"] for c in constraints if c["type"] == "ineq"]

        n_eq = len(eq_funs)
        n_ineq = len(ineq_funs)

        x = torch.tensor(np.asarray(x0, dtype=np.float64), dtype=self._dtype, device=self._device, requires_grad=True)

        # Dual variables
        lam_eq = torch.zeros(n_eq, dtype=self._dtype, device=self._device)
        lam_ineq = torch.zeros(n_ineq, dtype=self._dtype, device=self._device)
        mu = mu_init

        lbfgs_maxiter = maxiter or al_inner_iters

        best_x = x.detach().clone()
        best_obj = float("inf")
        max_viol = float("inf")

        for outer in range(al_outer_iters):
            _lam_eq = lam_eq.clone()
            _lam_ineq = lam_ineq.clone()
            _mu = mu

            # One LBFGS solve per outer iteration.  The LBFGS max_iter
            # controls how many quasi-Newton steps are taken; strong_wolfe
            # line search keeps each step well-conditioned.
            optimizer = torch.optim.LBFGS(
                [x],
                lr=1.0,
                max_iter=lbfgs_maxiter,
                line_search_fn="strong_wolfe",
                tolerance_grad=1e-10,
                tolerance_change=1e-12,
            )

            def closure(optimizer=optimizer, _lam_eq=_lam_eq, _mu=_mu, _lam_ineq=_lam_ineq):
                optimizer.zero_grad()
                # Project to box bounds before evaluating.  Use a very
                # tight epsilon -- the "safe where" pattern in the
                # information-theoretic functions handles exact zeros.
                with torch.no_grad():
                    x.data.clamp_(0.0, 1.0)

                obj = raw_objective(x)
                if not isinstance(obj, Tensor):
                    obj = torch.tensor(obj, dtype=self._dtype, device=self._device)
                loss = obj

                for i, cfun in enumerate(eq_funs):
                    ci = cfun(x)
                    if not isinstance(ci, Tensor):
                        ci = torch.tensor(ci, dtype=self._dtype, device=self._device)
                    loss = loss + _lam_eq[i] * ci + (_mu / 2.0) * ci**2

                for j, cfun in enumerate(ineq_funs):
                    hj = cfun(x)
                    if not isinstance(hj, Tensor):
                        hj = torch.tensor(hj, dtype=self._dtype, device=self._device)
                    slack = torch.min(
                        torch.zeros_like(hj),
                        hj - _lam_ineq[j] / _mu,
                    )
                    loss = loss + _lam_ineq[j] * slack + (_mu / 2.0) * slack**2

                loss.backward()
                return loss

            optimizer.step(closure)

            # Project to box bounds
            with torch.no_grad():
                x.data.clamp_(0.0, 1.0)

            # Update dual variables and check convergence
            with torch.no_grad():
                for i, cfun in enumerate(eq_funs):
                    ci = cfun(x)
                    if isinstance(ci, Tensor):
                        ci = ci.detach()
                    lam_eq[i] = lam_eq[i] + mu * float(ci)

                max_eq_viol = 0.0
                for cfun in eq_funs:
                    ci = cfun(x)
                    max_eq_viol = max(max_eq_viol, abs(float(ci.detach() if isinstance(ci, Tensor) else ci)))

                max_ineq_viol = 0.0
                for j, cfun in enumerate(ineq_funs):
                    hj = cfun(x)
                    if isinstance(hj, Tensor):
                        hj = hj.detach()
                    viol = max(0.0, -float(hj))
                    max_ineq_viol = max(max_ineq_viol, viol)
                    lam_ineq[j] = torch.clamp(lam_ineq[j] - mu * float(hj), min=0.0)

                max_viol = max(max_eq_viol, max_ineq_viol)

                obj_val = raw_objective(x)
                obj_val = float(obj_val.detach().cpu().item()) if isinstance(obj_val, Tensor) else float(obj_val)

                if obj_val < best_obj and max_viol < al_tol * 100:
                    best_obj = obj_val
                    best_x = x.detach().clone()

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

        x_opt = _to_numpy(best_x).astype(np.float64)
        fun_val = float(best_obj)
        success = max_viol < al_tol * 10

        return OptimizeResult(
            x=x_opt,
            fun=fun_val,
            success=success,
            message=f"AL: max_viol={max_viol:.2e}, mu={mu:.1f}",
        )

    def _optimize_shotgun_native(self, x0, niter, maxiter):
        """
        Multi-start optimization using the Augmented Lagrangian method.

        Parameters
        ----------
        x0 : ndarray
            Initial optimization vector.
        niter : int or None
            Number of random restarts.
        maxiter : int or None
            Maximum iterations per start.

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
            ic = self.construct_random_initial()
            starts.append(np.asarray(ic).flatten())

        if not starts:  # pragma: no cover
            return None

        raw_obj = self._torch_raw_objective
        results = []
        for i, s in enumerate(starts):
            logger.debug("Shotgun (torch AL): start {i}/{n}", i=i + 1, n=len(starts))
            res = self._al_minimize(s, maxiter=maxiter)
            results.append(res)

        if not results:  # pragma: no cover
            return None

        def _obj_val(r):
            v = raw_obj(r.x)
            if isinstance(v, Tensor):
                return float(v.detach().cpu().item())
            return float(np.asarray(v))

        return min(results, key=_obj_val)

    def _scipy_minimize_kwargs(self, x0, callback=False, maxiter=None):
        """
        Build ``minimizer_kwargs`` for scipy-based optimization (fallback).
        """
        icb = BasinHoppingInnerCallBack() if callback else None
        raw_objective = self._torch_raw_objective
        raw_constraints = self._torch_raw_constraints

        def _np_objective(x):
            val = raw_objective(x)
            if isinstance(val, Tensor):
                return float(val.detach().cpu().item())
            return float(np.asarray(val))

        wrapped_constraints = []
        for const in raw_constraints:
            _raw_fun = const["fun"]

            def _np_const_fun(x, _f=_raw_fun):
                val = _f(x)
                if isinstance(val, Tensor):
                    return float(val.detach().cpu().item())
                return float(np.asarray(val))

            wrapped_constraints.append({**const, "fun": _np_const_fun})

        minimizer_kwargs = {
            "bounds": [(0, 1)] * x0.size,
            "callback": icb,
            "constraints": wrapped_constraints,
            "options": {},
        }

        additions = deepcopy(self._additional_options)
        options = additions.pop("options", {})
        minimizer_kwargs["options"].update(options)
        minimizer_kwargs.update(additions)

        if maxiter:
            minimizer_kwargs["options"]["maxiter"] = maxiter

        self.objective = _np_objective
        return minimizer_kwargs, icb

    ###########################################################################
    # Native optimization dispatch

    def _optimization_backend_native(self, x0, niter, maxiter):
        """
        Dispatch to the native AL-based shotgun or basinhopping backend.

        Subclasses may override for different strategies (e.g. non-convex).
        """
        return self._optimize_shotgun_native(x0, niter, maxiter)

    ###########################################################################
    # Legacy helper methods (kept for compatibility)

    def _create_torch_objective(self, objective_func):
        """
        Create a wrapper that handles torch tensor conversion for the objective.

        Parameters
        ----------
        objective_func : callable
            The objective function that operates on torch tensors.

        Returns
        -------
        wrapper : callable
            A numpy-compatible wrapper.
        """

        def wrapper(x):
            x_tensor = _to_tensor(x, requires_grad=False, dtype=self._dtype).to(self._device)
            result = objective_func(x_tensor)
            if isinstance(result, Tensor):
                return float(result.item())
            return float(result)

        return wrapper

    def _create_torch_gradient(self, objective_func):
        """
        Create a gradient function using PyTorch autograd.

        Parameters
        ----------
        objective_func : callable
            The objective function that operates on torch tensors.

        Returns
        -------
        grad_func : callable
            A numpy-compatible gradient function.
        """

        def grad_func(x):
            x_tensor = _to_tensor(x, requires_grad=True, dtype=self._dtype).to(self._device)
            result = objective_func(x_tensor)
            if isinstance(result, Tensor):
                # Compute gradient
                result.backward()
                grad = x_tensor.grad
                if grad is not None:
                    return _to_numpy(grad)
            return np.zeros_like(x)

        return grad_func

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
        x0 = _to_numpy(x0).astype(np.float64)

        logger.info(
            "Starting PyTorch optimization: dim={dim}, niter={niter}, device={device}, dtype={dtype}",
            dim=x0.size,
            niter=niter,
            device=self._device,
            dtype=self._dtype,
        )

        # Store raw (torch-native) objective and constraints
        self._torch_raw_objective = self.objective
        self._torch_raw_constraints = list(self.constraints)

        # For very large problems, the AL method can be slow.
        # Fall back to scipy for those cases.
        _MAX_NATIVE_DIM = 200

        def _obj_val(r):
            """Evaluate raw objective on an OptimizeResult, returning a float."""
            v = self._torch_raw_objective(r.x)
            if isinstance(v, Tensor):
                return float(v.detach().cpu().item())
            return float(np.asarray(v))

        if TORCH_AVAILABLE and x0.size <= _MAX_NATIVE_DIM:
            # ---- native AL path (autograd gradients) ----
            logger.info("Using torch Augmented Lagrangian optimizer with autograd (dim={dim})", dim=x0.size)
            result = self._optimization_backend_native(x0, niter, maxiter)

            # Always also run scipy SLSQP and keep the better result.
            # The AL+LBFGS method is an interior method that can miss
            # boundary optima; scipy's SLSQP handles boundaries natively.
            minimizer_kwargs, icb = self._scipy_minimize_kwargs(x0, callback=callback, maxiter=maxiter)
            self._callback = BasinHoppingCallBack(minimizer_kwargs.get("constraints", {}), icb)
            result_scipy = self._optimization_backend(x0, minimizer_kwargs, niter)

            if result_scipy and (result is None or _obj_val(result_scipy) < _obj_val(result)):
                logger.info("scipy SLSQP found a better solution than AL")
                result = result_scipy
        else:
            # ---- scipy fallback path ----
            if x0.size > _MAX_NATIVE_DIM:
                logger.info(
                    "Problem dimension {dim} exceeds native limit {lim}; falling back to scipy SLSQP",
                    dim=x0.size,
                    lim=_MAX_NATIVE_DIM,
                )
            else:
                logger.info("Using scipy SLSQP fallback")
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

        obj_val = self._torch_raw_objective(self._optima)
        obj_val = float(obj_val.detach().cpu().item()) if isinstance(obj_val, Tensor) else float(np.asarray(obj_val))
        logger.info("PyTorch optimization complete: objective={obj}", obj=obj_val)

        return result

    def _optimize_shotgun(self, x0, minimizer_kwargs, niter):
        """
        Perform a non-convex optimization using a "shotgun" approach.

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

        # Use raw constraints if available, falling back to self.constraints
        raw_constraints = getattr(self, "_torch_raw_constraints", self.constraints)
        raw_objective = getattr(self, "_torch_raw_objective", self.objective)

        # Wrap for scipy (return numpy scalars)
        def _np_obj(x):
            val = raw_objective(x)
            if isinstance(val, Tensor):
                return float(val.detach().cpu().item())
            return float(np.asarray(val))

        wrapped_constraints = []
        for const in raw_constraints:
            _raw_fun = const["fun"]

            def _np_const_fun(x, _f=_raw_fun):
                val = _f(x)
                if isinstance(val, Tensor):
                    return float(val.detach().cpu().item())
                return float(np.asarray(val))

            wrapped_constraints.append({**const, "fun": _np_const_fun})

        minimizer_kwargs = {
            "bounds": Bounds(lb, ub, feasible),
            "tol": None,
            "callback": None,
            "constraints": wrapped_constraints,
        }

        if np.allclose(lb, ub):
            self._optima = x0
            return

        res = minimize(fun=_np_obj, x0=x0, **minimizer_kwargs)

        if res.success:
            logger.debug("Polishing successful: objective={obj}", obj=res.fun)
            self._optima = res.x.copy()

            if count < (res.x < cutoff).sum():
                self._polish(cutoff=cutoff)


class BaseConvexTorchOptimizer(BaseTorchOptimizer):
    """
    Implement convex optimization using PyTorch.
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


class BaseNonConvexTorchOptimizer(BaseTorchOptimizer):
    """
    Implement non-convex optimization using PyTorch.
    """

    _shotgun = False

    def _optimization_backend_native(self, x0, niter, maxiter):
        """
        Native AL backend for non-convex optimization.

        Uses a multi-start (shotgun) strategy with ``2 * niter`` random
        starting points plus the supplied ``x0``.  For the scipy fallback
        path, scipy's ``basinhopping`` is still used.
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
            The result of the optimization. Returns None if the optimization failed.
        """
        if niter is None:
            niter = self._default_hops

        logger.debug("Basin hopping (PyTorch): starting with niter={niter}", niter=niter)

        if self._shotgun:
            res_shotgun = self._optimize_shotgun(x0.copy(), minimizer_kwargs, self._shotgun)
            if res_shotgun:
                x0 = res_shotgun.x.copy()
        else:
            res_shotgun = None

        result = basinhopping(
            func=self.objective,
            x0=x0,
            minimizer_kwargs=minimizer_kwargs,
            niter=niter,
            accept_test=accept_test,
            callback=self._callback,
        )

        success, msg = basinhop_status(result)
        logger.info("Basin hopping (PyTorch) result: success={success}, message={msg}", success=success, msg=msg)
        if not success:  # pragma: no cover
            result = self._callback.minimum() or res_shotgun

        return result

    def _optimization_diffevo(self, x0, minimizer_kwargs, niter):  # pragma: no cover
        """
        Perform optimization using differential evolution.

        Parameters
        ----------
        x0 : np.ndarray, optional
            Initial optimization vector (unused by this method; kept for
            interface consistency).
        minimizer_kwargs : dict
            A dictionary of keyword arguments to pass to the optimizer.
        niter : int
            The population size for differential evolution.

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

        Parameters
        ----------
        x0 : ndarray, optional
            Initial optimization vector (unused by this method; kept for
            interface consistency).
        minimizer_kwargs : dict
            A dictionary of keyword arguments to pass to the optimizer.
        niter : int
            The number of iterations for the SHGO algorithm.

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

    _optimization_backend = _optimization_basinhopping


AuxVar = namedtuple("AuxVar", ["bases", "bound", "shape", "mask", "size"])


class BaseAuxVarTorchOptimizer(BaseNonConvexTorchOptimizer):
    """
    Base class that performs many methods related to optimizing auxiliary variables,
    using PyTorch for automatic differentiation.
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
            mask = torch.ones(shape, dtype=self._dtype, device=self._device) / bound
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
            self._full_slices.append(tuple(colon if j in relevant_vars | {var} else None for j in index))

        self._slices = []
        for i, (auxvar, var) in enumerate(zip(self._aux_vars, arvs, strict=True)):
            relevant_vars = auxvar.bases
            index = sorted(self._rvs | self._crvs | set(arvs[: i + 1]))
            self._slices.append(tuple(colon if j in relevant_vars | {var} else None for j in index))

    ###########################################################################
    # Constructing the joint distribution using PyTorch.

    def _construct_channels(self, x):
        """
        Construct the conditional distributions which produce the
        auxiliary variables.

        Parameters
        ----------
        x : torch.Tensor
            An optimization vector

        Yields
        ------
        channel : torch.Tensor
            A conditional distribution.
        """
        parts = [x[a:b] for a, b in self._parts]

        for part, auxvar in zip(parts, self._aux_vars, strict=True):
            channel = part.reshape(auxvar.shape)
            channel = channel / channel.sum(dim=-1, keepdim=True)
            channel = torch.where(torch.isnan(channel), auxvar.mask, channel)

            yield channel

    def construct_joint(self, x):
        """
        Construct the joint distribution using PyTorch.

        Parameters
        ----------
        x : torch.Tensor or np.ndarray
            An optimization vector.

        Returns
        -------
        joint : torch.Tensor
            The joint distribution resulting from the distribution passed
            in and the optimization vector.
        """
        x = _to_tensor(x, requires_grad=True, dtype=self._dtype).to(self._device)
        joint = self._pmf

        channels = list(self._construct_channels(x.clone()))

        for channel, slc in zip(channels, self._slices, strict=True):
            joint = joint.unsqueeze(-1) * channel[slc]

        return joint

    def _construct_joint_single(self, x):
        """
        Construct the joint distribution (optimized for single auxiliary variable).

        Parameters
        ----------
        x : torch.Tensor or np.ndarray
            An optimization vector.

        Returns
        -------
        joint : torch.Tensor
            The joint distribution resulting from the distribution passed
            in and the optimization vector.
        """
        x = _to_tensor(x, requires_grad=True, dtype=self._dtype).to(self._device)
        _, _, shape, mask, _ = self._aux_vars[0]
        channel = x.clone().reshape(shape)
        channel = channel / channel.sum(dim=-1, keepdim=True)
        channel = torch.where(torch.isnan(channel), mask, channel)

        joint = self._pmf.unsqueeze(-1) * channel[self._slices[0]]

        return joint

    def construct_full_joint(self, x):
        """
        Construct the full joint distribution using PyTorch.

        Parameters
        ----------
        x : torch.Tensor or np.ndarray
            An optimization vector.

        Returns
        -------
        joint : torch.Tensor
            The joint distribution resulting from the distribution passed
            in and the optimization vector.
        """
        x = _to_tensor(x, requires_grad=True, dtype=self._dtype).to(self._device)
        joint = self._full_pmf

        channels = list(self._construct_channels(x.clone()))

        for channel, slc in zip(channels, self._full_slices, strict=True):
            joint = joint.unsqueeze(-1) * channel[slc]

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
            "".join(flatten(alphabets))
            for bound in self._aux_bounds:
                alphabets += [(digits + ascii_letters)[:bound]]
            string = True
        except TypeError:
            for bound in self._aux_bounds:
                alphabets += [list(range(bound))]
            string = False

        joint = _to_numpy(self.construct_full_joint(x))
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
            Constraint which ensures that the auxiliary variables are a function
            of the random variables.

            Parameters
            ----------
            x : torch.Tensor or np.ndarray
                An optimization vector.

            Returns
            -------
            val : float
                The squared entropy (should be 0 for deterministic).
            """
            pmf = self.construct_joint(x)
            result = entropy(pmf) ** 2
            if isinstance(result, Tensor):
                return result
            return result

        return self._maybe_compile(constraint)

    ###########################################################################
    # Channel capacity methods

    def _channel_capacity(self, x):  # pragma: no cover
        """
        Compute the channel capacity of the mapping z -> z_bar.

        Parameters
        ----------
        x : torch.Tensor or np.ndarray
            An optimization vector.

        Returns
        -------
        ccs : [float]
            The channel capacity of each auxiliary variable.
        """
        x = _to_tensor(x, dtype=self._dtype).to(self._device)
        ccs = []
        for channel in self._construct_channels(x):
            ccs.append(channel_capacity(_to_numpy(channel))[0])
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
        """
        logger.debug("Post-processing (PyTorch): style={style}, minmax={minmax}", style=style, minmax=minmax)
        entropy = self._entropy(self._arvs)

        sign = +1 if minmax == "min" else -1

        def objective_entropy(x):
            """Post-process the entropy."""
            ent = entropy(self.construct_joint(x))
            return sign * ent

        def objective_channelcapacity(x):
            """Post-process the channel capacity."""
            cc = sum(self._channel_capacity(x))
            return sign * cc

        if style == "channel":
            objective = objective_channelcapacity
        elif style == "entropy":
            objective = self._maybe_compile(objective_entropy)
        else:
            msg = f"Style {style} is not understood."
            raise OptimizationException(msg)

        true_objective = self.objective(self._optima)

        def constraint_match_objective(x):
            """
            Constraint to ensure that the new solution is not worse than that
            found before.
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

        self.optimize(x0=self._optima.copy(), niter=niter, maxiter=maxiter)

        # and remove them again.
        self.constraints = self.constraints[:-1]
        if not self.constraints:
            del self.constraints

        self.objective = self.__old_objective
        del self.__old_objective


# Convenience functions
def is_torch_available():
    """
    Check if PyTorch is available for use.

    Returns
    -------
    available : bool
        True if PyTorch is installed and can be imported.
    """
    return TORCH_AVAILABLE


def is_cuda_available():
    """
    Check if CUDA is available for GPU acceleration.

    Returns
    -------
    available : bool
        True if CUDA is available.
    """
    if not TORCH_AVAILABLE:
        return False
    return torch.cuda.is_available()


def is_mps_available():
    """
    Check if MPS (Apple Silicon) is available for GPU acceleration.

    Returns
    -------
    available : bool
        True if MPS is available.
    """
    if not TORCH_AVAILABLE:
        return False
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def get_best_device():
    """
    Get the best available device for computation.

    Returns
    -------
    device : str
        The device string ('cuda', 'mps', or 'cpu').
    """
    if is_cuda_available():
        return "cuda"
    elif is_mps_available():
        return "mps"
    return "cpu"
