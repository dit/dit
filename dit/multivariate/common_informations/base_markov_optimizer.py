"""
Abstract base classes for Markov variable optimizers.

Supports pluggable optimization backends ('numpy', 'jax', 'torch') via the
``backend`` parameter. The default backend is 'numpy', which uses
``BaseAuxVarOptimizer`` from :mod:`dit.algorithms.optimization`. Selecting
'jax' or 'torch' substitutes the corresponding autodiff-enabled optimizer.
"""

from abc import abstractmethod

import numpy as np

from ...algorithms import BaseAuxVarOptimizer
from ...utils import unitful
from ..dual_total_correlation import dual_total_correlation
from ..entropy import entropy


__all__ = (
    'MarkovVarOptimizer',
    'MinimizingMarkovVarOptimizer',
    'make_markov_var_optimizer',
)


# ── Backend resolution ───────────────────────────────────────────────────

def _get_base_class(backend='numpy'):
    """
    Return the appropriate ``BaseAuxVarOptimizer`` class for *backend*.

    Parameters
    ----------
    backend : str
        One of ``'numpy'``, ``'jax'``, ``'torch'``.

    Returns
    -------
    cls : type
        The base auxiliary-variable optimizer class.

    Raises
    ------
    ValueError
        If *backend* is not recognised.
    """
    if backend == 'numpy':
        return BaseAuxVarOptimizer
    elif backend == 'jax':
        from ...algorithms.optimization_jax import BaseAuxVarJaxOptimizer
        return BaseAuxVarJaxOptimizer
    elif backend == 'torch':
        from ...algorithms.optimization_torch import BaseAuxVarTorchOptimizer
        return BaseAuxVarTorchOptimizer
    else:
        raise ValueError(
            f"Unknown backend: {backend!r}. "
            f"Choose from 'numpy', 'jax', 'torch'."
        )


_backend_class_cache = {}


def _make_backend_subclass(cls, backend):
    """
    Return a version of *cls* whose optimizer base uses *backend*.

    For ``backend='numpy'`` this is a no-op (returns *cls* unchanged).
    For other backends a new class is synthesised that combines the
    mixin logic from *cls*'s MRO with the requested backend base.

    Parameters
    ----------
    cls : type
        A concrete subclass of ``MarkovVarOptimizer`` (or of
        ``MinimizingMarkovVarOptimizer``).
    backend : str
        One of ``'numpy'``, ``'jax'``, ``'torch'``.

    Returns
    -------
    new_cls : type
        The optimizer class with the requested backend.
    """
    if backend == 'numpy':
        return cls

    cache_key = (cls, backend)
    if cache_key in _backend_class_cache:
        return _backend_class_cache[cache_key]

    Base = _get_base_class(backend)

    # Collect mixin classes from the MRO (order preserved).
    mixins = []
    for klass in cls.__mro__:
        if klass.__name__.endswith('Mixin'):
            mixins.append(klass)

    # Attributes defined directly on the concrete class (method overrides,
    # class variables like ``name`` / ``description``, etc.).
    attrs = {
        k: v for k, v in cls.__dict__.items()
        if not (k.startswith('__') and k.endswith('__'))
    }

    new_cls = type(cls.__name__, tuple(mixins) + (Base,), attrs)
    _backend_class_cache[cache_key] = new_cls
    return new_cls


def make_markov_var_optimizer(backend='numpy'):
    """
    Return a ``MarkovVarOptimizer`` class backed by *backend*.

    Parameters
    ----------
    backend : str
        One of ``'numpy'``, ``'jax'``, ``'torch'``.

    Returns
    -------
    cls : type
        A ``MarkovVarOptimizer`` subclass that uses the requested backend.
    """
    return _make_backend_subclass(MarkovVarOptimizer, backend)


# ── Mixin classes (backend-agnostic logic) ───────────────────────────────

class MarkovVarMixin:
    """
    Mixin containing all Markov-variable optimiser logic.

    Must be composed with a ``BaseAuxVarOptimizer``-compatible base class
    (e.g. via multiple inheritance).
    """

    name = ""
    description = ""

    def __init__(self, dist, rvs=None, crvs=None, bound=None, rv_mode=None):
        """
        Initialize the optimizer.

        Parameters
        ----------
        dist : Distribution
            The distribution to compute the auxiliary Markov variable, W, for.
        rvs : list, None
            A list of lists. Each inner list specifies the indexes of the random
            variables to render conditionally independent. If None, then all
            random variables are used, which is equivalent to passing
            ``rvs=dist.rvs``.
        crvs : list, None
            A single list of indexes specifying the random variables to
            condition on. If None, then no variables are conditioned on.
        bound : int, None
            Place an artificial bound on the size of W. If None, the
            theoretical bound from :meth:`compute_bound` is used.
        rv_mode : str, None
            Specifies how to interpret ``rvs`` and ``crvs``. Valid options are:
            ``{'indices', 'names'}``. If equal to ``'indices'``, then the
            elements of ``crvs`` and ``rvs`` are interpreted as random variable
            indices. If equal to ``'names'``, the elements are interpreted as
            random variable names. If ``None``, then the value of
            ``dist._rv_mode`` is consulted, which defaults to ``'indices'``.
        """
        super().__init__(dist, rvs=rvs, crvs=crvs, rv_mode=rv_mode)

        theoretical_bound = self.compute_bound()
        bound = min(bound, theoretical_bound) if bound else theoretical_bound

        rv_bounds = self._shape[1:-1]
        self._pmf_to_match = self._pmf.copy()

        # remove the rvs other than the first, they need to be generated by W
        # in order to satisfy the markov criteria:
        self._pmf = self._pmf.sum(axis=tuple(range(1, len(self._shape) - 1)))
        self._shape = self._pmf.shape
        self._all_vars = {0, 1}

        self._full_pmf = self._full_pmf.sum(axis=tuple(range(self._n + 1, len(self._full_shape) - 1)))
        self._full_shape = self._full_pmf.shape
        self._full_vars = tuple(range(self._n + 2))

        # back up where the rvs and crvs are, they need to be reflect
        # the above removals for the sake of adding auxvars:
        self.__rvs, self._rvs = self._rvs, {0}
        self.__crvs, self._crvs = self._crvs, {1}

        self._construct_auxvars([({0, 1}, bound)]
                                + [({1, 2}, s) for s in rv_bounds])

        # put rvs, crvs back:
        self._rvs = self.__rvs
        self._crvs = self.__crvs
        del self.__rvs
        del self.__crvs

        self._W = {1 + len(self._aux_vars)}

        # The constraint that the joint doesn't change.
        self.constraints += [{'type': 'eq',
                              'fun': self.constraint_match_joint,
                              },
                             ]

        self._default_hops = 5

        self._additional_options = {'options': {'maxiter': 1000,
                                                'ftol': 1e-6,
                                                'eps': 1.4901161193847656e-9,
                                                }
                                    }

    @abstractmethod
    def compute_bound(self):
        """
        Return a bound on the cardinality of the auxiliary variable.

        Returns
        -------
        bound : int
            The bound on the size of W.
        """
        pass

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
        joint = super().construct_joint(x)
        joint = np.moveaxis(joint, 1, -1)  # move crvs
        joint = np.moveaxis(joint, 1, -1)  # move W

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
        joint = super().construct_full_joint(x)
        joint = np.moveaxis(joint, self._n + 1, -1)  # move crvs
        joint = np.moveaxis(joint, self._n + 1, -1)  # move W
        return joint

    def constraint_match_joint(self, x):
        """
        Ensure that the joint distribution represented by the optimization
        vector matches that of the distribution.

        Parameters
        ----------
        x : np.ndarray
            An optimization vector.

        Returns
        -------
        delta : float
            The constraint residual; zero when the joint matches.
        """
        joint = self.construct_joint(x)
        joint = joint.sum(axis=-1)  # marginalize out w

        delta = (100 * (joint - self._pmf_to_match)**2).sum()

        return delta

    @classmethod
    def functional(cls):
        """
        Construct a functional form of the optimizer.

        Returns
        -------
        common_info : callable
            A function that computes the common information for a given
            distribution.
        """
        @unitful
        def common_info(dist, rvs=None, crvs=None, niter=None, maxiter=1000,
                        polish=1e-6, bound=None, rv_mode=None,
                        backend='numpy'):
            dtc = dual_total_correlation(dist, rvs, crvs, rv_mode)
            ent = entropy(dist, rvs, crvs, rv_mode)
            if np.isclose(dtc, ent):
                # Common informations are bound between the dual total
                # correlation and the joint entropy. Therefore, if the two
                # are equal, the common information is equal to them as well.
                return dtc

            actual_cls = _make_backend_subclass(cls, backend)
            ci = actual_cls(dist, rvs, crvs, bound, rv_mode)
            ci.optimize(niter=niter, maxiter=maxiter, polish=polish)
            return ci.objective(ci._optima)

        common_info.__doc__ = \
        """
        Computes the {name} common information, {description}.

        Parameters
        ----------
        dist : Distribution
            The distribution for which the {name} common information will be
            computed.
        rvs : list, None
            A list of lists. Each inner list specifies the indexes of the random
            variables used to calculate the {name} common information. If None,
            then it is calculated over all random variables, which is equivalent to
            passing ``rvs=dist.rvs``.
        crvs : list, None
            A single list of indexes specifying the random variables to condition
            on. If None, then no variables are conditioned on.
        niter : int > 0
            Number of basin hoppings to perform during the optimization.
        maxiter : int > 0
            The number of iterations of the optimization subroutine to perform.
        polish : False, float
            Whether to polish the result or not. If a float, this will perform a
            second optimization seeded with the result of the first, but with
            smaller tolerances and probabilities below polish set to 0. If
            False, don't polish.
        bound : int
            Bound the size of the Markov variable.
        rv_mode : str, None
            Specifies how to interpret ``rvs`` and ``crvs``. Valid options are:
            {{'indices', 'names'}}. If equal to ``'indices'``, then the elements
            of ``crvs`` and ``rvs`` are interpreted as random variable indices.
            If equal to ``'names'``, the elements are interpreted as random
            variable names. If ``None``, then the value of ``dist._rv_mode`` is
            consulted, which defaults to ``'indices'``.
        backend : str
            The optimization backend to use. One of ``'numpy'`` (default),
            ``'jax'``, or ``'torch'``.

        Returns
        -------
        ci : float
            The {name} common information.
        """.format(name=cls.name, description=cls.description)

        return common_info


class MinimizingMarkovVarMixin:
    """
    Mixin that adds auxiliary-variable minimisation on top of
    :class:`MarkovVarMixin`.
    """

    def optimize(self, x0=None, niter=None, maxiter=None, polish=1e-6,
                 callback=False, minimize=True, min_niter=15):
        """
        Run the optimization, optionally with auxiliary variable minimization.

        Parameters
        ----------
        x0 : np.ndarray, None
            The vector to initialize the optimization with. If None, a random
            vector is used.
        niter : int
            The number of times to basin hop in the optimization.
        maxiter : int
            The number of inner optimizer steps to perform.
        polish : False, float
            Whether to polish the result or not. If a float, this will perform a
            second optimization seeded with the result of the first, but with
            smaller tolerances and probabilities below polish set to 0. If
            False, don't polish.
        callback : bool
            Whether to utilize a callback or not.
        minimize : bool
            Whether to minimize the auxiliary variable or not.
        min_niter : int
            The number of basin hops to make during the minimization of the
            common variable.
        """
        # call the normal optimizer
        super().optimize(x0=x0,
                         niter=niter,
                         maxiter=maxiter,
                         polish=False,
                         callback=callback)
        if minimize:
            # minimize the entropy of W
            self._post_process(style='entropy', minmax='min', niter=min_niter, maxiter=maxiter)
        if polish:
            self._polish(cutoff=polish)


# ── Backward-compatible composed classes (numpy backend) ─────────────────

class MarkovVarOptimizer(MarkovVarMixin, BaseAuxVarOptimizer):
    """
    Abstract base class for constructing auxiliary variables which render a set
    of variables conditionally independent.

    Uses the default NumPy / SciPy optimization backend.  Pass
    ``backend='jax'`` or ``backend='torch'`` to :meth:`functional` (or use
    :func:`make_markov_var_optimizer`) for alternative backends.
    """
    pass


class MinimizingMarkovVarOptimizer(MinimizingMarkovVarMixin, MarkovVarOptimizer):  # pragma: no cover
    """
    Abstract base class for an optimizer which additionally minimizes the size
    of the auxiliary variable.

    Uses the default NumPy / SciPy optimization backend.
    """
    pass
