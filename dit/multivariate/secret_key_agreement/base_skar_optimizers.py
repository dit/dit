"""
Base classes for secret key agreement rate optimizers.

Supports pluggable optimization backends (``'numpy'``, ``'jax'``, ``'torch'``)
via the ``backend`` parameter on :meth:`functional` and on the standalone
public functions.

Each public base class is composed of a **Mixin** (containing all the
problem-specific logic and ``super()`` calls) and a concrete optimizer base.
This allows :func:`_make_backend_subclass` to substitute the optimizer base
while preserving the mixin logic — ``super()`` inside the mixin resolves to
whichever backend base is composed with it.
"""

from abc import abstractmethod

import numpy as np

from ...algorithms import BaseAuxVarOptimizer
from ...exceptions import ditException
from ...math import prod
from ...npdist import Distribution
from ...utils import unitful
from .._backend import _make_backend_subclass

__all__ = (
    'BaseIntrinsicMutualInformation',
    'BaseMoreIntrinsicMutualInformation',
    'BaseOneWaySKAR',
)


# ── Mixin classes (backend-agnostic logic) ────────────────────────────────

class OneWaySKARMixin:
    """
    Mixin containing one-way SKAR optimizer logic.

    Must be composed with a ``BaseAuxVarOptimizer``-compatible base class.
    """

    def __init__(self, dist, rv_x=None, rv_y=None, rv_z=None, rv_mode=None, bound_u=None, bound_v=None):
        """
        Initialize the optimizer.

        Parameters
        ----------
        dist : Distribution
            The distribution to compute the intrinsic mutual information of.
        rv_x : iterable
            The variables to consider `X`.
        rv_y : iterable
            The variables to consider `Y`.
        rv_z : iterable
            The variables to consider `Z`.
        rv_mode : str, None
            Specifies how to interpret `rvs` and `crvs`. Valid options are:
            {'indices', 'names'}. If equal to 'indices', then the elements of
            `crvs` and `rvs` are interpreted as random variable indices. If
            equal to 'names', the the elements are interpreted as random
            variable names. If `None`, then the value of `dist._rv_mode` is
            consulted, which defaults to 'indices'.
        bound_u : int, None
            Specifies a bound on the size of the auxiliary random variable. If
            None, then the theoretical bound is used.
        bound_v : int, None
            Specifies a bound on the size of the auxiliary random variable. If
            None, then the theoretical bound is used.
        """
        super().__init__(dist, [rv_x, rv_y], rv_z, rv_mode=rv_mode)

        theoretical_bound_u = self._get_u_bound()
        bound_u = min(bound_u, theoretical_bound_u) if bound_u else theoretical_bound_u

        theoretical_bound_v = self._get_v_bound()
        bound_v = min(bound_v, theoretical_bound_v) if bound_v else theoretical_bound_v

        self._construct_auxvars([({0}, bound_u), ({3}, bound_v)])
        self._x = {0}
        self._y = {1}
        self._z = {2}
        self._u = {3}
        self._v = {4}
        self._default_hops *= 2

    @abstractmethod
    def _get_u_bound(self):
        """
        Bound of |U|

        Returns
        -------
        bound : int
            The bound
        """
        pass

    @abstractmethod
    def _get_v_bound(self):
        """
        Bound of |V|

        Returns
        -------
        bound : int
            The bound
        """
        pass


class IntrinsicMIMixin:
    """
    Mixin containing intrinsic mutual information optimizer logic.

    Must be composed with a ``BaseAuxVarOptimizer``-compatible base class.
    """

    name = ""

    def __init__(self, dist, rvs=None, crvs=None, bound=None, rv_mode=None):
        """
        Initialize the optimizer.

        Parameters
        ----------
        dist : Distribution
            The distribution to compute the intrinsic mutual information of.
        rvs : list, None
            A list of lists. Each inner list specifies the indexes of the random
            variables used to calculate the intrinsic mutual information. If None,
            then it is calculated over all random variables, which is equivalent
            to passing `rvs=dist.rvs`.
        crvs : list
            A single list of indexes specifying the random variables to
            condition on.
        bound : int, None
            Specifies a bound on the size of the auxiliary random variable. If None,
            then the theoretical bound is used.
        rv_mode : str, None
            Specifies how to interpret `rvs` and `crvs`. Valid options are:
            {'indices', 'names'}. If equal to 'indices', then the elements of
            `crvs` and `rvs` are interpreted as random variable indices. If
            equal to 'names', the the elements are interpreted as random
            variable names. If `None`, then the value of `dist._rv_mode` is
            consulted, which defaults to 'indices'.
        """
        if not crvs:
            msg = "Intrinsic mutual informations require a conditional variable."
            raise ditException(msg)

        super().__init__(dist, rvs, crvs, rv_mode=rv_mode)

        crv_index = len(self._shape) - 1
        crv_size = self._shape[crv_index]
        bound = min([bound, crv_size]) if bound is not None else crv_size

        self._construct_auxvars([({crv_index}, bound)])

    def optimize(self, *args, **kwargs):
        """
        Perform the optimization.

        Parameters
        ----------
        x0 : np.ndarray, None
            Initial optimization vector. If None, use a random vector.
        niter : int, None
            The number of basin hops to perform while optimizing. If None,
            hop a number of times equal to the dimension of the conditioning
            variable(s).
        """
        result = super().optimize(*args, **kwargs)

        # test against known upper bounds as well, in case space wasn't well sampled.
        options = [self.construct_constant_initial(),  # mutual information
                   self.construct_copy_initial(),      # conditional mutual information
                   result.x,                           # found optima
                  ]

        self._optima = min(options, key=lambda opt: self.objective(opt))

    @classmethod
    def functional(cls):
        """
        Construct a functional form of the optimizer.
        """
        @unitful
        def intrinsic(dist, rvs=None, crvs=None, niter=None, bound=None,
                      rv_mode=None, backend='numpy'):
            actual_cls = _make_backend_subclass(cls, backend)
            opt = actual_cls(dist, rvs=rvs, crvs=crvs, rv_mode=rv_mode, bound=bound)
            opt.optimize(niter=niter)
            val = opt.objective(opt._optima)
            return float(val.detach().cpu().item()) if hasattr(val, 'detach') else float(val)

        intrinsic.__doc__ = \
        f"""
        Compute the intrinsic {cls.name}.

        Parameters
        ----------
        dist : Distribution
            The distribution to compute the intrinsic {cls.name} of.
        rvs : list, None
            A list of lists. Each inner list specifies the indexes of the random
            variables used to calculate the intrinsic {cls.name}. If None,
            then it is calculated over all random variables, which is equivalent
            to passing `rvs=dist.rvs`.
        crvs : list
            A single list of indexes specifying the random variables to
            condition on.
        niter : int
            The number of optimization iterations to perform.
        bound : int, None
            Bound on the size of the auxiliary variable. If None, use the
            theoretical bound.
        rv_mode : str, None
            Specifies how to interpret `rvs` and `crvs`. Valid options are:
            {{'indices', 'names'}}. If equal to 'indices', then the elements of
            `crvs` and `rvs` are interpreted as random variable indices. If
            equal to 'names', the the elements are interpreted as random
            variable names. If `None`, then the value of `dist._rv_mode` is
            consulted, which defaults to 'indices'.
        backend : str
            The optimization backend. One of ``'numpy'`` (default),
            ``'jax'``, or ``'torch'``.
        """

        return intrinsic


class MoreIntrinsicMIMixin:
    """
    Mixin containing reduced/minimal intrinsic MI optimizer logic.

    Must be composed with a ``BaseAuxVarOptimizer``-compatible base class.
    """

    name = ""

    def __init__(self, dist, rvs=None, crvs=None, bound=None, rv_mode=None):
        """
        Initialize the optimizer.

        Parameters
        ----------
        dist : Distribution
            The distribution to compute the intrinsic mutual information of.
        rvs : list, None
            A list of lists. Each inner list specifies the indexes of the random
            variables used to calculate the intrinsic mutual information. If None,
            then it is calculated over all random variables, which is equivalent
            to passing `rvs=dist.rvs`.
        crvs : list
            A single list of indexes specifying the random variables to
            condition on.
        bound : int, None
            Specifies a bound on the size of the auxiliary random variable. If None,
            then the theoretical bound is used.
        rv_mode : str, None
            Specifies how to interpret `rvs` and `crvs`. Valid options are:
            {'indices', 'names'}. If equal to 'indices', then the elements of
            `crvs` and `rvs` are interpreted as random variable indices. If
            equal to 'names', the the elements are interpreted as random
            variable names. If `None`, then the value of `dist._rv_mode` is
            consulted, which defaults to 'indices'.
        """
        if not crvs:
            msg = "Intrinsic mutual informations require a conditional variable."
            raise ditException(msg)

        super().__init__(dist, rvs, crvs, rv_mode=rv_mode)

        theoretical_bound = prod(self._shape)
        bound = min([bound, theoretical_bound]) if bound else theoretical_bound

        self._construct_auxvars([(self._rvs | self._crvs, bound)])

    @abstractmethod
    def measure(self, rvs, crvs):
        """
        Abstract method for computing the appropriate measure of generalized
        mutual information.

        Parameters
        ----------
        rvs : set
            The set of random variables.
        crvs : set
            The set of conditional random variables.

        Returns
        -------
        gmi : func
            The generalized mutual information.
        """
        pass

    @classmethod
    def functional(cls):
        """
        Construct a functional form of the optimizer.
        """
        @unitful
        def intrinsic(dist, rvs=None, crvs=None, niter=None, bounds=None,
                      rv_mode=None, backend='numpy'):
            if bounds is None:
                bounds = (2, 3, 4, None)

            actual_cls = _make_backend_subclass(cls, backend)
            candidates = []
            for bound in bounds:
                opt = actual_cls(dist, rvs=rvs, crvs=crvs, bound=bound, rv_mode=rv_mode)
                opt.optimize(niter=niter)
                val = opt.objective(opt._optima)
                val = float(val.detach().cpu().item()) if hasattr(val, 'detach') else float(val)
                candidates.append(val)
            return min(candidates)

        intrinsic.__doc__ = \
            f"""
            Compute the {cls.style} intrinsic {cls.name}.

            Parameters
            ----------
            dist : Distribution
                The distribution to compute the {cls.style} intrinsic {cls.name} of.
            rvs : list, None
                A list of lists. Each inner list specifies the indexes of the random
                variables used to calculate the intrinsic {cls.name}. If None,
                then it is calculated over all random variables, which is equivalent
                to passing `rvs=dist.rvs`.
            crvs : list
                A single list of indexes specifying the random variables to
                condition on.
            niter : int
                The number of optimization iterations to perform.
            bounds : [int], None
                Bounds on the size of the auxiliary variable. If None, use the
                theoretical bound. This is used to better sample smaller subspaces.
            rv_mode : str, None
                Specifies how to interpret `rvs` and `crvs`. Valid options are:
                {{'indices', 'names'}}. If equal to 'indices', then the elements of
                `crvs` and `rvs` are interpreted as random variable indices. If
                equal to 'names', the the elements are interpreted as random
                variable names. If `None`, then the value of `dist._rv_mode` is
                consulted, which defaults to 'indices'.
            backend : str
                The optimization backend. One of ``'numpy'`` (default),
                ``'jax'``, or ``'torch'``.
            """

        return intrinsic


class InnerTwoPartIMIMixin:
    """
    Mixin containing inner two-part intrinsic MI optimizer logic.

    Must be composed with a ``BaseAuxVarOptimizer``-compatible base class.
    """

    name = ""

    def __init__(self, dist, rvs=None, crvs=None, j=None, bound_u=None, bound_v=None, rv_mode=None):
        """
        Initialize the optimizer.

        Parameters
        ----------
        dist : Distribution
            The distribution to compute the intrinsic mutual information of.
        rvs : list, None
            A list of lists. Each inner list specifies the indexes of the random
            variables used to calculate the intrinsic mutual information. If
            None, then it is calculated over all random variables, which is
            equivalent to passing `rvs=dist.rvs`.
        crvs : list
            A single list of indexes specifying the random variables to
            condition on.
        j : list
            A list with a single index specifying the random variable to
            consider as J.
        bound_u : int, None
            Specifies a bound on the size of the U auxiliary random variable. If
            None, then the theoretical bound is used.
        bound_v : int, None
            Specifies a bound on the size of the V auxiliary random variable. If
            None, then the theoretical bound is used.
        rv_mode : str, None
            Specifies how to interpret `rvs` and `crvs`. Valid options are:
            {'indices', 'names'}. If equal to 'indices', then the elements of
            `crvs` and `rvs` are interpreted as random variable indices. If
            equal to 'names', the the elements are interpreted as random
            variable names. If `None`, then the value of `dist._rv_mode` is
            consulted, which defaults to 'indices'.
        """
        if not crvs:
            msg = "Intrinsic mutual informations require a conditional variable."
            raise ditException(msg)

        super().__init__(dist, rvs + [j], crvs, rv_mode=rv_mode)

        theoretical_bound_u = prod(self._shape[rv] for rv in self._rvs)
        bound_u = min([bound_u, theoretical_bound_u]) if bound_u else theoretical_bound_u

        theoretical_bound_v = prod(self._shape[rv] for rv in self._rvs)**2
        bound_v = min([bound_v, theoretical_bound_v]) if bound_v else theoretical_bound_v

        self._construct_auxvars([(self._rvs, bound_u),
                                 ({len(self._shape)}, bound_v),
                                 ])
        idx = min(self._arvs)
        self._j = {max(self._rvs)}
        self._u = {idx}
        self._v = {idx + 1}

    def _objective(self):
        """
        Maximize I[X:Y|J] + I[U:J|V] - I[U:Z|V], or its multivariate analog.

        Returns
        -------
        obj : func
            The objective function.
        """
        cmi1 = self._conditional_mutual_information(self._u, self._j, self._v)
        cmi2 = self._conditional_mutual_information(self._u, self._crvs, self._v)

        def objective(self, x):
            """
            Compute :math:`I[X:Y|J] + I[U:J|V] - I[U:Z|V]`

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

            # I[U:J|V]
            b = cmi1(pmf)

            # I[U:Z|V]
            c = cmi2(pmf)

            return -(b - c)

        return objective


class TwoPartIMIMixin:
    """
    Mixin containing two-part intrinsic MI optimizer logic.

    Must be composed with a ``BaseAuxVarOptimizer``-compatible base class.
    """

    name = ""

    def __init__(self, dist, rvs=None, crvs=None, bound_j=None, bound_u=None, bound_v=None, rv_mode=None):
        """
        Initialize the optimizer.

        Parameters
        ----------
        dist : Distribution
            The distribution to compute the intrinsic mutual information of.
        rvs : list, None
            A list of lists. Each inner list specifies the indexes of the random
            variables used to calculate the intrinsic mutual information. If
            None, then it is calculated over all random variables, which is
            equivalent to passing `rvs=dist.rvs`.
        crvs : list
            A single list of indexes specifying the random variables to
            condition on.
        bound_j : int, None
            Specifies a bound on the size of the J auxiliary random variable. If
            None, then the theoretical bound is used.
        bound_u : int, None
            Specifies a bound on the size of the U auxiliary random variable. If
            None, then the theoretical bound is used.
        bound_v : int, None
            Specifies a bound on the size of the V auxiliary random variable. If
            None, then the theoretical bound is used.
        rv_mode : str, None
            Specifies how to interpret `rvs` and `crvs`. Valid options are:
            {'indices', 'names'}. If equal to 'indices', then the elements of
            `crvs` and `rvs` are interpreted as random variable indices. If
            equal to 'names', the the elements are interpreted as random
            variable names. If `None`, then the value of `dist._rv_mode` is
            consulted, which defaults to 'indices'.
        """
        if not crvs:
            msg = "Intrinsic mutual informations require a conditional variable."
            raise ditException(msg)

        super().__init__(dist, rvs, crvs, rv_mode=rv_mode)

        theoretical_bound_j = prod(self._shape)
        bound_j = min([bound_j, theoretical_bound_j]) if bound_j else theoretical_bound_j

        self._construct_auxvars([(self._rvs | self._crvs, bound_j)])
        self._j = self._arvs

        self._bound_u = bound_u
        self._bound_v = bound_v

    def _objective(self):
        """
        Mimimize :math:`max(I[X:Y|J] + I[U:J|V] - I[U:Z|V])`, or its
        multivariate analog.

        Returns
        -------
        obj : func
            The objective function.
        """
        mmi = self.measure(self._rvs, self._j)

        def objective(self, x):
            """
            Compute max(I[X:Y|J] + I[U:J|V] - I[U:Z|V])

            Parameters
            ----------
            x : np.ndarray
                An optimization vector.

            Returns
            -------
            obj : float
                The value of the objective.

            TODO
            ----
            Save the optimal inner, so that full achieving joint can be constructed.
            """
            joint = self.construct_joint(x)
            joint_np = joint.detach().cpu().numpy() if hasattr(joint, 'detach') else np.asarray(joint)
            outcomes, pmf = zip(*[(o, p) for o, p in np.ndenumerate(joint_np)], strict=True)
            dist = Distribution(outcomes, pmf)

            inner_cls = _make_backend_subclass(
                InnerTwoPartIntrinsicMutualInformation,
                getattr(self, '_backend', 'numpy'),
            )
            inner = inner_cls(dist=dist,
                              rvs=[[rv] for rv in self._rvs],
                              crvs=self._crvs,
                              j=self._j,
                              bound_u=self._bound_u,
                              bound_v=self._bound_v,
                              )
            inner.optimize()
            opt = -inner.objective(inner._optima)

            a = mmi(joint)

            return a + opt

        return objective

    @classmethod
    def functional(cls):
        """
        Construct a functional form of the optimizer.
        """
        @unitful
        def two_part_intrinsic(dist, rvs=None, crvs=None, niter=None,
                               bound_j=None, bound_u=None, bound_v=None,
                               rv_mode=None, backend='numpy'):
            bounds = {
                (2, 2, 2),
                (bound_j, bound_u, bound_v),
            }

            actual_cls = _make_backend_subclass(cls, backend)
            candidates = []
            for b_j, b_u, b_v in bounds:
                opt = actual_cls(dist, rvs=rvs, crvs=crvs, bound_j=b_j,
                                 bound_u=b_u, bound_v=b_v, rv_mode=rv_mode)
                opt._backend = backend
                opt.optimize(niter=niter)
                val = opt.objective(opt._optima)
                val = float(val.detach().cpu().item()) if hasattr(val, 'detach') else float(val)
                candidates.append(val)

            return min(candidates)

        two_part_intrinsic.__doc__ = \
            f"""
            Compute the two-part intrinsic {cls.name}.

            Parameters
            ----------
            dist : Distribution
                The distribution to compute the two-part intrinsic {cls.name} of.
            rvs : list, None
                A list of lists. Each inner list specifies the indexes of the
                random variables used to calculate the intrinsic {cls.name}. If
                None, then it is calculated over all random variables, which is
                equivalent to passing `rvs=dist.rvs`.
            crvs : list
                A single list of indexes specifying the random variables to
                condition on.
            niter : int
                The number of optimization iterations to perform.
            bound_j : int, None
                Specifies a bound on the size of the J auxiliary random
                variable. If None, then the theoretical bound is used.
            bound_u : int, None
                Specifies a bound on the size of the U auxiliary random
                variable. If None, then the theoretical bound is used.
            bound_v : int, None
                Specifies a bound on the size of the V auxiliary random
                variable. If None, then the theoretical bound is used.
            rv_mode : str, None
                Specifies how to interpret `rvs` and `crvs`. Valid options are:
                {{'indices', 'names'}}. If equal to 'indices', then the elements
                of `crvs` and `rvs` are interpreted as random variable indices.
                If equal to 'names', the the elements are interpreted as random
                variable names. If `None`, then the value of `dist._rv_mode` is
                consulted, which defaults to 'indices'.
            backend : str
                The optimization backend. One of ``'numpy'`` (default),
                ``'jax'``, or ``'torch'``.
            """

        return two_part_intrinsic


# ── Backward-compatible composed classes (numpy backend) ──────────────────

class BaseOneWaySKAR(OneWaySKARMixin, BaseAuxVarOptimizer):
    """
    Compute lower bounds on the secret key agreement rate of the form:

    .. math::
        max_{V - U - X - YZ} objective()

    Uses the default NumPy / SciPy optimization backend.
    """

    construct_initial = BaseAuxVarOptimizer.construct_copy_initial


class BaseIntrinsicMutualInformation(IntrinsicMIMixin, BaseAuxVarOptimizer):
    """
    Compute a generalized intrinsic mutual information:

    .. math::
        IMI[X:Y|Z] = min_{p(z_bar|z)} I[X:Y|Z]

    Uses the default NumPy / SciPy optimization backend.
    """
    pass


class BaseMoreIntrinsicMutualInformation(MoreIntrinsicMIMixin, BaseAuxVarOptimizer):
    """
    Compute the reduced and minimal intrinsic mutual informations, upper bounds
    on the secret key agreement rate:

    .. math::
        I[X : Y \\downarrow\\downarrow\\downarrow Z] = min_U I[X:Y|U] + I[XY:U|Z]

    Uses the default NumPy / SciPy optimization backend.
    """
    pass


class BaseReducedIntrinsicMutualInformation(BaseMoreIntrinsicMutualInformation):
    """
    Compute the reduced intrinsic mutual information, a lower bound on the secret
    key agreement rate:

    .. math::
        I[X : Y \\Downarrow Z] = min_U I[X : Y \\downarrow ZU] + H[U]
    """

    style = "reduced"

    @property
    @staticmethod
    @abstractmethod
    def measure():
        pass

    def _objective(self, x):  # pragma: no cover
        """
        Minimize :math:`I[X:Y \\downarrow ZU] + H[U]`

        Parameters
        ----------
        x : ndarray
            An optimization vector.

        Returns
        -------
        obj : float
            The value of the objective function.
        """
        h = self._entropy(self._arvs)

        def objective(self, x):
            """
            Compute :math:`I[X:Y \\downarrow ZU] + H[U]`

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

            # I[X:Y \downarrow ZU]
            d = Distribution.from_ndarray(pmf)
            a = self.measure(dist=d, rvs=[[rv] for rv in self._rvs], crvs=self._crvs | self._arvs)

            # H[U]
            b = h(pmf)

            return a + b

        return objective


class BaseMinimalIntrinsicMutualInformation(BaseMoreIntrinsicMutualInformation):
    """
    Compute the minimal intrinsic mutual information, a upper bound on the
    secret key agreement rate:

    .. math::
        I[X : Y \\downarrow\\downarrow\\downarrow Z] = min_U I[X:Y|U] + I[XY:U|Z]
    """

    style = "minimal"

    def _objective(self):
        """
        Compute I[X:Y|U] + I[XY:U|Z], or its multivariate analog.

        Returns
        -------
        obj : func
            The objective function.
        """
        mmi = self.measure(self._rvs, self._arvs)
        cmi = self._conditional_mutual_information(self._rvs, self._arvs, self._crvs)

        def objective(self, x):
            """
            Compute :math:`I[X:Y|U] + I[XY:U|Z]`

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

            # I[X:Y|U]
            a = mmi(pmf)

            # I[XY:U|Z]
            b = cmi(pmf)

            return a + b

        return objective


class InnerTwoPartIntrinsicMutualInformation(InnerTwoPartIMIMixin, BaseAuxVarOptimizer):
    """
    Compute the two-part intrinsic mutual informations, an upper bound on the
    secret key agreement rate:

    .. math::
        I[X : Y \\downarrow\\downarrow\\downarrow\\downarrow Z] =
          inf_{J} min_{V - U - XY - ZJ} I[X:Y|J] + I[U:J|V] - I[U:Z|V]

    Uses the default NumPy / SciPy optimization backend.
    """
    pass


class BaseTwoPartIntrinsicMutualInformation(TwoPartIMIMixin, BaseAuxVarOptimizer):
    """
    Compute the two-part intrinsic mutual informations, an upper bound on the
    secret key agreement rate:

    .. math::
        I[X : Y \\downarrow\\downarrow\\downarrow\\downarrow Z] =
          inf_{J} min_{V - U - XY - ZJ} I[X:Y|J] + I[U:J|V] - I[U:Z|V]

    Uses the default NumPy / SciPy optimization backend.
    """
    pass
