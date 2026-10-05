"""
A computable relaxation of the two-part intrinsic mutual information, an upper
bound on the two-way secret key agreement rate.

For any ``J``, the two-part bound :cite:`gohari2010information,gohari2017comments`
gives

.. math::
    S[X : Y || Z] \\leq I[X : Y | J] + S[XY \\rightarrow J || Z]

where the second term is a one-way secret key agreement rate. Bounding that
rate by the intrinsic mutual information :cite:`maurer1997intrinsic`,
:math:`I[XY : J \\downarrow Z]`, yields

.. math::
    I[X : Y \\downarrow\\downarrow\\downarrow\\downarrow_r Z] =
        \\min_{p(j | xyz), p(\\overline{z} | z)} I[X : Y | J] + I[XY : J | \\overline{Z}]

which lies between the two-part and the minimal intrinsic mutual informations.
Unlike the two-part bound, it is a single joint minimization, so restricting
the auxiliary alphabets or stopping at a local optimum can only loosen it.
"""

import numpy as np

from ...algorithms import BaseAuxVarOptimizer
from ...algorithms.optimization import parallel_sweep
from ...exceptions import ditException
from ...math import prod
from ...utils import unitful
from .._backend import _make_backend_subclass

__all__ = (
    "RelaxedTwoPartIntrinsicMutualInformation",
    "relaxed_two_part_intrinsic_mutual_information",
)


class RelaxedTwoPartIMIMixin:
    """
    Mixin containing the relaxed two-part intrinsic mutual information logic.

    Must be composed with a ``BaseAuxVarOptimizer``-compatible base class.
    """

    _objective_bound = 0.0

    def __init__(self, dist, rvs=None, crvs=None, bound_j=None, bound_zbar=None):
        """
        Initialize the optimizer.

        Parameters
        ----------
        dist : Distribution
            The distribution of interest.
        rvs : list
            A list of two lists, the indices of X (Alice) and Y (Bob).
        crvs : list
            The indices of Z (Eve).
        bound_j : int, None
            Bound on the size of J. If None, use |X||Y||Z|.
        bound_zbar : int, None
            Bound on the size of the corrupted Z. If None, use |Z|, which
            suffices :cite:`christandl2003property`.
        """
        if not crvs:
            msg = "Intrinsic mutual informations require a conditional variable."
            raise ditException(msg)

        super().__init__(dist, rvs, crvs)
        self._inputs = (dist, rvs, crvs)

        default_j = prod(self._shape)
        bound_j = min([bound_j, default_j]) if bound_j else default_j

        crv_size = prod(self._shape[crv] for crv in self._crvs)
        bound_zbar = min([bound_zbar, crv_size]) if bound_zbar else crv_size

        self._construct_auxvars(
            [
                (self._rvs | self._crvs, bound_j),
                (self._crvs, bound_zbar),
            ]
        )
        j, zbar = sorted(self._arvs)
        self._j = {j}
        self._zbar = {zbar}

    def _objective(self):
        """
        Minimize I[X:Y|J] + I[XY:J|Zbar].

        Returns
        -------
        obj : func
            The objective function.
        """
        mi = self._total_correlation(self._rvs, self._j)
        cmi = self._conditional_mutual_information(self._rvs, self._j, self._zbar)

        def objective(self, x):
            """
            Compute :math:`I[X:Y|J] + I[XY:J|\\overline{Z}]`

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
            return mi(pmf) + cmi(pmf)

        return objective

    def _objective_gradient(self):
        """Gradient of ``I[X:Y|J] + I[XY:J|Zbar]`` w.r.t. the joint."""
        mi_grad = self._total_correlation_grad(self._rvs, self._j)
        cmi_grad = self._conditional_mutual_information_grad(self._rvs, self._j, self._zbar)

        def grad(pmf):
            return mi_grad(pmf) + cmi_grad(pmf)

        return grad

    def _minimal_initial(self, niter=None, rng=None):
        """
        Solve the minimal intrinsic mutual information with the same bound on J,
        and pair its optimal J with Zbar a copy of Z. The joint search over both
        channels otherwise often stalls above the minimal bound it relaxes.
        """
        from .minimal_intrinsic_mutual_informations import MinimalIntrinsicTotalCorrelation

        minimal_cls = _make_backend_subclass(MinimalIntrinsicTotalCorrelation, getattr(self, "_backend", "numpy"))
        minimal = minimal_cls(*self._inputs, bound=self._aux_bounds[0])
        minimal.optimize(niter=niter, rng=rng)
        j_channel = np.asarray(minimal._optima)
        _, copy_zbar = np.split(self.construct_copy_initial(), [self._parts[1][0]])
        return np.concatenate([j_channel, copy_zbar])

    def optimize(self, x0=None, niter=None, rng=None, **kwargs):
        """
        Perform the optimization, warm-started from the minimal intrinsic mutual
        information, then keep the best of that start, the constant and copy
        initials, and the optimum found.
        """
        if x0 is None:
            x0 = self._minimal_initial(niter=niter, rng=rng)

        result = super().optimize(x0=x0, niter=niter, rng=rng, **kwargs)

        options = [
            x0,
            self.construct_constant_initial(),
            self.construct_copy_initial(),
            result.x,
        ]

        self._optima = min(options, key=lambda opt: self.objective(opt))


class RelaxedTwoPartIntrinsicMutualInformation(RelaxedTwoPartIMIMixin, BaseAuxVarOptimizer):
    """
    Compute the relaxed two-part intrinsic mutual information, an upper bound
    on the secret key agreement rate:

    .. math::
        min_{p(j|xyz), p(zbar|z)} I[X:Y|J] + I[XY:J|Zbar]

    Uses the default NumPy / SciPy optimization backend.
    """

    pass


@unitful
def relaxed_two_part_intrinsic_mutual_information(
    dist, rvs, crvs, niter=None, bounds=None, bound_zbar=None, backend="numpy"
):
    """
    Compute the relaxed two-part intrinsic mutual information, an upper bound
    on the two-way secret key agreement rate that is never larger than the
    minimal intrinsic mutual information.

    Parameters
    ----------
    dist : Distribution
        The distribution of interest.
    rvs : list
        A list of two lists, the indices of X (Alice) and Y (Bob).
    crvs : list
        The indices of Z (Eve).
    niter : int, None
        The number of basin hops to perform.
    bounds : [int], None
        Bounds on the size of J to sweep over. Each yields a valid upper bound,
        and the smallest is returned. If None, use (2, 3, 4, None), where None
        means |X||Y||Z|.
    bound_zbar : int, None
        Bound on the size of the corrupted Z. If None, use |Z|.
    backend : str
        The optimization backend. One of ``'numpy'`` (default), ``'jax'``, or
        ``'torch'``.

    Returns
    -------
    rtpimi : float
        The relaxed two-part intrinsic mutual information.
    """
    if bounds is None:
        bounds = (2, 3, 4, None)

    actual_cls = _make_backend_subclass(RelaxedTwoPartIntrinsicMutualInformation, backend)

    def _run(bound_j, rng):
        opt = actual_cls(dist, rvs=rvs, crvs=crvs, bound_j=bound_j, bound_zbar=bound_zbar)
        opt._backend = backend
        opt.optimize(niter=niter, rng=rng)
        val = opt.objective(opt._optima)
        return float(val.detach().cpu().item()) if hasattr(val, "detach") else float(val)

    return min(parallel_sweep(_run, bounds))
