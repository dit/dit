"""
The secret key cost, or information of formation: the rate of secret key
needed to form a distribution by public communication
:cite:`renner2003bounds,winter2005secret`.
"""

import numpy as np

from ...algorithms import BaseAuxVarOptimizer
from ...exceptions import ditException
from ...utils import unitful

__all__ = ("secret_key_cost",)


class SecretKeyCost(BaseAuxVarOptimizer):
    """
    Compute

        min I[XY : V | U] such that XY - Z - U and X - UV - Y

    with ``|U| <= |Z| + 1`` and ``|V| <= |X||Y|`` :cite:`winter2005secret`, as
    stated in Theorem 5 of :cite:`chitambar2016private`.

    ``U`` is a channel from ``Z``. As for the Wyner common information, the
    second Markov chain is built in rather than imposed: ``V`` is a channel
    from ``XU`` and ``Y'`` a channel from ``UV``, so that ``X - UV - Y'``, and
    the constraint is that ``XY'U`` matches ``XYU``.
    """

    _shotgun = 5

    def __init__(self, dist, rvs, crvs, bound_u=None, bound_v=None):
        """
        Initialize the optimizer.

        Parameters
        ----------
        dist : Distribution
            The distribution of interest.
        rvs : iterable of iterables, len(rvs) == 2
            The variables representing Alice and Bob.
        crvs : iterable
            The variables representing Eve.
        bound_u : int, None
            Bound on the size of ``U``. If None, use ``|Z| + 1``.
        bound_v : int, None
            Bound on the size of ``V``. If None, use ``|X||Y|``.
        """
        if len(rvs) != 2:
            msg = "The secret key cost is defined for two parties."
            raise ditException(msg)
        if not crvs:
            msg = "The secret key cost requires an eavesdropper; without one it is the Wyner common information."
            raise ditException(msg)

        super().__init__(dist, rvs, crvs)

        nx, ny, nz = self._shape
        bound_u = min(bound_u, nz + 1) if bound_u else nz + 1
        bound_v = min(bound_v, nx * ny) if bound_v else nx * ny

        # axes: X, Y, Z, U, V, Y'
        self._construct_auxvars([({2}, bound_u), ({0, 3}, bound_v), ({3, 4}, ny)])

        self.constraints += [
            {
                "type": "eq",
                "fun": self.constraint_match_joint,
            },
        ]

        self._default_hops = 5

        self._additional_options = {
            "options": {
                "maxiter": 1000,
                "ftol": 1e-7,
            }
        }

    def _mismatch(self, joint):
        """
        The difference between ``p(x, y', u)`` and ``p(x, y, u)``, indexed
        ``(x, y, u)``.
        """
        return joint.sum(axis=(1, 2, 4)).transpose(0, 2, 1) - joint.sum(axis=(2, 4, 5))

    def constraint_match_joint(self, x):
        """
        Ensure that ``XY'U`` is distributed as ``XYU``.

        Parameters
        ----------
        x : np.ndarray
            An optimization vector.

        Returns
        -------
        delta : float
            The constraint residual; zero when the two match.
        """
        return 100 * (self._mismatch(self.construct_joint(x)) ** 2).sum()

    # NOTE: exact gradients are intentionally *not* wired here. The squared
    # residual has a vanishing gradient on the feasible set, and with exact
    # objective and constraint gradients (or the residual imposed as a vector of
    # constraints) SLSQP stalls at infeasible points, while finite differences
    # converge.

    def _objective(self):
        """
        The rate of secret key, ``I[XY' : V | U]``.

        Returns
        -------
        obj : func
            The objective function.
        """
        cmi = self._conditional_mutual_information({0, 5}, {4}, {3})

        def objective(self, x):
            """
            Compute I[XY' : V | U].

            Parameters
            ----------
            x : np.ndarray
                An optimization vector.

            Returns
            -------
            obj : float
                The value of the objective.
            """
            return cmi(self.construct_joint(x))

        return objective

    def construct_feasible_initial(self, degrade=False):
        """
        Construct the feasible point ``V = X``, ``Y' ~ p(y | x u)``, with
        ``U = Z`` (objective ``H[X | Z]``) or ``U`` constant (objective
        ``H[X]``).

        Parameters
        ----------
        degrade : bool
            If True, ``U`` is constant; otherwise ``U = Z``.

        Returns
        -------
        x : np.ndarray
            A feasible optimization vector.
        """
        cu, cv, _ = self._aux_vars
        nz, bound_u = cu.shape
        nx, _, bound_v = cv.shape
        channel_u = np.zeros(cu.shape)
        if degrade:
            channel_u[:, 0] = 1
        else:
            channel_u[np.arange(nz), np.minimum(np.arange(nz), bound_u - 1)] = 1
        channel_v = np.zeros(cv.shape)
        channel_v[np.arange(nx), :, np.arange(nx) % bound_v] = 1
        pxyu = np.einsum("xyz,zu->xyu", self._pmf, channel_u)
        pvyu = np.zeros((bound_v, pxyu.shape[1], bound_u))
        np.add.at(pvyu, np.arange(nx) % bound_v, pxyu)
        with np.errstate(divide="ignore", invalid="ignore"):
            channel_y = pvyu / pvyu.sum(axis=1, keepdims=True)
        channel_y = np.nan_to_num(channel_y, nan=1 / pxyu.shape[1]).transpose(2, 0, 1)
        return np.concatenate([channel_u.ravel(), channel_v.ravel(), channel_y.ravel()])

    def optimize(self, x0=None, *args, **kwargs):
        """
        Perform the optimization, starting from a feasible point by default.

        Parameters
        ----------
        x0 : np.ndarray, None
            Initial optimization vector. If None, use
            :meth:`construct_feasible_initial`.
        """
        if x0 is None:
            x0 = self.construct_feasible_initial()
        return super().optimize(x0, *args, **kwargs)


@unitful
def secret_key_cost(dist, rvs, crvs, niter=None, bound_u=None, bound_v=None, rng=None):
    """
    Compute the secret key cost :cite:`winter2005secret`, also known as the
    information of formation :cite:`renner2003bounds`: the minimal rate of
    secret key Alice and Bob need to produce ``XY`` by local operations and
    public communication, such that Eve could simulate the public transcript
    from ``Z``. It is

        min I[XY : V | U] such that XY - Z - U and X - UV - Y

    that is, the conditional Wyner common information ``C[X : Y | U]``
    minimized over degradations ``U`` of ``Z``. It is never less than the
    intrinsic mutual information :cite:`renner2003bounds`, and never more than
    either ``C[X : Y]`` or ``C[X : Y | Z]``.

    Parameters
    ----------
    dist : Distribution
        The distribution of interest.
    rvs : iterable of iterables, len(rvs) == 2
        The variables representing Alice and Bob.
    crvs : iterable
        The variables representing Eve.
    niter : int, None
        The number of basin hops to perform. If None, use 5.
    bound_u : int, None
        Bound on the size of ``U``. If None, use ``|Z| + 1``.
    bound_v : int, None
        Bound on the size of ``V``. If None, use ``|X||Y|``.
    rng : int, np.random.Generator, None
        Source of randomness.

    Returns
    -------
    skc : float
        The secret key cost.
    """
    opt = SecretKeyCost(dist, rvs, crvs, bound_u=bound_u, bound_v=bound_v)
    opt.optimize(niter=niter, rng=np.random.default_rng(rng))
    return max(float(opt.objective(opt._optima)), 0.0)
