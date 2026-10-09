"""
One-way secret key agreement rate. This is the rate at which Alice and Bob can
agree upon a secret key with Eve eavesdropping, if only Alice is permitted
to publicly communicate.
"""

import itertools
from math import comb

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, QhullError

from ...algorithms.optimization import parallel_sweep
from ...utils import unitful
from .._backend import _make_backend_subclass
from .base_skar_optimizers import BaseOneWaySKAR

__all__ = ("one_way_skar",)


def _simplex_grid(k, max_points):
    """Regular grid on the probability simplex over ``k`` symbols."""
    n = 1
    while comb(n + k, k - 1) <= max_points:
        n += 1
    pts = [c for c in itertools.product(range(n + 1), repeat=k - 1) if sum(c) <= n]
    return np.array([list(c) + [n - sum(c)] for c in pts], dtype=float) / n


def _entropy_rows(P):
    """Entropy, in bits, of each row of ``P``."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.where(P > 0, P * np.log2(P), 0.0).sum(axis=-1)


def _convex_envelope(Q, f):
    """Lower convex envelope of the values ``f`` at the simplex points ``Q``."""
    try:
        hull = ConvexHull(np.c_[Q[:, :-1], f])
    except QhullError:
        return f.copy()
    eq = hull.equations[hull.equations[:, -2] < -1e-12]
    slopes = -eq[:, :-2] / eq[:, -2:-1]
    offsets = -eq[:, -1] / eq[:, -2]
    env = np.full(len(Q), -np.inf)
    for i in range(0, len(eq), 2000):
        env = np.maximum(env, (Q[:, :-1] @ slopes[i : i + 2000].T + offsets[i : i + 2000]).max(axis=1))
    return np.minimum(env, f)


def _split(Q, values, target, maximize):
    """Mixture of grid points averaging to ``target`` that extremizes the average of ``values``."""
    res = linprog(-values if maximize else values, A_eq=Q.T, b_eq=target, bounds=(0, None), method="highs")
    if res.x is None:
        return None
    keep = res.x > 1e-12
    return res.x[keep], Q[keep]


class OneWaySKAR(BaseOneWaySKAR):
    """
    Compute the one-way secret key agreement rate:
        max_{V - U - X - YZ} I[U:Y|V] - I[U:Z|V]

    :cite:`ahlswede1993common`, with the cardinality bounds of
    :cite:`elgamal2011network` (Section 22.3).
    """

    def _get_u_bound(self):
        """
        |U| <= |X|^2: each of the |X| values of V needs its own |X| values of U.
        """
        return self._shape[0] ** 2

    def _get_v_bound(self):
        """
        |V| <= |X|
        """
        return self._shape[0]

    _envelope_max_alphabet = 4
    _envelope_grid_points = 4000

    def _envelope_seed(self):
        """
        Near-optimal auxiliary variables from a grid over Alice's simplex.

        With ``phi(q) = H[Y] - H[Z]`` when Alice's distribution is ``q``, the
        rate is the concave envelope of ``phi - vex(phi)`` evaluated at
        ``p(x)``, where ``vex`` is the convex envelope: ``V`` splits ``p(x)``
        into posteriors and ``U`` splits each of those again. This follows
        directly from the formula of :cite:`ahlswede1993common`; we know of no
        separate source for it. The optimum often puts posteriors on the
        boundary of the simplex, where local search from random starts rarely
        lands.

        Returns
        -------
        x : np.ndarray, None
            An optimization vector, or None if Alice's alphabet is too large
            or the auxiliary bounds cannot hold the split.
        """
        k = self._shape[0]
        bound_u, bound_v = self._aux_bounds
        px = self._pmf.sum(axis=(1, 2))
        if k > self._envelope_max_alphabet or np.any(px <= 0):
            return None
        channel_y = self._pmf.sum(axis=2) / px[:, None]
        channel_z = self._pmf.sum(axis=1) / px[:, None]
        Q = np.r_[_simplex_grid(k, self._envelope_grid_points), px[None, :]]
        phi = _entropy_rows(Q @ channel_y) - _entropy_rows(Q @ channel_z)

        if bound_v == 1:
            outer = (np.ones(1), px[None, :])
        else:
            outer = _split(Q, phi - _convex_envelope(Q, phi), px, maximize=True)
        if outer is None or len(outer[0]) > bound_v:
            return None

        columns, labels = [], []
        for v, (weight, posterior) in enumerate(zip(*outer, strict=True)):
            inner = _split(Q, phi, posterior, maximize=False)
            if inner is None:
                return None
            for mu, leaf in zip(*inner, strict=True):
                columns.append(weight * mu * leaf / px)
                labels.append(v)
        if len(columns) > bound_u:
            return None

        pu_x = np.zeros((k, bound_u))
        pu_x[:, : len(columns)] = np.array(columns).T
        pv_u = np.zeros((bound_u, bound_v))
        pv_u[np.arange(len(labels)), labels] = 1
        return np.r_[pu_x.ravel(), pv_u.ravel()]

    def _objective_gradient(self):
        """Gradient of the ``-(I[U:Y|V] - I[U:Z|V])`` objective w.r.t. the joint."""
        grad_a = self._conditional_mutual_information_grad(self._u, self._y, self._v)
        grad_b = self._conditional_mutual_information_grad(self._u, self._z, self._v)
        return lambda pmf: -(grad_a(pmf) - grad_b(pmf))

    def _objective(self):
        """
        Maximize I[U:Y|V] - I[U:Z|V]

        Returns
        -------
        obj : func
            The objective function.
        """
        # I[U:Y|V]
        cmi_a = self._conditional_mutual_information(self._u, self._y, self._v)
        # I[U:Z|V]
        cmi_b = self._conditional_mutual_information(self._u, self._z, self._v)

        def objective(self, x):
            """
            Compute I[U:Y] - I[U:X]

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

            a = cmi_a(pmf)
            b = cmi_b(pmf)

            return -(a - b)

        return objective


@unitful
def one_way_skar(dist, X, Y, Z, niter=None, bound_u=None, bound_v=None, backend="numpy"):
    """
    Compute the secret key agreement rate constrained to one-way communication.

    Parameters
    ----------
    dist : Distribution
        The distribution of interest.
    X : iterable
        The indices to consider as the X variable, Alice.
    Y : iterable
        The indices to consider as the Y variable, Bob.
    Z : iterable
        The indices to consider as the Z variable, Eve.
    niter : int, None
        The number of hops to perform during optimization.
    bound_u : int, None
        The bound to use on the size of the variable U. If none, use the
        theoretical bound of |X|^2.
    bound_v : int, None
        The bound to use on the size of the variable V. If none, use the
        theoretical bound of |X|.
    backend : str
        The optimization backend. One of ``'numpy'`` (default),
        ``'jax'``, or ``'torch'``.

    Returns
    -------
    owskar : float
        The necessary intrinsic mutual information.
    """
    actual_cls = _make_backend_subclass(OneWaySKAR, backend)

    def _run(bound, rng):
        nimi = actual_cls(dist, X, Y, Z, bound_u=bound_u, bound_v=bound)
        nimi.optimize(niter=niter, rng=rng)
        candidates = [nimi._optima]
        seed = nimi._envelope_seed() if backend == "numpy" else None
        if seed is not None:
            # Local search started from the seed can wander off it, so keep it as a candidate.
            nimi._optima = seed
            nimi._polish()
            candidates += [seed, nimi._optima]
        return max(-nimi.objective(x) for x in candidates)

    values = parallel_sweep(_run, sorted({1, 2, 3, bound_v}, key=lambda b: (b is None, b)))

    return max(values)
