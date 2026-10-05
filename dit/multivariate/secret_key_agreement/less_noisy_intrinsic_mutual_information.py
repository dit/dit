"""
The less-noisy intrinsic mutual information, an upper bound on the two-way
secret key agreement rate :cite:`pauwels2026bipartite`.
"""

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import minimize

from ...exceptions import ditException
from ...utils import unitful

__all__ = ("less_noisy_intrinsic_mutual_information",)


def _entropy(p):
    p = p[p > 0]
    return -np.sum(p * np.log2(p))


class LessNoisyIntrinsicMutualInformation:
    """
    Compute

        inf_{p(j|xy) : p(z|xy) is less noisy than p(j|xy)} I[X:Y|J]

    A channel ``p(j|t)`` is dominated by ``p(z|t)`` in the less-noisy sense iff
    ``q -> H[qW_Z] - H[qW_J]`` is concave over all input distributions ``q``
    :cite:`vandijk1997special`. If some direction ``v`` has ``v W_Z = 0`` but
    ``v W_J != 0``, that function has strictly positive curvature along ``v``,
    so every dominated channel factors as ``W_J = W_Z K`` for a real (possibly
    signed) matrix ``K``. The optimization is over ``K``, subject to ``W_Z K``
    being a channel and to the curvature condition, imposed on a sample of
    input distributions.
    """

    def __init__(self, dist, rvs, crvs, bound=None, rng=None):
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
        bound : int, None
            The size of ``J``. If None, use ``|Z|``.
        rng : int, np.random.Generator, None
            Source of randomness.
        """
        if len(rvs) != 2:
            msg = "The less-noisy intrinsic mutual information is defined for two parties."
            raise ditException(msg)
        if not crvs:
            msg = "Intrinsic mutual informations require a conditional variable."
            raise ditException(msg)

        self._rng = np.random.default_rng(rng)

        d = dist.coalesce([rvs[0], rvs[1], crvs]).copy(base="linear")
        d.make_dense()
        pxyz = d.pmf.reshape([len(a) for a in d.alphabet])
        nx, ny, nz = pxyz.shape
        pxy = pxyz.sum(axis=2)

        self._shape = (nx, ny)
        self._support = np.flatnonzero(pxy.ravel() > 0)
        self._pt = pxy.ravel()[self._support]
        wz = pxyz.reshape(nx * ny, nz)[self._support] / self._pt[:, None]
        self._wz = wz[:, wz.sum(axis=0) > 0]
        m, self._nz = self._wz.shape
        self._k = min(bound, self._nz) if bound else self._nz

        # directions w = v W_Z, with v summing to zero
        u, s, _ = np.linalg.svd(self._wz.T @ null_space(np.ones((1, m))), full_matrices=False)
        self._u = u[:, s > 1e-12]

        self._train = self._sample_inputs(200)
        self._check = np.vstack([self._train, self._sample_inputs(5000)])

    def _sample_inputs(self, n, eps=1e-9):
        """
        Sample Eve's output distributions ``q W_Z``, both from the interior and
        near the boundary of the input simplex.
        """
        m = len(self._pt)
        q = np.vstack(
            [
                self._rng.dirichlet(np.ones(m), n),
                self._rng.dirichlet(0.1 * np.ones(m), n),
                np.eye(m),
            ]
        )
        q = np.maximum(q, eps)
        q /= q.sum(axis=1, keepdims=True)
        return q @ self._wz

    def _ratios(self, K, R):
        """
        For each of Eve's output distributions ``r``, the largest ratio of the
        curvature of ``H[J]`` to that of ``H[Z]``. ``W_Z K`` is less noisy
        than ``W_Z`` iff every ratio is at most one.
        """
        u = self._u
        if u.shape[1] == 0:
            return np.zeros(len(R))
        mz = np.einsum("zd,nz,ze->nde", u, 1 / R, u)
        linv = np.linalg.inv(np.linalg.cholesky(mz))
        rk = R @ K
        g = u.T @ K
        keep = np.abs(g).max(axis=0) > 1e-10
        g, rk = g[:, keep], rk[:, keep]
        if (rk <= 0).any():
            return np.full(len(R), 1e6)
        mj = np.einsum("dj,nj,ej->nde", g, 1 / rk, g)
        return np.linalg.eigvalsh(linv @ mj @ np.swapaxes(linv, 1, 2))[:, -1]

    def _feasible(self, K, R, tol=1e-7):
        return bool((self._wz @ K >= -tol).all() and self._ratios(K, R).max() <= 1 + tol)

    def _unpack(self, x):
        K = np.empty((self._nz, self._k))
        K[:, :-1] = x.reshape(self._nz, self._k - 1)
        K[:, -1] = 1 - K[:, :-1].sum(axis=1)
        return K

    def objective(self, K):
        """
        Compute I[X:Y|J] for ``W_J = W_Z K``.

        Parameters
        ----------
        K : np.ndarray
            The matrix mapping Eve's channel to ``J``'s.

        Returns
        -------
        cmi : float
            The conditional mutual information.
        """
        wj = np.clip(self._wz @ K, 0, None)
        wj /= wj.sum(axis=1, keepdims=True)
        p = np.zeros((np.prod(self._shape), self._k))
        p[self._support] = self._pt[:, None] * wj
        p = p.reshape(*self._shape, self._k)
        return (
            _entropy(p.sum(axis=1).ravel())
            + _entropy(p.sum(axis=0).ravel())
            - _entropy(p.ravel())
            - _entropy(p.sum(axis=(0, 1)))
        )

    def _initial_points(self, n):
        """
        Starting points: coarse-grainings of Z and random channels from Z,
        all of which are degradations of Z and hence feasible.
        """
        points = [np.eye(self._nz, self._k)]
        points[0][self._k :, -1] = 1
        for i in range(n - 1):
            if i % 2:
                points.append(np.eye(self._k)[self._rng.integers(self._k, size=self._nz)])
            else:
                points.append(self._rng.dirichlet(0.5 * np.ones(self._k), self._nz))
        return [K[:, :-1].ravel() for K in points]

    def _local(self, x0):
        constraints = [
            {"type": "ineq", "fun": lambda x: (self._wz @ self._unpack(x)).ravel()},
            {"type": "ineq", "fun": lambda x: 1 - self._ratios(self._unpack(x), self._train)},
        ]
        res = minimize(
            lambda x: self.objective(self._unpack(x)),
            x0,
            method="SLSQP",
            constraints=constraints,
            options={"maxiter": 200, "ftol": 1e-10},
        )
        return self._unpack(res.x)

    def _repair(self, K):
        """
        Shrink ``K`` toward the constant channel until it passes the check on
        the larger input sample. The feasible set is convex and contains the
        constant channel.
        """
        if self._feasible(K, self._check):
            return K
        K0 = np.full_like(K, 1 / self._k)
        lo, hi = 0.0, 1.0
        for _ in range(40):
            mid = (lo + hi) / 2
            if self._feasible((1 - mid) * K0 + mid * K, self._check):
                lo = mid
            else:
                hi = mid
        return (1 - lo) * K0 + lo * K

    def optimize(self, niter=None):
        """
        Perform the optimization.

        Parameters
        ----------
        niter : int, None
            The number of local optimizations to run. If None, use 8.
        """
        K0 = np.full((self._nz, self._k), 1 / self._k)
        self._optima = K0
        if self._k == 1:
            return
        best = self.objective(K0)
        for x0 in self._initial_points(niter or 8):
            for K in (self._unpack(x0), self._repair(self._local(x0))):
                value = self.objective(K)
                if value < best:
                    best, self._optima = value, K


@unitful
def less_noisy_intrinsic_mutual_information(dist, rvs, crvs, niter=None, bound=None, rng=None):
    """
    Compute the less-noisy intrinsic mutual information :cite:`pauwels2026bipartite`:

        inf_{p(j|xy) : p(z|xy) is less noisy than p(j|xy)} I[X:Y|J]

    If Eve's channel is less noisy than that of ``J`` -- ``I[U:Z] >= I[U:J]``
    for every ``U - XY - ZJ`` and every input distribution -- then the secret
    key agreement rate against Z is at most that against J
    :cite:`gohari2017achieving`, which in turn is at most ``I[X:Y|J]``. Every
    degradation of Z is dominated, so this never exceeds the intrinsic mutual
    information, and it never falls below the inf-max upper bound of Gohari and
    Anantharam (2010), stated as Eq. (11) of :cite:`abin2026source`.

    Parameters
    ----------
    dist : Distribution
        The distribution of interest.
    rvs : iterable of iterables, len(rvs) == 2
        The variables representing Alice and Bob.
    crvs : iterable
        The variables representing Eve.
    niter : int, None
        The number of local optimizations to run. If None, use 8.
    bound : int, None
        The size of ``J``. If None, use ``|Z|``.
    rng : int, np.random.Generator, None
        Source of randomness.

    Returns
    -------
    lnimi : float
        The less-noisy intrinsic mutual information.

    Notes
    -----
    The less-noisy condition is checked on a finite sample of input
    distributions, so the returned value is a numerical estimate of an upper
    bound, not a certificate.
    """
    opt = LessNoisyIntrinsicMutualInformation(dist, rvs, crvs, bound=bound, rng=rng)
    opt.optimize(niter=niter)
    return float(opt.objective(opt._optima))
