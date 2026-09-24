"""
The shape function of Matveev & Romashchenko :cite:`matveev2026beyond`.

The shape function is the support function of the extension profile: it
records, for each direction ``alpha``, how far the profile extends,

.. math::

    S(\\alpha) = \\sup_W \\Big\\{ \\sum_i \\alpha_i H[X_i | W] - T[X_{0:n} | W] \\Big\\}.

For a pair this is the
``sup_W {alpha H[X|W] + beta H[Y|W] - I[X:Y|W]}`` of the reference.

Because ``T[X_{0:n}|W] = sum_i H[X_i|W] - H[X_{0:n}|W]`` and
``R_0 = I[X_{0:n}:W] = H[X_{0:n}] - H[X_{0:n}|W]``, the bracketed quantity is

.. math::

    \\sum_i \\alpha_i R_i - \\Big(R_0 + \\sum_i R_i - H[X_{0:n}]\\Big)
        = H[X_{0:n}] - \\Big(R_0 + \\sum_i (1 - \\alpha_i) R_i\\Big),

so the shape function is a *weighted rate query on the Gray-Wyner region*:

.. math::

    S(\\alpha) = H[X_{0:n}] - \\min_W \\Big[R_0 + \\sum_i (1 - \\alpha_i) R_i\\Big].

The weights ``(1, 1 - alpha_1, ..., 1 - alpha_n)`` are non-negative exactly
when every ``alpha_i <= 1``, which is why the shape function is studied on the
unit cube. Outside it the function is piecewise affine and fully determined by
the entropy profile, so nothing is lost.

Bounds
------
Two envelopes depend only on the entropy profile
:cite:`matveev2026beyond`. Evaluating the objective at the deterministic
probes ``W = X_{0:n}``, ``W = X_{-j}``, and ``W = .`` gives the attainable
points whose upper envelope is the lower bound,

.. math::

    S_{\\min}(\\alpha) = \\max\\Big\\{0,\\;
        \\max_j \\alpha_j H[X_j | X_{-j}],\\;
        \\sum_i \\alpha_i H[X_i] - T[X_{0:n}] \\Big\\},

while convexity of ``S`` together with ``S(0) = 0`` and
``S(e_j) = H[X_j | X_{-j}]`` gives the upper bound

.. math::

    S_{\\max}(\\alpha) = \\max\\Big\\{\\sum_j \\alpha_j H[X_j | X_{-j}],\\;
        \\sum_i \\alpha_i H[X_i] - T[X_{0:n}] \\Big\\}.

A pair attaining the lower bound everywhere is called *rigid*; a pair attains
the upper bound if and only if its mutual information is Gacs-Korner
extractable.
"""

from itertools import product

import numpy as np

from ...algorithms.optimization import parallel_sweep
from ...exceptions import ditException
from ...utils import flatten
from .network import GrayWynerNetwork

__all__ = ("ShapeFunction",)


class ShapeFunction:
    """
    The shape function of a tuple of random variables.

    Attributes
    ----------
    alphas : np.ndarray
        The sampled directions, shape ``(num_points, n)``.
    values : np.ndarray
        The shape function at each direction.
    minima, maxima : np.ndarray
        The ``S_min`` and ``S_max`` envelopes at each direction.
    """

    def __init__(
        self,
        dist,
        rvs=None,
        crvs=None,
        alphas=None,
        num=11,
        niter=None,
        maxiter=1000,
        bound=None,
    ):
        """
        Initialize and compute the shape function.

        Parameters
        ----------
        dist : Distribution
            The distribution of interest.
        rvs : list of lists, None
            The source groups. If None, each variable is its own source.
        crvs : list, None
            Variables to condition on.
        alphas : array-like, None
            The directions at which to evaluate, shape ``(num_points, n)``,
            each entry in ``[0, 1]``. If None, a default grid is used: the
            full ``num x num`` square for two sources, and the symmetric
            diagonal ``alpha_i = alpha`` with `num` points for more (a full
            grid would be exponential in the number of sources).
        num : int
            The resolution of the default grid.
        niter : int, None
            Number of basin hops per direction.
        maxiter : int
            Inner optimizer iterations.
        bound : int, None
            Optional cap on the cardinality of ``W``.
        """
        self.dist = dist.copy()
        self.rvs = [[i] for i in flatten(dist.rvs)] if rvs is None else rvs
        self.crvs = crvs
        self.n = len(self.rvs)

        if self.n < 2:
            msg = "The shape function requires at least two sources."
            raise ditException(msg)

        self._network = GrayWynerNetwork(dist, rvs=self.rvs, crvs=self.crvs, bound=bound)
        self._joint_entropy = self._network._joint_entropy
        self._marginal_entropies = np.asarray(self._network._marginal_entropies)

        from ...multivariate import entropy, total_correlation

        rv = list(flatten(self.rvs))
        self._total_correlation = float(total_correlation(dist, self.rvs, crvs))
        # H[X_j | X_{-j}], the value of S at the j-th unit direction.
        self._erasure_entropies = np.asarray(
            [
                self._joint_entropy - float(entropy(dist, [r for r in rv if r not in set(group)], crvs))
                for group in self.rvs
            ]
        )

        self.alphas = self._default_alphas(num) if alphas is None else np.atleast_2d(np.asarray(alphas, dtype=float))
        if self.alphas.shape[1] != self.n:
            msg = f"`alphas` must have {self.n} columns, got {self.alphas.shape[1]}."
            raise ditException(msg)
        if np.any(self.alphas < 0) or np.any(self.alphas > 1):
            msg = "`alphas` must lie in the unit cube."
            raise ditException(msg)

        self.label = getattr(dist, "name", "") or "shape"

        self.compute(niter=niter, maxiter=maxiter)

    def _default_alphas(self, num):
        """
        Build the default sampling grid.

        Parameters
        ----------
        num : int
            The grid resolution.

        Returns
        -------
        alphas : np.ndarray
            The directions, shape ``(num_points, n)``.
        """
        axis = np.linspace(0.0, 1.0, num)
        if self.n == 2:
            self.grid_shape = (num, num)
            return np.asarray(list(product(axis, repeat=2)))
        self.grid_shape = (num,)
        return np.repeat(axis[:, np.newaxis], self.n, axis=1)

    def minimum(self, alpha):
        """
        The ``S_min`` envelope, which depends only on the entropy profile.

        Parameters
        ----------
        alpha : array-like
            A direction, or an array of directions with `n` columns.

        Returns
        -------
        value : np.ndarray
            The lower bound at each direction.
        """
        alpha = np.atleast_2d(np.asarray(alpha, dtype=float))
        independent = alpha @ self._marginal_entropies - self._total_correlation
        singletons = (alpha * self._erasure_entropies).max(axis=1)
        return np.maximum(np.maximum(singletons, independent), 0.0)

    def maximum(self, alpha):
        """
        The ``S_max`` envelope, which depends only on the entropy profile.

        Parameters
        ----------
        alpha : array-like
            A direction, or an array of directions with `n` columns.

        Returns
        -------
        value : np.ndarray
            The upper bound at each direction.
        """
        alpha = np.atleast_2d(np.asarray(alpha, dtype=float))
        independent = alpha @ self._marginal_entropies - self._total_correlation
        subadditive = alpha @ self._erasure_entropies
        return np.maximum(subadditive, independent)

    def evaluate(self, alpha, niter=None, maxiter=1000, rng=None):
        """
        Evaluate the shape function in a single direction.

        Parameters
        ----------
        alpha : array-like
            A direction with `n` entries, each in ``[0, 1]``.
        niter : int, None
            Number of basin hops.
        maxiter : int
            Inner optimizer iterations.

        Returns
        -------
        value : float
            The shape function in that direction.
        """
        alpha = np.asarray(alpha, dtype=float)
        weights = 1.0 - alpha
        point = self._network.rate_point([1.0, *weights], niter=niter, maxiter=maxiter, rng=rng)
        value = self._joint_entropy - (point.common + weights @ np.asarray(point.private))
        # The attainable probes W = ., X_{-j}, X_{0:n} bound S from below;
        # a stalled optimizer can otherwise report an infeasible value.
        return float(max(value, self.minimum(alpha)[0]))

    def compute(self, niter=None, maxiter=1000):
        """
        Evaluate the shape function at every sampled direction.

        Parameters
        ----------
        niter : int, None
            Number of basin hops per direction.
        maxiter : int
            Inner optimizer iterations.
        """

        def _run(alpha, rng):
            return self.evaluate(alpha, niter=niter, maxiter=maxiter, rng=rng)

        self.values = np.asarray(parallel_sweep(_run, list(self.alphas)))
        self.minima = self.minimum(self.alphas)
        self.maxima = self.maximum(self.alphas)

    def rigidity(self, tol=1e-3):
        """
        Where the shape function sits between its two envelopes.

        A fraction of zero at every direction means the pair is *rigid*: its
        extension profile is as small as the entropy profile permits. A
        fraction of one means the mutual information is Gacs-Korner
        extractable.

        Parameters
        ----------
        tol : float
            Directions where ``S_max - S_min`` is below this are Shannon-forced
            and are excluded from the summary.

        Returns
        -------
        summary : dict
            The min, mean, and max of the normalized position, plus booleans
            reporting rigidity and maximality.
        """
        spread = self.maxima - self.minima
        free = spread > tol

        if not free.any():  # pragma: no cover
            return {"min": 0.0, "mean": 0.0, "max": 0.0, "rigid": True, "maximal": True}

        fractions = (self.values[free] - self.minima[free]) / spread[free]
        return {
            "min": float(fractions.min()),
            "mean": float(fractions.mean()),
            "max": float(fractions.max()),
            "rigid": bool(fractions.max() < tol),
            "maximal": bool(fractions.min() > 1 - tol),
        }

    def plot(self):  # pragma: no cover
        """
        Plot the shape function.

        Returns
        -------
        fig : plt.Figure
            The resulting figure.
        """
        from .plotting import GrayWynerPlotter

        return GrayWynerPlotter.plot_shape(self)
