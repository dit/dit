"""
Auxiliary-variable optimizer for the generalized Gray-Wyner network.

The Gray-Wyner network (Gray & Wyner 1974) has a single encoder observing a
correlated source vector ``(X_1, ..., X_n)`` and ``n`` decoders. The encoder
emits one *common* message (rate ``R_0``, sent to every decoder) and ``n``
*private* messages (rate ``R_i`` to decoder ``i``). Decoder ``i`` reconstructs
``X_i`` to within distortion ``D_i``.

The achievable rate region (lossless: Gray & Wyner 1974; lossy: Viswanatha,
Akyol, & Rose 2014) is the set of ``(R_0, R_1, ..., R_n)`` for which there is
an auxiliary variable ``W`` with

    R_0 >= I(X_1, ..., X_n : W)
    R_i >= R_{X_i | W}(D_i)        for each i,

where ``R_{X_i | W}(D_i)`` is the conditional rate-distortion function of
``X_i`` given ``W``. In the lossless case (``D_i = 0`` with a Hamming
distortion) this reduces to ``R_i >= H(X_i | W)``.

Because the region is convex, its lower boundary is traced by minimizing a
weighted sum of rates ``lambda_0 R_0 + sum_i lambda_i R_i`` over ``W`` (and, in
the lossy case, the reconstruction test channels ``q(x_hat_i | x_i, w)``
subject to the distortion budgets). Sweeping the weights sweeps the Pareto
surface.
"""

from collections import namedtuple

import numpy as np

from ...algorithms import BaseAuxVarOptimizer
from ...exceptions import ditException
from ...math import prod

__all__ = (
    "GrayWynerOptimizer",
    "GrayWynerPoint",
)


GrayWynerPoint = namedtuple("GrayWynerPoint", ["common", "private"])


def hamming_matrix(k):
    """
    Construct a ``k x k`` Hamming distortion matrix.

    Parameters
    ----------
    k : int
        The alphabet size.

    Returns
    -------
    d : np.ndarray
        The distortion matrix, ``1 - I``.
    """
    return 1 - np.eye(k)


class GrayWynerOptimizer(BaseAuxVarOptimizer):
    """
    Minimize a weighted sum of Gray-Wyner rates over the common auxiliary
    variable ``W`` (and reconstruction test channels for lossy decoders).
    """

    name = "gray-wyner"

    def __init__(self, dist, lambdas, rvs=None, crvs=None, distortions=None, bounds=None, markov=False, bound=None):
        """
        Initialize the optimizer.

        Parameters
        ----------
        dist : Distribution
            The distribution of interest.
        lambdas : iterable of float
            The non-negative weights ``(lambda_0, lambda_1, ..., lambda_n)``
            placed on the common rate and each private rate. Must have length
            ``n + 1`` where ``n`` is the number of source groups.
        rvs : list of lists, None
            The source groups ``X_1, ..., X_n``. If None, each variable of
            `dist` is treated as its own source.
        crvs : list, None
            Variables to condition the entire network on. If None, none.
        distortions : list, None
            A length-``n`` list giving, for each decoder, either ``None``
            (lossless reconstruction) or a square distortion matrix
            ``d[x_i, x_hat_i]``. If None, every decoder is lossless.
        bounds : list, None
            A length-``n`` list of per-decoder distortion budgets ``D_i``.
            Ignored for lossless decoders. If None, every budget is 0.
        markov : bool
            If True, constrain the sources to be conditionally independent
            given ``W`` (``I(X_1 ; ... ; X_n | W) = 0``). This is the
            constraint defining the (lossy) Wyner common information; it is not
            imposed for the general rate region.
        bound : int, None
            An optional cap on the cardinality of ``W``. If None, the
            Caratheodory-style bound from :meth:`compute_bound` is used.
        """
        super().__init__(dist, rvs=rvs, crvs=crvs)

        self._sources = sorted(self._rvs)
        n = len(self._sources)

        self._lambdas = np.asarray(lambdas, dtype=float)
        if self._lambdas.size != n + 1:
            msg = f"`lambdas` must have length {n + 1} (n + 1), got {self._lambdas.size}."
            raise ditException(msg)
        if np.any(self._lambdas < 0):
            msg = "`lambdas` must be non-negative."
            raise ditException(msg)

        distortions = [None] * n if distortions is None else list(distortions)
        budgets = [0.0] * n if bounds is None else [float(b) for b in bounds]
        if len(distortions) != n or len(budgets) != n:
            msg = f"`distortions` and `bounds` must each have length {n}."
            raise ditException(msg)

        # A decoder is lossy only when it has both a distortion matrix and a
        # positive budget; otherwise the (exact, robust) lossless path is used.
        self._lossy = [d is not None and b > 0 for d, b in zip(distortions, budgets, strict=True)]
        self._distortions = distortions
        self._budgets = budgets

        w_bound = self.compute_bound()
        w_bound = min(bound, w_bound) if bound else w_bound

        w_idx = len(self._all_vars)
        self._W = {w_idx}

        # For the general rate region the reconstruction at decoder i may use
        # its private description, so the test channel is q(x_hat_i | x_i, w)
        # and the private rate is the conditional rate-distortion I(X_i:X_hat_i
        # | W). For the Wyner common information (`markov`) the reconstruction
        # must be a function of the common message W alone, q(x_hat_i | w), so
        # that the distortion budget genuinely couples to the choice of W.
        self._markov = markov
        auxvars = [(set(self._rvs | self._crvs), w_bound)]
        self._recon_idx = {}
        next_idx = w_idx + 1
        for i, lossy in zip(self._sources, self._lossy, strict=True):
            if lossy:
                k_i = self._shape[i]
                recon_bases = {w_idx} if markov else {i, w_idx}
                auxvars.append((recon_bases, k_i))
                self._recon_idx[i] = next_idx
                next_idx += 1

        self._construct_auxvars(auxvars)

        # Rate functions.
        self._rate_common = self._conditional_mutual_information(self._rvs, self._W, self._crvs)
        self._rate_private = []
        self._distortion_funcs = []
        for i, lossy in zip(self._sources, self._lossy, strict=True):
            cond = self._W | self._crvs
            if lossy:
                xhat = {self._recon_idx[i]}
                self._rate_private.append(self._conditional_mutual_information({i}, xhat, cond))
                self._distortion_funcs.append(self._make_distortion(i, self._recon_idx[i], self._distortions[i]))
            else:
                self._rate_private.append(self._entropy({i}, cond))
                self._distortion_funcs.append(None)

        # Distortion budget constraints for lossy decoders.
        for i, lossy in zip(self._sources, self._lossy, strict=True):
            if lossy:
                self.constraints.append(
                    {
                        "type": "ineq",
                        "fun": self._make_distortion_constraint(i),
                    }
                )

        # Optional conditional-independence constraint (Wyner common info).
        if markov and len(self._sources) > 1:
            tc = self._total_correlation(self._rvs, self._W | self._crvs)
            self.constraints.append(
                {
                    "type": "eq",
                    "fun": lambda x: tc(self.construct_joint(x)),
                }
            )

        self._default_hops = max(self._default_hops, 10)

    def compute_bound(self):
        """
        Caratheodory-style cardinality bound on ``W``.

        Returns
        -------
        bound : int
            The bound on the alphabet size of ``W``.
        """
        source_card = prod(self._shape[i] for i in self._sources)
        crv_card = prod(self._shape[c] for c in self._crvs) if self._crvs else 1
        return int(source_card * crv_card + 1)

    def _make_distortion(self, source, xhat, dmatrix):
        """
        Build a function computing the average distortion for a lossy decoder.

        Parameters
        ----------
        source : int
            The index of the source variable ``X_i``.
        xhat : int
            The index of the reconstruction variable ``X_hat_i``.
        dmatrix : np.ndarray
            The distortion matrix ``d[x_i, x_hat_i]``.

        Returns
        -------
        distortion : func
            A function mapping a joint pmf to the average distortion.
        """
        dmatrix = np.asarray(dmatrix, dtype=float)
        keep = {source, xhat}
        idx_sum = tuple(sorted(self._all_vars - keep))

        def distortion(pmf):
            # `source < xhat` always (sources are low indices, auxvars high),
            # so the surviving axes are ordered (x_i, x_hat_i).
            p = pmf.sum(axis=idx_sum)
            return float((p * dmatrix).sum())

        return distortion

    def _make_distortion_constraint(self, source):
        """
        Build a scipy-style inequality constraint ``D_i - <d> >= 0``.

        Parameters
        ----------
        source : int
            The index of the source variable ``X_i``.

        Returns
        -------
        constraint : func
            A function mapping an optimization vector to the constraint slack.
        """
        pos = self._sources.index(source)
        budget = self._budgets[pos]
        distortion = self._distortion_funcs[pos]

        def constraint(x):
            pmf = self.construct_joint(x)
            return budget - distortion(pmf)

        return constraint

    def rates(self, x=None):
        """
        Compute the Gray-Wyner rate point for an optimization vector.

        Parameters
        ----------
        x : np.ndarray, None
            An optimization vector. If None, use ``self._optima``.

        Returns
        -------
        point : GrayWynerPoint
            The common rate and the tuple of private rates.
        """
        if x is None:
            x = self._optima
        pmf = self.construct_joint(x)
        common = float(self._rate_common(pmf))
        private = tuple(float(rate(pmf)) for rate in self._rate_private)
        return GrayWynerPoint(common=common, private=private)

    def _objective(self):
        """
        The weighted sum of rates.

        Returns
        -------
        obj : func
            The objective function.
        """
        rate_common = self._rate_common
        rate_private = self._rate_private
        lambdas = self._lambdas

        def objective(self, x):
            pmf = self.construct_joint(x)
            obj = lambdas[0] * rate_common(pmf)
            for w, rate in zip(lambdas[1:], rate_private, strict=True):
                obj = obj + w * rate(pmf)
            return obj

        return objective
