"""
The beta-approximate common information (information-correlation function).

Generalizes Wyner and Gacs-Korner common informations using conditional
maximal correlation as a privacy/commonness constraint.

.. math::

    C_{\\beta}(X_1 : \\ldots : X_n | Z) =
        \\inf_{P_{U|X_1 \\ldots X_n Z} :\\;
              \\max_{i \\neq j} \\rho_m(X_i; X_j | U, Z) \\le \\beta}
        I(X_1 \\ldots X_n ; U | Z)

Special cases:
    - beta = 0  =>  Wyner common information
    - beta -> 1 =>  Gacs-Korner common information
    - beta >= rho_m(X;Y) => 0

Reference
---------
L. Yu, H. Li, and C. W. Chen, "Generalized Common Informations: Measuring
Commonness by the Conditional Maximal Correlation," arXiv:1610.09289v3, 2017.
"""

from itertools import combinations

import numpy as np

from ...algorithms import BaseAuxVarOptimizer
from ...divergences.maximum_correlation import maximum_correlation
from ...helpers import normalize_rvs
from ...math import prod
from ...utils import unitful
from ..dual_total_correlation import dual_total_correlation

__all__ = ("beta_common_information",)


def _maxcorr_from_2d(pXY):
    """Maximal correlation from a 2-d joint pmf (no conditioning)."""
    pX = pXY.sum(axis=1, keepdims=True)
    pY = pXY.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        Q = pXY / (np.sqrt(pX) * np.sqrt(pY))
    Q = np.where(np.isfinite(Q), Q, 0.0)
    if min(Q.shape) < 2:
        return 0.0
    s = np.linalg.svd(Q, compute_uv=False)
    return float(s[1]) if len(s) > 1 else 0.0


def _conditional_maxcorr_from_3d(pXY_Z):
    """
    Compute rho_m(X; Y | Z) from a 3-dimensional array P(X, Y, Z).

    Uses the SVD characterization: rho_m = max_{z: P(z)>0} lambda_2(Q_z).
    Vectorised over all conditioning values using batched SVD.

    Parameters
    ----------
    pXY_Z : np.ndarray, shape (|X|, |Y|, |Z|)

    Returns
    -------
    rho : float
    """
    dx, dy, dz = pXY_Z.shape
    if dx < 2 or dy < 2:
        return 0.0

    pZ = pXY_Z.sum(axis=(0, 1))  # (dz,)
    live = pZ > 1e-15
    if not live.any():
        return 0.0

    # P(x,y|z) for live z, shape (n_live, dx, dy)
    pxy_gz = pXY_Z[:, :, live].transpose(2, 0, 1) / pZ[live, np.newaxis, np.newaxis]
    px_gz = pxy_gz.sum(axis=2, keepdims=True)  # (n_live, dx, 1)
    py_gz = pxy_gz.sum(axis=1, keepdims=True)  # (n_live, 1, dy)

    with np.errstate(divide="ignore", invalid="ignore"):
        Q_batch = pxy_gz / (np.sqrt(px_gz) * np.sqrt(py_gz))
    Q_batch = np.where(np.isfinite(Q_batch), Q_batch, 0.0)

    # Batched SVD: (n_live, dx, dy) -> singular values (n_live, min(dx,dy))
    s_all = np.linalg.svd(Q_batch, compute_uv=False)
    if s_all.shape[1] < 2:
        return 0.0

    return float(s_all[:, 1].max())


def _max_pairwise_maxcorr(joint, rvs, cond_axes):
    """
    Compute max_{i!=j in rvs} rho_m(X_i; X_j | cond_axes) from a joint pmf.

    Parameters
    ----------
    joint : np.ndarray
        The joint pmf; axes correspond to variable indices in ascending order.
    rvs : set of int
        The random variable indices.
    cond_axes : set of int
        The conditioning variable indices (e.g. crvs | arvs).

    Returns
    -------
    max_rho : float
    """
    rv_list = sorted(rvs)

    if len(rv_list) < 2:
        return 0.0

    max_rho = 0.0

    for i, j in combinations(rv_list, 2):
        other_rvs = tuple(rv for rv in rv_list if rv != i and rv != j)
        pij = joint.sum(axis=other_rvs) if other_rvs else joint

        if pij.ndim <= 2:
            rho = _maxcorr_from_2d(pij)
        else:
            pair_shape = pij.shape[:2]
            cond_size = int(np.prod(pij.shape[2:]))
            if cond_size == 0:
                continue
            pij_3d = pij.reshape(pair_shape[0], pair_shape[1], cond_size)
            rho = _conditional_maxcorr_from_3d(pij_3d)

        max_rho = max(max_rho, rho)

    return max_rho


class BetaCommonInformation(BaseAuxVarOptimizer):
    """
    Compute the beta-approximate common information:

    .. math::

        C_{\\beta} = \\inf_{P_{U|XZ}:\\;
                     \\max_{i \\neq j} \\rho_m(X_i; X_j | U, Z) \\le \\beta}
                    I(X; U | Z)

    Parameters
    ----------
    dist : Distribution
        The distribution of interest.
    rvs : list of lists, None
        The variable groupings.
    crvs : list, None
        Conditioning variables.
    beta : float
        The maximal-correlation threshold, 0 <= beta <= 1.
    bound : int, None
        Cardinality bound on the auxiliary variable U.
    """

    _PENALTY_WEIGHT = 200

    def __init__(self, dist, rvs=None, crvs=None, beta=0, bound=None):
        self._beta = beta
        self._objective_bound = dual_total_correlation(dist, rvs, crvs)
        super().__init__(dist, rvs=rvs, crvs=crvs)

        theoretical_bound = self.compute_bound()
        bound = min(bound, theoretical_bound) if bound else theoretical_bound

        self._construct_auxvars([(self._rvs | self._crvs, bound)])

        self.constraints += [
            {
                "type": "ineq",
                "fun": self.constraint_maximal_correlation,
            },
        ]

        self._default_hops = 5

        self._additional_options = {
            "options": {
                "maxiter": 1000,
                "ftol": 1e-6,
                "eps": 1.4901161193847656e-9,
            }
        }

    def compute_bound(self):
        """
        Cardinality bound on U from the Caratheodory-Fenchel theorem
        (Lemma 15a): |U| <= prod(|X_i|) + 1.

        Returns
        -------
        bound : int
        """
        return prod(self._shape[i] for i in self._rvs) + 1

    def constraint_maximal_correlation(self, x):
        """
        Inequality constraint: beta - max_{pairs} rho_m(X_i; X_j | U, Z).

        The constraint is satisfied (>= 0) when the maximum pairwise
        conditional maximal correlation does not exceed beta.

        Parameters
        ----------
        x : np.ndarray
            Optimization vector.

        Returns
        -------
        slack : float
            Non-negative when the constraint is satisfied.
        """
        joint = self.construct_joint(x)
        cond_axes = self._crvs | self._arvs
        max_rho = _max_pairwise_maxcorr(joint, self._rvs, cond_axes)
        return self._beta - max_rho

    def true_objective(self, x):
        """
        The unpenalized mutual information I(X_1...X_n ; U | Z).

        Parameters
        ----------
        x : np.ndarray
            An optimization vector.

        Returns
        -------
        mi : float
            The conditional mutual information.
        """
        pmf = self.construct_joint(x)
        return self._cmi(pmf)

    def _objective(self):
        """
        I(X_1...X_n ; U | Z) augmented with a quadratic penalty for
        violating the maximal-correlation constraint.  The penalty steers
        the optimizer toward the feasible region even when the constraint
        surface is non-smooth.

        Returns
        -------
        obj : callable
            The objective function.
        """
        self._cmi = self._conditional_mutual_information(self._rvs, self._arvs, self._crvs)
        beta = self._beta
        rvs = self._rvs
        cond_axes = self._crvs | self._arvs
        w = self._PENALTY_WEIGHT

        def objective(self, x):
            """
            Compute I[rvs : U | crvs] + penalty.

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
            mi = self._cmi(pmf)
            max_rho = _max_pairwise_maxcorr(pmf, rvs, cond_axes)
            violation = max(0.0, max_rho - beta)
            return mi + w * violation * violation

        return objective


@unitful
def beta_common_information(dist, beta, rvs=None, crvs=None, niter=None, maxiter=1000, polish=1e-6, bound=None):
    """
    Compute the beta-approximate common information (information-correlation
    function) of *dist*.

    .. math::

        C_{\\beta}(X_1 : \\ldots : X_n | Z) =
            \\inf_{P_{U|X_1 \\ldots X_n Z} :\\;
                  \\max_{i \\neq j} \\rho_m(X_i; X_j | U, Z) \\le \\beta}
            I(X_1 \\ldots X_n ; U | Z)

    This generalises Wyner common information (beta=0) and, in the limit
    beta -> 1, Gacs-Korner common information.  When beta >= rho_m(X;Y)
    the value is 0 because no auxiliary variable is needed.

    Parameters
    ----------
    dist : Distribution
        The distribution from which the common information is computed.
    beta : float
        The maximal-correlation threshold, 0 <= beta <= 1.
    rvs : list of lists, None
        A list of lists. Each inner list specifies the indexes of the random
        variables for one group. If None, each outcome coordinate forms its
        own group (equivalent to ``rvs=dist.rvs``).
    crvs : list, None
        A single list of indexes specifying the random variables to condition
        on. If None, no conditioning is applied.
    niter : int, None
        Number of basin-hopping restarts.
    maxiter : int
        Maximum iterations per local optimisation.
    polish : float, False
        If a float, perform a second optimisation pass with probabilities
        below this threshold zeroed out.  If False, skip polishing.
    bound : int, None
        Artificial bound on the cardinality of U.  If None, the theoretical
        bound from Lemma 15a is used.

    Returns
    -------
    c_beta : float
        The beta-approximate common information.
    """
    rvs, crvs = normalize_rvs(dist, rvs, crvs)

    if beta >= 1:
        return 0.0

    # Fast path: C_beta = 0 when beta >= max pairwise rho_m.
    all_below = True
    for i, j in combinations(range(len(rvs)), 2):
        rho = maximum_correlation(dist, [rvs[i], rvs[j]], crvs)
        if beta < rho - 1e-10:
            all_below = False
            break
    if all_below:
        return 0.0

    bci = BetaCommonInformation(dist, rvs, crvs, beta=beta, bound=bound)
    bci.optimize(niter=niter, maxiter=maxiter, polish=polish)
    return bci.true_objective(bci._optima)
