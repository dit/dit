"""
Stochastic Gacs-Korner common information. See "Gacs-Korner Common Information
Variational Autoencoder" for details.
"""

import numpy as np

from ...algorithms import BaseAuxVarOptimizer
from ...helpers import normalize_rvs
from ...utils import unitful

__all__ = (
    "StochasticGKCommonInformation",
    "stochastic_gk_common_information",
)


class StochasticGKCommonInformation(BaseAuxVarOptimizer):
    """
    Compute the stochastic Gacs-Korner common information: the maximum
    I(X_i; Z) over stochastic variables Z satisfying p(Z|X_i) = p(Z|X_j)
    for all jointly occurring (X_i, X_j).
    """

    name = ""
    description = ""

    def __init__(self, dist, rvs=None, crvs=None, bound=None):
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
            `rvs=dist.rvs`.
        crvs : list, None
            A single list of indexes specifying the random variables to
            condition on. If None, then no variables are conditioned on.
        bound : int
            Place an artificial bound on the size of W.
        """
        super().__init__(dist, rvs=rvs, crvs=crvs)

        theoretical_bound = self.compute_bound()
        bound = min(bound, theoretical_bound) if bound else theoretical_bound

        self._construct_auxvars([({0}, bound)])

        self.constraints += [
            {
                "type": "eq",
                "fun": self.constraint_match_conditional_distributions,
            },
        ]

    def compute_bound(self):
        """
        Return a bound on the cardinality of the auxiliary variable.

        Returns
        -------
        bound : int
            The bound on the size of W.
        """
        return 2 * min(self._shape[: len(self._rvs)]) + 1

    def constraint_match_conditional_distributions(self, x):
        """
        Ensure that p(z|x_i) = p(z|x_j) for all i, j.

        Parameters
        ----------
        x : np.ndarray
            An optimization vector.
        """
        joint = self.construct_joint(x)

        rv_joint = joint.sum(axis=tuple(self._crvs | self._arvs))

        idxs = [idx for idx, support in np.ndenumerate(~np.isclose(rv_joint, 0.0)) if support]

        marginals = []
        for rv in sorted(self._rvs):
            others = tuple(self._rvs - {rv})
            p_xyz = joint.sum(axis=others)
            p_xy = p_xyz.sum(axis=2, keepdims=True)
            p_z_g_xy = np.where(p_xy > 0, p_xyz / p_xy, 0.0)
            marginals.append(p_z_g_xy)

        delta = 0
        target_marginal = marginals[0]
        for idx in idxs:
            for i, m in zip(idx[1:], marginals[1:], strict=True):
                delta += ((target_marginal[idx[0]] - m[i]) ** 2).sum()

        return 100 * delta

    def _objective(self):
        """
        The mutual information between the auxiliary random variable and `rvs`.

        Returns
        -------
        obj : func
            The objective function.
        """
        conditional_mutual_information = self._conditional_mutual_information({min(self._rvs)}, self._arvs, self._crvs)

        def objective(self, x):
            """
            Compute I[rv_i : W | crvs]

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
            return -conditional_mutual_information(pmf)

        return objective


@unitful
def stochastic_gk_common_information(dist, rvs=None, crvs=None, niter=None, maxiter=1000, polish=1e-6, bound=None):
    """
    Compute the stochastic Gacs-Korner common information of `dist`. This is
    the maximum I(X_i; Z) over stochastic variables Z satisfying
    p(Z|X_i) = p(Z|X_j) for all jointly occurring (X_i, X_j). When Z is
    restricted to deterministic functions, this recovers the classical
    Gacs-Korner common information.

    Parameters
    ----------
    dist : Distribution
        The distribution from which the stochastic Gacs-Korner common
        information is computed.
    rvs : list, None
        A list of lists. Each inner list specifies the indexes of the random
        variables used to calculate the common information. If None, then it
        is calculated over all random variables, which is equivalent to
        passing `rvs=dist.rvs`.
    crvs : list, None
        A single list of indexes specifying the random variables to condition
        on. If None, then no variables are conditioned on.

    Returns
    -------
    SGK : float
        The stochastic Gacs-Korner common information.
    """
    rvs, crvs = normalize_rvs(dist, rvs, crvs)

    # dtc = dual_total_correlation(dist, rvs, crvs)
    # ent = entropy(dist, rvs, crvs)
    # if np.isclose(dtc, ent):
    #     return dtc

    sgkci = StochasticGKCommonInformation(dist, rvs, crvs, bound)
    sgkci.optimize(niter=niter, maxiter=maxiter, polish=polish)
    return -sgkci.objective(sgkci._optima)
