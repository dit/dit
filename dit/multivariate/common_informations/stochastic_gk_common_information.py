"""
Stochastic Gacs-Korner common information. See "Gacs-Korner Common Information
Variational Autoencoder" for details.
"""

import numpy as np

from ...algorithms import BaseAuxVarOptimizer
from ...helpers import normalize_rvs
from ...utils import unitful

__all__ = (
    'StochasticGKCommonInformation',
)


class StochasticGKCommonInformation(BaseAuxVarOptimizer):
    """
    Abstract base class for constructing auxiliary variables which render a set
    of variables conditionally independent.
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
            `rvs=dist.rvs`.
        crvs : list, None
            A single list of indexes specifying the random variables to
            condition on. If None, then no variables are conditioned on.
        bound : int
            Place an artificial bound on the size of W.
        rv_mode : str, None
            Specifies how to interpret `rvs` and `crvs`. Valid options are:
            {'indices', 'names'}. If equal to 'indices', then the elements of
            `crvs` and `rvs` are interpreted as random variable indices. If
            equal to 'names', the the elements are interpreted as random
            variable names. If `None`, then the value of `dist._rv_mode` is
            consulted, which defaults to 'indices'.
        """
        super().__init__(dist, rvs=rvs, crvs=crvs, rv_mode=rv_mode)

        theoretical_bound = self.compute_bound()
        bound = min(bound, theoretical_bound) if bound else theoretical_bound

        self._construct_auxvars([({0}, bound)])

        self.constraints += [{'type': 'eq',
                              'fun': self.constraint_match_conditional_distributions,
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
        return 2 * min(self._shape[:len(self._rvs)]) + 1

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
        for rv in self._rvs:
            others = tuple(self._rvs - {rv})
            p_xyz = joint.sum(axis=others)
            p_xy = p_xyz.sum(axis=2, keepdims=True)
            p_z_g_xy = p_xyz / p_xy
            marginals.append(p_z_g_xy)

        delta = 0
        target_marginal = marginals[0]
        for idx in idxs:
            for i, m in zip(idx[1:], marginals[1:], strict=True):
                delta += ((target_marginal[idx[0]] - m[i])**2).sum()

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
def stochastic_gk_common_information(dist, rvs=None, crvs=None, niter=None, maxiter=1000, polish=1e-6, bound=None, rv_mode=None):
    """
    Compute the functional common information, F, of `dist`. It is the entropy
    of the smallest random variable W such that all the variables in `rvs` are
    rendered independent conditioned on W, and W is a function of `rvs`.

    Parameters
    ----------
    dist : Distribution
        The distribution from which the functional common information is
        computed.
    rvs : list, None
        A list of lists. Each inner list specifies the indexes of the random
        variables used to calculate the total correlation. If None, then the
        total correlation is calculated over all random variables, which is
        equivalent to passing `rvs=dist.rvs`.
    crvs : list, None
        A single list of indexes specifying the random variables to condition
        on. If None, then no variables are conditioned on.
    rv_mode : str, None
        Specifies how to interpret `rvs` and `crvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `crvs` and `rvs` are interpreted as random variable indices. If equal
        to 'names', the the elements are interpreted as random variable names.
        If `None`, then the value of `dist._rv_mode` is consulted, which
        defaults to 'indices'.

    Returns
    -------
    F : float
        The functional common information.
    """
    rvs, crvs, rv_mode = normalize_rvs(dist, rvs, crvs, rv_mode)

    # dtc = dual_total_correlation(dist, rvs, crvs, rv_mode)
    # ent = entropy(dist, rvs, crvs, rv_mode)
    # if np.isclose(dtc, ent):
    #     return dtc

    sgkci = StochasticGKCommonInformation(dist, rvs, crvs, bound, rv_mode)
    sgkci.optimize(niter=niter, maxiter=maxiter, polish=polish)
    return -sgkci.objective(sgkci._optima)
