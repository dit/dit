"""
Compute the hypercontractivity coefficient:
    s*(X||Y) = max_{U - X - Y} I[U:Y]/I[U:X]
"""

from __future__ import division

import numpy as np

from ..algorithms import BaseAuxVarOptimizer
from ..exceptions import ditException
from ..helpers import normalize_rvs
from ..multivariate.entropy import entropy
from ..multivariate.total_correlation import total_correlation


class HypercontractivityCoefficient(BaseAuxVarOptimizer):
    """
    Computes the hypercontractivity coefficient:

        max_{U - X - Y} I[U:Y]/I[U:X]
    """

    _shotgun = True

    def __init__(self, dist, rv_x=None, rv_y=None, bound=None, rv_mode=None):
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
        self._x = {0}
        self._y = {1}
        self._u = {3}
        super(HypercontractivityCoefficient, self).__init__(dist, [rv_x, rv_y], [], rv_mode=rv_mode)

        theoretical_bound = self._full_shape[self._proxy_vars[0]] + 1
        bound = min(bound, theoretical_bound) if bound else theoretical_bound

        self._construct_auxvars([({0}, bound)])

    def _objective(self):
        """
        The hypercontractivity coefficient to minimize.

        Returns
        -------
        obj : func
            The objective function.
        """
        mi_a = self._mutual_information(self._u, self._y)
        mi_b = self._mutual_information(self._u, self._x)

        def objective(self, x):
            """
            Compute I[U:Y]/I[U:X]

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
            a = mi_a(pmf)
            b = mi_b(pmf)
            return -(a/b) if not np.isclose(b, 0.0) else np.inf

        return objective

    construct_initial = BaseAuxVarOptimizer.construct_random_initial


def hypercontractivity_coefficient(dist, rvs, bound=None, niter=None, rv_mode=None):
    """
    Computes the hypercontractivity coefficient:

        s*(X||Y) = max_{U - X - Y} I[U:Y]/I[U:X]

    Parameters
    ----------
    dist : Distribution
        The distribution of interest.
    rvs : iterable of iterables, len(rvs) == 2
        The variables to compute the hypercontractivity coefficient of.
        Order is important.
    bound : int, None
        An external bound on the size of `U`. If None, |U| <= |X|+1.
    niter : int, None
        The number of basin-hopping steps to perform. If None, use the default.
    rv_mode : str, None
        Specifies how to interpret `rvs` and `crvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `crvs` and `rvs` are interpreted as random variable indices. If
        equal to 'names', the the elements are interpreted as random
        variable names. If `None`, then the value of `dist._rv_mode` is
        consulted, which defaults to 'indices'.

    Returns
    -------
    hc : float
        The hypercontractivity coefficient.
    """
    rvs, _, rv_mode = normalize_rvs(dist, rvs, None, rv_mode)

    if len(rvs) != 2:
        msg = 'Hypercontractivity coefficient can only be computed for 2 variables, not {}.'.format(len(rvs))
        raise ditException(msg)

    # test some special cases:
    if np.isclose(total_correlation(dist, rvs), 0.0):
        return 0.0
    elif np.isclose(entropy(dist, rvs[1], rvs[0]), 0.0):
        return 1.0
    else:
        hc = HypercontractivityCoefficient(dist, rvs[0], rvs[1], bound=bound, rv_mode=rv_mode)
        hc.optimize(niter=niter)
        return -hc.objective(hc._optima)
