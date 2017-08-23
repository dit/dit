"""
The I_proj measure as proposed by Harder et al.
"""

from __future__ import division

import numpy as np
from scipy.optimize import minimize

from .pid import BaseBivariatePID

from .. import Distribution
from ..divergences.pmf import relative_entropy
from ..exceptions import ditException


class MinDKLOptimizer(object):
    """
    An optimizer to find the minimum D_KL(p||q) given p and a
    restriction on the domain of q.
    """

    def __init__(self, dist, domain):
        """
        Initialize the optimizer.

        Parameters
        ----------
        dist : Distribution
            The distribution `p`.
        domain : list of lists
            The pmfs defining the domain over which `q` is optimized.
        """
        self._dist = dist
        self._p = dist.pmf
        self._domain = np.stack(domain)
        self._domain_inv = np.linalg.pinv(self._domain)

    def _q(self, x):
        """
        Transform `x` into a distribution.

        Parameters
        ----------
        x : np.ndarray
            Optimization vector

        Returns
        -------
        q : np.ndarray
            The distribution resulting from `x`.
        """
        q = np.dot(x, self._domain)
        q /= q.sum()
        return q

    def objective(self, x):
        """
        The objective to minimize, D(p||q).

        Parameters
        ----------
        x : np.ndarray
            The optimization vector.

        Returns
        -------
        dkl : float
            The Kullback-Leibler divergence.
        """
        q = self._q(x)
        dkl = relative_entropy(self._p, q)
        return dkl

    def optimize(self):
        """
        Perform the optmization.

        Notes
        -----
        The optimization is convex, so we use sp.optimize.minimize.
        """
        x0 = np.dot(self._p, self._domain_inv)

        bounds = [(0, 1)] * x0.size

        res = minimize(fun=self.objective,
                       x0=x0,
                       method='L-BFGS-B',
                       bounds=bounds,
                       options={'maxiter': 1000,
                                'ftol': 1e-7,
                                'eps': 1.4901161193847656e-08,
                                },
                       )

        if not res.success: # pragma: no cover
            msg = "Optimization failed: {}".format(res.message)
            raise ditException(msg)

        self._optima = res.x

    def construct_dist(self, q=None):
        """
        Construct a distribution from a vector.

        Paramters
        ---------
        q : np.ndarray, None
            The vector to turn in to a distribution. If None, use self._optima.

        Returns
        -------
        dist : Distribution
            The distribution of `q`.
        """
        if q is None: # pragma: no cover
            q = self._q(self._optima)

        dist = Distribution(self._dist.outcomes, q)

        return dist


def min_dkl(dist, domain):
    """
    Given a distribution and a domain, find the minimum D(p||q) where
    p is `dist` and q is in `domain`.

    Paramters
    ---------
    dist : Distribution
        The distribution for p.
    domain : list of lists
        The set of points whose closure q must live in.

    Returns
    -------
    dkl : float
        The minimum D(p||q) with q restricted to `domain`.
    """
    dist = dist.copy()
    dist.make_dense()
    optimizer = MinDKLOptimizer(dist, domain)
    optimizer.optimize()
    return optimizer.construct_dist()


def projected_information(dist, X, Y, Z):
    """
    I_Z^pi(X \searrow Y)

    Paramters
    ---------
    dist : Distribution
        The distribution to compute the projected information from.
    X : iterable
        The aggregate variable X.
    Y : iterable
        The aggregate variable Y.
    Z : iterable
        The aggregate variable Z.

    Returns
    -------
    pi : float
        The projected information.
    """
    p_z_ys = dist.condition_on(rvs=Z, crvs=Y)[1]
    for d in p_z_ys:
        d.make_dense()
    domain = [d.pmf for d in p_z_ys]
    p_xz = dist.marginal(X + Z)
    p_z = dist.marginal(Z)
    p_x, p_z_xs = dist.condition_on(rvs=Z, crvs=X)

    vals = []
    for x, p_z_x in zip(p_x.outcomes, p_z_xs):
        p_proj_z = min_dkl(p_z_x, domain)
        for z in p_z.outcomes:
            vals.append(p_xz[x + z] * np.log2(p_proj_z[z] / p_z[z]))
    val = np.nansum(vals)

    return val


def i_proj(d, inputs, output):
    """
    Compute I_proj(inputs : output) = min{PI(X \searrow Y), PI(Y \searrow X)}

    Parameters
    ----------
    d : Distribution
        The distribution to compute i_proj for.
    inputs : iterable of iterables, len(inputs) == 2
        The input variables.
    output : iterable
        The output variable.

    Returns
    -------
    iproj : float
        The value of I_proj.
    """
    if len(inputs) != 2: # pragma: no cover
        msg = "This method needs exact two inputs, {} given.".format(len(inputs))
        raise ditException(msg)

    pi_0 = projected_information(d, inputs[0], inputs[1], output)
    pi_1 = projected_information(d, inputs[1], inputs[0], output)
    return min(pi_0, pi_1)


class PID_Proj(BaseBivariatePID):
    """
    The Harder et al partial information decomposition.
    """
    _name = "I_proj"
    _measure = staticmethod(i_proj)
