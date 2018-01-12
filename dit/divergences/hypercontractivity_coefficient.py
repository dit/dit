"""
Compute the hypercontractivity coefficient:
    s*(X||Y) = max_{U - X - Y} I[U:Y]/I[U:X]
"""

from __future__ import division

from string import ascii_letters, digits

import numpy as np
from scipy.optimize import basinhopping, minimize

from .. import Distribution, insert_rvf, modify_outcomes
from ..exceptions import ditException
from ..helpers import flatten, normalize_rvs, parse_rvs
from ..multivariate.entropy import entropy
from ..multivariate.total_correlation import total_correlation
from ..utils.optimization import (BasinHoppingCallBack,
                                  Uniquifier,
                                  accept_test,
                                  basinhop_status,
                                  colon
                                  )


class HypercontractivityCoefficient(object):
    """
    Computes the hypercontractivity coefficient:

        max_{U - X - Y} I[U:Y]/I[U:X]
    """

    def __init__(self, dist, rv_x=None, rv_y=None, rv_mode=None, bound_u=None):
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
        rv_mode : str, None
            Specifies how to interpret `rvs` and `crvs`. Valid options are:
            {'indices', 'names'}. If equal to 'indices', then the elements of
            `crvs` and `rvs` are interpreted as random variable indices. If
            equal to 'names', the the elements are interpreted as random
            variable names. If `None`, then the value of `dist._rv_mode` is
            consulted, which defaults to 'indices'.
        bound_u : int, None
            Specifies a bound on the size of the auxiliary random variable. If None,
            then the theoretical bound is used.
        """
        self._dist = dist.copy(base='linear')
        self._alphabet = self._dist.alphabet
        self._rv_x = parse_rvs(self._dist, rv_x, rv_mode)[1]
        self._rv_y = parse_rvs(self._dist, rv_y, rv_mode)[1]
        self._dist = modify_outcomes(self._dist, tuple)

        # compress all random variables down to single vars
        self._unqs = [Uniquifier() for _ in range(2)]
        for unq, var in zip(self._unqs, [self._rv_x, self._rv_y]):
            self._dist = insert_rvf(self._dist, lambda x: (unq(tuple(x[i] for i in var)),))

        self._shape = list(map(len, self._dist.alphabet))
        self._dist.make_dense()
        self._full_pmf = self._dist.pmf.reshape(self._shape)

        n = dist.outcome_length()
        self._pmf = self._full_pmf.sum(axis=tuple(range(n)))

        self._all_vars = set(range(3))
        self._x = {0}
        self._y = {1}
        self._u = {2}

        theoretical_bound_u = self._shape[-2] + 1
        if bound_u is not None:
            self._bound_u = min(bound_u, theoretical_bound_u)
        else:
            self._bound_u = theoretical_bound_u

        self._default_hops = 2*self._bound_u

        self._mask_u = np.ones([self._shape[-2], 1, self._bound_u]) / self._bound_u
        self._size_u = self._shape[-2] * self._bound_u

    def construct_random_initial(self):
        """
        Construct a random optimization vector.

        Returns
        -------
        x : ndarray
            A random optimization vector.
        """
        channel_u = np.random.random([self._shape[-2], self._bound_u]).flatten()
        return channel_u

    def construct_copy_initial(self):
        """
        Construct a random optimization vector.

        Returns
        -------
        x : ndarray
            An identity optimization vector.
        """
        channel_u = np.eye(self._shape[-2], self._bound_u).flatten()
        return channel_u

    def construct_joint(self, x, full=False):
        """
        Construct the joint distribution.

        Parameters
        ----------
        x : ndarray
            An optimization vector.

        Returns
        -------
        joint : float
            The joint distribution resulting from the distribution passed
            in and the optimization vector.
        """
        channel_u = x.reshape([self._shape[-2], 1, self._bound_u])
        channel_u /= channel_u.sum(axis=tuple(self._u), keepdims=True)
        channel_u[np.isnan(channel_u)] = self._mask_u[np.isnan(channel_u)]

        if full:
            slc = [np.newaxis] * (len(self._shape) - 2) + [colon] * 3
            joint = self._full_pmf[..., np.newaxis] * channel_u[slc]
        else:
            slc = [colon] * 3
            joint = self._pmf[..., np.newaxis] * channel_u[slc]

        return joint

    def _mutual_information(self, pmf, X, Y):
        """
        Compute the mutual information, I[X:Y].

        Parameters
        ----------
        pmf : np.ndarray
            The joint probability distributions.
        X : collection
            The indices to consider as the X variable.
        Y : collection
            The indices to consider as the Y variable.

        Returns
        -------
        mi : float
            The mutual information.
        """
        p_XY = pmf.sum(axis=tuple(self._all_vars - (X | Y)), keepdims=True)
        p_X = pmf.sum(axis=tuple(self._all_vars - X), keepdims=True)
        p_Y = pmf.sum(axis=tuple(self._all_vars - Y), keepdims=True)

        mi = np.nansum(p_XY * np.log2(p_XY / (p_X * p_Y)))

        return mi

    def objective(self, x):
        """
        The hypercontractivity coefficient to minimize.

        Parameters
        ----------
        x : ndarray
            An optimization vector.

        Returns
        -------
        obj : float
            The value of the objective function.
        """
        pmf = self.construct_joint(x)

        # I[U:Y]
        a = self._mutual_information(pmf, self._u, self._y)

        # I[U:X]
        b = self._mutual_information(pmf, self._u, self._x)

        return -(a/b) if not np.isclose(b, 0.0) else np.inf

    def optimize(self, x0=None, nhops=None, polish=1e-8):
        """
        Perform the optimization.

        Parameters
        ----------
        x0 : ndarray, None
            Initial optimization vector. If None, use a random vector.
        nhops : int, None
            The number of basin hops to perform while optimizing. If None,
            hop a number of times equal to the dimension of the conditioning
            variable(s).
        polish : float
            If `polish` > 0, the found minima is improved by removing small
            components and optimized with lower tolerances.
        """
        if x0 is not None:
            x = x0
        else:
            x = self.construct_copy_initial()

        if nhops is None:
            nhops = self._default_hops

        minimizer_kwargs = {'method': 'L-BFGS-B',
                            'bounds': [(0, 1)] * x.size,
                            }

        res1 = minimize(fun=self.objective,
                        x0=x,
                        **minimizer_kwargs
                        )

        self._callback = BasinHoppingCallBack({}, None)

        res2 = basinhopping(func=self.objective,
                            x0=res1.x if res1.success else x,
                            minimizer_kwargs=minimizer_kwargs,
                            niter=nhops,
                            accept_test=accept_test,
                            )

        success, msg = basinhop_status(res2)
        if success:
            self._optima = res2.x
        else: # pragma: no cover
            minimum = self._callback.minimum()
            if minimum is not None:
                self._optima = minimum
            elif res1.success:
                self._optima = res1.x
            else:
                print(res1)
                print(res2)
                msg = "No minima found."
                raise Exception(msg)

        if polish:
            self._polish(cutoff=polish)

    def _polish(self, cutoff=1e-8):
        """
        Improve the solution found by the optimizer.

        Parameters
        ----------
        cutoff : float
            Set probabilities lower than this to zero, reducing the total
            optimization dimension.
        """
        x0 = self._optima
        count = (x0 < cutoff).sum()
        x0[x0 < cutoff] = 0

        kwargs = {'method': 'L-BFGS-B',
                  'bounds': [(0, 0) if np.isclose(x, 0) else (0, 1) for x in x0],
                  'tol': None,
                  'callback': None,
                  'options': {'maxiter': 1000,
                              'ftol': 15e-11,
                              'eps': 1.4901161193847656e-12,
                              },
                  }

        res = minimize(fun=self.objective,
                       x0=x0,
                       **kwargs
                       )

        self._optima = res.x

        if count < (res.x < cutoff).sum():
            self._polish(cutoff=cutoff)

    def construct_distribution(self, x=None, cutoff=1e-5):
        """
        Construct the distribution.

        Parameters
        ----------
        x : ndarray
            An optimization vector.
        cutoff : float
            Ignore probabilities smaller than this.

        Returns
        -------
        d : Distribution
            The original distribution, plus U and V.
        """
        if x is None:
            x = self._optima

        alphabets = list(self._alphabet)

        try:
            # if all outcomes are strings, make new variable strings too.
            ''.join(flatten(alphabets))
            alphabets += [(digits + ascii_letters)[:self._bound_u]]
            string = True
        except:
            alphabets += [list(range(self._bound_u))]
            string = False

        joint = self.construct_joint(x, full=True)
        joint = joint.sum(axis=tuple(range(-3, -1)))
        outcomes, pmf = zip(*[(o, p) for o, p in np.ndenumerate(joint) if p > cutoff])
        outcomes = [tuple(a[i] for i, a in zip(o, alphabets)) for o in outcomes]

        if string:
            outcomes = [''.join(o) for o in outcomes]

        # normalize, in case cutoffs removed a significant amount of pmf
        pmf = np.asarray(pmf)
        pmf /= pmf.sum()

        d = Distribution(outcomes, pmf)
        return d


def hypercontractivity_coefficient(dist, rvs, rv_mode=None, bound_u=None, nhops=None):
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
    rv_mode : str, None
        Specifies how to interpret `rvs` and `crvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `crvs` and `rvs` are interpreted as random variable indices. If
        equal to 'names', the the elements are interpreted as random
        variable names. If `None`, then the value of `dist._rv_mode` is
        consulted, which defaults to 'indices'.
    bound_u : int, None
        An external bound on the size of `U`. If None, |U| <= |X|+1.
    nhops : int, None
        The number of basin-hopping steps to perform. If None, use the default.

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
        hc = HypercontractivityCoefficient(dist, rvs[0], rvs[1], rv_mode=rv_mode, bound_u=bound_u)
        hc.optimize(nhops=nhops)
        return -hc.objective(hc._optima)
