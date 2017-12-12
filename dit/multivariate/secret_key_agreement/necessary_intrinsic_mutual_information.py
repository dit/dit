"""

"""

from __future__ import division

import numpy as np

from ... import Distribution, insert_rvf, modify_outcomes
from .. import total_correlation
from ...helpers import flatten, parse_rvs
from ...utils.optimization import Uniquifier, accept_test, colon

__all__ = [
    'lower_intrinsic_mutual_information',
    'necessary_intrinsic_mutual_information',
]

def lower_intrinsic_mutual_information(dist, rvs, crvs, rv_mode=None):
    """
    Compute a trivial lower-bound on the secret key agreement rate.

    Parameters
    ----------
    dist : Distribution
    rvs : iterable of iterables
    crvs : iterable
    rv_mode : str, None
        Specifies how to interpret `rvs` and `crvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `crvs` and `rvs` are interpreted as random variable indices. If
        equal to 'names', the the elements are interpreted as random
        variable names. If `None`, then the value of `dist._rv_mode` is
        consulted, which defaults to 'indices'.

    Returns
    -------
    lb : float
        A lower-bound on the secret key agreement rate.
    """
    a = total_correlation(dist, rvs, rv_mode=rv_mode)
    b = total_correlation(dist, [rvs[0], crvs], rv_mode=rv_mode)
    c = total_correlation(dist, [rvs[1], crvs], rv_mode=rv_mode)
    return max([0.0, a-b, a-c])


class NecessaryIntrinsicMutualInformation(object):
    """
    Compute a lower bound on the secret key agreement rate:

        I_X[X:Y \\uparrow\\uparrow Z] = max_{V - U - X - YZ} I[U:Y|V] - I[U:Z|V]
    """

    def __init__(self, dist, rv_x=None, rv_y=None, rv_z=None, rv_mode=None, bound_u=None, bound_v=None):
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
        rv_z : iterable
            The variables to consider `Z`.
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
        bound_v : int, None
            Specifies a bound on the size of the auxiliary random variable. If None,
            then the theoretical bound is used.
        """
        self._dist = dist.copy(base='linear')
        self._alphabet = self._dist.alphabet
        self._rv_x = parse_rvs(self._dist, rv_x, rv_mode)[1]
        self._rv_y = parse_rvs(self._dist, rv_y, rv_mode)[1]
        self._rv_z = parse_rvs(self._dist, rv_z, rv_mode)[1]
        self._dist = modify_outcomes(self._dist, tuple)

        # compress all random variables down to single vars
        self._unqs = [Uniquifier() for _ in range(3)]
        for unq, var in zip(self._unqs, [self._rv_x, self._rv_y, self._rv_z]):
            self._dist = insert_rvf(self._dist, lambda x: (unq(tuple(x[i] for i in var)),))

        self._shape = list(map(len, self._dist.alphabet))
        self._dist.make_dense()
        self._full_pmf = self._dist.pmf.reshape(self._shape)

        n = dist.outcome_length()
        self._pmf = self._full_pmf.sum(axis=tuple(range(n)))

        self._all_vars = set(range(5))
        self._x = {0}
        self._y = {1}
        self._z = {2}
        self._u = {3}
        self._v = {4}

        theoretical_bound_u = self._shape[-3]  # |X|
        if bound_u is not None:
            self._bound_u = min(bound_u, theoretical_bound_u)
        else:
            self._bound_u = theoretical_bound_u

        theoretical_bound_v = self._shape[-3] ** 2  # |X|**2
        if bound_v is not None:
            self._bound_v = min(bound_v, theoretical_bound_v)
        else:
            self._bound_v = theoretical_bound_v

        self._default_hops = self._bound_u * self._bound_v

        self._mask_u = np.ones([self._shape[-3], 1, 1, self._bound_u]) / self._bound_u
        self._mask_v = np.ones([self._shape[-3], 1, 1, self._bound_u, self._bound_v]) / self._bound_v
        self._size_u = self._shape[-3] * self._bound_u
        self._size_v = self._size_u * self._bound_v

    def construct_random_initial(self):
        """
        Construct a random optimization vector.

        Returns
        -------
        x : ndarray
            A random optimization vector.
        """
        channel_u = np.random.random([self._shape[-3], self._bound_u]).flatten()
        channel_v = np.random.random([self._shape[-3], self._bound_u, self._bound_v]).flatten()
        return np.concatenate([channel_u, channel_v], axis=0)

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
        u_part = x[:self._size_u]
        v_part = x[self._size_u:]

        channel_u = u_part.reshape([self._shape[-3], 1, 1, self._bound_u])
        channel_u /= channel_u.sum(axis=tuple(self._u), keepdims=True)
        channel_u[np.isnan(channel_u)] = self._mask_u[np.isnan(channel_u)]

        channel_v = v_part.reshape([self._shape[-3], 1, 1, self._bound_u, self._bound_v])
        channel_v /= channel_v.sum(axis=tuple(self._v), keepdims=True)
        channel_v[np.isnan(channel_v)] = self._mask_v[np.isnan(channel_v)]

        if full:
            slc = [np.newaxis] * (len(self._shape) - 3) + [colon] * 4
            joint = self._full_pmf[..., np.newaxis] * channel_u[slc]
            slc += [colon]
            joint = joint[..., np.newaxis] * channel_v[slc]
        else:
            slc = [colon] * 4
            joint = self._pmf[..., np.newaxis] * channel_u[slc]
            slc += [colon]
            joint = joint[..., np.newaxis] * channel_v[slc]

        return joint

    def _conditional_mutual_information(self, pmf, X, Y, Z):
        """
        Compute the conditional mutual information, I[X:Y|Z].

        Parameters
        ----------
        pmf : np.ndarray
            The joint probability distributions.
        X : collection
            The indices to consider as the X variable.
        Y : collection
            The indices to consider as the Y variable.
        Z : collection
            The indices to consider as the Z variable.

        Returns
        -------
        cmi : float
            The conditional mutual information.
        """
        p_XYZ = pmf.sum(axis=tuple(self._all_vars - (X | Y | Z)), keepdims=True)
        p_XZ = pmf.sum(axis=tuple(self._all_vars - (X | Z)), keepdims=True)
        p_YZ = pmf.sum(axis=tuple(self._all_vars - (Y | Z)), keepdims=True)
        p_Z = pmf.sum(axis=tuple(self._all_vars - Z), keepdims=True)

        cmi = np.nansum(p_XYZ * np.log2(p_Z * p_XYZ / p_XZ / p_YZ))

        return cmi

    def objective(self, x):
        """
        The multivariate mutual information to minimize.

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

        # I[U:Y|V]
        a = self._conditional_mutual_information(pmf, self._u, self._y, self._v)

        # I[U:Z|V]
        b = self._conditional_mutual_information(pmf, self._u, self._z, self._v)

        return -(a - b)

    def optimize(self, x0=None, nhops=None, polish=1e-6):
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
        from scipy.optimize import basinhopping

        if x0 is not None:
            x = x0
        else:
            x = self.construct_random_initial()

        if nhops is None:
            nhops = self._default_hops

        minimizer_kwargs = {'method': 'L-BFGS-B',
                            'bounds': [(0, 1)] * x.size,
                            }

        res = basinhopping(func=self.objective,
                           x0=x,
                           minimizer_kwargs=minimizer_kwargs,
                           niter=nhops,
                           accept_test=accept_test,
                           )

        self._optima = res.x

        if polish:
            self._polish(cutoff=polish)

    def _polish(self, cutoff=1e-6):
        """
        Improve the solution found by the optimizer.

        Parameters
        ----------
        cutoff : float
            Set probabilities lower than this to zero, reducing the total
            optimization dimension.
        """
        from scipy.optimize import minimize

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
            The original distribution, plus `z_bar'.
        """
        if x is None:
            x = self._optima

        alphabets = list(self._alphabet)

        try:
            # if all outcomes are strings, make new variable strings too.
            ''.join(flatten(alphabets))
            alphabets += [(digits + ascii_letters)[:self._bound_u]]
            alphabets += [(digits + ascii_letters)[:self._bound_v]]
            string = True
        except:
            alphabets += [list(range(self._bound_u))]
            alphabets += [list(range(self._bound_v))]
            string = False

        joint = self.construct_joint(x, full=True)
        joint = joint.sum(axis=tuple(range(-4, -1)))
        outcomes, pmf = zip(*[(o, p) for o, p in np.ndenumerate(joint) if p > cutoff])
        outcomes = [tuple(a[i] for i, a in zip(o, alphabets)) for o in outcomes]

        if string:
            outcomes = [''.join(o) for o in outcomes]

        # normalize, in case cutoffs removed a significant amount of pmf
        pmf = np.asarray(pmf)
        pmf /= pmf.sum()

        d = Distribution(outcomes, pmf)
        return d


def necessary_intrinsic_mutual_information(dist, rvs, crvs, rv_mode=None):
    """
    Compute a non-trivial lower bound on secret key agreement rate.

    Paramters
    ---------
    dist : Distribution
    rvs : iterable of iterables
    crvs : iterable
    rv_mode : str, None
        Specifies how to interpret `rvs` and `crvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `crvs` and `rvs` are interpreted as random variable indices. If
        equal to 'names', the the elements are interpreted as random
        variable names. If `None`, then the value of `dist._rv_mode` is
        consulted, which defaults to 'indices'.

    Returns
    -------
    """
    first = NecessaryIntrinsicMutualInformation(dist, rvs[0], rvs[1], crvs, rv_mode=rv_mode)
    first.optimize()
    A = -first.objective(first._optima)

    second = NecessaryIntrinsicMutualInformation(dist, rvs[1], rvs[0], crvs, rv_mode=rv_mode)
    second.optimize()
    B = -second.objective(second._optima)

    return max(A, B)
