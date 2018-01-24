"""
Base class for the calculation of reduced and minimal intrinsic informations.
"""

from __future__ import division

from six import with_metaclass

from abc import ABCMeta, abstractmethod
from string import digits, ascii_letters

import numpy as np

from ... import Distribution, insert_rvf, modify_outcomes
from ...exceptions import ditException
from ...helpers import flatten, normalize_rvs, parse_rvs
from ...math import prod
from ...utils.optimization import Uniquifier, accept_test, colon


class BaseMoreIntrinsicMutualInformation(with_metaclass(ABCMeta, object)):
    """
    Compute the minimal intrinsic mutual information, a lower bound on the secret
    key agreement rate:

        I[X : Y \downarrow\downarrow\downarrow Z] = min_U I[X:Y|U] + I[XY:U|Z]
    """

    name = ""

    def __init__(self, dist, rvs=None, crvs=None, rv_mode=None, bound=None):
        """
        Initialize the optimizer.

        Parameters
        ----------
        dist : Distribution
            The distribution to compute the intrinsic mutual information of.
        rvs : list, None
            A list of lists. Each inner list specifies the indexes of the random
            variables used to calculate the intrinsic mutual information. If None,
            then it is calculated over all random variables, which is equivalent
            to passing `rvs=dist.rvs`.
        crvs : list
            A single list of indexes specifying the random variables to
            condition on.
        rv_mode : str, None
            Specifies how to interpret `rvs` and `crvs`. Valid options are:
            {'indices', 'names'}. If equal to 'indices', then the elements of
            `crvs` and `rvs` are interpreted as random variable indices. If
            equal to 'names', the the elements are interpreted as random
            variable names. If `None`, then the value of `dist._rv_mode` is
            consulted, which defaults to 'indices'.
        bound : int, None
            Specifies a bound on the size of the auxiliary random variable. If None,
            then the theoretical bound is used.
        """
        self._dist = dist.copy(base='linear')
        self._alphabet = self._dist.alphabet
        rvs, crvs, self._rv_mode = normalize_rvs(self._dist, rvs, crvs, rv_mode)
        self._rvs = [parse_rvs(self._dist, rv, rv_mode)[1] for rv in rvs]
        self._crvs = parse_rvs(self._dist, crvs, rv_mode)[1]
        self._dist = modify_outcomes(self._dist, tuple)

        if not self._crvs:
            msg = "Intrinsic mutual informations require a conditional variable."
            raise ditException(msg)

        # compress all random variables down to single vars
        self._unqs = [Uniquifier() for _ in range(3)]
        for unq, var in zip(self._unqs, rvs + [crvs]):
            self._dist = insert_rvf(self._dist, lambda x: (unq(tuple(x[i] for i in var)),))

        self._shape = list(map(len, self._dist.alphabet))
        self._dist.make_dense()
        self._full_pmf = self._dist.pmf.reshape(self._shape)

        n = dist.outcome_length()
        self._pmf = self._full_pmf.sum(axis=tuple(range(n)))

        self._all_vars = set(range(len(rvs + [crvs]) + 1))
        self._rvs = set(range(len(rvs)))
        self._crvs = {len(rvs)}
        self._U = {len(rvs) + 1}

        self._num_vars = len(self._rvs) + 1

        theoretical_bound = prod(self._shape[n:])
        if bound is not None:
            self._bound = min(bound, theoretical_bound)
        else:
            self._bound = theoretical_bound

        self._default_hops = self._bound

        self._mask = np.ones(self._shape[n:] + [self._bound]) / self._bound

    def construct_random_initial(self):
        """
        Construct a random optimization vector.

        Returns
        -------
        x : ndarray
            A random optimization vector.
        """
        x = np.random.random(self._shape[-self._num_vars:] + [self._bound]).flatten()
        return x

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
        channel = x.reshape(self._shape[-self._num_vars:] + [self._bound])
        channel /= channel.sum(axis=tuple(self._U), keepdims=True)
        channel[np.isnan(channel)] = self._mask[np.isnan(channel)]

        if full:
            slc = [np.newaxis] * (len(self._shape) - self._num_vars) + [colon] * len(self._all_vars)
            joint = self._full_pmf[..., np.newaxis] * channel[slc]
        else:
            joint = self._pmf[..., np.newaxis] * channel

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

    @abstractmethod
    def measure(self, pmf, rvs, crvs):
        """
        Abstract method for computing the appropriate measure of generalized
        mutual information.

        Parameters
        ----------

        """
        pass

    @abstractmethod
    def objective(self, x):
        """
        The objective of the minimization.

        Parameters
        ----------
        x : np.ndarray
            The optimization vector.

        Returns
        -------
        obj : float
            The objective to minimize
        """
        pass

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
            alphabets += [(digits + ascii_letters)[:self._bound]]
            string = True
        except:
            alphabets += [list(range(self._bound))]
            string = False

        joint = self.construct_joint(x, full=True)
        joint = joint.sum(axis=tuple(range(-self._num_vars - 1, -1)))
        outcomes, pmf = zip(*[(o, p) for o, p in np.ndenumerate(joint) if p > cutoff])
        outcomes = [tuple(a[i] for i, a in zip(o, alphabets)) for o in outcomes]

        if string:
            outcomes = [''.join(o) for o in outcomes]

        # normalize, in case cutoffs removed a significant amount of pmf
        pmf = np.asarray(pmf)
        pmf /= pmf.sum()

        d = Distribution(outcomes, pmf)
        return d

    @classmethod
    def functional(cls):
        """
        Construct a functional form of the optimizer.
        """

        def intrinsic(dist, rvs=None, crvs=None, rv_mode=None, nhops=None, bounds=(2, 3, 4, None)):
            candidates = []
            for bound in bounds:
                opt = cls(dist, rvs, crvs, rv_mode, bound)
                opt.optimize(nhops=nhops)
                candidates.append(opt.objective(opt._optima))
            return min(candidates)

        intrinsic.__doc__ = \
            """
            Compute the {type} intrinsic {name}.

            Parameters
            ----------
            dist : Distribution
                The distribution to compute the {type} intrinsic {name} of.
            rvs : list, None
                A list of lists. Each inner list specifies the indexes of the random
                variables used to calculate the intrinsic {name}. If None,
                then it is calculated over all random variables, which is equivalent
                to passing `rvs=dist.rvs`.
            crvs : list
                A single list of indexes specifying the random variables to
                condition on.
            rv_mode : str, None
                Specifies how to interpret `rvs` and `crvs`. Valid options are:
                {{'indices', 'names'}}. If equal to 'indices', then the elements of
                `crvs` and `rvs` are interpreted as random variable indices. If
                equal to 'names', the the elements are interpreted as random
                variable names. If `None`, then the value of `dist._rv_mode` is
                consulted, which defaults to 'indices'.
            bound : int, None
                Bound on the size of the auxiliary variable. If None, use the
                theoretical bound.
            """.format(name=cls.name, type=cls.type)

        return intrinsic
