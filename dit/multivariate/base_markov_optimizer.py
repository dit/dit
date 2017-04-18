"""
Abstract base classes
"""

from __future__ import division

from abc import ABCMeta, abstractmethod

import numpy as np

from .. import Distribution
from ..helpers import flatten, normalize_rvs
from ..math import close
from . import dual_total_correlation, entropy
from ..utils.optimization import (BasinHoppingCallBack,
                                  BasinHoppingInnerCallBack,
                                  accept_test,
                                  basinhop_status)

class MarkovVarOptimizer(object):
    """
    Abstract base class for constructing auxiliary variables which render a set
    of variables conditionally independent.
    """
    __metaclass__ = ABCMeta

    name = ""
    description = ""

    def __init__(self, dist, rvs=None, crvs=None, rv_mode=None):
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
        rv_mode : str, None
            Specifies how to interpret `rvs` and `crvs`. Valid options are:
            {'indices', 'names'}. If equal to 'indices', then the elements of
            `crvs` and `rvs` are interpreted as random variable indices. If
            equal to 'names', the the elements are interpreted as random
            variable names. If `None`, then the value of `dist._rv_mode` is
            consulted, which defaults to 'indices'.
        """
        self._rvs, self._crvs, self._rv_mode = normalize_rvs(dist, rvs, crvs, rv_mode)
        self._others = list(set(flatten(dist.rvs)) - \
                            set(flatten(self._rvs)) - \
                            set(flatten(self._crvs)))
        self._dist = dist.copy()
        sizes = list(map(len, self._dist.alphabet))

        # compute the sizes of the RVs
        self._rv_sizes = [ [ sizes[i] for i in rv ] for rv in self._rvs ]
        self._rv_lens = list(map(len, self._rv_sizes))
        self._sizes = [0] + np.cumsum(self._rv_lens).tolist()

        # compute the bound on the auxiliary variable
        self._bound = self.compute_bound()

        # compute the size of the conditional variables, including the auxiliary
        # variable.
        self._crv_size = [ self._bound ] + [ sizes[i] for i in self._crvs ]
        self._crv_len = len(self._crv_size)

        # compute the shapes of each conditional distribution
        self._shapes = [self._rv_sizes[0] + self._crv_size] + \
                       [self._crv_size + rv for rv in self._rv_sizes[1:]]
        self._splits = np.cumsum([ np.prod(s) for s in self._shapes ])[:-1]

        # compute the pmf of rvs[0] and the full joint
        self._dist.make_dense()
        self._pmf = self._dist.marginal(self._rvs[0], rv_mode=self._rv_mode).pmf
        self._pmf = self._pmf.reshape([sizes[i] for i in self._rvs[0]])
        self._true_joint = self._dist.pmf.reshape(sizes)

        # the set of indices to normalize
        idxs = [ list(range(len(shape))) for shape in self._shapes ]
        self._idxs = [ idxs[0][self._rv_lens[0]:] ] + \
                     [ idx[self._crv_len:] for idx in idxs[1:]]

        # construct the conditional distribution of `others`, so that they can
        # be reconstructed.
        if len(self._others) == 0:
            self._others_cdist = None
        elif len(self._others) == 1:
            self._others_cdist = self._true_joint / np.sum(self._true_joint, axis=self._others[0], keepdims=True)
            self._others_cdist[np.isnan(self._others_cdist)] = 1
        else:
            self._others_cdist = self._true_joint / np.sum(self._true_joint, axis=tuple(self._others), keepdims=True)
            self._others_cdist[np.isnan(self._others_cdist)] = 1

        # The constraint that the joint doesn't change.
        self.constraints = [{'type': 'eq',
                             'fun': self.constraint_match_joint,
                            },
                           ]

    @abstractmethod
    def compute_bound(self): # pragma: no cover
        """
        Return a bound on the cardinality of the auxiliary variable.

        Returns
        -------
        bound : int
            The bound.
        """
        pass

    @abstractmethod
    def objective(self, x): # pragma: no cover
        """
        Compute the optimization objective function.

        Parameters
        ----------
        x : ndarray
            An optimization vector.
        """
        pass

    @staticmethod
    def row_normalize(mat, axis=1):
        """
        Row-normalize `mat`, so that it is a valid conditional probability
        distribution. The normalization is done in place.

        Parameters
        ----------
        mat : ndarray
            The matrix to row-normalize in place.
        axis : [int]
            Axes to sum over. Defaults to 1.
        """
        try:
            # np.sum will take tuples, but not of length 1.
            if len(axis) == 1:
                axis = axis[0]
            else:
                axis = tuple(axis)
        except TypeError: # pragma: no cover
            pass
        mat /= np.sum(mat, axis=axis, keepdims=True)

    def construct_random_initial(self):
        """
        Construct a random optimization vector.

        Returns
        -------
        x : ndarray
            An optimization vector.
        """
        # make cdists like p(x_i | w)
        cdists = [ np.random.random(size=shape) for shape in self._shapes ]

        for cdist, axes in zip(cdists, self._idxs):
            self.row_normalize(cdist, axes)

        # smash them together
        x = np.concatenate([ cdist.flatten() for cdist in cdists ], axis=0)

        return x

    def construct_cdists(self, x, normalize=False):
        """
        Given an optimization vector, construct the conditional distributions it
        represents.

        Parameters
        ----------
        x : ndarray
            An optimization vector.
        normalize : bool
            Whether to normalize the conditional distributions or not.
        """
        if normalize: # pragma: no cover
            x = x.copy()

        cdists = np.split(x, self._splits)
        cdists = [ cd.reshape(s) for cd, s in zip(cdists, self._shapes) ]
        cdists = [ np.squeeze(cdist) for cdist in cdists ]

        if normalize: # pragma: no cover
            for cdist, axes in zip(cdists, self._idxs):
                self.row_normalize(cdist, axes)

        return cdists

    def construct_joint(self, x):
        """
        Given an optimization vector, construct the joint distribution it
        represents.

        Parameters
        ----------
        x : ndarray
            An optimization vector.
        """
        cdists = self.construct_cdists(x)

        colon = slice(None, None)

        # get p(x_0, w)
        slc = [colon]*len(self._pmf.shape) + [np.newaxis]*self._crv_len
        joint = self._pmf[slc] * cdists[0]

        # go from p(x_0, w, ...) to p(x_0, w, ..., x_i)
        for i, cdist in enumerate(cdists[1:]):
            slc1 = [Ellipsis] + [np.newaxis]*(len(cdist.shape)-self._crv_len)
            slc2 = [np.newaxis]*self._rv_lens[0] + \
                   [colon]*self._crv_len + \
                   [np.newaxis]*self._sizes[i] + \
                   [colon]*(len(cdist.shape)-self._crv_len)
            joint = joint[slc1] * cdist[slc2]

        # reorder and make gaps for `others`
        joint = np.moveaxis(joint, self._rv_lens[0], -1)
        slc = [Ellipsis] + [np.newaxis]*len(self._others) + [colon]
        joint = joint[slc]
        finish = self._rvs[0] + self._crvs + list(flatten(self._rvs[1:])) + \
                 sorted(self._others)
        start = list(range(len(finish)))
        joint = np.moveaxis(joint, start, finish)

        # add others
        if self._others_cdist is not None:
            joint = joint * self._others_cdist[..., np.newaxis]

        return joint

    def constraint_match_joint(self, x):
        """
        Ensure that the joint distribution represented by the optimization
        vector matches that of the distribution.

        Parameters
        ----------
        x : ndarray
            An optimization vector.
        """
        joint = self.construct_joint(x)
        joint = joint.sum(axis=-1) # marginalize out w

        delta = (100*(joint - self._true_joint)**2).sum()

        return delta

    def construct_markov_var(self, x):
        """
        Construct the auxiliary Markov variable.

        Parameters
        ----------
        x : ndarray
            An optimization vector.
        """
        joint = self.construct_joint(x)
        markov_var = joint.sum(axis=tuple(range(len(joint.shape)-1)))
        return markov_var

    def entropy(self, x):
        """
        Compute the entropy of the auxiliary Markov variable.

        Parameters
        ----------
        x : ndarray
            An optimization vector.
        """
        markov_var = self.construct_markov_var(x)
        ent = -np.nansum(markov_var * np.log2(markov_var))
        return ent

    def mutual_information(self, x):
        """
        Computes the mutual information between the original variables and the
        auxilary Markov variable.

        Parameters
        ----------
        x : ndarray
            An optimization vector.
        """
        joint = self.construct_joint(x)

        # p(rvs)
        others = joint.sum(axis=-1, keepdims=True)

        # p(w)
        axes = tuple(range(len(joint.shape)-1))
        markov_var = joint.sum(axis=axes, keepdims=True)

        mut_info = np.nansum(joint * np.log2(joint / (markov_var * others)))

        return mut_info

    def optimize(self, x0=None, nhops=5, jacobian=False, polish=1e-6, callback=False):
        """
        Perform the optimization.

        Parameters
        ----------
        x0 : ndarray, None
            The vector to initialize the optimization with. If None, a random
            vector is used.
        nhops : int
            The number of times to basin hop in the optimization.
        jacobian : bool
            Whether to use an Jacobians computed with numdifftools or not.
            Defaults to False. The use of numdifftools can greatly slow down the
            speed of the algorithms.
        polish : False, float
            Whether to polish the result or not. If a float, this will perform a
            second optimization seeded with the result of the first, but with
            smaller tolerances and probabilities below polish set to 0. If
            False, don't polish.

        Notes
        -----
        """
        from scipy.optimize import basinhopping

        if x0 is not None:
            x = x0
        else:
            x = self.construct_random_initial()

        if callback:
            icb = BasinHoppingInnerCallBack()
        else:
            icb = None

        minimizer_kwargs = {'method': 'SLSQP',
                            'bounds': [(0, 1)]*x.size,
                            'constraints': self.constraints,
                            'tol': None,
                            'callback': icb,
                            'options': {'maxiter': 1000,
                                        'ftol': 5e-07,
                                        'eps': 1.4901161193847656e-09,
                                       },
                           }

        # this makes things slower!
        # TODO: use ndt.nd_algopy, which may be significantly faster.
        if jacobian: # pragma: no cover
            import numdifftools as ndt

            minimizer_kwargs['jac'] = ndt.Jacobian(self.objective)
            for const in minimizer_kwargs['constraints']:
                const['jac'] = ndt.Jacobian(const['fun'])

        self._callback = BasinHoppingCallBack(minimizer_kwargs['constraints'], icb)

        res = basinhopping(func=self.objective,
                           x0=x,
                           minimizer_kwargs=minimizer_kwargs,
                           niter=nhops,
                           callback=self._callback,
                           accept_test=accept_test,
                          )

        success, msg = basinhop_status(res)
        if success:
            self._optima = res.x
        else: # pragma: no cover
            minimum = self._callback.minimum()
            if minimum is not None:
                self._optima = minimum
            # else:
            #     raise Exception(msg)

        if polish:
            self._polish(cutoff=polish, jacobian=jacobian)

    def _polish(self, cutoff=1e-6, jacobian=False):
        """
        Improve the solution found by the optimizer.

        Parameters
        ----------
        cutoff : float
            Set probabilities lower than this to zero, reducing the total
            optimization dimension.
        jacobian : bool
            Whether to use an Jacobians computed with numdifftools or not.
            Defaults to False. The use of numdifftools can greatly slow down the
            speed of the algorithms.
        """
        from scipy.optimize import minimize

        x0 = self._optima
        count = (x0 < cutoff).sum()
        x0[x0 < cutoff] = 0

        kwargs = {'method': 'SLSQP',
                  'bounds': [(0, 0) if close(x, 0) else (0, 1) for x in x0],
                  'constraints': self.constraints,
                  'tol': None,
                  'callback': None,
                  'options': {'maxiter': 1000,
                              'ftol': 15e-11,
                              'eps': 1.4901161193847656e-12,
                             },
                 }

        # this makes things slower!
        if jacobian: # pragma: no cover
            import numdifftools as ndt

            kwargs['jac'] = ndt.Jacobian(self.objective)
            for const in kwargs['constraints']:
                const['jac'] = ndt.Jacobian(const['fun'])

        res = minimize(fun=self.objective,
                       x0=x0,
                       **kwargs
                      )

        self._optima = res.x

        if count < (res.x < cutoff).sum():
            self._polish(cutoff=cutoff, jacobian=jacobian)

    def construct_distribution(self, idx=-1, cutoff=1e-5):
        """
        Construct a distribution with of the initial joint and the auxiliar
        variable.

        Parameters
        ----------
        idx : int
            The location to place the auxiliar variable in the distribution.
        cutoff : float
            Consider probabilities less than this as zero, and renormalize.
        """
        joint = self.construct_joint(self._optima)

        # move w to specified index
        if idx == -1:
            idx = len(joint.shape)-1
        joint = np.moveaxis(joint, -1, idx)

        # trim small probabilities
        joint /= joint.sum()
        joint[joint < cutoff] = 0
        joint /= joint.sum()

        # this code sucks
        # it makes w's alphabet go from, e.g. (0, 3, 6, 7) to (0, 1, 2, 3)
        outcomes, pmf = zip(*[ (o, p) for o, p in np.ndenumerate(joint) if p > 0 ])
        outcomes = list(outcomes)
        symbol_map = {}
        for i, outcome in enumerate(outcomes):
            outcome = list(outcome)
            sym = outcome[idx]
            if sym not in symbol_map:
                symbol_map[sym] = len(symbol_map)
            outcome[idx] = symbol_map[sym]
            outcomes[i] = tuple(outcome)

        d = Distribution(outcomes, pmf)
        return d

    @classmethod
    def functional(cls):
        """
        Construct a functional form of the optimizer.
        """

        def common_info(dist, rvs=None, crvs=None, rv_mode=None, nhops=5, polish=1e-6):
            dtc = dual_total_correlation(dist, rvs, crvs, rv_mode)
            ent = entropy(dist, rvs, crvs, rv_mode)
            if close(dtc, ent):
                # Common informations are bound between the dual total correlation and the joint
                # entropy. Therefore, if the two are equal, the common information is equal to them
                # as well.
                return dtc

            ci = cls(dist, rvs, crvs, rv_mode)
            ci.optimize(nhops=nhops, polish=polish)
            return ci.objective(ci._optima)

        common_info.__doc__ = \
        """
        Computes the {name} common information, {description}.

        Parameters
        ----------
        dist : Distribution
            The distribution for which the {name} common information will be
            computed.
        rvs : list, None
            A list of lists. Each inner list specifies the indexes of the random
            variables used to calculate the {name} common information. If None,
            then it calculated over all random variables, which is equivalent to
            passing `rvs=dist.rvs`.
        crvs : list, None
            A single list of indexes specifying the random variables to condition
            on. If None, then no variables are conditioned on.
        rv_mode : str, None
            Specifies how to interpret `rvs` and `crvs`. Valid options are:
            {{'indices', 'names'}}. If equal to 'indices', then the elements of
            `crvs` and `rvs` are interpreted as random variable indices. If equal
            to 'names', the the elements are interpreted as random variable names.
            If `None`, then the value of `dist._rv_mode` is consulted, which
            defaults to 'indices'.
        nhops : int > 0
            Number of basin hoppings to perform during the optimization.
        polish : False, float
            Whether to polish the result or not. If a float, this will perform a
            second optimization seeded with the result of the first, but with
            smaller tolerances and probabilities below polish set to 0. If
            False, don't polish.

        Returns
        -------
        ci : float
            The {name} common information.
        """.format(name=cls.name, description=cls.description)

        return common_info


class MinimizingMarkovVarOptimizer(MarkovVarOptimizer):
    """
    Abstract base class for an optimizer which additionally minimizes the size
    of the auxiliary variable.
    """
    __metaclass__ = ABCMeta

    def minimize_aux_var(self, njumps=25, jacobian=False):
        """
        Minimize the entropy of the auxilary variable without compromizing the
        objective.

        Parameters
        ----------
        njumps : int
            The number of basin hops to make during the optimization.
        jacobian : bool
            Whether to use an Jacobians computed with numdifftools or not.
            Defaults to False. The use of numdifftools can greatly slow down the
            speed of the algorithms.
        """
        from scipy.optimize import basinhopping

        true_objective = self.objective(self._optima)

        def constraint_match_objective(x):
            obj = abs(self.objective(x) - true_objective)**1.5
            return obj

        constraint = {'type': 'eq',
                      'fun': constraint_match_objective,
                     }

        minimizer_kwargs = {'method': 'SLSQP',
                            'bounds': [(0, 1)]*self._optima.size,
                            'constraints': self.constraints + [constraint],
                            'tol': None,
                            'callback': None,
                            'options': {'maxiter': 1000,
                                        'ftol': 1e-06,
                                        'eps': 1.4901161193847656e-08,
                                       },
                           }

        # this makes things slower!
        if jacobian: # pragma: no cover
            import numdifftools as ndt

            minimizer_kwargs['jac'] = ndt.Jacobian(self.entropy)
            for const in minimizer_kwargs['constraints']:
                const['jac'] = ndt.Jacobian(const['fun'])

        callback = BasinHoppingCallBack(minimizer_kwargs['constraints'])

        res = basinhopping(func=self.entropy,
                           x0=self._optima,
                           minimizer_kwargs=minimizer_kwargs,
                           niter=njumps,
                           callback=callback,
                           accept_test=accept_test,
                          )

        if basinhop_status(res)[0]:
            self._optima = res.x
        else:
            if callback.candidates:
                self._optima = min(callback.candidates)[1]

    def optimize(self, x0=None, nhops=5, jacobian=False, polish=1e-6, callback=False, minimize=False, njumps=15):
        """
        Parameters
        ----------
        x0 : ndarray, None
            The vector to initialize the optimization with. If None, a random
            vector is used.
        nhops : int
            The number of times to basin hop in the optimization.
        nhops : int
        jacobian : bool
            Whether to use an Jacobians computed with numdifftools or not.
            Defaults to False. The use of numdifftools can greatly slow down the
            speed of the algorithms.
        polish : False, float
            Whether to polish the result or not. If a float, this will perform a
            second optimization seeded with the result of the first, but with
            smaller tolerances and probabilities below polish set to 0. If
            False, don't polish.
        minimize : bool
            Whether to minimize the auxiliary variable or not.
        njumps : int
            The number of basin hops to make during the optimization.
        """
        # call the normal optimizer
        super(MinimizingMarkovVarOptimizer, self).optimize(x0=x0, nhops=nhops, jacobian=jacobian, polish=False, callback=callback)
        if minimize:
            # minimize the entropy of W
            self.minimize_aux_var(jacobian=jacobian, njumps=njumps)
        if polish:
            self._polish(cutoff=polish, jacobian=jacobian)
