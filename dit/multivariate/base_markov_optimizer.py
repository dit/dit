"""
"""

from __future__ import division

from abc import ABCMeta, abstractmethod

import numpy as np

from ..helpers import flatten, normalize_rvs
from ..math import close


class MarkovVarOptimizer(object):
    """
    Abstract base class for constructing auxiliary variables which render a set
    of variables conditionally independent.
    """
    __metaclass__ = ABCMeta

    def __init__(self, dist, rvs=None, crvs=None, rv_mode=None):
        """
        Parameters
        ----------
        dist : Distribution
            The distribution to compute the auxiliary Markov variable, W, for.
        """
        self._rvs, self._crvs, self._rv_mode = normalize_rvs(dist, rvs, crvs, rv_mode)
        self._others = list(set(flatten(dist.rvs)) - \
                            set(flatten(self._rvs)) - \
                            set(flatten(self._crvs)))
        self._dist = dist.copy()
        sizes = list(map(len, self._dist.alphabet))

        self._rv_sizes = [ [ sizes[i] for i in rv ] for rv in self._rvs ]
        self._rv_lens = list(map(len, self._rv_sizes))

        self._bound = self.compute_bound()

        self._crv_size = [ self._bound ] + [ sizes[i] for i in self._crvs ]
        self._crv_len = len(self._crv_size)

        self._shapes = [self._rv_sizes[0] + self._crv_size] + \
                       [self._crv_size + rv for rv in self._rv_sizes[1:]]
        self._splits = np.cumsum([ np.prod(s) for s in self._shapes ])[:-1]

        self._dist.make_dense()
        self._pmf = self._dist.marginal(self._rvs[0], rv_mode=self._rv_mode).pmf
        self._pmf = self._pmf.reshape([sizes[i] for i in self._rvs[0]])
        self._true_joint = self._dist.pmf.reshape(sizes)

        idxs = [ list(range(len(shape))) for shape in self._shapes ]
        self._idxs = [ idxs[0][self._rv_lens[0]:] ] + \
                     [ idx[self._crv_len:] for idx in idxs[1:]]

        if len(self._others) == 0:
            self._others_cdist = None
        elif len(self._others) == 1:
            self._others_cdist = self._true_joint / np.sum(self._true_joint, axis=self._others[0], keepdims=True)
            self._others_cdist[np.isnan(self._others_cdist)] = 1
        else:
            self._others_cdist = self._true_joint / np.sum(self._true_joint, axis=tuple(self._others), keepdims=True)
            self._others_cdist[np.isnan(self._others_cdist)] = 1

        self.constraints = [{'type': 'eq',
                             'fun': self.constraint_match_joint,
                            },
                           ]

    @abstractmethod
    def compute_bound(self):
        """
        Return a bound on the alphabet size W
        """
        pass

    @abstractmethod
    def objective(self, x):
        """
        Compute the optimization objective function.

        Parameters
        ----------
        x : ndarray of shape (n,)
            A vector of conditional probabilities. See
            `construct_random_initial` for its structure.
        """
        pass

    @staticmethod
    def row_normalize(mat, axis=1):
        """
        Row-normalize `mat`, so that it is a valid conditional probability
        distribution.

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
        except:
            pass
        mat /= np.sum(mat, axis=axis, keepdims=True)

    def construct_random_initial(self):
        """
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
        """
        if normalize:
            x = x.copy()

        cdists = np.split(x, self._splits)
        cdists = [ cd.reshape(s) for cd, s in zip(cdists, self._shapes) ]
        cdists = [ np.squeeze(cdist) for cdist in cdists ]

        if normalize:
            for cdist, axes in zip(cdists, self._idxs):
                self.row_normalize(cdist, axes)

        return cdists

    def construct_joint(self, x):
        """
        Given an optimization vector, construct the joint distribution it
        represents.

        Parameters
        ----------
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
                   [np.newaxis]*([0] + np.cumsum(self._rv_lens).tolist())[i] + \
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
        """
        joint = self.construct_joint(x)
        markov_var = joint.sum(axis=tuple(range(len(joint.shape)-1)))
        return markov_var

    def entropy(self, x):
        """
        Compute the entropy of the auxiliary Markov variable.

        Parameters
        ----------
        """
        markov_var = self.construct_markov_var(x)
        ent = -np.nansum(markov_var * np.log2(markov_var))
        return ent

    def mutual_information(self, x):
        """
        Computes the mutual information between the original variables and the auxilary Markov variable.

        Parameters
        ----------
        """
        joint = self.construct_joint(x)

        # p(w)
        markov_var = joint.sum(axis=-1, keepdims=True)

        # p(x_0, ... x_n)
        others = joint.sum(axis=tuple(range(len(joint.shape)-1)), keepdims=True)

        mut_info = np.nansum(joint * np.log2(joint / (markov_var * others)))

        return mut_info

    def optimize(self, x0=None, nhops=5, jacobian=False, polish=True):
        """
        Perform the optimization.

        Parameters
        ----------

        Notes
        -----
        """
        from scipy.optimize import basinhopping

        if x0 is not None:
            x = x0
        else:
            x = self.construct_random_initial()

        def accept_test(**kwargs):
            x = kwargs['x_new']
            tmax = bool(np.all(x <= 1))
            tmin = bool(np.all(x >= 0))
            return tmin and tmax

        minimizer_kwargs = {'method': 'SLSQP',
                            'bounds': [(0, 1)]*x.size,
                            'constraints': self.constraints,
                            'tol': None,
                            'callback': None,
                            'options': {'maxiter': 1000,
                                        'ftol': 5e-07, # default: 1e-06,
                                        'eps': 1.4901161193847656e-09, #defeault: 1.4901161193847656e-08,
                                       },
                           }

        # this makes things slower!
        if jacobian:
            import numdifftools as ndt

            minimizer_kwargs['jac'] = ndt.Jacobian(self.objective)
            for const in minimizer_kwargs['constraints']:
                const['jac'] = ndt.Jacobian(const['fun'])

        res = basinhopping(func=self.objective,
                           x0=x,
                           minimizer_kwargs=minimizer_kwargs,
                           niter=nhops,
                           accept_test=accept_test,
                          )

        self._res = res

        if not res.lowest_optimization_result.success:
            raise Exception(res.lowest_optimization_result.message)

        if polish:
            self._polish(jacobian=jacobian)

    def _polish(self, cutoff=1e-6, jacobian=False):
        """
        Parameters
        ----------
        """
        from scipy.optimize import minimize

        x0 = self._res.x
        count = (x0 < cutoff).sum()
        x0[x0 < cutoff] = 0

        minimizer_kwargs = {'method': 'SLSQP',
                            'bounds': [(0, 0) if close(x, 0) else (0, 1) for x in x0],
                            'constraints': self.constraints,
                            'tol': None,
                            'callback': None,
                            'options': {'maxiter': 1000,
                                        'ftol': 15e-11, # default: 1e-06,
                                        'eps': 1.4901161193847656e-12, #defeault: 1.4901161193847656e-08,
                                       },
                           }

        # this makes things slower!
        if jacobian:
            import numdifftools as ndt

            minimizer_kwargs['jac'] = ndt.Jacobian(self.objective)
            for const in minimizer_kwargs['constraints']:
                const['jac'] = ndt.Jacobian(const['fun'])

        res = minimize(fun=self.objective,
                       x0=x0,
                       **minimizer_kwargs
                      )

        self._res = res

        if count < (res.x < cutoff).sum():
            self._polish(cutoff=cutoff, jacobian=jacobian)

    def construct_distribution(self, idx=-1, cutoff=1e-5):
        """
        Parameters
        ----------
        """
        joint = self.construct_joint(self._res.x)

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

        d = D(outcomes, pmf)
        return d


class MinimizingMarkovVarOptimizer(MarkovVarOptimizer):
    """
    """

    def minimize_aux_var(self, njumps=25, jacobian=False):
        """
        Minimize the entropy of the auxilary variable without compromizing the
        objective.
        """
        from scipy.optimize import basinhopping

        true_objective = self._res.fun

        def constraint_match_objective(x):
            obj = abs(self.objective(x) - true_objective)**1.5
            return obj

        constraint = {'type': 'eq',
                      'fun': constraint_match_objective,
                     }

        def accept_test(**kwargs):
            x = kwargs['x_new']
            tmax = bool(np.all(x <= 1))
            tmin = bool(np.all(x >= 0))
            return tmin and tmax

        minimizer_kwargs = {'method': 'SLSQP',
                            'bounds': [(0, 1)]*self._res.x.size,
                            'constraints': self.constraints + [constraint],
                            'tol': None,
                            'callback': None,
                            'options': {'maxiter': 1000,
                                        'ftol': 1e-06,
                                        'eps': 1.4901161193847656e-08,
                                       },
                           }

        # this makes things slower!
        if jacobian:
            import numdifftools as ndt

            minimizer_kwargs['jac'] = ndt.Jacobian(self.entropy)
            for const in minimizer_kwargs['constraints']:
                const['jac'] = ndt.Jacobian(const['fun'])

        res = basinhopping(func=self.entropy,
                           x0=self._res.x,
                           minimizer_kwargs=minimizer_kwargs,
                           niter=njumps,
                           accept_test=accept_test,
                          )

        if res.lowest_optimization_result.success:
            self._res = res
        else:
            pass
            # raise Exception(res.lowest_optimization_result.message)

    def optimize(self, x0=None, nhops=5, jacobian=False, polish=True, minimize=False, njumps=15):
        """
        Parameters
        ----------
        x0 :
        nhops : int
        jacobian : bool
        polish : bool
        minimize : bool
        njumps : int
        """
        super(MinimizingMarkovVarOptimizer, self).optimize(x0=x0, nhops=nhops, jacobian=jacobian, polish=False)
        if minimize:
            self.minimize_aux_var(jacobian=jacobian, njumps=njumps)
        if polish:
            self._polish(jacobian=jacobian)
