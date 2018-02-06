"""
Abstract base classes
"""

from __future__ import division

from abc import abstractmethod

import numpy as np

from ... import Distribution
from ...algorithms import BaseAuxVarOptimizer
from ...helpers import flatten, normalize_rvs
from ...utils.optimization import colon
from ..dual_total_correlation import dual_total_correlation
from ..entropy import entropy


class MarkovVarOptimizerOld(BaseAuxVarOptimizer):
    """
    Abstract base class for constructing auxiliary variables which render a set
    of variables conditionally independent.
    """

    name = ""
    description = ""

    _objective = lambda: None

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
        theoretical_bound = self.compute_bound()
        self._bound = min(bound, theoretical_bound) if bound is not None else theoretical_bound

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
        idxs = [list(range(len(shape))) for shape in self._shapes]
        self._idxs = [idxs[0][self._rv_lens[0]:]] + \
                     [idx[self._crv_len:] for idx in idxs[1:]]

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

        self._finish = self._rvs[0] + self._crvs + sum(self._rvs[1:], []) + \
                 sorted(self._others)
        self._start = list(range(len(self._finish)))

        self._default_hops = 10
        self._aux_bounds = [self._bound]
        self._arvs = {len(self.construct_joint(self.construct_random_initial()).shape)-1}
        self._proxy_vars = tuple()
        self._optvec_size = self._crv_len

        self._additional_options = {'options': {'maxiter': 1000,
                                                'ftol': 5e-7,
                                                'eps': 1.4901161193847656e-9,
                                                }
                                    }

    @abstractmethod
    def compute_bound(self):
        """
        Return a bound on the cardinality of the auxiliary variable.

        Returns
        -------
        bound : int
            The bound on the size of W.
        """
        pass

    @staticmethod
    def row_normalize(mat, axis=1):
        """
        Row-normalize `mat`, so that it is a valid conditional probability
        distribution. The normalization is done in place.

        Parameters
        ----------
        mat : np.ndarray
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
        except TypeError:  # pragma: no cover
            pass
        mat /= np.sum(mat, axis=axis, keepdims=True)

    def construct_random_initial(self):
        """
        Construct a random optimization vector.

        Returns
        -------
        x : np.ndarray
            An optimization vector.
        """
        # make cdists like p(x_i | w)
        cdists = [ np.random.random(size=shape) for shape in self._shapes ]

        for cdist, axes in zip(cdists, self._idxs):
            self.row_normalize(cdist, axes)

        # smash them together
        x = np.concatenate([cdist.flatten() for cdist in cdists], axis=0)

        return x

    def construct_cdists(self, x, normalize=False):
        """
        Given an optimization vector, construct the conditional distributions it
        represents.

        Parameters
        ----------
        x : np.ndarray
            An optimization vector.
        normalize : bool
            Whether to normalize the conditional distributions or not.
        """
        if normalize:  # pragma: no cover
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
        x : np.ndarray
            An optimization vector.
        """
        cdists = self.construct_cdists(x)

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
        joint = np.moveaxis(joint, self._start, self._finish)

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
        x : np.ndarray
            An optimization vector.
        """
        joint = self.construct_joint(x)
        joint = joint.sum(axis=-1)  # marginalize out w

        delta = (100*(joint - self._true_joint)**2).sum()

        return delta

    def construct_markov_var(self, x):
        """
        Construct the auxiliary Markov variable.

        Parameters
        ----------
        x : np.ndarray
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
        x : np.ndarray
            An optimization vector.
        """
        markov_var = self.construct_markov_var(x)
        ent = -np.nansum(markov_var * np.log2(markov_var))
        return ent

    def mutual_information(self, x):
        """
        Computes the mutual information between the original variables and the
        auxiliary Markov variable.

        Parameters
        ----------
        x : np.ndarray
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

    def construct_distribution(self, idx=-1, cutoff=1e-5):
        """
        Construct a distribution with of the initial joint and the auxiliar
        variable.

        Parameters
        ----------
        idx : int
            The location to place the auxiliary variable in the distribution.
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

        def common_info(dist, rvs=None, crvs=None, niter=None, maxiter=1000, polish=1e-6, bound=None, rv_mode=None):
            dtc = dual_total_correlation(dist, rvs, crvs, rv_mode)
            ent = entropy(dist, rvs, crvs, rv_mode)
            if np.isclose(dtc, ent):
                # Common informations are bound between the dual total correlation and the joint
                # entropy. Therefore, if the two are equal, the common information is equal to them
                # as well.
                return dtc

            ci = cls(dist, rvs, crvs, bound, rv_mode)
            ci.optimize(niter=niter, maxiter=maxiter, polish=polish)
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
        niter : int > 0
            Number of basin hoppings to perform during the optimization.
        maxiter : int > 0
            The number of iterations of the optimization subroutine to perform.
        polish : False, float
            Whether to polish the result or not. If a float, this will perform a
            second optimization seeded with the result of the first, but with
            smaller tolerances and probabilities below polish set to 0. If
            False, don't polish.
        bound : int
            Bound the size of the Markov variable.
        rv_mode : str, None
            Specifies how to interpret `rvs` and `crvs`. Valid options are:
            {{'indices', 'names'}}. If equal to 'indices', then the elements of
            `crvs` and `rvs` are interpreted as random variable indices. If equal
            to 'names', the the elements are interpreted as random variable names.
            If `None`, then the value of `dist._rv_mode` is consulted, which
            defaults to 'indices'.

        Returns
        -------
        ci : float
            The {name} common information.
        """.format(name=cls.name, description=cls.description)

        return common_info

    construct_initial = construct_random_initial


class MarkovVarOptimizerNew(BaseAuxVarOptimizer):
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
        super(MarkovVarOptimizer, self).__init__(rvs=rvs, crvs=crvs, rv_mode=rv_mode)

        theoretical_bound = self.compute_bound()
        bound = min(bound, theoretical_bound) if bound else theoretical_bound

        self._construct_auxvars([({0}, bound),])

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
        theoretical_bound = self.compute_bound()
        self._bound = min(bound, theoretical_bound) if bound is not None else theoretical_bound

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
        idxs = [list(range(len(shape))) for shape in self._shapes]
        self._idxs = [idxs[0][self._rv_lens[0]:]] + \
                     [idx[self._crv_len:] for idx in idxs[1:]]

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

        self._finish = self._rvs[0] + self._crvs + sum(self._rvs[1:], []) + sorted(self._others)
        self._start = list(range(len(self._finish)))

        self._default_hops = 10
        self._aux_bounds = [self._bound]
        self._arvs = {len(self.construct_joint(self.construct_random_initial()).shape)-1}
        self._proxy_vars = tuple()
        self._optvec_size = self._crv_len

        self._additional_options = {'options': {'maxiter': 1000,
                                                'ftol': 5e-7,
                                                'eps': 1.4901161193847656e-9,
                                                }
                                    }

    @abstractmethod
    def compute_bound(self):
        """
        Return a bound on the cardinality of the auxiliary variable.

        Returns
        -------
        bound : int
            The bound on the size of W.
        """
        pass

    @staticmethod
    def row_normalize(mat, axis=1):
        """
        Row-normalize `mat`, so that it is a valid conditional probability
        distribution. The normalization is done in place.

        Parameters
        ----------
        mat : np.ndarray
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
        except TypeError:  # pragma: no cover
            pass
        mat /= np.sum(mat, axis=axis, keepdims=True)

    def construct_random_initial(self):
        """
        Construct a random optimization vector.

        Returns
        -------
        x : np.ndarray
            An optimization vector.
        """
        # make cdists like p(x_i | w)
        cdists = [ np.random.random(size=shape) for shape in self._shapes ]

        for cdist, axes in zip(cdists, self._idxs):
            self.row_normalize(cdist, axes)

        # smash them together
        x = np.concatenate([cdist.flatten() for cdist in cdists], axis=0)

        return x

    def construct_cdists(self, x, normalize=False):
        """
        Given an optimization vector, construct the conditional distributions it
        represents.

        Parameters
        ----------
        x : np.ndarray
            An optimization vector.
        normalize : bool
            Whether to normalize the conditional distributions or not.
        """
        if normalize:  # pragma: no cover
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
        x : np.ndarray
            An optimization vector.
        """
        cdists = self.construct_cdists(x)

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
        joint = np.moveaxis(joint, self._start, self._finish)

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
        x : np.ndarray
            An optimization vector.
        """
        joint = self.construct_joint(x)
        joint = joint.sum(axis=-1)  # marginalize out w

        delta = (100*(joint - self._true_joint)**2).sum()

        return delta

    def construct_markov_var(self, x):
        """
        Construct the auxiliary Markov variable.

        Parameters
        ----------
        x : np.ndarray
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
        x : np.ndarray
            An optimization vector.
        """
        markov_var = self.construct_markov_var(x)
        ent = -np.nansum(markov_var * np.log2(markov_var))
        return ent

    def mutual_information(self, x):
        """
        Computes the mutual information between the original variables and the
        auxiliary Markov variable.

        Parameters
        ----------
        x : np.ndarray
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

    def construct_distribution(self, idx=-1, cutoff=1e-5):
        """
        Construct a distribution with of the initial joint and the auxiliar
        variable.

        Parameters
        ----------
        idx : int
            The location to place the auxiliary variable in the distribution.
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

        def common_info(dist, rvs=None, crvs=None, niter=None, maxiter=1000, polish=1e-6, bound=None, rv_mode=None):
            dtc = dual_total_correlation(dist, rvs, crvs, rv_mode)
            ent = entropy(dist, rvs, crvs, rv_mode)
            if np.isclose(dtc, ent):
                # Common informations are bound between the dual total correlation and the joint
                # entropy. Therefore, if the two are equal, the common information is equal to them
                # as well.
                return dtc

            ci = cls(dist, rvs, crvs, bound, rv_mode)
            ci.optimize(niter=niter, maxiter=maxiter, polish=polish)
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
        niter : int > 0
            Number of basin hoppings to perform during the optimization.
        maxiter : int > 0
            The number of iterations of the optimization subroutine to perform.
        polish : False, float
            Whether to polish the result or not. If a float, this will perform a
            second optimization seeded with the result of the first, but with
            smaller tolerances and probabilities below polish set to 0. If
            False, don't polish.
        bound : int
            Bound the size of the Markov variable.
        rv_mode : str, None
            Specifies how to interpret `rvs` and `crvs`. Valid options are:
            {{'indices', 'names'}}. If equal to 'indices', then the elements of
            `crvs` and `rvs` are interpreted as random variable indices. If equal
            to 'names', the the elements are interpreted as random variable names.
            If `None`, then the value of `dist._rv_mode` is consulted, which
            defaults to 'indices'.

        Returns
        -------
        ci : float
            The {name} common information.
        """.format(name=cls.name, description=cls.description)

        return common_info

    construct_initial = construct_random_initial


MarkovVarOptimizer = MarkovVarOptimizerOld


class MinimizingMarkovVarOptimizer(MarkovVarOptimizer):
    """
    Abstract base class for an optimizer which additionally minimizes the size
    of the auxiliary variable.
    """

    def optimize(self, x0=None, niter=None, polish=1e-6, callback=False, minimize=False, min_niter=15):
        """
        Parameters
        ----------
        x0 : np.ndarray, None
            The vector to initialize the optimization with. If None, a random
            vector is used.
        niter : int
            The number of times to basin hop in the optimization.
        polish : False, float
            Whether to polish the result or not. If a float, this will perform a
            second optimization seeded with the result of the first, but with
            smaller tolerances and probabilities below polish set to 0. If
            False, don't polish.
        callback : bool
            Whether to utilize a callback or not.
        minimize : bool
            Whether to minimize the auxiliary variable or not.
        min_niter : int
            The number of basin hops to make during the minimization of the common variable.
        """
        # call the normal optimizer
        super(MinimizingMarkovVarOptimizer, self).optimize(x0=x0, niter=niter, polish=False, callback=callback)
        if minimize:
            # minimize the entropy of W
            self._post_process(style='entropy', minmax='min', niter=min_niter)
        if polish:
            self._polish(cutoff=polish)
