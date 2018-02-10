"""
Abstract base classes
"""

from __future__ import division

from abc import abstractmethod

import numpy as np

from ... import Distribution
from ...algorithms import BaseAuxVarOptimizer
from ..dual_total_correlation import dual_total_correlation
from ..entropy import entropy


class MarkovVarOptimizer(BaseAuxVarOptimizer):
    """
    Abstract base class for constructing auxiliary variables which render a set
    of variables conditionally independent.
    """

    name = ""
    description = ""

    construct_initial = BaseAuxVarOptimizer.construct_random_initial

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
        super(MarkovVarOptimizer, self).__init__(dist, rvs=rvs, crvs=crvs, rv_mode=rv_mode)

        theoretical_bound = self.compute_bound()
        bound = min(bound, theoretical_bound) if bound else theoretical_bound

        rv_bounds = self._shape[1:-1]
        self._pmf_to_match = self._pmf.copy()

        # remove the rvs other than the first, they need to be generated by W
        # in order to satisfy the markov criteria:
        self._pmf = self._pmf.sum(axis=tuple(range(1, len(self._shape)-1)))
        self._shape = self._pmf.shape
        self._all_vars = {0, 1}

        self._full_pmf = self._full_pmf.sum(axis=tuple(range(self._n + 1, len(self._full_shape)-1)))
        self._full_shape = self._full_pmf.shape
        self._full_vars = tuple(range(self._n + 2))

        # back up where the rvs and crvs are, they need to be reflect
        # the above removals for the sake of adding auxvars:
        self.__rvs, self._rvs = self._rvs, {0}
        self.__crvs, self._crvs = self._crvs, {1}

        self._construct_auxvars([({0, 1}, bound)] +
                                [({1, 2}, s) for s in rv_bounds])

        # put rvs, crvs back:
        self._rvs = self.__rvs
        self._crvs = self.__crvs
        del self.__rvs
        del self.__crvs

        self._W = {1 + len(self._aux_vars)}

        # The constraint that the joint doesn't change.
        self.constraints = [{'type': 'eq',
                             'fun': self.constraint_match_joint,
                             },
                            ]

        self._default_hops = 10

        self._additional_options = {'options': {'maxiter': 1000,
                                                'ftol': 1e-6,
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

    def construct_joint(self, x):
        """
        Construct the joint distribution.

        Parameters
        ----------
        x : np.ndarray
            An optimization vector.

        Returns
        -------
        joint : np.ndarray
            The joint distribution resulting from the distribution passed
            in and the optimization vector.
        """
        joint = super(MarkovVarOptimizer, self).construct_joint(x)
        joint = np.moveaxis(joint, 1, -1)  # move crvs
        joint = np.moveaxis(joint, 1, -1)  # move W

        return joint

    def construct_full_joint(self, x):
        """
        Construct the joint distribution.

        Parameters
        ----------
        x : np.ndarray
            An optimization vector.

        Returns
        -------
        joint : np.ndarray
            The joint distribution resulting from the distribution passed
            in and the optimization vector.
        """
        joint = super(MarkovVarOptimizer, self).construct_full_joint(x)
        joint = np.moveaxis(joint, self._n + 1, -1)  # move crvs
        joint = np.moveaxis(joint, self._n + 1, -1)  # move W
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

        delta = (100*(joint - self._pmf_to_match)**2).sum()

        return delta

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


class MinimizingMarkovVarOptimizer(MarkovVarOptimizer):
    """
    Abstract base class for an optimizer which additionally minimizes the size
    of the auxiliary variable.
    """

    def optimize(self, x0=None, niter=None, maxiter=None, polish=1e-6, callback=False, minimize=True, min_niter=15):
        """
        Parameters
        ----------
        x0 : np.ndarray, None
            The vector to initialize the optimization with. If None, a random
            vector is used.
        niter : int
            The number of times to basin hop in the optimization.
        maxiter : int
            The number of inner optimizer steps to perform.
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
        super(MinimizingMarkovVarOptimizer, self).optimize(x0=x0,
                                                           niter=niter,
                                                           maxiter=maxiter,
                                                           polish=False,
                                                           callback=callback)
        if minimize:
            # minimize the entropy of W
            self._post_process(style='entropy', minmax='min', niter=min_niter, maxiter=maxiter)
        if polish:
            self._polish(cutoff=polish)
