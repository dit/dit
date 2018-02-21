"""
Base class for the calculation of reduced and minimal intrinsic informations.
"""

from __future__ import division

from abc import abstractmethod

from ...algorithms import BaseAuxVarOptimizer
from ...exceptions import ditException
from ...math import prod
from ...utils import unitful

__all__ = [
    'BaseIntrinsicMutualInformation',
    'BaseMoreIntrinsicMutualInformation',
]


class BaseIntrinsicMutualInformation(BaseAuxVarOptimizer):
    """
    Compute a generalized intrinsic mutual information:

        IMI[X:Y|Z] = min_{p(z_bar|z)} I[X:Y|Z]
    """

    name = ""

    construct_initial = BaseAuxVarOptimizer.construct_random_initial

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
        if not crvs:
            msg = "Intrinsic mutual informations require a conditional variable."
            raise ditException(msg)

        super(BaseIntrinsicMutualInformation, self).__init__(dist, rvs, crvs, rv_mode=rv_mode)

        crv_index = len(self._shape) - 1
        crv_size = self._shape[crv_index]
        bound = min([bound, crv_size]) if bound is not None else crv_size

        self._construct_auxvars([({crv_index}, bound)])

    def optimize(self, *args, **kwargs):
        """
        Perform the optimization.

        Parameters
        ----------
        x0 : np.ndarray, None
            Initial optimization vector. If None, use a random vector.
        niter : int, None
            The number of basin hops to perform while optimizing. If None,
            hop a number of times equal to the dimension of the conditioning
            variable(s).
        """
        result = super(BaseIntrinsicMutualInformation, self).optimize(*args, **kwargs)

        # test against known upper bounds as well, in case space wasn't well sampled.
        options = [self.construct_constant_initial(),  # mutual information
                   self.construct_copy_initial(),      # conditional mutual information
                   result.x,                           # found optima
                  ]

        self._optima = min(options, key=lambda opt: self.objective(opt))

    @classmethod
    def functional(cls):
        """
        Construct a functional form of the optimizer.
        """
        @unitful
        def intrinsic(dist, rvs=None, crvs=None, niter=None, bound=None, rv_mode=None):
            opt = cls(dist, rvs, crvs, rv_mode, bound=bound)
            opt.optimize(niter=niter)
            return opt.objective(opt._optima)

        intrinsic.__doc__ = \
        """
        Compute the intrinsic {name}.

        Parameters
        ----------
        dist : Distribution
            The distribution to compute the intrinsic {name} of.
        rvs : list, None
            A list of lists. Each inner list specifies the indexes of the random
            variables used to calculate the intrinsic {name}. If None,
            then it is calculated over all random variables, which is equivalent
            to passing `rvs=dist.rvs`.
        crvs : list
            A single list of indexes specifying the random variables to
            condition on.
        niter : int
            The number of optimization iterations to perform.
        bound : int, None
            Bound on the size of the auxiliary variable. If None, use the
            theoretical bound.
        rv_mode : str, None
            Specifies how to interpret `rvs` and `crvs`. Valid options are:
            {{'indices', 'names'}}. If equal to 'indices', then the elements of
            `crvs` and `rvs` are interpreted as random variable indices. If
            equal to 'names', the the elements are interpreted as random
            variable names. If `None`, then the value of `dist._rv_mode` is
            consulted, which defaults to 'indices'.
        """.format(name=cls.name)

        return intrinsic


class BaseMoreIntrinsicMutualInformation(BaseAuxVarOptimizer):
    """
    Compute the reduced and minimal intrinsic mutual informations, upper bounds on the secret
    key agreement rate:

        I[X : Y \downarrow\downarrow\downarrow Z] = min_U I[X:Y|U] + I[XY:U|Z]
    """

    name = ""

    construct_initial = BaseAuxVarOptimizer.construct_random_initial

    def __init__(self, dist, rvs=None, crvs=None, bound=None, rv_mode=None):
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
        if not crvs:
            msg = "Intrinsic mutual informations require a conditional variable."
            raise ditException(msg)

        super(BaseMoreIntrinsicMutualInformation, self).__init__(dist, rvs, crvs, rv_mode=rv_mode)

        theoretical_bound = prod(self._shape)
        bound = min([bound, theoretical_bound]) if bound else theoretical_bound

        self._construct_auxvars([(self._rvs | self._crvs, bound)])

    @abstractmethod
    def measure(self, rvs, crvs):
        """
        Abstract method for computing the appropriate measure of generalized
        mutual information.

        Parameters
        ----------
        rvs : set
            The set of random variables.
        crvs : set
            The set of conditional random variables.

        Returns
        -------
        gmi : func
            The generalized mutual information.
        """
        pass

    @classmethod
    def functional(cls):
        """
        Construct a functional form of the optimizer.
        """

        def intrinsic(dist, rvs=None, crvs=None, niter=None, bounds=(2, 3, 4, None), rv_mode=None):
            candidates = []
            for bound in bounds:
                opt = cls(dist, rvs, crvs, bound, rv_mode)
                opt.optimize(niter=niter)
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
            niter : int
                The number of optimization iterations to perform.
            bounds : [int], None
                Bounds on the size of the auxiliary variable. If None, use the
                theoretical bound. This is used to better sample smaller subspaces.
            rv_mode : str, None
                Specifies how to interpret `rvs` and `crvs`. Valid options are:
                {{'indices', 'names'}}. If equal to 'indices', then the elements of
                `crvs` and `rvs` are interpreted as random variable indices. If
                equal to 'names', the the elements are interpreted as random
                variable names. If `None`, then the value of `dist._rv_mode` is
                consulted, which defaults to 'indices'.
            """.format(name=cls.name, type=cls.type)

        return intrinsic
