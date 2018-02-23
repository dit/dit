"""
Information measures based on Mike DeWeese's multivariate mutual information.
"""
from itertools import product

from ..algorithms import BaseAuxVarOptimizer
from ..distconst import RVFunctions, insert_rvf
from ..helpers import normalize_rvs
from ..utils import extended_partition, partitions, unitful


__all__ = [
    'deweese_coinformation',
    'deweese_total_correlation',
    'deweese_dual_total_correlation',
    'deweese_caekl_mutual_information',
]


def deweese_constructor(mmi):
    """
    Construct a DeWeese-like multivariate mutual information.

    Parameters
    ----------
    mmi : func
        A multivariate mutual information.

    Returns
    -------
    deweese_mmi : func
        A DeWeese'd form of `mmi`.
    """
    @unitful
    def deweese(dist, rvs=None, crvs=None, return_opt=False, rv_mode=None):
        """
        Compute the DeWeese form of {name}.

        Parameters
        ----------
        dist : Distribution
            The distribution to work with.
        rvs : iter of iters, None
            The variables of interest. If None, use all.
        crvs : iter, None
            The variables to condition on. If None, none.
        return_opt : bool
            Whether to return the distribution containing the
            variable functions or not. Defaults to False.
        rv_mode : str, None
            Specifies how to interpret `rvs` and `crvs`. Valid options are:
            {{'indices', 'names'}}. If equal to 'indices', then the elements of
            `crvs` and `rvs` are interpreted as random variable indices. If
            equal to 'names', the the elements are interpreted as random
            variable names. If `None`, then the value of `dist._rv_mode` is
            consulted, which defaults to 'indices'.

        Returns
        -------
        val : float
            The value of the DeWeese {name}
        opt_d : Distribution
            The distribution with the functions achieving `val`.
            Only returned if `return_opt` is True.
        """
        rvs, crvs, rv_mode = normalize_rvs(dist, rvs, crvs, rv_mode)

        dist = dist.coalesce(rvs + [crvs])

        new_rvs = [[i + len(rvs) + 1] for i, _ in enumerate(rvs)]

        new_crvs = [dist.outcome_length() - 1]

        rvf = RVFunctions(dist)

        def all_funcs():
            """
            A generator to construct all possible functions of the variables.

            Yields
            ------
            d : Distribution
                A distribution with additional indices corresponding
                to functions of those variables.
            """
            partss = [partitions(set([(o[i],) for o in dist.outcomes])) for i, _ in enumerate(rvs)]
            for parts in product(*partss):
                d = dist.copy()
                for i, part in enumerate(parts):
                    new_part = extended_partition(d.outcomes, [i], part, d._outcome_ctor)
                    d = insert_rvf(d, rvf.from_partition(new_part))
                yield d

        possibilities = ((mmi(d, rvs=new_rvs, crvs=new_crvs), d) for d in all_funcs())

        opt_val, opt_d = max(possibilities, key=lambda t: t[0])

        if return_opt:
            return opt_val, opt_d
        else:
            return opt_val

    deweese.__doc__ = deweese.__doc__.format(name=mmi.__name__)

    return deweese


class BaseDeWeeseOptimizer(BaseAuxVarOptimizer):
    """
    An optimizer for DeWeese-style multivariate mutual informations.
    """
    construct_initial = BaseAuxVarOptimizer.construct_copy_initial

    _shotgun = True

    def __init__(self, dist, rvs=None, crvs=None, deterministic=False, rv_mode=None):
        """
        Initialize the optimizer.

        Parameters
        ----------
        dist : Distribution
            The distribution to optimize.
        rvs : iter of iters
            The random variables of interest.
        crvs : iter
            The random variables to condition on.
        deterministic : bool
            Whether the functions to optimize over should be
            deterministic or not. Defaults to False.
        rv_mode : str, None
            Specifies how to interpret `rvs` and `crvs`. Valid options are:
            {{'indices', 'names'}}. If equal to 'indices', then the elements of
            `crvs` and `rvs` are interpreted as random variable indices. If
            equal to 'names', the the elements are interpreted as random
            variable names. If `None`, then the value of `dist._rv_mode` is
            consulted, which defaults to 'indices'.
        """
        super(BaseDeWeeseOptimizer, self).__init__(dist, rvs=rvs, crvs=crvs, rv_mode=rv_mode)
        self._construct_auxvars([({rv}, size) for rv, size in zip(self._rvs, self._shape)])

        if deterministic:
            self.constraints = [{'type': 'eq',
                                 'fun': self._constraint_deterministic(),
                                 },
                                ]
            self._default_hops *= 2

    @classmethod
    def functional(cls):
        """
        Construct a functional form of this optimizer.

        Returns
        -------
        function : func
            A function which constructs this optimizer and performs the optimization.
        """
        @unitful
        def function(dist, rvs=None, crvs=None, niter=None, deterministic=False, rv_mode=None):
            """
            Compute the DeWeese {name}.

            Parameters
            ----------
            dist : Distribution
                The distribution of interest.
            rvs : iter of iters, None
                The random variables of interest. If None, use all.
            crvs : iter, None
                The variables to condition on. If None, none.
            niter : int, None
                If specified, the number of optimization steps to perform.
            deterministic : bool
                Whether the functions to optimize over should be
                deterministic or not. Defaults to False.
            rv_mode : str, None
                Specifies how to interpret `rvs` and `crvs`. Valid options are:
                {{'indices', 'names'}}. If equal to 'indices', then the elements of
                `crvs` and `rvs` are interpreted as random variable indices. If
                equal to 'names', the the elements are interpreted as random
                variable names. If `None`, then the value of `dist._rv_mode` is
                consulted, which defaults to 'indices'.

            Returns
            -------
            val : float
                The value of the DeWeese {name}.
            """
            opt = cls(dist, rvs=rvs,
                            crvs=crvs,
                            rv_mode=rv_mode,
                            deterministic=deterministic)
            opt.optimize(niter=niter)
            return -opt.objective(opt._optima)

        function.__doc__ = function.__doc__.format(name=cls.name)

        return function


class DeWeeseCoInformation(BaseDeWeeseOptimizer):
    """
    The DeWeese Co-Information:
        I_D[X_0 : ... : X_n | Y] = max_{p(x'_i | x_i)} I[X'_0 : ... : X'_n | Y]
    """

    name = 'coinformation'

    def _objective(self):
        """
        The conditional co-information.

        Returns
        -------
        obj : func
            The objective function.
        """
        coi = self._coinformation(rvs=self._arvs, crvs=self._crvs)

        def objective(self, x):
            """
            The conditional coinformation.

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
            return -coi(pmf)

        return objective


deweese_coinformation = DeWeeseCoInformation.functional()


class DeWeeseTotalCorrelation(BaseDeWeeseOptimizer):
    """
    The DeWeese Total Correlation:
        T_D[X_0 : ... : X_n | Y] = max_{p(x'_i | x_i)} T[X'_0 : ... : X'_n | Y]
    """

    name = 'total correlation'

    def _objective(self):
        """
        The conditional total correlation.

        Returns
        -------
        obj : func
            The objective function.
        """
        tc = self._total_correlation(rvs=self._arvs, crvs=self._crvs)

        def objective(self, x):
            """
            The conditional total correlation.

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
            return -tc(pmf)

        return objective


deweese_total_correlation = DeWeeseTotalCorrelation.functional()


class DeWeeseDualTotalCorrelation(BaseDeWeeseOptimizer):
    """
    The DeWeese Dual Total Correlation:
        B_D[X_0 : ... : X_n | Y] = max_{p(x'_i | x_i)} B[X'_0 : ... : X'_n | Y]
    """

    name = 'dual total correlation'

    def _objective(self):
        """
        The conditional dual total correlation.

        Returns
        -------
        obj : func
            The objective function.
        """
        dtc = self._dual_total_correlation(rvs=self._arvs, crvs=self._crvs)

        def objective(self, x):
            """
            The conditional dual total correlation.

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
            return -dtc(pmf)

        return objective


deweese_dual_total_correlation = DeWeeseDualTotalCorrelation.functional()


class DeWeeseCAEKLMutualInformation(BaseDeWeeseOptimizer):
    """
    The DeWeese CAEKL Mutual Information:
        J_D[X_0 : ... : X_n | Y] = max_{p(x'_i | x_i)} J[X'_0 : ... : X'_n | Y]
    """

    name = 'caekl mutual information'

    def _objective(self):
        """
        The conditional caekl mutual information.

        Returns
        -------
        obj : func
            The objective function.
        """
        caekl = self._caekl_mutual_information(rvs=self._arvs, crvs=self._crvs)

        def objective(self, x):
            """
            The conditional caekl mutual information.

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
            return -caekl(pmf)

        return objective


deweese_caekl_mutual_information = DeWeeseCAEKLMutualInformation.functional()
