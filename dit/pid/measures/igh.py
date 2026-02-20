"""
The redundancy measure of Griffith & Ho.
"""

from ...algorithms.optimization import BaseAuxVarOptimizer, BaseConvexOptimizer, OptimizationException
from ...math import prod
from ...shannon import entropy, mutual_information
from ..pid import BasePID

__all__ = ("PID_GH",)


class GHOptimizer(BaseConvexOptimizer, BaseAuxVarOptimizer):
    """
    Optimizer for the Griffith & Ho redundancy.
    """

    construct_initial = BaseAuxVarOptimizer.construct_copy_initial

    def __init__(self, dist, sources, target, bound=None, rv_mode=None):
        """
        Initialize the optimizer.

        Parameters
        ----------
        dist : Distribution
            The distribution to compute i_gh of.
        sources : list, None
            A list of lists. Each inner list specifies the indexes of the random
            variables used to calculate the intrinsic mutual information. If None,
            then it is calculated over all random variables, which is equivalent
            to passing `rvs=dist.rvs`.
        target : list
            A single list of indexes specifying the random variables to
            treat as the target.
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
        super().__init__(dist, sources, target, rv_mode=rv_mode)

        q_size = prod(self._shape) + 1
        bound = min([bound, q_size]) if bound is not None else q_size

        self._construct_auxvars([(self._rvs | self._crvs, bound)])

        entropies = [entropy(dist, source + target) for source in sources]
        mutual_informations = [mutual_information(dist, source, target) for source in sources]
        trivial = all(abs(h - i) < 1e-6 for h, i in zip(entropies, mutual_informations, strict=True))
        if not trivial:
            self.constraints += [
                {
                    "type": "eq",
                    "fun": self.constraint_markov_chains(),
                },
            ]

        self._additional_options = {
            "options": {
                "maxiter": 2500,
                "ftol": 1e-6,
                "eps": 1.4901161193847656e-9,
            }
        }

    def constraint_markov_chains(self):
        """
        Constrain I[Q : Y | Xi] = 0.

        Parameters
        ----------
        x : np.ndarray
            An optimization vector.
        """
        cmis = [self._conditional_mutual_information(self._crvs, self._arvs, {rv}) for rv in self._rvs]

        def constraint(x):
            pmf = self.construct_joint(x)
            return sum(cmi(pmf) for cmi in cmis)

        return constraint

    def _objective(self):
        """
        The mutual information between the target and the auxiliary variable.

        Returns
        -------
        obj : func
            The objective function.
        """
        mi = self._mutual_information(self._arvs, self._crvs)

        def objective(self, x):
            """
            Compute I[Q : Y]

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
            return -mi(pmf)

        return objective


class PID_GH(BasePID):
    """
    The I_GH measure defined by Griffith & Ho.
    """

    _name = "I_GH"

    @staticmethod
    def _measure(d, sources, target):
        """
        Compute I_GH(sources : target) =
            \\max I[Q : target] such that I[Q : Y | Xi] = 0

        Parameters
        ----------
        d : Distribution
            The distribution to compute i_gh for.
        sources : iterable of iterables
            The source variables.
        target : iterable
            The target variable.

        Returns
        -------
        igh : float
            The value of I_GH.
        """
        if len(sources) == 1:
            return mutual_information(d, sources[0], target)
        md = d.coalesce(sources)
        upper_bound = prod(len(a) for a in md.alphabet) + 1
        for bound in range(upper_bound, md.outcome_length(), -1):
            try:
                gho = GHOptimizer(d, sources, target, bound=bound)
                gho.optimize()
                break
            except OptimizationException:
                continue
        q = len(sources) + 1
        od = gho.construct_distribution()
        return mutual_information(od, target, [q])
