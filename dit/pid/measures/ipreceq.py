"""
Add the I_\preceq measure as defined by Kolchinsky.
"""

import numpy as np

from ...algorithms.optimization import (BaseAuxVarOptimizer,
                                        BaseConvexOptimizer,
                                        OptimizationException)
from ...shannon import mutual_information
from ..pid import BasePID


__all__ = (
    'PID_Preceq',
)


class KolchinskyOptimizer(BaseConvexOptimizer, BaseAuxVarOptimizer):
    """
    An optimizer to find the greatest I[Q:Y] such that p(q|y) is a garbling
    of each p(xi|y).
    """

    def __init__(self, dist, sources, target, bound=None, rv_mode=None):
        """
        Initialize the optimizer.

        Parameters
        ----------
        dist : Distribution
            The distribution to compute the i_preceq of.
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

        q_size = sum(self._shape[:-1]) - len(sources) + 1
        bound = min([bound, q_size]) if bound is not None else q_size

        auxvars = [(self._crvs, bound)]
        for rv in sorted(self._rvs):
            auxvars.append(({rv}, bound))
        self._construct_auxvars(auxvars)

        self.constraints += [{'type': 'eq',
                              'fun': self.constraint_garbling,
                              },
                             ]

    def constraint_garbling(self, x):
        """
        Constrane p(q | y) to be a garbling of each p(xi | y).

        Parameters
        ----------
        x : np.ndarray
            An optimization vector.

        Returns
        -------
        delta : float >= 0
            The deviation from Blackwell ordering.
        """
        rvs = sorted(self._rvs)
        arvs = sorted(self._arvs)

        joint = self.construct_joint(x)

        p_q_y = joint.sum(axis=tuple(rvs + arvs[1:])).T
        p_q_g_y = p_q_y / p_q_y.sum(axis=0, keepdims=True)
        p_q_g_y /= p_q_g_y.sum(axis=0, keepdims=True)

        delta = 0

        for i in range(len(rvs)):
            other_rvs = rvs[:i] + rvs[i + 1:]
            other_arvs = arvs[:i + 1] + arvs[i + 2:]

            p_q_xi = joint.sum(axis=tuple(other_rvs + other_arvs + list(self._crvs))).T
            p_q_g_xi = p_q_xi / p_q_xi.sum(axis=0, keepdims=True)
            p_q_g_xi /= p_q_g_xi.sum(axis=0, keepdims=True)

            p_xi_y = joint.sum(axis=tuple(other_rvs + arvs))
            p_xi_g_y = p_xi_y / p_xi_y.sum(axis=0, keepdims=True)
            p_xi_g_y /= p_xi_g_y.sum(axis=0, keepdims=True)

            s_q_g_y = (p_q_g_xi[:, :, np.newaxis] * p_xi_g_y[np.newaxis, :, :]).sum(axis=1)

            delta += abs(s_q_g_y - p_q_g_y).sum()**2

        return delta

    def _objective(self):
        """
        The mutual information between the target and the auxiliary variable.

        Returns
        -------
        obj : func
            The objective function.
        """
        q = {sorted(self._arvs)[0]}
        mi = self._mutual_information(q, self._crvs)

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


class PID_Preceq(BasePID):
    """
    The I_\preceq measure defined by Kolchinsky.
    """

    _name = "I_≼"

    @staticmethod
    def _measure(d, sources, target):
        """
        Compute I_preceq(inputs : output) =
            \\max I[Q : output] such that p(q|y) \preceq p(xi|y)

        Parameters
        ----------
        d : Distribution
            The distribution to compute i_min for.
        sources : iterable of iterables
            The source variables.
        target : iterable
            The target variable.

        Returns
        -------
        ipreceq : float
            The value of I_preceq.
        """
        if len(sources) == 1:
            return mutual_information(d, sources[0], target)
        md = d.coalesce(sources)
        upper_bound = sum(len(a) for a in md.alphabet) - md.outcome_length() + 1
        for bound in [None] + list(range(upper_bound, md.outcome_length(), -1)):
            try:
                ko = KolchinskyOptimizer(d, sources, target, bound=bound)
                ko.optimize(polish=1e-8)
                break
            except OptimizationException:
                continue
        q = len(sources) + 1
        od = ko.construct_distribution().marginal(range(q + 1))
        return mutual_information(od, target, [q])
