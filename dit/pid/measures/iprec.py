"""
Add the I_\\prec measure as defined in

A Kolchinsky, A Novel Approach to the Partial Information Decomposition, Entropy, 2022.
https://arxiv.org/abs/1908.08642

Given a joint distribution p_{Y,X_1,X_2}, this redundancy measure is
given by the solution to the following optimization problem:

R = max_{s_{Q|Y}} I_s(Y;Q)
             s.t. s_{Q|Y} ⪯ p_{X_i|Y} ∀i

This can in turn be re-written as:
 R = max_{s_{Q|Y},s_{Q|X_1}, .., s_{Q|X_n}} I_s(Y;Q)
     s.t. ∀i,q,y : Σ_{x_i} s(q|x_i) p(x_i,y) = s(q|y)p(y)

This is the maximization of a convex function subject to a system of linear
constraints. It is the same quantity as the degradation intersection
information of `ideg` (Gomes & Figueiredo's notation for this measure), and by
default it is computed exactly by the same
linear program over the extreme rays of the cone of common degradations. If
the cardinality of Q is fixed, the maximum is found exactly from the vertices
of that program's feasible polytope, by grouping the extreme rays of each into
the outcomes of Q.
"""

import numpy as np

from ...shannon import mutual_information
from ..pid import BasePID
from .ideg import _degradation_lp, _restricted_degradation

__all__ = ("PID_Prec",)


def _solution(pjoint, target_rvndx, probs_y, n_q=None):
    """
    Compute I_prec and an optimal Q from the extreme rays of the cone of common
    degradations: by their linear program, or with Q restricted to `n_q` outcomes.
    """
    channels = []
    for rvndx in range(len(pjoint.rvs)):
        if rvndx == target_rvndx:
            continue
        pYSource = pjoint.marginal([rvndx, target_rvndx])
        joint = np.array(
            [
                [pYSource[pYSource._outcome_ctor((x, y))] for x in pjoint.alphabet[rvndx]]
                for y in pjoint.alphabet[target_rvndx]
            ]
        )
        channels.append(joint / probs_y[:, None])

    if n_q is None:
        v_opt, lambdas = _degradation_lp(channels, probs_y)
    else:
        v_opt, lambdas = _restricted_degradation(channels, probs_y, n_q)

    sol = {"p(Q,Y)": (probs_y[:, None] * (channels[0] @ lambdas[0])).T}
    for rvndx, lam in enumerate(lambdas):
        sol[f"p(Q|X{rvndx})"] = lam.T
    return v_opt, sol


class PID_Prec(BasePID):
    """
    The :math:`I_{\\prec}` measure defined by Kolchinsky.

    This is the same measure as :class:`~dit.pid.PID_Deg`, the degradation
    intersection information of Gomes & Figueiredo. By default it is
    computed exactly, by a linear program over the extreme rays of the
    cone of common degradations. Passing ``n_q`` fixes the cardinality of the
    redundancy variable; the maximum is then found exactly from the vertices of
    that program's feasible polytope.
    """

    _name = "I_≺"

    @staticmethod
    def get_solution_information(d, sources, target, n_q=None):
        """
        Compute :math:`I_{\\prec}(\\text{inputs} : \\text{output}) =
        \\max I[Q : \\text{output}]` such that :math:`Q \\prec X_i`, and
        return information about the value and optimal Q.

        Parameters
        ----------
        d : Distribution
            The distribution to compute I_prec for.
        sources : iterable of iterables
            The source variables.
        target : iterable
            The target variable.
        n_q : int (default None)
            The cardinality of the redundancy random variable Q. If not specified,
            the exact maximum is computed by a linear program over the extreme rays
            of the cone of common degradations, and Q needs at most the cardinality
            from Theorem A1 in the paper. If specified, the maximum over Q with this
            cardinality is computed exactly, by grouping the extreme rays in the
            support of each vertex of the program's feasible polytope into at most
            n_q outcomes; a cardinality below that of Theorem A1 may only give a lower
            bound on the Blackwell redundancy.

        Returns
        -------
        iprec : float
            The value of I_prec.
        sol : dict
            Dictionary containing solution information
        """
        pjoint = d.coalesce(list(sources) + [target])
        target_rvndx = len(pjoint.rvs) - 1
        pY = pjoint.marginal([target_rvndx])
        # pY has a single (coalesced) variable, so each outcome is the 1-tuple (y,).
        probs_y = np.array([pY[(y,)] for y in pjoint.alphabet[target_rvndx]])
        n_y = len(pY)

        if not (n_q is None or (isinstance(n_q, int) and n_q >= 1)):
            raise Exception("Parameter n_q should be None or positive integer")

        if n_y <= 1 or n_q == 1:
            # Trivial case where target or redundancy has a single outcome, redundancy has to be 0
            return 0, {}

        for rvndx in range(len(pjoint.rvs)):
            mP = pjoint.marginal([rvndx])  # the marginal distribution over the current R.V.
            if len(mP._outcomes_index) != len(pjoint.alphabet[rvndx]):
                raise Exception("All marginals should have full support (to proceed, drop outcomes with 0 probability)")

        return _solution(pjoint, target_rvndx, probs_y, n_q)

    @classmethod
    def _measure(cls, d, sources, target, n_q=None):
        """
        See information for get_solution_information method.
        """
        if len(sources) == 1:
            return mutual_information(d, sources[0], target)
        v_opt, _ = cls.get_solution_information(d, sources, target, n_q)
        return v_opt
