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
information of `ideg`, and by default it is computed exactly by the same
linear program over the extreme rays of the cone of common degradations. If
the cardinality of Q is fixed, the maximum is instead found by enumerating the
vertices of the polytope of degradation channels and evaluating I(Y;Q) at
each one.
"""

import numpy as np
from scipy.special import entr

from ...algorithms.convex_maximization import maximize_convex_function
from ...shannon import mutual_information
from ..pid import BasePID
from .ideg import _degradation_lp

__all__ = ("PID_Prec",)


def _lp_solution(pjoint, target_rvndx, probs_y):
    """
    Compute I_prec and an optimal Q by the linear program over the extreme rays of the
    cone of common degradations.
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

    v_opt, lambdas = _degradation_lp(channels, probs_y)

    sol = {"p(Q,Y)": (probs_y[:, None] * (channels[0] @ lambdas[0])).T}
    for rvndx, lam in enumerate(lambdas):
        sol[f"p(Q|X{rvndx})"] = lam.T
    return v_opt, sol


class PID_Prec(BasePID):
    """
    The :math:`I_{\\prec}` measure defined by Kolchinsky.

    This is the degradation intersection information of
    :class:`~dit.pid.PID_Deg`, for any number of sources. By default it is
    computed exactly, by the same linear program over the extreme rays of the
    cone of common degradations. Passing ``n_q`` fixes the cardinality of the
    redundancy variable and finds the maximum by vertex enumeration instead.
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
            The distribution to compute i_min for.
        sources : iterable of iterables
            The source variables.
        target : iterable
            The target variable.
        n_q : int (default None)
            The cardinality of the redundancy random variable Q. If not specified,
            the exact maximum is computed by a linear program over the extreme rays
            of the cone of common degradations, and Q needs at most the cardinality
            from Theorem A1 in the paper. If specified, the maximum over Q with this
            cardinality is computed by enumerating the vertices of the polytope of
            degradation channels (which requires pypoman); a cardinality below that
            of Theorem A1 may only give a lower bound on the Blackwell redundancy.

        Returns
        -------
        iprec : float
            The value of I_prec.
        sol : dict
            Dictionary containing solution information

        Returns
        -------
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

        if n_q is None:
            return _lp_solution(pjoint, target_rvndx, probs_y)

        # variablesQgiven holds ppl variables that represent conditional probability
        # values of s(Q=q|X_i=x_i) for different sources i, as well as s(Q=q|Y=y)
        # for the target Y (recall that Q is our redundancy random variable)
        variablesQgiven = {}

        var_ix = 0  # counter for tracking how many variables we've created

        # Iterate over all the random variables (R.V.s): i.e., all the sources + the target
        for rvndx, _rv in enumerate(pjoint.rvs):
            variablesQgiven[rvndx] = {}

            # Iterate over outcomes of current R.V.
            for v_ix, _v in enumerate(pjoint.alphabet[rvndx]):
                sum_to_one = 0
                for q in range(n_q):
                    # represents s(Q=q|X_rvndx=v) if rvndx != target_rvndx
                    #        and s(Q=q|Y=v)       if rvndx == target_rvndx
                    variablesQgiven[rvndx][(q, v_ix)] = var_ix
                    var_ix += 1

        num_vars = var_ix

        A_eq, b_eq = [], []  # linear constraints Ax =b
        A_ineq, b_ineq = [], []  # linear constraints Ax<=b

        for rvndx, _rv in enumerate(pjoint.rvs):
            for v_ix, _v in enumerate(pjoint.alphabet[rvndx]):
                sum_to_one = np.zeros(num_vars)
                for q in range(n_q):
                    var_ix = variablesQgiven[rvndx][(q, v_ix)]

                    # Non-negative constraint on each variable
                    z = np.zeros(num_vars)
                    z[var_ix] = -1
                    A_ineq.append(z)
                    b_ineq.append(0)

                    sum_to_one[var_ix] = 1

                if rvndx != target_rvndx:
                    # Linear constraint that enforces Σ_q s(Q=q|X_rvndx=v) = 1
                    A_eq.append(sum_to_one)
                    b_eq.append(1)

        # Now we add the constraint:
        #    ∀i,q,y : Σ_{x_i} s(q|x_i) p(x_i,y) = s(q|y)p(y)
        for rvndx, _rv in enumerate(pjoint.rvs):
            if rvndx == target_rvndx:
                continue

            # Compute joint marginal of target Y and source X_rvndx
            pYSource = pjoint.marginal(
                [
                    rvndx,
                    target_rvndx,
                ],
            )
            for q in range(n_q):
                for y_ix, y in enumerate(pjoint.alphabet[target_rvndx]):
                    z = np.zeros(num_vars)
                    cur_marg = 0.0
                    for x_ix, x in enumerate(pjoint.alphabet[rvndx]):
                        pXY = pYSource[pYSource._outcome_ctor((x, y))]
                        cur_marg += pXY
                        z[variablesQgiven[rvndx][(q, x_ix)]] = pXY
                    z[variablesQgiven[target_rvndx][(q, y_ix)]] = -cur_marg
                    A_eq.append(z)
                    b_eq.append(0)

        # Define a matrix sol_mx that allows for fast mapping from solution vector
        # returned by ppl (i.e., a particular extreme point of our polytope) to a
        # joint distribution over Q and Y
        mul_mx = np.zeros((num_vars, n_q * n_y))
        for (q, y_ix), k in variablesQgiven[target_rvndx].items():
            mul_mx[k, q * n_y + y_ix] += probs_y[y_ix]

        H_Y = entr(probs_y).sum()
        ln2 = np.log(2)

        def objective(x):  # efficient calculation of mutual information
            # Map solution vector x to joint distribution over Q and Y
            pQY = x.dot(mul_mx).reshape((n_q, n_y))
            pQY += 1e-12  # Remove small negative values that may appear due to numerical issues
            probs_q = pQY.sum(axis=1)
            H_YgQ = entr(pQY / probs_q[:, None]).sum(axis=1).dot(probs_q)
            v = (H_Y - H_YgQ) / ln2
            if np.isclose(v, 0):
                v = 0
            return v

        # The following uses ppl to turn our system of linear inequalities into a
        # set of extreme points of the corresponding polytope. It then calls
        # objective on each extreme point
        x_opt, v_opt = maximize_convex_function(
            f=objective, A_eq=np.array(A_eq), b_eq=np.array(b_eq), A_ineq=np.array(A_ineq), b_ineq=np.array(b_ineq)
        )

        # Return mutual information I(Q;Y) and solution information
        sol = {}
        sol["p(Q,Y)"] = x_opt.dot(mul_mx).reshape((n_q, n_y))

        # Compute conditional distributions of Q given each source X_i
        for rvndx in range(len(pjoint.rvs) - 1):
            pX = pjoint.marginal(
                [
                    rvndx,
                ]
            )
            cK = f"p(Q|X{rvndx})"
            sol[cK] = np.zeros((n_q, len(pX.alphabet[0])))
            for (q, v_ix), k in variablesQgiven[rvndx].items():
                sol[cK][q, v_ix] = x_opt[k]

        return v_opt, sol

    @classmethod
    def _measure(cls, d, sources, target, n_q=None):
        """
        See information for get_solution_information method.
        """
        if len(sources) == 1:
            return mutual_information(d, sources[0], target)
        v_opt, _ = cls.get_solution_information(d, sources, target, n_q)
        return v_opt
