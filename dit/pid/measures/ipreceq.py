"""
Add the I_\\preceq measure as defined in 

A Kolchinsky, A novel approach to multivariate redundancy and synergy
https://arxiv.org/abs/1908.08642

Given a joint distribution p_{Y,X_1,X_2}, this redundancy measure is 
given by the solution to the following optimization problem:

R = min_{s_{Q|Y}} I_s(Y;Q) 
             s.t. s_{Q|Y} ⪯ p_{X_i|Y} ∀i

This can in turn be re-written as:
 R = min_{s_{Q|Y},s_{Q|X_1}, .., s_{Q|X_n}} I_s(Y;Q)
     s.t. ∀i,q,y : Σ_{x_i} s(q|x_i) p(x_i,y) = s(q|y)p(y)

Note that this is optimization problem involes a maximization of 
a convex function subject to a system of linear constraints.  This 
system of linear constraints defines a convex polytope, and the maximum 
of the function will lie at one of the vertices of the polytope.  We 
proceed by finding these vertices, iterating over them, and finding 
the vertex with the largest value of I(Y;Q).
"""

import numpy as np

from ...algorithms.convex_maximization import maximize_convex_function
from ..pid import BasePID
from ...shannon import mutual_information, entropy


__all__ = (
    'PID_Preceq',
)



class PID_Preceq(BasePID):
    """
    The I_\\preceq measure defined by Kolchinsky.
    """

    _name = "I_≼"

    @staticmethod
    def _measure(d, sources, target, eps=1e-8, return_solution_information=False):
        """
        Compute I_preceq(inputs : output) =
            \\max I[Q : output] such that p(q|y) \\preceq p(xi|y)

        Parameters
        ----------
        d : Distribution
            The distribution to compute i_min for.
        sources : iterable of iterables
            The source variables.
        target : iterable
            The target variable.
        eps : float
            Rounding error. We must round our conditional probability distributions
            to rationals (the library ppl requires rationals)
        return_solution_information : bool 
            if True, returns solution information in terms of joint distribution p(Q,Y) and 
            conditional distributions p(Q|X_i)

        Returns 
        -------
        ipreceq : float
            The value of I_preceq.
        sol : dict (only if return_solution_information=True)
            Dictionary containing solution information

        Returns
        -------
        """
        if len(sources) == 1:
            return mutual_information(d, sources[0], target)

        pjoint      = d.coalesce(list(sources) + [target,])
        target_rvndx = len(pjoint.rvs) - 1
        pY           = pjoint.marginal([target_rvndx,], rv_mode='indices')
        n_y          = len(pY)
        
        if n_y <= 1:
            # Trivial case where target has a single outcome, redundancy has to be 0
            return 0, {}
        
        # variablesQgiven holds ppl variables that represent conditional probability
        # values of s(Q=q|X_i=x_i) for different sources i, as well as s(Q=q|Y=y)
        # for the target Y (recall that Q is our redundancy random variable)
        variablesQgiven = {} 

        var_ix        = 0  # counter for tracking how many variables we've created
        

        # Calculate the maximum number of outcomes we will require for Q
        n_q = sum([len(alphabet)-1 for rvndx, alphabet in enumerate(pjoint.alphabet)
                                 if rvndx != target_rvndx]) + 1
                                 
        # Iterate over all the random variables (R.V.s): i.e., all the sources + the target 
        for rvndx, rv in enumerate(pjoint.rvs):
            variablesQgiven[rvndx] = {}

            mP = pjoint.marginal([rvndx,]) # the marginal distribution over the current R.V.
            if len(mP._outcomes_index) != len(pjoint.alphabet[rvndx]):
                raise Exception('All marginals should have full support ' +
                                '(to proceed, drop outcomes with 0 probability)')

            # Iterate over outcomes of current R.V.
            for v in pjoint.alphabet[rvndx]:
                sum_to_one = 0 
                for q in range(n_q):
                    # represents s(Q=q|X_rvndx=v) if rvndx != target_rvndx
                    #        and s(Q=q|Y=v)       if rvndx == target_rvndx
                    variablesQgiven[rvndx][(q, v)] = var_ix 
                    var_ix += 1
        
        num_vars = var_ix 

        A_eq  , b_eq   = [], []  # linear constraints Ax =b
        A_ineq, b_ineq = [], []  # linear constraints Ax<=b

        for rvndx, rv in enumerate(pjoint.rvs):
            for v in pjoint.alphabet[rvndx]:
                sum_to_one = np.zeros(num_vars, dtype='int')
                for q in range(n_q):
                    var_ix = variablesQgiven[rvndx][(q, v)]

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
        for rvndx, rv in enumerate(pjoint.rvs):
            if rvndx == target_rvndx:
                continue

            # Compute joint marginal of target Y and source X_rvndx
            pYSource = pjoint.marginal([rvndx,target_rvndx,], rv_mode='indices')
            for q in range(n_q):
                for y in pjoint.alphabet[target_rvndx]:
                    z = np.zeros(num_vars, dtype='int')
                    cur_mult   = 0.  # multiplier to make everything rational, and make rounded values add up to 1
                    for x in pjoint.alphabet[rvndx]:
                        # We divide by eps and round, and then multiply by cur_mult,
                        # to make everything rational
                        pXY        = int( pYSource[pYSource._outcome_ctor((x,y))] / eps )
                        cur_mult   += pXY
                        z[variablesQgiven[rvndx][(q, x)]] = pXY 
                    z[variablesQgiven[target_rvndx][(q, y)]] = -cur_mult
                    A_eq.append(z)
                    b_eq.append(0)

        # Define a matrix sol_mx that allows for fast mapping from solution vector 
        # returned by ppl (i.e., a particular extreme point of our polytope) to a 
        # joint distribution over Q and Y
        mul_mx = np.zeros((num_vars, n_q*n_y))
        y_ixs  = {}
        for (q,y), k in variablesQgiven[target_rvndx].items():
            if y not in y_ixs: 
                y_ixs[y] = len(y_ixs)
            mul_mx[k, q*n_y + y_ixs[y]] += pY[y]

        H_Y = entropy(pjoint, rvs=[target_rvndx,], rv_mode='indices')
        if pjoint.is_log():
            normConst = np.log(pjoint.get_base(numerical=True))
        else:
            normConst = np.log(2)

        def entr(x):
            x = x + 1e-18
            return -x*np.log(x)            

        def objective(x):  # efficient calculation of mutual information
            # Map solution vector x to joint distribution over Q and Y
            pQY     = x.dot(mul_mx).reshape((n_q,n_y))
            probs_q = pQY.sum(axis=1) + 1e-18
            H_YgQ   = entr(pQY/probs_q[:,None]).sum(axis=1).dot(probs_q)
            return H_Y-H_YgQ/normConst

        # The following uses ppl to turn our system of linear inequalities into a 
        # set of extreme points of the corresponding polytope. It then calls 
        # get_solution_val on each extreme point
        x_opt, v_opt = maximize_convex_function(
            f=objective,
            A_eq=np.array(A_eq, dtype='int'), 
            b_eq=np.array(b_eq, dtype='int'), 
            A_ineq=np.array(A_ineq, dtype='int'), 
            b_ineq=np.array(b_ineq, dtype='int'))

        # Return mutual information I(Q;Y) and possibly solution information
        if return_solution_information:
            sol = {}
            sol['p(Q,Y)'] = x_opt.dot(mul_mx).reshape((n_q,n_y))

            # Compute conditional distributions of Q given each source X_i
            for rvndx in range(len(pjoint.rvs)-1):
                pX = pjoint.marginal([rvndx,])
                cK = 'p(Q|X%d)'%rvndx
                sol[cK] = np.zeros( (n_q, len(pX.alphabet[0]) ) )
                for (q,v), k in variablesQgiven[rvndx].items():
                    v_ix = pX._outcomes_index[v]
                    sol[cK][q,v_ix] = x_opt[k]

            return v_opt, sol
        else:
            return v_opt
