"""
Another way to do maxent without using the convex solver from CVXOPT.

This uses the Frank-Wolfe algorithm:

    http://en.wikipedia.org/wiki/Frank%E2%80%93Wolfe_algorithm

"""
from __future__ import print_function

from itertools import combinations

import logging

from debtcollector import removals

import numpy as np

import dit
from dit.helpers import RV_MODES
from dit.utils import basic_logger

from .optutil import (
    as_full_rank, prepare_dist, op_runner, frank_wolfe
)
from .maxentropy import (
    marginal_constraints, marginal_constraints_generic, isolate_zeros_generic
)

__all__ = [
    # 'marginal_maxent_dists',
]


def initial_point_generic(dist, rvs, A=None, b=None, isolated=None, **kwargs):
    """
    Find an initial point in the interior of the feasible set.

    """
    from cvxopt import matrix
    from cvxopt.modeling import variable

    if isolated is None:
        variables = isolate_zeros_generic(dist, rvs)
    else:
        variables = isolated

    if A is None or b is None:
        A, b = marginal_constraints_generic(dist, rvs)

        # Reduce the size of A so that only nonzero elements are searched.
        # Also make it full rank.
        Asmall = A[:, variables.nonzero] # pylint: disable=no-member
        Asmall, b, rank = as_full_rank(Asmall, b)
        Asmall = matrix(Asmall)
        b = matrix(b)
    else:
        # Assume they are already CVXOPT matrices
        if A.size[1] == len(variables.nonzero): # pylint: disable=no-member
            Asmall = A
        else:
            msg = 'A must be the reduced equality constraint matrix.'
            raise Exception(msg)

    n = len(variables.nonzero) # pylint: disable=no-member
    x = variable(n)
    t = variable()

    tols = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    for tol in tols:
        constraints = []
        constraints.append((-tol <= Asmall * x - b))
        constraints.append((Asmall * x - b <= tol))
        constraints.append((x >= t))

        # Objective to minimize
        objective = -t

        opt = op_runner(objective, constraints, **kwargs)
        if opt.status == 'optimal':
            #print("Found initial point with tol={}".format(tol))
            break
    else:
        msg = 'Could not find valid initial point: {}'
        raise Exception(msg.format(opt.status))

    # Grab the optimized x
    optvariables = opt.variables()
    if len(optvariables[0]) == n:
        xopt = optvariables[0].value
    else:
        xopt = optvariables[1].value

    # Turn values close to zero to be exactly equal to zero.
    xopt = np.array(xopt)[:, 0]
    xopt[np.abs(xopt) < tol] = 0
    xopt /= xopt.sum()

    # Do not build the full vector since this is input to the reduced
    # optimization problem.
    #xx = np.zeros(len(dist.pmf))
    #xx[variables.nonzero] = xopt

    return xopt, opt


def initial_point(dist, k, A=None, b=None, isolated=None, **kwargs):
    """
    Find an initial point in the interior of the feasible set.

    """
    n_variables = dist.outcome_length()

    if m > n_variables:
        msg = "Cannot constrain {0}-way marginals"
        msg += " with only {1} random variables."
        msg = msg.format(m, n_variables)
        raise ValueError(msg)

    rvs = list(combinations(range(n_variables), m))
    kwargs['rv_mode'] = 'indices'

    return initial_point_generic(dist, rvs, A, b, isolated, **kwargs)

def check_feasibility(dist, k, **kwargs):
    """
    Checks feasibility by solving the minimum residual problem:

        minimize: max(abs(A x - b))

    If the value of the objective is close to zero, then we know that we
    can match the constraints, and so, the problem is feasible.

    """
    from cvxopt import matrix
    from cvxopt.modeling import variable

    A, b = marginal_constraints(dist, k)
    A = matrix(A)
    b = matrix(b)

    n = len(dist.pmf)
    x = variable(n)
    t = variable()

    c1 = (-t <= A * x - b)
    c2 = (A * x - b <= t)
    c3 = (x >= 0)

    objective = t
    constraints = [c1, c2, c3]

    opt = op_runner(objective, constraints, **kwargs)
    if opt.status != 'optimal':
        raise Exception('Not feasible')

    return opt

def negentropy(p):
    """
    Entropy which operates on vectors of length N.

    """
    # This works fine even if p is a n-by-1 cvxopt.matrix.
    return np.nansum(p * np.log2(p))

def marginal_maxent_generic(dist, rvs, **kwargs):
    from cvxopt import matrix

    verbose = kwargs.get('verbose', False)
    logger = basic_logger('dit.maxentropy', verbose)

    rv_mode = kwargs.pop('rv_mode', None)

    A, b = marginal_constraints_generic(dist, rvs, rv_mode)

    # Reduce the size of A so that only nonzero elements are searched.
    # Also make it full rank.
    variables = isolate_zeros_generic(dist, rvs)
    Asmall = A[:, variables.nonzero] # pylint: disable=no-member
    Asmall, b, rank = as_full_rank(Asmall, b)
    Asmall = matrix(Asmall)
    b = matrix(b)

    # Set cvx info level based on logging.INFO level.
    if logger.isEnabledFor(logging.INFO):
        show_progress = True
    else:
        show_progress = False

    logger.info("Finding initial distribution.")
    initial_x, _ = initial_point_generic(dist, rvs, A=Asmall, b=b,
                                         isolated=variables,
                                         show_progress=show_progress)
    initial_x = matrix(initial_x)
    objective = negentropy

    # We optimize the reduced problem.

    # For the gradient, we are going to keep the elements we know to be zero
    # at zero. Generally, the gradient is: log2(x_i) + 1 / ln(b)
    nonzero = variables.nonzero # pylint: disable=no-member
    ln2 = np.log(2)
    def gradient(x):
        # This operates only on nonzero elements.

        xarr = np.asarray(x)
        # All of the optimization elements should be greater than zero
        # But occasional they might go slightly negative or zero.
        # In those cases, we will just set the gradient to zero and keep the
        # value fixed from that point forward.
        bad_x = xarr <= 0
        grad = np.log2(xarr) + 1 / ln2
        grad[bad_x] = 0
        return matrix(grad)

    logger.info("Finding maximum entropy distribution.")
    x, obj = frank_wolfe(objective, gradient, Asmall, b, initial_x, **kwargs)
    x = np.asarray(x).transpose()[0]

    # Rebuild the full distribution.
    xfinal = np.zeros(A.shape[1])
    xfinal[nonzero] = x

    return xfinal, obj#, Asmall, b, variables

def marginal_maxent(dist, k, **kwargs):
    n_variables = dist.outcome_length()

    if k > n_variables:
        msg = "Cannot constrain {0}-way marginals"
        msg += " with only {1} random variables."
        msg = msg.format(m, n_variables)
        raise ValueError(msg)

    rv_mode = kwargs.pop('rv_mode', None)

    if rv_mode is None:
        rv_mode = dist._rv_mode

    if rv_mode in [RV_MODES.NAMES, 'names']:
        vars = dist.get_rv_names()
        rvs = list(combinations(vars, k))
    else:
        rvs = list(combinations(range(n_variables), k))

    kwargs['rv_mode'] = rv_mode

    return marginal_maxent_generic(dist, rvs, **kwargs)

@removals.remove(message="Please use the version in dit.algorithms.scipy_optmizers instead.",
                 version="1.0.0.dev8")
def marginal_maxent_dists(dist, k_max=None, maxiters=1000, tol=1e-3, verbose=False):
    """
    Return the marginal-constrained maximum entropy distributions.

    Parameters
    ----------
    dist : distribution
        The distribution used to constrain the maxent distributions.
    k_max : int
        The maximum order to calculate.

    """
    dist = prepare_dist(dist)

    n_variables = dist.outcome_length()
    symbols = dist.alphabet[0]

    if k_max is None:
        k_max = n_variables

    outcomes = list(dist._sample_space)

    # Optimization for the k=0 and k=1 cases are slow since you have to optimize
    # the full space. We also know the answer in these cases.

    # This is safe since the distribution must be dense.
    k0 = dit.Distribution(outcomes, [1]*len(outcomes), base='linear', validate=False)
    k0.normalize()

    k1 = dit.product_distribution(dist)

    dists = [k0, k1]
    for k in range(k_max + 1):
        if verbose:
            print("Constraining maxent dist to match {0}-way marginals.".format(k))

        if k in [0, 1, n_variables]:
            continue

        kwargs = {'maxiters': maxiters, 'tol': tol, 'verbose': verbose}
        pmf_opt, opt = marginal_maxent(dist, k, **kwargs)
        d = dit.Distribution(outcomes, pmf_opt)
        d.make_sparse()
        dists.append(d)

    # To match the all-way marginal is to match itself. Again, this is a time
    # savings decision, even though the optimization should be fast.
    if k_max == n_variables:
        dists.append(dist)

    return dists
