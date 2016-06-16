"""
Frank-Wolfe algorithm.

"""
# 2015-04-15
#
# We place this in its own module since using:
#
#   from __future__ import division
#
# seems to cause some issues with `cvxopt.matrix`, causing a TypeError.
#
# >>> from cvxopt import matrix
# >>> x = matrix([[1,2],[3,5]])
# >>> x / 3
# TypeError: unsupported operand type(s) for /: 'cvxopt.base.matrix' and 'int'

from __future__ import print_function

import logging
import numpy as np

from dit.utils import basic_logger

def frank_wolfe(objective, gradient, A, b, initial_x,
                maxiters=2000, tol=1e-4, clean=True, verbose=None):
    """
    Uses the Frank--Wolfe algorithm to minimize the convex objective.

    Minimization is subject to the linear equality constraint: A x = b.

    Assumes x should be nonnegative.

    Parameters
    ----------
    objective : callable
        The objective function. It would receive a ``cvxopt`` matrix for the
        input `x` and return the value of the objective function.
    gradient : callable
        The gradient function. It should receive a ``cvxopt`` matrix for the
        input `x` and return the value of the gradient evaluated at `x`.
    A : matrix
        A ``cvxopt`` matrix specifying the LHS linear equality constraints.
    b : matrix
        A ``cvxopt`` matrix specifying the RHS linear equality constraints.
    initial_x : matrix
        A ``cvxopt`` matrix specifying the initial `x` to use.
    maxiters : int
        The maximum number of iterations to perform. If convergence was not
        reached after the last iteration, a warning is issued and the current
        value of `x` is returned.
    tol : float
        The tolerance used to determine when we have converged to the optimum.
    clean : bool
        Occasionally, the iteration process will take nonnegative values to be
        ever so slightly negative. If ``True``, then we forcibly make such
        values equal to zero and renormalize the vector. This is an application
        specific decision and is probably not more generally useful.
    verbose : int
        An integer representing the logging level ala the ``logging`` module.
        If `None`, then (effectively) the log level is set to `WARNING`. For
        a bit more information, set this to `logging.INFO`. For a bit less,
        set this to `logging.ERROR`, or perhaps 100.

    """
    # Function level import to avoid circular import.
    from dit.algorithms.optutil import op_runner

    # Function level import to keep cvxopt dependency optional.
    # All variables should be cvxopt variables, not NumPy arrays
    from cvxopt.modeling import variable

    # Set up a custom logger.
    logger = basic_logger('dit.frankwolfe', verbose)

    # Set cvx info level based on logging.DEBUG level.
    if logger.isEnabledFor(logging.DEBUG):
        show_progress = True
    else:
        show_progress = False

    assert (A.size[1] == initial_x.size[0])

    n = initial_x.size[0]
    x = initial_x
    xdiff = 0

    TOL = 1e-7
    verbosechunk = maxiters / 10
    for i in range(maxiters):
        obj = objective(x)
        grad = gradient(x)

        xbar = variable(n)

        new_objective = grad.T * xbar
        constraints = []
        constraints.append((xbar >= 0))
        constraints.append((-TOL <= A * xbar - b))
        constraints.append((A * xbar - b <= TOL))

        logger.debug('FW Iteration: {}'.format(i))
        opt = op_runner(new_objective, constraints, show_progress=show_progress)
        if opt.status != 'optimal':
            msg = '\tFrank-Wolfe: Did not find optimal direction on '
            msg += 'iteration {}: {}'
            msg = msg.format(i, opt.status)
            logger.info(msg)

        # Calculate optimality gap
        xbar_opt = opt.variables()[0].value
        opt_bd = grad.T * (xbar_opt - x)

        msg = "i={:6}  obj={:10.7f}  opt_bd={:10.7f}  xdiff={:12.10f}"
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(msg.format(i, obj, opt_bd[0, 0], xdiff))
            logger.debug("")
        elif i % verbosechunk == 0:
            logger.info(msg.format(i, obj, opt_bd[0, 0], xdiff))

        xnew = (i * x + 2 * xbar_opt) / (i + 2)
        xdiff = np.linalg.norm(xnew - x)
        x = xnew

        if xdiff < tol:
            obj = objective(x)
            break
    else:
        msg = "Only converged to xdiff={:12.10f} after {} iterations. "
        msg += "Desired: {}"
        logger.warn(msg.format(xdiff, maxiters, tol))

    xopt = np.array(x)

    if clean:
        xopt[np.abs(xopt) < tol] = 0
        xopt /= xopt.sum()

    return xopt, obj
