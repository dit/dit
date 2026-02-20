"""
Code for maximizing a convex function over a polytope, as defined
by a set of linear equalities and inequalities.
"""

import numpy as np
import scipy

from .optutil import as_full_rank

__all__ = ("maximize_convex_function",)


def maximize_convex_function(f, A_ineq, b_ineq, A_eq=None, b_eq=None):
    """
    Maximize a convex function over a polytope. This function uses the fact that
    the maximum of a convex function over a polytope will be achieved at one of
    the extreme points of the polytope.

    The maximization is done by taking a system of linear inequalities, using the
    pypoman library to create a list of extreme points, and then evaluating the
    objective function on each point.

        Parameters
        ----------
        f : function
            Objective function to maximize
        A_ineq : matrix
            Specifies inequalities matrix, should be num_inequalities x num_variables
        b_ineq : array
            Specifies inequalities vector, should be num_inequalities long
        A_eq : matrix
            Specifies equalities matrix, should be num_equalities x num_variables
        b_eq : array
            Specifies equalities vector, should be num_equalities long

        Returns tuple optimal_extreme_point, maximum_function_value

    """

    best_x, best_val = None, -np.inf

    A_ineq = A_ineq.astype("float")
    b_ineq = b_ineq.astype("float")

    A_ineq, b_ineq, _ = as_full_rank(A_ineq, b_ineq)

    if A_eq is not None:
        # pypoman doesn't support equality constraints. We remove equality
        # constraints by doing a coordinate transformation.

        A_eq = A_eq.astype("float")
        b_eq = b_eq.astype("float")

        A_eq, b_eq, _ = as_full_rank(A_eq, b_eq)

        # Get one solution that satisfies A x0 = b
        x0 = np.linalg.lstsq(A_eq, b_eq, rcond=None)[0]
        assert np.abs(A_eq.dot(x0) - b_eq).max() < 1e-5

        # Get projector onto null space of A, it satisfies AZ=0 and Z^T Z=I
        Z = scipy.linalg.null_space(A_eq)
        # Now every solution can be written as x = x0 + Zq, since A x = A x0 = b

        # Inequalities get transformed as
        #   A'x <= b'  --->  A'(x0 + Zq) <= b --> (A'Z)q \le b - A'x0

        b_ineq = b_ineq - A_ineq.dot(x0)
        A_ineq = A_ineq.dot(Z)

        transform = lambda q: Z.dot(q) + x0

    else:
        transform = lambda x: x

    import pypoman

    extreme_points = pypoman.compute_polytope_vertices(A_ineq, b_ineq)

    for v in extreme_points:
        x = transform(v)
        val = f(x)
        if val > best_val:
            best_x, best_val = x, val

    if best_x is None:
        raise Exception("No extreme points found!")

    return best_x, best_val
