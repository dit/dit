"""
Code for maximizing a convex function over a polytope, as defined
by a set of linear equalities and inequalities.

This uses the fact that the maximum of a convex function over a 
polytope will be achieved at one of the extreme points of the polytope.

Thus, the maximization is done by taking a system of linear inequalities, 
using the Parma Polyhedral Library (pplpy) to create a list of extreme
points, and then evaluating the objective function on each point.
"""

import numpy as np

__all__ = (
    'maximize_convex_function',
    'polytope_extremepoint_iterator',
)


def polytope_extremepoint_iterator(A_ineq, b_ineq, A_eq=None, b_eq=None):
    """
    Iterator over extreme points of polytope defined by linear equalities
    and inequalities, A_ineq x <= b_ineq, A_eq x = b_eq. 

    This uses the Parma Polyhedral Library (PPL). Because PPL expects all
    data to be rational, we enforce that inequalities and equalities are 
    specified by integer-valued matrices.

    Parameters
    ----------
    A_ineq : np.array
        Inequalities matrix. Data type should be int, 
        shape should be (num_inequalities x num_variables)
    b_ineq : np.array
        Inequalities values. Data type should be int, 
        shape should be (num_inequalities)
    A_eq : np.array
        Equalities matrix. Data type should be int, 
        shape should be (num_equalities x num_variables)
    b_eq : np.array
        Equalities values. Data type should be int, 
        shape should be (num_equalities)
    """

    try:
        import ppl
    except ImportError:
        raise Exception("""
Convex maximization code requires the Parma Polyhedra Library (PPL) to 
be installed. Normally, this can be done with
   pip install pplpy cysignals gmpy2 
Please see https://gitlab.com/videlec/pplpy for more information.
""")

    def get_num_cons(A, b):
        # Check data for validity and return number of constraints 
        if A is None:
            assert(b is None or len(b) == 0)
            num_cons = 0
        else:
            assert(isinstance(A,np.ndarray))
            assert(isinstance(b,np.ndarray))
            assert(np.issubdtype(A.dtype, np.integer))
            assert(np.issubdtype(b.dtype, np.integer))
            num_cons = A.shape[0]
            assert(num_cons == len(b))
        return num_cons

    num_ineq_cons = get_num_cons(A_ineq, b_ineq)
    num_eq_cons   = get_num_cons(A_eq  , b_eq)

    if num_eq_cons == 0 and num_ineq_cons == 0:
        raise Exception("Must specify at least one inequality or equality constrants")

    if num_eq_cons > 0 and num_ineq_cons > 0:
        assert(A_eq.shape[1] == A_ineq.shape[1])

    num_vars = (A_eq if num_eq_cons > 0 else A_ineq).shape[1]

    ppl_vars = [ppl.Variable(i) for i in range(num_vars)]

    cs = ppl.Constraint_System()
    for rowndx in range(num_ineq_cons):
        if np.all(A_ineq[rowndx] == 0):
            if b_ineq[rowndx]<0:
                raise Exception('Inequality constraint %d involves no variables and is unsatisfiable' % rowndx)
            else:
                continue # trivial constraint
                
        lhs = sum([coef*ppl_vars[i]
                   for i, coef in enumerate(A_ineq[rowndx]) if coef != 0])
        cs.insert(lhs <= b_ineq[rowndx] )

    for rowndx in range(num_eq_cons):
        if np.all(A_eq[rowndx] == 0):
            if b_eq[rowndx]!=0:
                raise Exception('Equality constraint %d involves no variables and is unsatisfiable' % rowndx)
            else:
                continue # trivial constraint

        lhs = sum([coef*ppl_vars[i]
                   for i, coef in enumerate(A_eq[rowndx]) if coef != 0])
        cs.insert(lhs == b_eq[rowndx] )

    # convert linear inequalities into a list of extreme points
    poly_from_constraints = ppl.C_Polyhedron(cs)
    all_generators = poly_from_constraints.minimized_generators()

    for p in all_generators:
        if not p.is_point():
            raise Exception('Returned solution not a point: %s. '%p + 
                'Typically this means that linear constraints specify a cone, not a polytope')
            
        # Convert a solution vector in ppl format to a numpy array
        x = np.fromiter(p.coefficients(), dtype='double')
        x = x/float(p.divisor())
        yield x


def maximize_convex_function(f, A_ineq, b_ineq, A_eq=None, b_eq=None):
    """
    Maximize a convex function over a polytope.

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

    extreme_points = polytope_extremepoint_iterator(
        A_ineq=A_ineq, b_ineq=b_ineq, A_eq=A_eq, b_eq=b_eq)

    for x in extreme_points:
        val = f(x)
        if val > best_val:
            best_x, best_val = x, val
            
    if best_x is None:
        raise Exception('No extreme points found!')

    return best_x, best_val

