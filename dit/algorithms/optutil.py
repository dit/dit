"""
Various utilities that can be helpful for optimization problems.

"""
from __future__ import division, print_function

from collections import defaultdict
import itertools

import numpy as np
import dit

from .frankwolfe import frank_wolfe


def as_full_rank(A, b):
    """
    From a linear system Ax = b, return Bx = c such that B has full rank.

    In CVXOPT, linear constraints are denoted as: Ax = b. A has shape (p, n)
    and must have full rank. x has shape (n, 1), and so b has shape (p, 1).
    Let's assume that we have:

        rank(A) = q <= n

    This is a typical situation if you are doing optimization, where you have
    an under-determined system and are using some criterion for selecting out
    a particular solution. Now, it may happen that q < p, which means that some
    of your constraint equations are not independent. Since CVXOPT requires
    that A have full rank, we must isolate an equivalent system Bx = c which
    does have full rank. We use SVD for this. So A = U \Sigma V^*, where
    U is (p, p), \Sigma is (p, n) and V^* is (n, n). Then:

        \Sigma V^* x = U^{-1} b

    We take B = \Sigma V^* and c = U^T b, where we use U^T instead of U^{-1}
    for computational efficiency (and since U is orthogonal). But note, we
    take only the cols of U (which are rows in U^{-1}) and rows of \Sigma that
    have nonzero singular values.

    Parameters
    ----------
    A : array-like, shape (p, n)
        The LHS for the linear constraints.
    b : array-like, shape (p,) or (p, 1)
        The RHS for the linear constraints.

    Returns
    -------
    B : array-like, shape (q, n)
        The LHS for the linear constraints.
    c : array-like, shape (q,) or (q, 1)
        The RHS for the linear constraints.
    rank : int
        The rank of B.

    """
    try:
        from scipy.linalg import svd
    except ImportError:
        from numpy.linalg import svd

    import scipy.linalg as splinalg

    A = np.atleast_2d(A)
    b = np.asarray(b)

    U, S, Vh = svd(A)
    Smat = splinalg.diagsvd(S, A.shape[0], A.shape[1])

    # See np.linalg.matrix_rank
    tol = S.max() * max(A.shape) * np.finfo(S.dtype).eps
    rank = np.sum(S > tol)

    B = np.dot(Smat, Vh)[:rank]
    c = np.dot(U.transpose(), b)[:rank]

    return B, c, rank


class CVXOPT_Template(object):
    """
    Template for convex minimization on probability distributions.

    """
    def __init__(self, dist, tol=None, prng=None):
        """
        Initialize optimizer.

        Parameters
        ----------
        dist : distribution
            The distribution that is used during optimization.
        tol : float | None
            The desired convergence tolerance.
        prng : RandomState
            A NumPy-compatible pseudorandom number generator.

        """
        dist = prepare_dist(dist)
        self.dist = dist
        self.pmf = dist.pmf
        self.n_variables = dist.outcome_length()

        self.n_symbols = len(dist.alphabet[0])
        self.n_elements = len(self.pmf)

        if prng is None:
            prng = np.random.RandomState()
        self.prng = prng

        if tol is None:
            tol = {}
        self.tol = tol

        self.init()


    def init(self):

        # Dimension of optimization variable
        self.n = len(self.pmf)

        # Number of nonlinear constraints
        self.m = 0

        self.prep()
        self.build_function()
        self.build_gradient_hessian()
        self.build_linear_inequality_constraints()
        self.build_linear_equality_constraints()
        self.build_F()

    def prep(self):
        pass

    def build_function(self):
        self.func = lambda x: x.sum()


    def build_gradient_hessian(self):

        import numdifftools

        self.gradient = numdifftools.Gradient(self.func)
        self.hessian = numdifftools.Hessian(self.func)


    def build_linear_inequality_constraints(self):
        from cvxopt import matrix

        # Dimension of optimization variable
        n = self.n

        # Nonnegativity constraint
        #
        # We have M = N = 0 (no 2nd order cones or positive semidefinite cones)
        # So, K = l where l is the dimension of the nonnegative orthant. Thus,
        # we have l = n.
        G = matrix(-1 * np.eye(n))   # G should have shape: (K,n) = (n,n)
        h = matrix(np.zeros((n,1)))  # h should have shape: (K,1) = (n,1)

        self.G = G
        self.h = h


    def build_linear_equality_constraints(self):
        from cvxopt import matrix

        # Normalization constraint only
        A = [np.ones(self.n_elements)]
        b = [1]

        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float)

        self.A = matrix(A)
        self.b = matrix(b)  # now a column vector


    def initial_dist(self):
        return self.prng.dirichlet([1] * self.n)


    def build_F(self):
        from cvxopt import matrix

        n = self.n
        m = self.m

        def F(x=None, z=None):
            # x has shape: (n,1)   and is the distribution
            # z has shape: (m+1,1) and is the Hessian of f_0

            if x is None and z is None:
                d = self.initial_dist()
                return (m, matrix(d))

            xarr = np.array(x)[:, 0]

            # Verify that x is in domain.
            # Does G,h and A,b take care of this?
            #
            if np.any(xarr > 1) or np.any(xarr < 0):
                return None
            if not np.allclose(np.sum(xarr), 1, **self.tol):
                return None

            # Derivatives
            f = self.func(xarr)
            Df = self.gradient(xarr)
            Df = matrix(Df.reshape((1, n)))

            if z is None:
                return (f, Df)
            else:
                # Hessian
                H = self.hessian(xarr)
                H = matrix(H)
                return (f, Df, z[0] * H)

        self.F = F


    def optimize(self, **kwargs):
        """
        Options:

            show_progress=False,
            maxiters=100,
            abstol=1e-7,
            reltol=1e-6,
            feastol=1e-7,
            refinement=0 if m=0 else 1

        """
        from cvxopt.solvers import cp, options

        old_options = options.copy()
        out = None

        try:
            options.clear()
            options.update(kwargs)
            with np.errstate(divide='ignore', invalid='ignore'):
                result = cp(F=self.F,
                            G=self.G,
                            h=self.h,
                            dims={'l':self.G.size[0], 'q':[], 's':[]},
                            A=self.A,
                            b=self.b)
        except:
            raise
        else:
            self.result = result
            out = np.asarray(result['x'])
        finally:
            options.clear()
            options.update(old_options)

        return out


class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)


def prepare_dist(dist):
    if not isinstance(dist._sample_space, dit.samplespace.CartesianProduct):
        dist = dit.expanded_samplespace(dist, union=True)

    if not dist.is_dense():
        if len(dist._sample_space) > 1e4:
            import warnings
            msg = "Sample space has more than 10k elements."
            msg += " This could be slow."
            warnings.warn(msg)
        dist.make_dense()

    # We also need linear probabilities.
    dist.set_base('linear')

    return dist


def op_runner(objective, constraints, **kwargs):
    """
    Minimize the objective specified by the constraints.

    This safely let's you pass options to the solver and restores their values
    once the optimization process has completed.

    The objective must be linear in the variables.
    This uses cvxopt.modeling.

    """
    from cvxopt.solvers import options
    from cvxopt.modeling import variable, op

    old_options = options.copy()

    opt = op(objective, constraints)

    try:
        options.clear()
        options.update(kwargs)
        # Ignore 0 log 0 warnings.
        with np.errstate(divide='ignore', invalid='ignore'):
            opt.solve()
    except:
        raise
    finally:
        options.clear()
        options.update(old_options)

    return opt
