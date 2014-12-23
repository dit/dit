"""
Maximum entropy with marginal distribution constraints.

"""

from __future__ import print_function
from __future__ import division

import itertools
import bisect

import numpy as np
import dit

from dit.abstractdist import AbstractDenseDistribution

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
    take only the cols of U and rows of \Sigma that have nonzero singular
    values.

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
        from scipy import linalg
    except ImportError:
        from numpy import linalg

    A = np.atleast_2d(A)

    U, S, Vh = linalg.svd(A)

    tol = S.max() * max(A.shape) * np.finfo(S.dtype).eps
    rank = np.sum(S > tol)

    Ut = U[:, :rank].transpose()

    # See scipy.linalg.diagsvd for details.
    # Note that we only take the first 'rank' rows/cols of S.
    part = np.diag(S[:rank])
    typ = part.dtype.char
    D = np.r_['-1', part, np.zeros((rank, A.shape[1] - rank), typ)]

    B = np.dot(D, Vh)
    c = np.dot(Ut, b)

    return B, c, rank


class PhantomArray(object):
    """
    A simple wrapper around a sparse pmf specified by a lookup table.

    The wrapper provides NumPy

    """
    def __init__(self, lookup, n_variables, n_symbols):
        self.lookup = lookup
        self.n_variables = n_variables
        self.n_symbols = n_symbols
        self.n_elements = n_symbols ** n_variables

    def __getitem__(self, idx):
        try:
            idx[0]
        except:
            return self.lookup.get(idx, 0)
        else:
            # Iterable. Return a NumPy array of the elements.
            return np.array([self.lookup.get(i, 0) for i in idx])

    def __len__(self):
        return self.n_elements

def cartesian_product_view(dist):
    """
    Return a dense Cartesian product view of the `dist`.

    In a Cartesian product view, we union the sample space of each random
    variable and then use it in a Cartesian product that defines the sample
    space for the entire distribution.

    Parameters
    ----------
    dist : distribution
        The distribution to make dense.

    Returns
    -------
    pa : PhantomArray
        The dense, Cartesian product representation of the pmf of `dist`.
    n_variables : int
        The number of random variables in the distribution.
    n_symbols : int
        The number of symbols in the sample space for a single random variable.

    Examples
    --------
    >>> import dit
    >>> d = dit.Distribution(['00', '10'], [.5, .5])
    >>> pa, n, k = cartesian_product_view(d)
    >>> pa[[0,1,2,3]]
    array([ 0.5, 0., 0.5, 0. ])

    """
    symbols = list(sorted(set.union(*map(set, dist.alphabet))))
    n_variables = dist.outcome_length()
    n_symbols = len(symbols)

    lookup = {}
    for outcome, p in dist.zipped():
        index = 0
        for i, symbol in enumerate(outcome):
            idx = bisect.bisect_left(symbols, symbol)
            index += idx * n_symbols ** (n_variables - 1 - i)
        lookup[index] = p

    pmf = PhantomArray(lookup, n_variables, n_symbols)
    return pmf, n_variables, n_symbols


def linear_constraints(pmf, n_variables, n_symbols, m):
    """
    Returns `A` and `b` in `A x = b`, for a system of marginal constraints.

    The resulting matrix `A` is not guaranteed to have full rank.

    Parameters
    ----------
    pmf : array-like, shape ( n_symbols ** n_variables, )
        The probability mass function of the distribution. The pmf must have
        a Cartesian product sample space with the same sample space used for
        each random variable.
    n_variables : int
        The number of random variables.
    n_symbols : int
        The number of symbols that each random variable can be.
    m : int
        The size of the marginals to constrain. When `m=2`, pairwise marginals
        are constrained to equal the pairwise marginals in `pmf`. When `m=3`,
        three-way marginals are constrained to equal those in `pmf.

    Returns
    -------
    A : array-like, shape (p, q)
        The matrix defining the marginal equality constraints and also the
        normalization constraint. The number of rows is:
            p = C(n_variables, m) * n_symbols ** m + 1
        where C() is the choose formula.. The number of columns is:
            q = n_symbols ** n_variables

    b : array-like, (p,)
        The RHS of the linear equality constraints.

    """
    if m > n_variables:
        msg = "Cannot constrain {0}-way marginals"
        msg += " with only {1} random variables."
        msg = msg.format(m, n_variables)
        raise ValueError(msg)

    d = AbstractDenseDistribution(n_variables, n_symbols)

    # Begin with the normalization constraint.
    A = [ np.ones(d.n_elements) ]
    b = [ 1 ]

    # Now add all the marginal constraints.
    if m > 0:
        cache = {}
        for rvs in itertools.combinations(range(n_variables), m):
            for idx in d.parameter_array(rvs, cache=cache):
                bvec = np.zeros(d.n_elements)
                bvec[idx] = 1
                A.append(bvec)
                b.append(pmf[idx].sum())

    A = np.asarray(A)
    b = np.asarray(b)

    return A, b


def constraint_rank(dist, m):
    """
    Returns the rank of the linear constraint matrix.

    """
    pmf, n_variables, n_symbols = cartesian_product_view(dist)
    A, b = linear_constraints(pmf, n_variables, n_symbols, m)
    C, d, rank = as_full_rank(A, b)
    return rank
