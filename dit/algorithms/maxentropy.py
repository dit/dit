"""
Maximum entropy with marginal distribution constraints.

Note: We are actually doing the maximum entropy optimization. So we have not
built in the fact that the solution is an exponential family.

Also, this doesn't seem to work that well in practice. The optimization
simply fails to converge for many distributions. Xor() works great, but And()
fails to converge for 2-way marginals. Random distributions seem to work.
Jittering the distributions sometimes helps.

We might need to assume the exponential form and then fit the params to match
the marginals. Perhaps exact gradient and Hessians might help, or maybe even
some rescaling of the linear constraints.

"""

from __future__ import print_function
from __future__ import division

import itertools
import bisect

import numpy as np

import dit

from dit.abstractdist import AbstractDenseDistribution

__all__ = [
    'MarginalMaximumEntropy',
    'MomentMaximumEntropy',
    'marginal_maxent_dists',
    'moment_maxent_dists',
]


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
    b = np.asarray(b)

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


def marginal_constraints(pmf, n_variables, n_symbols, m):
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

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    return A, b


def marginal_constraint_rank(dist, m):
    """
    Returns the rank of the marginal constraint matrix.

    """
    dist = dit.expanded_samplespace(dist)
    dist.make_dense()
    n_variables = dist.outcome_length()
    n_symbols = len(dist.alphabet[0])
    pmf = dist.pmf

    A, b = marginal_constraints(pmf, n_variables, n_symbols, m)
    C, d, rank = as_full_rank(A, b)
    return rank


def moment(f, pmf, center=0, n=1):
    """
    Return the nth moment of `f` about `center`, distributed by `pmf`.

    Explicitly:   \sum_i (f(i) - center)**n p(i)

    Note, `pmf` is the joint distribution. So n=1 can be used even when
    calculating covariances such as <xx> and <xy>. The first would actually
    be a 2nd moment, while the second would be a mixed 1st moment.

    Parameters
    ----------
    f : array-like
        The numerical values assigned to each outcome of `p`.
    pmf : array-like
        The pmf for a distribution, linear-distributed values.
    center : float
        Calculate a centered moment.
    n : int
        The moment to calculate.

    """
    return ((f - center)**n * pmf).sum()


def moment_constraints(pmf, n_variables, m, symbol_map, with_replacement=True):
    """
    Returns `A` and `b` in `A x = b`, for an Ising-like system.

    If without replacement, we include only m-way first-moment constraints
    where each element is distinct. So <xx> and <yy> would not be included if
    n_variables=2 and m=2.

    The function we take means of is:  f(x) = \prod_i x_i

    The resulting matrix `A` is not guaranteed to have full rank.

    Parameters
    ----------
    pmf : array-like, shape ( n_symbols ** n_variables, )
        The probability mass function of the distribution. The pmf must have
        a Cartesian product sample space with the same sample space used for
        each random variable.
    n_variables : int
        The number of random variables.
    m : int | list
        The size of the moments to constrain. When `m=2`, pairwise means
        are constrained to equal the pairwise means in `pmf`. When `m=3`,
        three-way means are constrained to equal those in `pmf.
        If m is a list, then include all m-way moments in the list.
    symbol_map : array-like
        A mapping from the ith symbol to a real number that is to be used in
        the calculation of moments. For example, symbol_map=[-1, 1] corresponds
        to the typical Ising model.
    with_replacement : bool
        If `True`, variables are selected with replacement. The standard Ising
        does not select with replacement, and so terms like <xx>, <yy> do not
        appear for m=2. When `True`, we are constraining the entire moment
        matrix.

    Returns
    -------
    A : array-like, shape (p, q)
        The matrix defining the marginal equality constraints and also the
        normalization constraint. The number of rows is:
            p = C(n_variables, m) * n_symbols ** m + 1
        where C() is the choose formula. The number of columns is:
            q = n_symbols ** n_variables

    b : array-like, (p,)
        The RHS of the linear equality constraints.

    """
    n_symbols = len(symbol_map)
    d = AbstractDenseDistribution(n_variables, n_symbols)

    if len(pmf) != d.n_elements:
        msg = 'Length of `pmf` != n_symbols ** n_variables. Symbol map: {0!r}'
        raise ValueError(msg.format(symbol_map))

    # Begin with the normalization constraint.
    A = [ np.ones(d.n_elements) ]
    b = [ 1 ]


    try:
        m[0]
    except TypeError:
        mvals = [m]
    except IndexError:
        # m is empty list
        pass
    else:
        mvals = m

    if with_replacement:
        combinations = itertools.combinations_with_replacement
    else:
        combinations = itertools.combinations

    # Now add all the moment constraints.
    for m in mvals:
        if m < 1:
            continue

        outcomes = list(itertools.product(symbol_map, repeat=n_variables))
        outcomes = np.asarray(outcomes)
        for rvs in combinations(range(n_variables), m):
            # Make it a list for NumPy indexing
            rvs = list(rvs)
            f = np.array([outcome[rvs].prod() for outcome in outcomes])
            mean = moment(f, pmf, n=1)
            A.append(f)
            b.append(mean)

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    return A, b


def moment_constraint_rank(dist, m, symbol_map=None, cumulative=True, with_replacement=True):
    """
    Returns the rank of the moment constraint matrix.

    """
    if cumulative:
        mvals = range(m + 1)
    else:
        mvals = [m]

    dist = dit.expanded_samplespace(dist)
    dist.make_dense()
    n_variables = dist.outcome_length()
    n_symbols = len(dist.alphabet[0])
    pmf = dist.pmf

    # Symbol map
    if symbol_map is None:
        symbol_map = range(n_symbols)

    A, b = moment_constraints(pmf, n_variables, mvals, symbol_map,
                              with_replacement=with_replacement)
    C, d, rank = as_full_rank(A, b)

    return rank


def ising_constraint_rank(dist, m, symbol_map=None, cumulative=True):
    """
    Returns the rank of the Ising constraint matrix.

    """
    return moment_constraint_rank(dist, m, symbol_map, cumulative, with_replacement=False)


def negentropy(p):
    """
    Entropy which operates on vectors of length N.

    """
    return np.nansum(p * np.log2(p))


class MaximumEntropy(object):
    """
    Find maximum entropy distribution.

    """
    def __init__(self, dist, prng=None):
        """
        Initialize optimizer.

        Parameters
        ----------
        dist : distribution
            The distribution used to specify the marginal constraints.

        """
        if not isinstance(dist._sample_space, dit.samplespace.CartesianProduct):
            dist = dit.expanded_samplespace(dist, union=True)

        if not dist.is_dense():
            dist.make_dense()

        self.dist = dist
        self.pmf = dist.pmf
        self.n_variables = dist.outcome_length()
        self.n_symbols = len(dist.alphabet[0])
        self.n_elements = len(dist)

        if prng is None:
            prng = np.random.RandomState()
        self.prng = prng

        self.tol = {}

        self.initial = None
        self.init()


    def init(self):

        # Dimension of optimization variable
        self.n = len(self.pmf)

        # Number of nonlinear constraints
        self.m = 0

        self.build_gradient_hessian()
        self.build_nonnegativity_constraints()
        self.build_linear_equality_constraints()
        self.build_F()


    def build_gradient_hessian(self):

        import numdifftools

        self.func = negentropy
        self.gradient = numdifftools.Gradient(negentropy)
        self.hessian = numdifftools.Hessian(negentropy)


    def build_nonnegativity_constraints(self):
        from cvxopt import matrix

        # Dimension of optimization variable
        n = self.n

        # Nonnegativity constraint
        #
        # We have M = N = 0 (no 2nd order cones or positive semidefinite cones)
        # So, K = l where l is the dimension of the nonnegative orthant. Thus,
        # we have l = n.
        G = matrix( -1 * np.eye(n) )   # G should have shape: (K,n) = (n,n)
        h = matrix( np.zeros((n,1)) )  # h should have shape: (K,1) = (n,1)

        self.G = G
        self.h = h


    def build_linear_equality_constraints(self):
        from cvxopt import matrix

        # Normalization constraint only
        A = [ np.ones(self.n_elements) ]
        b = [ 1 ]

        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float)

        self.A = matrix(A)
        self.b = matrix(b)  # now a column vector


    def build_F(self):
        from cvxopt import matrix

        n = self.n
        m = self.m

        def F(x=None, z=None):
            # x has shape: (n,1)   and is the distribution
            # z has shape: (m+1,1) and is the Hessian of f_0

            if x is None and z is None:
                # Initial point is the original distribution.
                #d = self.pmf[np.arange(self.n_elements)]
                d = self.initial_dist()
                #d = np.ones(n) / n
                return (m, matrix(d))

            xarr = np.array(x)[:,0]

            # Verify that x is in domain.
            # Does G,h and A,b take care of this?
            #
            if np.any(xarr > 1) or np.any(xarr < 0):
                return None
            if not np.allclose(np.sum(xarr), 1, **self.tol):
                return None

            # Using automatic differentiators for now.
            f = self.func(xarr)
            Df = self.gradient(xarr)
            Df = matrix(Df.reshape((1, n)))

            if z is None:
                return (f, Df)
            H = self.hessian(xarr)
            H = matrix(H)

            return (f, Df, z[0] * H)

        self.F = F


    def initial_dist(self):
        return self.prng.dirichlet([1] * self.n)


    def optimize(self, show_progress=False):
        from cvxopt.solvers import cp, options

        old = options.get('show_progress', None)
        out = None

        try:
            options['show_progress'] = show_progress
            with np.errstate(divide='ignore', invalid='ignore'):
                result = cp(F=self.F,
                            G=self.G,
                            h=self.h,
                            dims={'l':self.n, 'q':[], 's':[]},
                            A=self.A,
                            b=self.b)
        except:
            raise
        else:
            self.result = result
            out = np.asarray(result['x'])
        finally:
            if old is None:
                del options['show_progress']
            else:
                options['show_progress'] = old

        return out


class MarginalMaximumEntropy(MaximumEntropy):
    """
    Find maximum entropy distribution subject to k-way marginal constraints.

    k=0 should reproduce the behavior of MaximumEntropy.

    """
    def __init__(self, dist, k, prng=None):
        """
        Initialize optimizer.

        Parameters
        ----------
        dist : distribution
            The distribution used to specify the marginal constraints.
        k : int
            The number of variables in the constrained marginals.

        """
        self.k = k
        super(MarginalMaximumEntropy, self).__init__(dist, prng=prng)


    def build_linear_equality_constraints(self):
        from cvxopt import matrix

        # Dimension of optimization variable
        n = self.n

        args = (self.pmf, self.n_variables, self.n_symbols, self.k)
        A, b = marginal_constraints(*args)
        A, b, rank = as_full_rank(A, b)
        if rank > n:
            raise ValueError('More independent constraints than parameters.')

        A = matrix(A)
        b = matrix(b)  # now a column vector

        self.A = A
        self.b = b

class MomentMaximumEntropy(MaximumEntropy):
    """
    Find maximum entropy distribution subject to k-way marginal constraints.

    k=0 should reproduce the behavior of MaximumEntropy.

    """
    def __init__(self, dist, k, symbol_map, cumulative=True, with_replacement=True, prng=None):
        """
        Initialize optimizer.

        Parameters
        ----------
        dist : distribution
            The distribution used to specify the marginal constraints.
        k : int
            The number of variables in the constrained marginals.
        symbol_map : list
            The mapping from states to real numbers. This is used while taking
            moments.
        cumulative : bool
            If `True`, include all moments less than or equal to `k`.
        with_replacement : bool
            If `True`, then variables are selected for moments with replacement.
            The standard Ising model selects without replacement.
        prng : RandomState
            A pseudorandom number generator.

        """
        self.k = k
        self.symbol_map = symbol_map
        self.cumulative = cumulative
        self.with_replacement = with_replacement
        super(MomentMaximumEntropy, self).__init__(dist, prng=prng)


    def build_linear_equality_constraints(self):
        from cvxopt import matrix

        # Dimension of optimization variable
        n = self.n

        if self.cumulative:
            k = range(self.k + 1)
        else:
            k = [self.k]

        args = (self.pmf, self.n_variables, k, self.symbol_map)
        kwargs = {'with_replacement': self.with_replacement}
        A, b = moment_constraints(*args, **kwargs)
        A, b, rank = as_full_rank(A, b)
        if rank > n:
            raise ValueError('More independent constraints than parameters.')

        A = matrix(A)
        b = matrix(b)  # now a column vector

        self.A = A
        self.b = b


def marginal_maxent_dists(dist, k_max=None, jitter=True, show_progress=True):
    """
    Return the marginal-constrained maximum entropy distributions.

    Parameters
    ----------
    dist : distribution
        The distribution used to constrain the maxent distributions.
    k_max : int
        The maximum order to calculate.
    jitter : bool | float
        When `True` or a float, we perturb the distribution slightly before
        proceeding. This can sometimes help with convergence.
    show-progress : bool
        If `True`, show convergence progress to stdout.

    """
    dist = dit.expanded_samplespace(dist, union=True)
    dist.make_dense()

    if jitter:
        # This is sometimes necessary. If your distribution does not have
        # full support than convergence can be difficult to come by.
        dist.pmf = dit.math.pmfops.jittered(dist.pmf)

    pmf = dist.pmf
    n_variables = dist.outcome_length()
    n_symbols = len(dist.alphabet[0])
    symbols = dist.alphabet[0]

    if k_max is None:
        k_max = n_variables

    outcomes = list(dist._product(symbols, repeat=n_variables))

    dists = []
    for k in range(k_max + 1):
        print()
        print("Constraining maxent dist to match {0}-way marginals.".format(k))
        print()
        opt = MarginalMaximumEntropy(dist, k)
        pmf_opt = opt.optimize(show_progress=show_progress)
        d = dit.Distribution(outcomes, pmf_opt)
        dists.append(d)

    return dists


def moment_maxent_dists(dist, symbol_map, k_max=None, jitter=True,
                        with_replacement=True, show_progress=True):
    """
    Return the marginal-constrained maximum entropy distributions.

    Parameters
    ----------
    dist : distribution
        The distribution used to constrain the maxent distributions.
    symbol_map : iterable
        A list whose elements are the real values that each state is assigned
        while calculating moments. Typical values are [-1, 1] or [0, 1].
    k_max : int
        The maximum order to calculate.
    jitter : bool | float
        When `True` or a float, we perturb the distribution slightly before
        proceeding. This can sometimes help with convergence.
    with_replacement : bool
        If `True`, then variables are selected for moments with replacement.
        The standard Ising model selects without replacement.
    show-progress : bool
        If `True`, show convergence progress to stdout.

    """
    dist = dit.expanded_samplespace(dist, union=True)
    dist.make_dense()

    if jitter:
        # This is sometimes necessary. If your distribution does not have
        # full support than convergence can be difficult to come by.
        dist.pmf = dit.math.pmfops.jittered(dist.pmf)

    pmf = dist.pmf
    n_variables = dist.outcome_length()
    n_symbols = len(dist.alphabet[0])
    symbols = dist.alphabet[0]

    if k_max is None:
        k_max = n_variables

    outcomes = list(dist._product(symbols, repeat=n_variables))

    if with_replacement:
        text = 'with replacement'
    else:
        text = 'without replacement'

    dists = []
    for k in range(k_max + 1):
        msg = "Constraining maxent dist to match {0}-way moments, {1}."
        print()
        print(msg.format(k, text))
        print()
        opt = MomentMaximumEntropy(dist, k, symbol_map, with_replacement=with_replacement)
        pmf_opt = opt.optimize(show_progress=show_progress)
        d = dit.Distribution(outcomes, pmf_opt)
        dists.append(d)

    return dists


