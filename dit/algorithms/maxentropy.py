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


TODO:

This code for moment-based maximum entropy needs to be updated so that it can
handle any Cartesian product sample space, rather than just homogeneous ones.

"""

from __future__ import division, print_function

import itertools

import numpy as np

import dit

from dit.abstractdist import AbstractDenseDistribution, get_abstract_dist

from .optutil import as_full_rank, CVXOPT_Template

__all__ = [
    'MarginalMaximumEntropy',
    'MomentMaximumEntropy',
    # Use version provided by maxentropyfw.py
    #'marginal_maxent_dists',
    'moment_maxent_dists',
]


def marginal_constraints(dist, m, with_normalization=True):
    """
    Returns `A` and `b` in `A x = b`, for a system of marginal constraints.

    The resulting matrix `A` is not guaranteed to have full rank.

    Parameters
    ----------
    dist : distribution
        The distribution from which the marginal constraints are constructed.

    m : int
        The size of the marginals to constrain. When `m=2`, pairwise marginals
        are constrained to equal the pairwise marginals in `pmf`. When `m=3`,
        three-way marginals are constrained to equal those in `pmf.

    with_normalization : bool
        If true, include a constraint for normalization.

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
    assert dist.is_dense()
    assert dist.get_base() == 'linear'

    pmf = dist.pmf

    d = get_abstract_dist(dist)

    if m > d.n_variables:
        msg = "Cannot constrain {0}-way marginals"
        msg += " with only {1} random variables."
        msg = msg.format(m, d.n_variables)
        raise ValueError(msg)

    A = []
    b = []

    # Begin with the normalization constraint.
    if with_normalization:
        A.append( np.ones(d.n_elements) )
        b.append( 1 )

    # Now add all the marginal constraints.
    if m > 0:
        cache = {}
        for rvs in itertools.combinations(range(d.n_variables), m):
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

    A, b = marginal_constraints(dist, m)
    _, _, rank = as_full_rank(A, b)
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
    _, _, rank = as_full_rank(A, b)

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


class MaximumEntropy(CVXOPT_Template):
    """
    Find maximum entropy distribution.

    """
    def build_function(self):
        self.func = negentropy


class MarginalMaximumEntropy(MaximumEntropy):
    """
    Find maximum entropy distribution subject to k-way marginal constraints.

    k=0 should reproduce the behavior of MaximumEntropy.

    """
    def __init__(self, dist, k, tol=None, prng=None):
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
        super(MarginalMaximumEntropy, self).__init__(dist, tol=tol, prng=prng)


    def build_linear_equality_constraints(self):
        from cvxopt import matrix

        # Dimension of optimization variable
        n = self.n

        A, b = marginal_constraints(self.dist, self.k)
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
    def __init__(self, dist, k, symbol_map, cumulative=True, with_replacement=True, tol=None, prng=None):
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
        tol : float | None
            The desired convergence tolerance.
        prng : RandomState
            A pseudorandom number generator.

        """
        self.k = k
        self.symbol_map = symbol_map
        self.cumulative = cumulative
        self.with_replacement = with_replacement
        super(MomentMaximumEntropy, self).__init__(dist, tol=tol, prng=prng)


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

    n_variables = dist.outcome_length()
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

    n_variables = dist.outcome_length()
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
