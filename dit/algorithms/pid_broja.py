 # -*- coding: utf-8 -*-

from __future__ import division

from collections import defaultdict
import itertools

import numpy as np
import dit

from dit.abstractdist import AbstractDenseDistribution, get_abstract_dist
from dit.algorithms.optutil import CVXOPT_Template, as_full_rank, Bunch

def prepare_dist(dist, sources, target, rv_mode=None):
    """
    Prepares a ``dit`` distribution for the optimization process.

    The requires that we:

    1. Remove any random variables not part of the sources or the target.
    2. Make sure the sample space is a Cartesian product sample space.
    3. Make the pmf representation dense.

    Parameters
    ----------
    dist : dit distribution
        The original distribution from which the optimization problem is
        extracted.
    sources : list of lists
        The sources random variables. Each random variable specifies a list of
        random variables in `dist` that define a source.
    target : list
        The random variables in `dist` that define the target.
    rv_mode : str, None
        Specifies how to interpret the elements of each source and the target.
        Valid options are: {'indices', 'names'}. If equal to 'indices', then
        the elements of each source and the target are interpreted as random
        variable indices. If equal to 'names', the the elements are interpreted
        as random variable names. If `None`, then the value of `dist._rv_mode`
        is consulted.

    Returns
    -------
    d : dit distribution
        A reduced distribution where the first n random variables specify
        the n sources and the last random variable specifies the target.

    """
    if not dist.is_joint():
        msg = "The information measure requires a joint distribution."
        raise ditException(msg)

    # Simplify the distribution by removing irrelevant rvs and giving each
    # source their own singleton index. Similarly for the target rvs.
    rvs = sources + [target]
    d = dist.coalesce(rvs, rv_mode=rv_mode)

    # Fix sample space and make dense.
    d = dit.algorithms.optutil.prepare_dist(d)

    return d

def marginal_constraints(dist, k, normalization=True):
    """
    Builds the k-way marginal constraints.

    This assumes the target random variable is the last one.
    All others are source random variables.
    Each constraint involves (k-1) sources and the target random variable.

    Explicitly, we demand that:

        p( source_{k-1}, target ) = q( source_{k-1}, target)

    For each (k-1)-combination of the sources, the distribution involving those
    sources and the target random variable must match the true distribution p
    for all possible candidate distributions.

    For unique information, k=2 is used, but we allow more general constraints.

    """
    assert dist.is_dense()
    assert dist.get_base() == 'linear'

    pmf = dist.pmf

    d = get_abstract_dist(dist)
    n_variables = d.n_variables
    n_elements = d.n_elements

    #
    # Linear equality constraints (these are not independent constraints)
    #
    A = []
    b = []

    # Normalization: \sum_i q_i = 1
    if normalization:
        A.append( np.ones(n_elements) )
        b.append( 1 )

    # Random variables
    rvs = range(n_variables)
    target_rvs = tuple(rvs[-1:])
    source_rvs = tuple(rvs[:-1])

    marginal_size = k
    submarginal_size = marginal_size - 1
    assert submarginal_size >= 1

    # p( source_{k-1}, target ) = q( source_{k-1}, target )
    cache = {}
    for subrvs in itertools.combinations(source_rvs, submarginal_size):
        rvs = subrvs + target_rvs
        marray = d.parameter_array(rvs, cache=cache)
        for idx in marray:
            # Convert the sparse nonzero elements to a dense boolean array
            bvec = np.zeros(n_elements)
            bvec[idx] = 1
            A.append(bvec)
            b.append(pmf[idx].sum())

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    return A, b


def extra_constraints(dist, k):
    """
    Builds a list of additional constraints using the specific properties
    of the random variables. The goal is to determine if there are any
    (input, output) pairs such that p(input, output) = q(input, output)
    for all q in the feasible set.

    This can happen in a few ways:

    1. If any marginal probability is zero, then all joint probabilities which
       contributed to that marginal probability must also be zero.

    2. Now suppose we to find when p({x_i}, y) = q({x_i}, y).
       For every (k-1)-subset, g_k(x), we have:

          p(g_k(x), y) = q(g_k(x), y)

       So we have:

          p(y) = q(y)
          p(g_k(x) | y) = q(g_k(x) | y)

       Now suppose that for every i, we had:

          p(x_i | y) = \delta(x_i, f_i(y))

       Then,

          q(x_i | y) = \delta(x_i, f_i(y))

       as well since we match all k-way marginals with k >= 2. E.g., For k=4,
       we have: p( x_1, x_2, x_3, y ) = q( x_1, x_2, x_3, y ) which implies
       that p( x_1 | y ) = q( x_1 | y). So generally, we have:

          p({x_i} | y) = q({x_i} | y)

       for each i. Then, since p(y) = q(y), we also have:

          p({x_i}, y) = q({x_i}, y).

       Note, we do not require that y be some function of the x_i.

    """
    assert dist.is_dense()
    assert dist.get_base() == 'linear'

    d = get_abstract_dist(dist)
    n_variables = d.n_variables
    n_elements = d.n_elements

    rvs = range(n_variables)
    target_rvs = tuple(rvs[-1:])
    source_rvs = tuple(rvs[:-1])

    marginal_size = k
    submarginal_size = marginal_size - 1
    assert submarginal_size >= 1

    ### Find values that are fixed at zero

    # This finds marginal probabilities that are zero and infers that the
    # joint probabilities that contributed to it are also zero.
    zero_elements = np.zeros(n_elements, dtype=int)
    cache = {}
    pmf = dist.pmf
    for subrvs in itertools.combinations(source_rvs, submarginal_size):
        rvs = subrvs + target_rvs
        marray = d.parameter_array(rvs, cache=cache)
        for idx in marray:
            # Convert the sparse nonzero elements to a dense boolean array
            bvec = np.zeros(n_elements, dtype=int)
            bvec[idx] = 1
            p = pmf[idx].sum()
            if np.isclose(p, 0):
                zero_elements += bvec

    ### Find values that match the original joint.

    # First identify each p(input_i | output) = 1
    determined = defaultdict(lambda : [0] * len(source_rvs))
    for i, source_rv in enumerate(source_rvs):
        md, cdists = dist.condition_on(target_rvs, rvs=[source_rv])
        for target_outcome, cdist in zip(md.outcomes, cdists):
            # cdist is dense
            if np.isclose(cdist.pmf, 1).sum() == 1:
                # Then p(source_rv | target_rvs) = 1
                determined[target_outcome][i] = 1

    is_determined = {}
    for outcome, det_vector in determined.items():
        if all(det_vector):
            is_determined[outcome] = True
        else:
            is_determined[outcome] = False

    # Need to find joint indexes j for which all p(a_i | b) = 1.
    # For these j, p_j = q_j.
    determined = {}
    for i, (outcome, p) in enumerate(dist.zipped()):
        if not (p > 0):
            continue
        target = dist._outcome_ctor([outcome[target] for target in target_rvs])
        if is_determined.get(target, False):
            determined[i] = p

    ###

    fixed = {}
    zeros = []
    nonzeros = []
    for i, is_zero in enumerate(zero_elements):
        if is_zero:
            fixed[i] = 0
            zeros.append(i)
        else:
            nonzeros.append(i)
    for i, p in determined.items():
        fixed[i] = p

    fixed = sorted(fixed.items())
    fixed_indexes, fixed_values = zip(*fixed)
    free = [i for i in range(n_elements) if i not in set(fixed_indexes)]
    fixed_nonzeros = [i for i in fixed_indexes if i in set(nonzeros)]

    # all indexes   = free + fixed
    # all indexes   = nonzeros + zeros
    # fixed         = fixed_nonzeros + zeros
    # nonzero >= free
    variables = Bunch(
        free=free,
        fixed=fixed_indexes,
        fixed_values=fixed_values,
        fixed_nonzeros=fixed_nonzeros,
        zeros=zeros,
        nonzeros=nonzeros)

    return variables

class PID_BROJA(CVXOPT_Template):
    """
    An optimizer for the partial information framework that restricts to
    matching pairwise marginals for each input with the output.

    Inputs are: X_0, X_1, ..., X_n
    Output is: Y

    This is based off:

        Bertschinger, N.; Rauh, J.; Olbrich, E.; Jost, J.; Ay, N.
        Quantifying Unique Information. Entropy 2014, 16, 2161-2183.

    """
    def __init__(self, dist, sources, target, k=2, rv_mode=None, extra_constraints=True, tol=None, prng=None):
        """
        Initialize an optimizer for the partial information framework.

        Parameters
        ----------
        dist : distribution
            The distribution used to calculate the partial information.
        sources : list of lists
            The sources random variables. Each random variable specifies a list
            of random variables in `dist` that define a source.
        target : list
            The random variables in `dist` that define the target.
        k : int
            The size of the marginals that are constrained to equal marginals
            from `dist`. For the calculation of unique information, we use k=2.
        rv_mode : str, None
            Specifies how to interpret the elements of each source and the
            target. Valid options are: {'indices', 'names'}. If equal to
            'indices', then the elements of each source and the target are
            interpreted as random variable indices. If equal to 'names', the
            elements are interpreted as random variable names. If `None`, then
            the value of `dist._rv_mode` is consulted.
        extra_constraints : bool
            When possible, additional constraints beyond the required marginal
            constraints are added to the optimization problem. These exist
            values of the input and output that satisfy p(inputs | outputs) = 1
            In that case, p(inputs, outputs) is equal to q(inputs, outputs) for
            all q in the feasible set.
        tol : float | None
            The desired convergence tolerance.
        prng : RandomState
            A NumPy-compatible pseudorandom number generator.


        """
        # Store the original parameters in case we want to construct an
        # "uncoalesced" distribution from the optimial distribution.
        self.dist_original = dist
        self._params = Bunch(sources=sources, target=target, rv_mode=rv_mode)

        self.dist = prepare_dist(dist, sources, target, rv_mode=rv_mode)
        self.k = k
        self.extra_constraints = extra_constraints

        super(PID_BROJA, self).__init__(self.dist, tol=tol, prng=prng)

        self.pmf_copy = self.pmf.copy()


    def prep(self):
        # We are going to remove all zero and fixed elements.
        # This means we will no longer be optimizing H[B | {A_i}] but some
        # some constant shift from that function. Since the elements don't
        # change, technically, we don't need to even include the terms having
        # to do with the fixed elements. But y_i = (M x)_i depends on having
        # the full vector x. So when we calculate the objective, we will
        # need to reconstruct the full x vector and calculate
        if self.extra_constraints:
            self.vartypes = extra_constraints(self.dist, self.k)

            # Update the effective number of parameters
            # This will carryover and fix the inequality constraints.
            self.n = len(self.vartypes.free)
        else:
            self.vartypes = None

    def build_function(self):

        # Build the matrix which calculates the marginal of the inputs.
        from scipy.linalg import block_diag
        Bsize = len(self.dist.alphabet[-1])
        Osize = np.prod(map(len, self.dist.alphabet[:-1]))
        block = np.ones((Bsize, Bsize))
        blocks = [block] * Osize
        M = block_diag(*blocks)

        M_free = M[:, self.vartypes.free]
        fnz = self.vartypes.fixed_nonzeros
        M_fnz = M[:, fnz]
        x_fnz = self.pmf[fnz]
        y_fnz_offset = np.dot(M_fnz, x_fnz)

        self.M_free = M_free
        self.y_fnz_offset = y_fnz_offset

        def negH_YgXis(pmf):
            y_free = np.dot(M_free, pmf)
            y = y_free + y_fnz_offset
            nonzeros = self.vartypes.nonzeros
            y_nonzero = y[nonzeros]
            self.pmf_copy[self.vartypes.free] = pmf
            x_nonzero = self.pmf_copy[nonzeros]
            terms = x_nonzero * np.log2(x_nonzero / y_nonzero)
            return np.nansum(terms)

        self.func = negH_YgXis

    def build_linear_equality_constraints(self):
        from cvxopt import matrix

        A, b = marginal_constraints(self.dist, self.k)

        # Reduce the size of the constraint matrix
        if self.extra_constraints:
            Asmall = A[:, self.vartypes.free]
            # The shape of b is unchanged, but fixed nonzero values modify it.
            fnz = self.vartypes.fixed_nonzeros
            b = b - np.dot(A[:, fnz], self.pmf[fnz])
        else:
            Asmall = A

        if Asmall.shape[1] == 0:
            Asmall = None
            b = None
        else:
            Asmall, b, rank = as_full_rank(Asmall, b)
            if rank > Asmall.shape[1]:
                msg = 'More independent constraints than free parameters.'
                raise ValueError(msg)

            Asmall = matrix(Asmall)
            b = matrix(b)  # now a column vector

        self.A = Asmall
        self.b = b

    def initial_dist(self):
        d = self.prng.dirichlet([1] * self.n)
        """
        #x = InitialPoint(self.dist)

        from pymisc import to_mma

        print("Finding initial feasible point.")
        d = x.optimize(maxiters=10, show_progress=True)[:, 0]
        #d = pmf_AND2(.37, .12)
        print d.round(3)
        print "Done"
        print
        """
        return d


def entropy(p):
    return -np.nansum(p * np.log2(p))

def cmi(pmf):
    # Binary rvs: p(X_0, X_1, Y)
    # f0(p) := I[ X_0 : Y | X_1 ] = H[ Y | X_1] - H[Y | X_0, X_1]
    n_variables = 3
    n_symbols = 2
    d = AbstractDenseDistribution(n_variables, n_symbols)


    H_X0X1Y = entropy(pmf)

    idx_X0X1 = d.parameter_array([0, 1])
    p_X0X1 = np.array([ pmf[idx].sum() for idx in idx_X0X1 ])
    H_X0X1 = entropy(p_X0X1)

    idx_X1 = d.parameter_array([1])
    p_X1 = np.array([ pmf[idx].sum() for idx in idx_X1 ])
    H_X1 = entropy(p_X1)

    idx_X1Y = d.parameter_array([1, 2])
    p_X1Y = np.array([ pmf[idx].sum() for idx in idx_X1Y ])
    H_X1Y = entropy(p_X1Y)

    cmi = H_X1Y - H_X1 - H_X0X1Y + H_X0X1
    return -cmi

def negH_YgX1X2(pmf):
    """
    Calculate -H[Y | X_0, X_1].

    """
    n_variables = 3
    n_symbols = 2
    d = AbstractDenseDistribution(n_variables, n_symbols)

    H_X0X1Y = entropy(pmf)

    idx_X0X1 = d.parameter_array([0, 1])
    p_X0X1 = np.array([ pmf[idx].sum() for idx in idx_X0X1 ])
    H_X0X1 = entropy(p_X0X1)

    H_YgX0X1 = H_X0X1Y - H_X0X1

    return -H_YgX0X1

def pi_decomp(d, d_opt):
    u0 = dit.multivariate.coinformation(d_opt, [[2],[0]], [1])
    u1 = dit.multivariate.coinformation(d_opt, [[2],[1]], [0])
    mi0 = dit.shannon.mutual_information(d_opt, [0], [2])
    rdn = mi0 - u0
    mi = dit.shannon.mutual_information(d, [0,1], [2])
    syn = mi - mi0 - u1

    return syn, u0, u1, rdn

def pmf_AND(x1):
    """
    Conforming pmfs for the AND distribution.

    """
    return np.array([x1, 0, 1/2 - x1, 0, 1/2 - x1, 0, -1/4 + x1, 1/4])

def AND2():
    d = dit.example_dists.And()
    d.make_dense()
    d.pmf = np.array([3,1,3,1,3,1,1,3], dtype=float)  / 16
    return d

def pmf_AND2(x1, x2):
    return np.array([x1, x2, 3/8 - x1, 1/8 - x2, 3/8 - x1, 1/8 - x2, -1/8 + x1, 1/8 + x2])


def main():
    d = dit.example_dists.And()
    d = AND2()
    x = Optimizer(d, negH_YgX1X2)

    #x = Optimizer(d, cmi)
    print d
    print
    pmf_opt = x.optimize(show_progress=True)

    mi = dit.shannon.mutual_information(d, [0,1], [2])
    d_opt = dit.Distribution(d.outcomes, pmf_opt)
    print
    print d_opt

    print
    print "I[X1, X2 : Y] = ", mi
    print
    print "syn, u1, u2, rdn"
    print pi_decomp(d, d_opt)

def dice():
    # DoF: 85, 36, 15, 10, 5, 0
    d = dit.example_dists.summed_dice(1, 5)
    x = PID_BROJA(d, [[0], [1]], [2], extra_constraints=True)
    #pmf_opt = x.optimize(show_progress=True)
    return x


if __name__ == '__main__':
    dice()
    pass
