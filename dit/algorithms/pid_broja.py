# -*- coding: utf-8 -*-
"""
Partial information decompositions
    and
Maxent decompositions of I[sources : target]
"""


from __future__ import division, print_function

import logging
from collections import defaultdict
import itertools

import numpy as np
import dit

from dit.utils import basic_logger
from dit.abstractdist import get_abstract_dist
from dit.algorithms.optutil import (
    CVXOPT_Template, as_full_rank, Bunch, op_runner, frank_wolfe
)
from dit.exceptions import ditException

__all__ = ['unique_informations', 'k_informations', 'k_synergy']


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

def marginal_constraints(dist, k, normalization=True, source_marginal=False):
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
        A.append(np.ones(n_elements))
        b.append(1)

    # Random variables
    rvs = range(n_variables)
    target_rvs = tuple(rvs[-1:])
    source_rvs = tuple(rvs[:-1])

    try:
        k, source_rvs = k
        source_rvs = tuple(source_rvs)
    except TypeError:
        pass

    assert k >= 1
    marginal_size = k
    submarginal_size = marginal_size - 1
    #assert submarginal_size >= 1

    # p( source_{k-1}, target ) = q( source_{k-1}, target )
    cache = {}
    for subrvs in itertools.combinations(source_rvs, submarginal_size):
        marg_rvs = subrvs + target_rvs
        marray = d.parameter_array(marg_rvs, cache=cache)
        for idx in marray:
            # Convert the sparse nonzero elements to a dense boolean array
            bvec = np.zeros(n_elements)
            bvec[idx] = 1
            A.append(bvec)
            b.append(pmf[idx].sum())

    if source_marginal:
        marray = d.parameter_array(source_rvs, cache=cache)
        for idx in marray:
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

    2. Now suppose we want to find out when p({x_i}, y) = q({x_i}, y).

       For every (k-1)-subset, g_k(x), we have:

          p(g_k(x), y) = q(g_k(x), y)

       So we have:

          p(y) = q(y)
          p(g_k(x) | y) = q(g_k(x) | y)    provided k > 1.

       If k = 1, we cannot proceed.

       Now suppose that for every i, we had:

          p(x_i | y) = \delta(x_i, f_i(y))

       Then, it follows that:

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

    try:
        # Ignore source restrictions.
        k, _ = k
    except TypeError:
        pass

    marginal_size = k
    assert k >= 1
    submarginal_size = marginal_size - 1
    #assert submarginal_size >= 1

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
    determined = defaultdict(lambda: [0] * len(source_rvs))

    # If there is no source rv because k=1, then nothing can be determined.
    if submarginal_size:
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

    if fixed:
        fixed = sorted(fixed.items())
        fixed_indexes, fixed_values = list(zip(*fixed))
        fixed_indexes = list(fixed_indexes)
        fixed_values = list(fixed_values)
    else:
        fixed_indexes = []
        fixed_values = []

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

class MaximumConditionalEntropy(CVXOPT_Template):
    """
    An optimizer for the unique information.

    Inputs are: X_0, X_1, ..., X_n
    Output is: Y

    We find a distribution that matches all pairwise marginals for each input
    with the output: P(X_i, Y), that maximizes the H[Y | X_1, ..., X_n].

    This is based off:

        Bertschinger, N.; Rauh, J.; Olbrich, E.; Jost, J.; Ay, N.
        Quantifying Unique Information. Entropy 2014, 16, 2161-2183.

    """
    def __init__(self, dist, sources, target, k=2, rv_mode=None,
                 extra_constraints=True, source_marginal=False, tol=None,
                 prng=None, verbose=None):
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
            Note that these marginals include the target random variable.
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
        source_marginal : bool
            If `True`, also require that the source marginal distribution
            p(X_1, ..., X_n) is matched. This will yield a distribution such
            that S^k := H(q) - H(p) is the information that is not captured
            by matching the k-way marginals that include the target. k=1
            is the mutual information between the sources and the target.
        tol : float | None
            The desired convergence tolerance.
        prng : RandomState
            A NumPy-compatible pseudorandom number generator.
        verbose : int
            An integer representing the logging level ala the ``logging``
            module. If `None`, then (effectively) the log level is set to
            `WARNING`. For a bit more information, set this to `logging.INFO`.
            For a bit less, set this to `logging.ERROR`, or perhaps 100.

        """
        self.logger = basic_logger('dit.pid_broja', verbose)

        # Store the original parameters in case we want to construct an
        # "uncoalesced" distribution from the optimial distribution.
        self.dist_original = dist
        self._params = Bunch(sources=sources, target=target, rv_mode=rv_mode)

        self.dist = prepare_dist(dist, sources, target, rv_mode=rv_mode)
        self.k = k
        self.extra_constraints = extra_constraints
        self.source_marginal = source_marginal
        self.verbose = verbose

        super(MaximumConditionalEntropy, self).__init__(self.dist, tol=tol, prng=prng)

    def prep(self):
        # We are going to remove all zero and fixed elements.
        # This means we will no longer be optimizing H[B | {A_i}] but some
        # some constant shift from that function. Since the elements don't
        # change, technically, we don't need to even include the terms having
        # to do with the fixed elements. But y_i = (M x)_i depends on having
        # the full vector x. So when we calculate the objective, we will
        # need to reconstruct the full x vector and calculate
        self.pmf_copy = self.pmf.copy()

        if self.extra_constraints:
            self.vartypes = extra_constraints(self.dist, self.k)

            # Update the effective number of parameters
            # This will carryover and fix the inequality constraints.
            self.n = len(self.vartypes.free) # pylint: disable=no-member
            fnz = self.vartypes.fixed_nonzeros # pylint: disable=no-member
            self.normalization = 1 - self.pmf_copy[fnz].sum()
        else:
            warn = "This might not work if there are too many "
            warn += "optimization variables."
            self.logger.warn(warn)

            indexes = np.arange(len(self.pmf))
            variables = Bunch(
                free=indexes,
                fixed=[],
                fixed_values=[],
                fixed_nonzeros=[],
                zeros=[],
                nonzeros=indexes)
            self.vartypes = variables
            self.normalization = 1

    def build_function(self):

        # Build the matrix which calculates the marginal of the inputs.
        from scipy.linalg import block_diag
        Bsize = len(self.dist.alphabet[-1])
        Osize = np.prod([len(a) for a in self.dist.alphabet[:-1]])
        block = np.ones((Bsize, Bsize))
        blocks = [block] * Osize
        self.M = M = block_diag(*blocks)

        # Construct submatrices used for free and fixed nonzero parameters.
        M_col_free = M[:, self.vartypes.free] # pylint: disable=no-member
        fnz = self.vartypes.fixed_nonzeros # pylint: disable=no-member
        M_col_fnz = M[:, fnz]
        x_fnz = self.pmf[fnz]
        self.y_partial_fnz = y_partial_fnz = np.dot(M_col_fnz, x_fnz)

        def negH_YgXis(x_free):
            """
            Calculates -H[Y | X_1, \ldots, X_n] using only the free variables.

            """
            # Convert back to a NumPy 1D array
            try:
                x_free = np.asarray(x_free).transpose()[0]
            except IndexError:
                assert (x_free.size[1] == 0)
                x_free = np.array([])

            y_partial_free = np.dot(M_col_free, x_free)
            # y_partial_zero == 0, so no need to calculate or add it.
            y = y_partial_free + y_partial_fnz
            nonzeros = self.vartypes.nonzeros # pylint: disable=no-member
            y_nonzero = y[nonzeros]
            self.pmf_copy[self.vartypes.free] = x_free # pylint: disable=no-member
            x_nonzero = self.pmf_copy[nonzeros]
            terms = x_nonzero * np.log2(x_nonzero / y_nonzero)
            return np.nansum(terms)

        self.func = negH_YgXis

    def build_linear_equality_constraints(self):
        from cvxopt import matrix

        smarg = self.source_marginal
        A, b = marginal_constraints(self.dist, self.k, source_marginal=smarg)
        self.A_full = A
        self.b_full = b

        # Reduce the size of the constraint matrix
        if self.extra_constraints:
            Asmall = A[:, self.vartypes.free] # pylint: disable=no-member
            # The shape of b is unchanged, but fixed nonzero values modify it.
            fnz = self.vartypes.fixed_nonzeros # pylint: disable=no-member
            b = b - np.dot(A[:, fnz], self.pmf[fnz])
        else:
            Asmall = A

        if Asmall.shape[1] == 0:
            # No free parameters
            Asmall = None
            b = None
        else:
            #print(Asmall[0], b[0])
            Asmall, b, rank = as_full_rank(Asmall, b)
            #print(Asmall[0], b[0])
            if rank > Asmall.shape[1]:
                msg = 'More independent constraints than free parameters.'
                raise ValueError(msg)

            Asmall = matrix(Asmall)
            b = matrix(b)  # now a column vector

        self.A = Asmall
        self.b = b

    def build_gradient_hessian(self):

        from cvxopt import matrix

        def gradient(x_free):
            """Return the gradient of the free elements only.

            The gradient is:

                (\grad f)_j = \log_b x_j / y_j  + 1 / \log_b
                             - 1 / \log_b \sum_i x_i / y_i M_{ij}

            Our task here is return the elements corresponding to the
            free indexes of the grad(x).

            """
            # Convert back to a NumPy 1D array
            x_free = np.asarray(x_free).transpose()[0]

            nonzeros = self.vartypes.nonzeros # pylint: disable=no-member
            free = self.vartypes.free # pylint: disable=no-member

            # First, we need to obtain y_nonzero.
            M_col_free = self.M[:, self.vartypes.free] # pylint: disable=no-member
            y_partial_free = np.dot(M_col_free, x_free)
            y = y_partial_free + self.y_partial_fnz
            y_nonzero = y[nonzeros]

            # We also need x_nonzero
            self.pmf_copy[self.vartypes.free] = x_free # pylint: disable=no-member
            x_nonzero = self.pmf_copy[nonzeros]

            # The last term:
            #   \sum_i  x_i / y_i M_{ij}
            # will have nonzero contributions only from the nonzero rows.
            M_row_nonzero = self.M[nonzeros, :]
            last = (x_nonzero / y_nonzero)[:, np.newaxis] * M_row_nonzero
            last = last.sum(axis=0)
            last_free = 1 / np.log(2) * last[free]

            # Indirectly:
            #grad = x / y + 1/np.log(2) - 1/np.log(2) * last
            #grad_free = grad[free]

            # Directly:
            grad_free = x_free / y[free] + 1 / np.log(2) - last_free

            return matrix(grad_free)

        self.gradient = gradient

        def hessian(x_free):
            raise NotImplementedError

        self.hessian = hessian


    def initial_dist(self):
        """
        Find an initial point in the interior of the feasible set.

        """
        from cvxopt import matrix
        from cvxopt.modeling import variable

        A = self.A
        b = self.b

        # Assume they are already CVXOPT matrices
        if self.vartypes and A.size[1] != len(self.vartypes.free): # pylint: disable=no-member
            msg = 'A must be the reduced equality constraint matrix.'
            raise Exception(msg)

        # Set cvx info level based on logging.INFO level.
        if self.logger.isEnabledFor(logging.INFO):
            show_progress = True
        else:
            show_progress = False

        n = len(self.vartypes.free) # pylint: disable=no-member
        x = variable(n)
        t = variable()

        tols = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
        for tol in tols:
            constraints = []
            constraints.append((-tol <= A * x - b))
            constraints.append((A * x - b <= tol))
            constraints.append((x >= t))

            # Objective to minimize
            objective = -t

            opt = op_runner(objective, constraints, show_progress=show_progress)
            if opt.status == 'optimal':
                #print("Found initial point with tol={}".format(tol))
                break
        else:
            msg = 'Could not find valid initial point: {}'
            raise Exception(msg.format(opt.status))

        # Grab the optimized x. Perhaps there is a more reliable way to get
        # x rather than t. For now,w e check the length.
        optvariables = opt.variables()
        if len(optvariables[0]) == n:
            xopt = optvariables[0].value
        else:
            xopt = optvariables[1].value

        # Turn values close to zero to be exactly equal to zero.
        xopt = np.array(xopt)[:, 0]
        xopt[np.abs(xopt) < tol] = 0
        # Normalize properly accounting for fixed nonzero values.
        xopt /= xopt.sum()
        xopt *= self.normalization

        # Do not build the full vector since this is input to the reduced
        # optimization problem.
        #xx = np.zeros(len(dist.pmf))
        #xx[variables.nonzero] = xopt

        return xopt, opt

    def optimize(self, **kwargs):
        from cvxopt import matrix

        objective = self.func
        gradient = self.gradient

        if self.A is None:
            # No free constraints.
            assert (len(self.vartypes.fixed) == len(self.pmf)) # pylint: disable=no-member
            self.logger.info("No free parameters. Optimization unnecessary.")
            self.pmf_copy[self.vartypes.fixed] = self.vartypes.fixed_values # pylint: disable=no-member
            xfinal = self.pmf_copy.copy()
            xfinal_free = xfinal[self.vartypes.free] # pylint: disable=no-member
            opt = self.func(matrix(xfinal_free))
            return xfinal, opt

        A = matrix(self.A)
        b = matrix(self.b)
        self.logger.info("Finding initial distribution.")
        initial_x = matrix(self.initial_dist()[0])


        m = "Optimizing from initial distribution using Frank-Wolfe algorithm."
        self.logger.info(m)

        # Set logging level for Frank-Wolfe if we are at logging.INFO level.
        if self.logger.isEnabledFor(logging.INFO):
            verbose = logging.INFO
        else:
            verbose = None
        if verbose not in kwargs:
            kwargs['verbose'] = verbose

        x, obj = frank_wolfe(objective, gradient, A, b, initial_x, **kwargs)
        # x is currently a matrix
        x = np.asarray(x).transpose()[0]

        # Subnormalize it.
        x *= self.normalization
        self.xfinal = x

        # Rebuild the full distribution as a NumPy array
        self.pmf_copy[self.vartypes.free] = x # pylint: disable=no-member
        xfinal_full = self.pmf_copy.copy()

        return xfinal_full, obj

def pi_decomp(d, d_opt):
    u0 = dit.multivariate.coinformation(d_opt, [[2], [0]], [1])
    u1 = dit.multivariate.coinformation(d_opt, [[2], [1]], [0])
    mi0 = dit.shannon.mutual_information(d_opt, [0], [2])
    mi0_test = dit.shannon.mutual_information(d, [0], [2])
    rdn = mi0 - u0
    mi = dit.shannon.mutual_information(d, [0, 1], [2])
    syn = mi - mi0 - u1

    return syn, u0, u1, rdn

def calculate_synergy(pmf_opt, ui):
    d = ui.dist
    d_opt = d.copy()
    d_opt.pmf[:] = pmf_opt
    # Original sources
    original_sources = ui._params.sources
    sources = list(range(len(original_sources)))
    target = [len(sources)]
    mi = dit.multivariate.coinformation(d, [sources, target])
    mi_opt = dit.multivariate.coinformation(d_opt, [sources, target])
    return mi - mi_opt

def dice(a, b):
    # DoF: 85, 36, 15, 10, 5, 0
    d = dit.example_dists.summed_dice(a, b)
    x = MaximumConditionalEntropy(d, [[0], [1]], [2], extra_constraints=True)
    pmf_opt, obj = x.optimize()
    d_opt = x.dist.copy()
    d_opt.pmf[:] = pmf_opt
    print(pi_decomp(x.dist, d_opt))
    return x

def demo():
    import matplotlib.pyplot as plt

    _, axes = plt.subplots(1, 3)
    bvals = range(1, 7)
    avals = np.linspace(0, 1, num=10)
    for b in bvals:
        decomps = []
        for a in avals:
            print("**** {}, {} *****".format(a, b))
            d = dit.example_dists.summed_dice(a, b)
            x = MaximumConditionalEntropy(d, [[0], [1]], [2],
                                          extra_constraints=True, verbose=20)
            pmf_opt, _ = x.optimize()
            d_opt = x.dist.copy()
            d_opt.pmf[:] = pmf_opt
            decomps.append(pi_decomp(x.dist, d_opt))
        decomps = np.asarray(decomps)
        # redundancy
        axes[0].plot(avals, decomps[:, -1], label="{}".format(b))
        # unique
        axes[1].plot(avals, decomps[:, -2], label="{}".format(b))
        # synergy
        axes[2].plot(avals, decomps[:, 0], label="{}".format(b))

    axes[0].set_title('Redundancy')
    axes[1].set_title('Unique')
    axes[2].set_title('Synergy')

    plt.show()

def k_synergy(d, sources, target, k=2, rv_mode=None, extra_constraints=True,
              tol=None, prng=None, verbose=None):
    """
    Returns the k-synergy.

    The k-synergy is the amount of I[sources : target] that is not captured by
    matching the k-way marginals, which include the source, and the marginal
    source distribution. When k=1, we only match p(target) and the source
    marginal, p(source marginal). So the k-synergy will be equal to the mutual
    information I[sources: target].

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
    verbose : int
        An integer representing the logging level ala the ``logging``
        module. If `None`, then (effectively) the log level is set to
        `WARNING`. For a bit more information, set this to `logging.INFO`.
        For a bit less, set this to `logging.ERROR`, or perhaps 100.

    Returns
    -------
    ui : array-like
        The unique information that each source has about the target rv.
    rdn : float
        The redundancy calculated from the optimized distribution.
    syn : float
        The synergy calculated by subtracting the optimized mutual information
        I[sources : target] from the true mutual information.
    mi_orig : float
        The total mutual information between the sources and the target for
        the original (true) distribution.
    mi_opt : float
        The total mutual information between the sources and the target for
        the optimized distribution.

    """
    x = MaximumConditionalEntropy(d, sources, target, k=k, rv_mode=rv_mode,
                                  extra_constraints=extra_constraints,
                                  source_marginal=True, tol=tol, prng=prng,
                                  verbose=verbose)
    pmf, _ = x.optimize()
    d_orig = x.dist
    d_opt = d_orig.copy()
    d_opt.pmf[:] = pmf
    H_opt = dit.multivariate.entropy(d_opt)
    H_orig = dit.multivariate.entropy(d_orig)
    return H_opt - H_orig

def k_informations(d, sources, target, rv_mode=None, extra_constraints=True,
                   tol=None, prng=None, verbose=None):
    """
    Returns the amount of I[sources:target] captured by matching k-way
    marginals that include the target and the source marginal distribution.

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
    verbose : int
        An integer representing the logging level ala the ``logging``
        module. If `None`, then (effectively) the log level is set to
        `WARNING`. For a bit more information, set this to `logging.INFO`.
        For a bit less, set this to `logging.ERROR`, or perhaps 100.

    Returns
    -------
    infos : array-like, (len(sources),)
        The k-way informations. infos[i] corresponds to the additional
        amount of information gained about the total mutual information
        in moving from matching the (i+1)-way marginals to the (i+2)-way
        marginals (plus the source marginals). So infos[0] is how much
        you gain about I[sources:target] in moving from 1-way to 2-way
        marginals (with source) and source marginals.

    """
    nonkinfos = []
    x = MaximumConditionalEntropy(d, sources, target, k=1, rv_mode=rv_mode,
                                  extra_constraints=extra_constraints,
                                  source_marginal=True, tol=tol, prng=prng,
                                  verbose=verbose)
    n = x.dist.outcome_length()
    while x.k <= n:
        pmf, _ = x.optimize()
        d_orig = x.dist
        d_opt = d_orig.copy()
        d_opt.pmf[:] = pmf
        H_opt = dit.multivariate.entropy(d_opt)
        H_orig = dit.multivariate.entropy(d_orig)
        nonkinfo = H_opt - H_orig
        nonkinfos.append(nonkinfo)
        x.k += 1
        x.init()

    nonkinfos.reverse()
    diffs = np.diff(nonkinfos)
    return np.asarray(list(reversed(diffs)))

def unique_informations(d, sources, target, k=2, rv_mode=None,
                        extra_constraints=True, tol=None, prng=None,
                        verbose=None):
    """
    Returns the unique information each source has about the target.

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
    verbose : int
        An integer representing the logging level ala the ``logging``
        module. If `None`, then (effectively) the log level is set to
        `WARNING`. For a bit more information, set this to `logging.INFO`.
        For a bit less, set this to `logging.ERROR`, or perhaps 100.

    Returns
    -------
    ui : array-like
        The unique information that each source has about the target rv.
    rdn : float
        The redundancy calculated from the optimized distribution.
    syn : float
        The synergy calculated by subtracting the optimized mutual information
        I[sources : target] from the true mutual information.
    mi_orig : float
        The total mutual information between the sources and the target for
        the original (true) distribution.
    mi_opt : float
        The total mutual information between the sources and the target for
        the optimized distribution.

    Notes
    -----
    The nonunique information would be `mi_orig - ui.sum()`.

    """
    x = MaximumConditionalEntropy(d, sources, target, k=k, rv_mode=rv_mode,
                                  extra_constraints=extra_constraints, tol=tol,
                                  prng=prng, verbose=verbose)
    pmf, _ = x.optimize()
    d_orig = x.dist
    d_opt = d_orig.copy()
    d_opt.pmf[:] = pmf
    n = d_opt.outcome_length()
    uis = []
    for rv in range(n-1):
        others = list(range(rv)) + list(range(rv+1, n-1))
        ui = dit.multivariate.coinformation(d_opt, [[rv], [n-1]], others)
        uis.append(ui)
    mi_opt = dit.multivariate.coinformation(d_opt, [[n-1], list(range(n-1))])
    mi_orig = dit.multivariate.coinformation(d_orig, [[n-1], list(range(n-1))])
    rdn = dit.multivariate.coinformation(d_opt, [[i] for i in range(n)])
    syn = mi_orig - mi_opt
    return np.array(uis), rdn, syn, mi_orig, mi_opt


if __name__ == '__main__':
    z = dice(1, 3)
