"""
Abstract implementation of dense random vectors.

"""

import itertools
import numpy as np

__all__ = [
    'AbstractDenseDistribution',
    'distribution_constraint',
    'brute_marginal_array',
    'get_abstract_dist',
]

class AbstractDenseDistribution(object):
    """
    An abstract, dense distribution.

    We can think of this as the distribution for a random vector, where the
    the sample space for each random variable is identical. If L is the length
    of the vector and K is the size of the sample space for each random
    variable, then the distribution's pmf is an array consisting of `K**L`
    probabilities. We parameterize this pmf in the following way:

        d[i] = Pr( 'i'th lexicographically ordered word )

    For example, suppose L = 2 and K = 2, with alphabet {0,1}.  Then, our
    distribution is:

        d = Pr(X_0, X_1)

    where:

        d[0] = Pr(00)
        d[1] = Pr(01)
        d[2] = Pr(10)
        d[3] = Pr(11)

    ---

    This class provides convenient lookups for any subset of random variables
    in terms of the parameters.  For example,

        Pr(X_0 = 0) = d[0] + d[1]
        Pr(X_0 = 1) = d[2] + d[3]
        Pr(X_1 = 0) = d[0] + d[2]
        Pr(X_1 = 1) = d[1] + d[3]

    Thus, we can represent the random variables as matrices:

        X_0 -> [[0,1],[2,3]]
        X_1 -> [[0,2],[1,3]]

    Applications of this class are numerous.  For example, now we can quickly
    obtain any marginal distributions from a dense representation of a joint
    distribution. It also allows us to write down constraint equations when
    declaring distributions equal to each other. For example, suppose we wanted
    P(X_0) = P(X_1).  This corresponds to:

        d[0] + d[1] = d[0] + d[2]
        d[2] + d[3] = d[1] + d[3]

    which can be represented as a matrix equation `np.dot(A,d) = b` with:

        A = [[0,1,-1,0],    b = [[0],
             [0,-1,1,0]]         [0]]

    """
    def __init__(self, n_variables, n_symbols):
        """
        Initialize the abstract distribution.

        Parameters
        ----------
        n_variables : int
            The number of random variables.
        n_symbols : int
            The number of symbols in sample space of every random variable.

        """
        self.n_variables = n_variables
        self.n_symbols = n_symbols
        self.n_elements = n_symbols ** n_variables

        self._initialize_singletons(n_variables, n_symbols)

    def _initialize_singletons(self, n_variables, n_symbols):
        """
        Populates the singleton marginal distribution matrices.

            P(X_i)  for i = 0, ..., L-1

        """
        # For each X_t, we have an array of shape ( |A|, |A|^(L-1) )
        # giving a total of |A|^L elements. The columns break into |A|^t
        # blocks, each block having |A| rows and |A|^(L-t-1) columns.
        # To populate the array, we proceed by block, by row, by column.
        shape = (n_variables, n_symbols, n_symbols ** (n_variables - 1))
        rvs = np.empty(shape, dtype=int)
        rows = list(range(n_symbols))
        for t in range(n_variables):
            Xt = rvs[t]
            blocks = range(n_symbols**t)
            blockCols = n_symbols**(n_variables - t - 1)
            cols = range(blockCols)
            locations = itertools.product(blocks, rows, cols)
            for idx, (block, row, col) in enumerate(locations):
                i, j = row, block * blockCols + col
                Xt[i, j] = idx

        # Convert rvs to a 2D array of sets.
        # This makes it quicker to form marginals.
        self.rvs = rvs_set = np.empty((n_variables, n_symbols), dtype=object)
        indexes = itertools.product(range(n_variables), range(n_symbols))
        for index in indexes:
            rvs_set[index] = set(rvs[index])

    def parameter_array(self, indexes, cache=None):
        """
        Returns a 2D NumPy array representing the distribution on `indexes`.

        For example,  indexes=[0,1] returns an array representing P(X_0,X_1)
        in terms of the parameters of the joint distribution.

        Parameters
        ----------
        indexes : list or set
            A list or set of integers, specifying which indexes should be
            included in the distribution.
        cache : dict or None
            If you intend on calculating the parameter arrays for a large number
            of possible indexes, then pass in the same dictionary to cache each
            time and the arrays will be calculated efficiently.

        Returns
        -------
        p : NumPy array, shape (m,n)
            The representation of the distribution in terms of the parameters
            of the original distribution. The number of rows, m, is equal to:
                m = self.n_symbols ** len(indexes)
                n = self.n_symbols ** (self.n_variables - len(indexes))

        """
        if cache is None:
            # Then we use an internal cache for this call only.
            cache = {}

        indexes = set(indexes)
        if min(indexes) < 0 or max(indexes) >= self.n_variables:
            msg = 'Invalid indexes: ' + str(indexes)
            raise Exception(msg)
        indexes = tuple(sorted(indexes))

        # We want to return 2D arrays, but for efficiency reasons, the cache
        # must store 1D arrays of sets.  Thus, we make 'calculate' a closure.

        def calculate(indexes):
            """
            Internal function which calculates parameter arrays.

            """
            # The singleton random variables have already been computed.
            if len(indexes) == 1:
                idx = next(iter(indexes))
                cache[indexes] = p = self.rvs[idx]

            # If indexes are consecutive from zero, then we can do these easily.
            elif (indexes[0] == 0) and (np.all(np.diff(indexes) == 1)):
                p = np.arange( self.n_symbols**self.n_variables )
                shape = (self.n_symbols**len(indexes),
                         self.n_symbols**(self.n_variables - len(indexes)))
                p = p.reshape(shape)
                cache[indexes] = p = np.array([set(row) for row in p])

            else:
                # We take intersections to find the parameters for each word.
                # To avoid repeatedly taking intersections of the same sets, we
                # need to implement this recursively from the left.

                # Note, we catch len(indexes) == 1 earlier.
                left = calculate(indexes[:-1])
                right = calculate(indexes[-1:])

                # The new p is a Cartestian product of the row intersections.
                p = np.empty(len(left) * len(right), dtype=object)
                for i, (rowL, rowR) in enumerate(itertools.product(left, right)):
                    p[i] = rowL.intersection(rowR)
                cache[indexes] = p

            return p

        if indexes in cache:
            p = cache[indexes]
        else:
            p = calculate(indexes)

        # p is a 1D array of sets. Convert it to a 2D array.
        p = np.array([sorted(element) for element in p])

        # If each set is not of the same length, then NumPy will create
        # a 1D array with dtype=object.  This should not happen.
        assert len(p.shape) == 2, "Parameter array is not 2D!"

        return p

    def marginal(self, indexes):
        """
        Returns an abstract representation of a marginal distribution.

        Parameters
        ----------
        indexes : list or set
            A list or set of integers, specifying which indexes to keep. In
            truth, the index values do not matter since the new distribution
            object only needs to know the word length. However, we do some
            checks to make sure the indexes are valid.

        Returns
        -------
        d : AbstractDenseDistribution
            The new abstract representation of the marignal distribution.

       """
        indexes = set(indexes)
        if min(indexes) < 0 or max(indexes) >= self.n_variables:
            msg = 'Invalid indexes.'
            raise Exception(msg)

        d = AbstractDenseDistribution(len(indexes), self.n_symbols)
        return d

def distribution_constraint(indexes1, indexes2, distribution):
    """
    Returns an array representing an equality constraint on two distributions.

    Suppose indexes1=(0,1)
            indexes2=(1,2)

    Then, we are demanding that Pr(X_0 X_1) = Pr(X_1 X_2).  The indexes are
    assumed to come from some larger joint distribution of length `n_variables`,
    which is parametrized as a vector d.  Each d[i] corresponds to the
    probability of a single word of length `n_variables`, lexicographically
    ordered. Thus, an equality distribution constraint provides
    `n_symbols ** len(indexes1)` equations and satisfies:

        np.dot(A,d) = 0

    where A is the matrix of coeffecients defining the constraints. Note, it is
    not generically true that the rows of this matrix are linearly independent.

    Parameters
    ----------
    indexes1 : tuple
        A tuple of integers representing the indexes of the first distribution.
        The length of indexes1 must equal the length of indexes2.
    indexes2 : tuple
        A tuple of integers representing the indexes of the second distribution.
        The length of indexes1 must equal the length of indexes2.
    distribution : AbstractDenseDistribution
        An abstract joint distribution compatible with the indexes.

    Returns
    -------
    A : NumPy array, shape (m,n)
        The constraint matrix A.  The number of rows, m, is equal to
        `n_symbols ** len(indexes1)`. The number of columns, n, is
        equal to the number of parameters in the joint distribution.
    b : NumPy array, shape (n,)
        An array of zeros.
    """
    if len(set(indexes1)) != len(set(indexes2)):
        raise Exception("Incompatible distributions.")

    cache = {}
    d1 = distribution.parameter_array(indexes1, cache=cache)
    d2 = distribution.parameter_array(indexes2, cache=cache)
    A = np.zeros((len(d1), distribution.n_elements), dtype=int)
    b = np.zeros(distribution.n_elements, dtype=int)

    for idx, (w1, w2) in enumerate(zip(d1, d2)):
        symdiff = set.symmetric_difference(set(w1), set(w2))
        vec = [1 if i in w1 else -1 for i in symdiff]
        A[(idx,), tuple(symdiff)] = vec

    return A, b

def brute_marginal_array(d, rvs, rv_mode=None):
    """A brute force computation of the marginal array.

    The parameter array tells which elements of the joint pmf must be summed
    to yield each element of the marginal distribution specified by `indexes`.

    This is more general than AbstractDenseDistribution since it allows the
    alphabet to vary with each random variable. It still requires a Cartestian
    product sample space, however.

    TODO: Expand this to construct arrays for coalescings as well.

    """
    from dit.helpers import parse_rvs, RV_MODES

    # We need to filter the indexes for duplicates, etc. So that we can be
    # sure that when we query the joint outcome, we have the right indexes.
    rvs, indexes = parse_rvs(d, rvs, rv_mode, unique=True, sort=True)
    marginal = d.marginal(indexes, rv_mode=RV_MODES.INDICES)

    shape = (len(marginal._sample_space), len(d._sample_space))
    arr = np.zeros(shape, dtype=bool)

    mindex = marginal._sample_space.index
    for i, joint_outcome in enumerate(d._sample_space):
        outcome = [joint_outcome[j] for j in indexes]
        outcome = marginal._outcome_ctor(outcome)
        idx = mindex(outcome)
        arr[idx, i] = 1

    # Now we need to turn this into a sparse matrix where there are only as
    # many columns as there are nonzero elements in each row.

    # Apply nonzero, use [1] to get only the columns
    nz = np.nonzero(arr)[1]
    n_rows = len(marginal._sample_space)
    arr = nz.reshape((n_rows, len(nz) // n_rows))

    return arr

def get_abstract_dist(dist):
    """
    Returns an abstract representation of the distribution.

    For now, it hacks in a way to deal with non-homogeneous Cartesian product
    sample spaces.

    """
    if dist.is_homogeneous():
        n_variables = dist.outcome_length()
        n_symbols = len(dist.alphabet[0])
        d = AbstractDenseDistribution(n_variables, n_symbols)
    else:
        class D(object):
            n_variables = dist.outcome_length()
            n_elements = np.prod(list(map(len, dist.alphabet)))
            def parameter_array(self, indexes, cache=None):
                return brute_marginal_array(dist, indexes, rv_mode='indexes')
        d = D()

    return d
