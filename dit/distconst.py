"""
Specialized distribution constructors.
"""

import contextlib
from collections import defaultdict
from itertools import combinations, permutations, product
from math import comb
from random import randint

import numpy as np

from .distribution import Distribution
from .exceptions import ditException
from .helpers import normalize_rvs, parse_rvs
from .utils import digits, powerset
from .utils.deprecation import renamed_kwarg
from .validate import validate_pmf

__all__ = (
    "mixture_distribution",
    "mixture_distribution2",
    "noisy",
    "erasure",
    "modify_outcomes",
    "random_scalar_distribution",
    "random_distribution",
    "simplex_grid",
    "uniform_distribution",
    "uniform_scalar_distribution",
    "insert_rvf",
    "RVFunctions",
    "product_distribution",
    "kirkwood_approximation",
    "lancaster_approximation",
    "fejer_riesz_approximation",
    "uniform",
    "uniform_like",
    "all_dist_structures",
    "random_dist_structure",
    "DistributionEnumerator",
)


def mixture_distribution(dists, weights, merge=False):
    """
    Create a mixture distribution: :math:`\\sum p_i d_i`

    Parameters
    ----------
    dists: [Distribution]
        List of distributions to mix.  Each distribution is assumed to have
        the same base and sample space.

    weights: [float]
        List of weights to use while mixing `dists`.  The weights are assumed
        to be probability represented in the base of the distributions.

    merge: bool
        If `True` then distributions will be mixed even if they do not share
        the same sample space. The idea is that each of the input distributions
        is reinterpreted on a common, merged sample space. If `False`, then
        an exception will be raised if incompatible distributions are mixed.

    Returns
    -------
    mix: Distribution
        The mixture distribution.

    Raises
    ------
    DitException
        Raised if there `dists` and `weights` have unequal lengths.
    InvalidNormalization
        Raised if the weights do not sum to unity.
    InvalidProbability
        Raised if the weights are not valid probabilities.
    IncompatibleOutcome
        Raised if the sample spaces for each distribution are not compatible.

    """
    weights = np.asarray(weights)
    if len(dists) != len(weights):
        msg = "Length of `dists` and `weights` must be equal."
        raise ditException(msg)

    ops = dists[0].ops
    validate_pmf(weights, ops)

    if merge:
        vals = lambda o: [(ops.mult(w, d[o]) if o in d else 0) for w, d in zip(weights, dists, strict=True)]
    else:
        vals = lambda o: [ops.mult(w, d[o]) for w, d in zip(weights, dists, strict=True)]

    outcomes = set().union(*[d.outcomes for d in dists])
    pmf = [ops.add_reduce(np.array(vals(o))) for o in outcomes]
    mix = dists[0].__class__(tuple(outcomes), pmf, base=ops.get_base())
    return mix


def mixture_distribution2(dists, weights):
    """
    Create a mixture distribution: :math:`\\sum p_i d_i`

    This version assumes that the pmf for each distribution is of the same
    form, and as a result, will be faster than `mixture_distribution`.
    Explicitly, it assumes that the sample space is ordered exactly the same
    for each distribution and that the outcomes currently represented in the
    pmf are the same as well. Using it in any other case will result in
    incorrect output or an exception.

    Parameters
    ----------
    dists: [Distribution]
        List of distributions to mix.  Each distribution is assumed to have
        the same base and sample space.

    weights: [float]
        List of weights to use while mixing `dists`.  The weights are assumed
        to be probability represented in the base of the distributions.

    Returns
    -------
    mix: Distribution
        The mixture distribution.

    Raises
    ------
    DitException
        Raised if there `dists` and `weights` have unequal lengths.
    InvalidNormalization
        Raised if the weights do not sum to unity.
    InvalidProbability
        Raised if the weights are not valid probabilities.
    IncompatibleDistribution
        Raised if the sample spaces for each distribution are not compatible.

    """
    weights = np.asarray(weights)
    if len(dists) != len(weights):
        msg = "Length of `dists` and `weights` must be equal."
        raise ditException(msg)

    # Also just quickly make sure that the pmfs have the same length. In
    # general, NumPy should give a value error complaining that it cannot
    # broadcast the smaller array. But if a pmf has length 1, then it can
    # be broadcast. This would make it harder to detect errors.
    shapes = {dist.pmf.shape for dist in dists}
    if len(shapes) != 1:
        raise ValueError("All pmfs must have the same length.")

    ops = dists[0].ops
    validate_pmf(weights, ops)

    pmf = ops.mult(dists[0].pmf, weights[0])
    for dist, weight in zip(dists[1:], weights[1:], strict=True):
        pmf = ops.add(pmf, ops.mult(dist.pmf, weight))
    mix = dists[0].copy()
    mix.pmf = pmf
    return mix


def noisy(dist, noise=1 / 2):
    """
    Construct a noisy version of `dist`.

    Parameters
    ----------
    dist : Distribution
        The distribution to fuzz.
    noise : float, 0 <= `noise` <= 1
        The noise level.

    Returns
    -------
    fuzzy: Distribution
        The noisy distribution.
    """
    fuzz = uniform(list(product(*dist.alphabet)))
    fuzzy = mixture_distribution([dist, fuzz], [1 - noise, noise], merge=True)
    return fuzzy


def erasure(dist, epsilon=1 / 2):
    """
    Construct a version of `dist` where each variable has been passed through
    an erasure channel.

    Parameters
    ----------
    dist : Distribution
        The distribution to fuzz.
    epsilon : float, 0 <= `epsilon` <= 1
        The erasure probability.

    Returns
    -------
    erased : Distribution
        The erased distribution.
    """
    outcomes = defaultdict(float)
    n = dist.outcome_length()

    for outcome, prob in dist.zipped():
        for o in product(*zip(outcome, ("_",) * n, strict=True)):
            count = o.count("_")
            outcomes[tuple(o)] += prob * epsilon**count * (1 - epsilon) ** (n - count)

    return Distribution(outcomes)


def modify_outcomes(dist, ctor):
    """
    Returns `dist` but with modified outcomes, after passing them to `ctor`.

    Parameters
    ----------
    dist : Distribution, Distribution
        The distribution to be modified.

    ctor : callable
        The constructor that receives an existing outcome and returns a new
        modified outcome.

    Returns
    -------
    d : Distribution, Distribution
        The modified distribution.

    Examples
    --------
    Convert joint tuple outcomes to strings.
    >>> d = dit.uniform_distribution(3, ['0', '1'])
    >>> d2 = dit.modify_outcomes(d, lambda x: ''.join(x))

    Increment scalar outcomes by 1.
    >>> d = dit.uniform_scalar_distribution(5)
    >>> d2 = dit.modify_outcomes(d, lambda x: x + 1)
    """
    outcomes = tuple(map(ctor, dist.outcomes))
    ops = dist.ops
    newdist = {}
    for outcome, p in zip(outcomes, dist.pmf, strict=True):
        newdist[outcome] = ops.add(p, newdist.get(outcome, ops.zero))
    outcomes = list(newdist.keys())
    pmf = np.array(list(newdist.values()))
    d = dist.__class__(outcomes, pmf, base=dist.get_base())
    return d


@renamed_kwarg("prng", "rng")
def random_scalar_distribution(n, base=None, alpha=None, rng=None):
    """
    Returns a random scalar distribution over `n` outcomes.

    The distribution is sampled uniformly over the space of distributions on
    the `n`-simplex. If `alpha` is not `None`, then the distribution is
    sampled from the Dirichlet distribution with parameter `alpha`.

    Parameters
    ----------
    n : int | list
        The number of outcomes, or a list containing the outcomes.

    base : float, 'linear', 'e'
        The desired base for the distribution probabilities.

    alpha : list | None
        The concentration parameters defining that the Dirichlet distribution
        used to draw the random distribution. If `None`, then each of the
        concentration parameters are set equal to 1.

    """
    import dit.math

    rng = dit.math.as_random_state(rng)

    try:
        nOutcomes = len(n)
    except TypeError:
        nOutcomes = n

    d = uniform_scalar_distribution(n)
    if alpha is None:
        alpha = np.ones(len(d))
    elif len(alpha) != nOutcomes:
        raise ditException("Number of concentration parameters must be `n`.")

    pmf = rng.dirichlet(alpha)
    d.pmf = pmf

    # Maybe we should use ditParams['base'] when base is None?
    if base is not None:
        d.set_base(base)

    return d


@renamed_kwarg("prng", "rng")
def random_distribution(outcome_length, alphabet_size, base=None, alpha=None, rng=None):
    """
    Returns a random distribution drawn uniformly from the simplex.

    The distribution is sampled uniformly over the space of distributions on
    the `n`-simplex, where `n` is equal to `alphabet_size**outcome_length`.
    If `alpha` is not `None`, then the distribution is sampled from the
    Dirichlet distribution with parameter `alpha`.

    Parameters
    ----------
    outcome_length : int
        The length of the outcomes.

    alphabet_size : int, list
        The alphabet used to construct the outcomes of the distribution. If an
        integer, then the alphabet will consist of integers from 0 to k-1 where
        k is the alphabet size.  If a list, then the elements are used as the
        alphabet.

    base : float, 'linear', 'e'
        The desired base for the distribution probabilities.

    alpha : list | None
        The concentration parameters defining that the Dirichlet distribution
        used to draw the random distribution. If `None`, then each of the
        concentration parameters are set equal to 1.

    Returns
    -------
    d : Distribution.
        A uniform sampled distribution.

    """
    import dit.math

    rng = dit.math.as_random_state(rng)

    d = uniform_distribution(outcome_length, alphabet_size)

    if alpha is None:
        alpha = np.ones(len(d))
    elif len(alpha) != len(d):
        raise ditException("Invalid number of concentration parameters.")

    pmf = rng.dirichlet(alpha)
    d.pmf = pmf

    # Maybe we should use ditParams['base'] when base is None?
    if base is not None:
        d.set_base(base)

    return d


def simplex_grid(length, subdivisions, using=None, inplace=False):
    """Returns a generator over distributions, determined by a grid.

    The grid is "triangular" in Euclidean space.

    The total number of points on the grid is::

        (subdivisions + length - 1)! / (subdivisions)! / (length-1)!

    and is equivalent to the total number of ways ``n`` indistinguishable items
    can be placed into ``k`` distinguishable slots, where n=`subdivisions` and
    k=`length`.

    Parameters
    ----------
    length : int
        The number of elements in each distribution. The dimensionality
        of the simplex is length-1.
    subdivisions : int
        The number of subdivisions for the interval [0, 1]. Each component
        will take on values at the boundaries of the subdivisions. For example,
        one subdivision means each component can take the values 0 or 1 only.
        Two subdivisions corresponds to :math:`[[0, 1/2], [1/2, 1]]` and thus,
        each component can take the values 0, 1/2, or 1. A common use case is
        to exponentially increase the number of subdivisions at each level.
        That is, subdivisions would be: 2**0, 2**1, 2**2, 2**3, ...
    using : None, callable, or distribution
        If None, then scalar distributions on integers are yielded. If `using`
        is a distribution, then each yielded distribution is a copy of `using`
        with its pmf set appropriately. For other callables, a tuple of the
        pmf is passed to the callable and then yielded.
    inplace : bool
        If `True`, then each yielded distribution is the same Python object,
        but with a new probability mass function. If `False`, then each yielded
        distribution is a unique Python object and can be safely stored for
        other calculations after the generator has finished. This keyword has
        an effect only when `using` is None or some distribution.

    Examples
    --------
    >>> list(dit.simplex_grid(2, 2, using=tuple))
    [(0.0, 1.0), (0.25, 0.75), (0.5, 0.5), (0.75, 0.25), (1.0, 0.0)]

    """
    from dit.math.combinatorics import slots

    if subdivisions < 1:
        raise ditException("`subdivisions` must be greater than or equal to 1")
    elif length < 1:
        raise ditException("`length` must be greater than or equal to 1")

    gen = slots(int(subdivisions), int(length), normalized=True)

    if using is None:
        using = random_scalar_distribution(length)

    if using is tuple:
        for pmf in gen:
            yield pmf
    elif not isinstance(using, Distribution):
        for pmf in gen:
            yield using(pmf)
    else:
        using.make_dense()
        if length != len(using.pmf):
            raise Exception("`length` must match the length of pmf")

        if inplace:
            d = using
            for pmf in gen:
                d.pmf = pmf
                yield d
        else:
            for pmf in gen:
                d = using.copy()
                d.pmf = pmf
                yield d


def uniform_scalar_distribution(n, base=None):
    """
    Returns a uniform scalar distribution over `n` outcomes.

    Parameters
    ----------
    n : int, list
        If an integer, then the outcomes are integers from 0 to n-1. If a list
        then the elements are treated as the outcomes.

    Returns
    -------
    d : Distribution
        A uniform scalar distribution.

    """
    try:
        nOutcomes = len(n)
        outcomes = n
    except TypeError:
        nOutcomes = n
        outcomes = tuple(range(n))

    pmf = [1 / nOutcomes] * nOutcomes
    d = Distribution(outcomes, pmf, base="linear")

    # Maybe we should use ditParams['base'] when base is None?
    if base is not None:
        d.set_base(base)

    return d


def uniform_distribution(outcome_length, alphabet_size, base=None):
    """
    Returns a uniform distribution.

    Parameters
    ----------
    outcome_length : int
        The length of the outcomes.

    alphabet_size : int, list of lists
        The alphabets used to construct the outcomes of the distribution. If an
        integer, then the alphabet for each random variable will be the same,
        consisting of integers from 0 to k-1 where k is the alphabet size.
        If a list, then the elements are used as the alphabet for each random
        variable.  If the list has a single element, then it will be used
        as the alphabet for each random variable.

    base : float, 'linear', 'e'
        The desired base for the distribution probabilities.

    Returns
    -------
    d : Distribution.
        A uniform distribution.

    Examples
    --------
    Each random variable has the same standardized alphabet: [0,1]
    >>> d = dit.uniform_distribution(2, 2)

    Each random variable has its own alphabet.
    >>> d = dit.uniform_distribution(2, [[0,1],[1,2]])

    Both random variables have ['H','T'] as an alphabet.
    >>> d = dit.uniform_distribution(2, [['H','T']])

    """
    try:
        int(alphabet_size)
    except TypeError as err:
        # Assume it is a list of lists.
        alphabet = alphabet_size

        # Autoextend if only one alphabet is provided.
        if len(alphabet) == 1:
            alphabet = [alphabet[0]] * outcome_length
        elif len(alphabet) != outcome_length:
            raise TypeError("outcome_length does not match number of rvs.") from err
    else:
        # Build the standard alphabet.
        alphabet = [tuple(range(alphabet_size))] * outcome_length

    try:
        Z = np.prod(list(map(len, alphabet)))
        with contextlib.suppress(Exception):
            # for some reason numpypy.prod returns a list, and pypy can't handle
            #   multiplying a list by a numpy float.
            Z = int(Z[0])
    except TypeError as err:
        raise TypeError("alphabet_size must be an int or list of lists.") from err

    pmf = [1 / Z] * Z
    outcomes = tuple(product(*alphabet))
    d = Distribution(outcomes, pmf, base="linear")

    # Maybe we should use ditParams['base'] when base is None?
    if base is not None:
        d.set_base(base)

    return d


def uniform_like(dist):
    """
    Returns a uniform distribution with the same outcome length, alphabet size, and base as `dist`.

    Parameters
    ----------
    dist : Distribution
        The distribution to mimic.
    """
    outcome_length = dist.outcome_length()
    alphabet_size = dist.alphabet
    base = dist.get_base()
    return uniform_distribution(outcome_length, alphabet_size, base)


def uniform(outcomes, base=None):
    """
    Produces a uniform distribution over `outcomes`.

    Parameters
    ----------
    outcomes : iterable
        The set of outcomes with which to construct the distribution.

    base : float, 'linear', 'e'
        The desired base for the distribution probabilities.

    Returns
    -------
    d : Distribution
        A uniform distribution over `outcomes`.

    Raises
    ------
    ditException
        Raised if the elements of `outcomes` do not all have the same length.
    TypeError
        Raised if `outcomes` is not iterable.
    """
    outcomes = list(outcomes)
    length = len(outcomes)
    pmf = [1 / length] * length
    d = Distribution(outcomes, pmf)

    # Maybe we should use ditParams['base'] when base is None?
    if base is not None:
        d.set_base(base)

    return d


def insert_rvf(d, func, index=-1):
    """
    Returns a new distribution with an added random variable at index `index`.

    The new random variable must be a function of the other random variables.
    By this, we mean that the entropy of the new random variable conditioned
    on the original random variables should be zero.

    Parameters
    ----------
    dist : Distribution
        The distribution used to construct the new distribution.
    func : callable | list of callable
        A function which takes a single argument---the value of the previous
        random variables---and returns a new random variable. Note, the return
        value will be added to the outcome using `__add__`, and so it should be
        a hashable, orderable sequence (as every outcome must be). If a list of
        callables is provided, then multiple random variables are added
        simultaneously and will appear in the same order as the list.
    index : int
        The index at which to insert the random variable. A value of -1 is
        will append the random variable to the end.

    Returns
    -------
    d : Distribution
        The new distribution.

    Examples
    --------
    >>> d = dit.Distribution(['00', '01', '10', '11'], [1/4]*4)
    >>> def xor(outcome):
    ...    return str(int(outcome[0] != outcome[1]))
    ...
    >>> d2 = dit.insert_rvf(d, xor)
    >>> d.outcomes
    ('000', '011', '101', '110')

    """
    try:
        func[0]
    except TypeError:
        funcs = [func]
    else:
        funcs = func

    partial_outcomes = [map(func, d.outcomes) for func in funcs]

    # Now "flatten" the new contributions.
    partial_outcomes = [
        d._outcome_ctor([o for o_list in outcome for o in o_list]) for outcome in zip(*partial_outcomes, strict=True)
    ]

    new_outcomes = zip(d.outcomes, partial_outcomes, strict=True)
    if index == -1:
        outcomes = [old + new for old, new in new_outcomes]
    else:
        outcomes = [old[:index] + new + old[index:] for old, new in new_outcomes]

    d2 = Distribution(outcomes, d.pmf.copy(), base=d.get_base())
    return d2


class RVFunctions:
    """
    Helper class for building new random variables.

    Each new random variable is a function of the existing random variables.
    So for each outcome in the original distribution, there can be only one
    possible value for the new random variable.

    Some methods may make assumptions about the sample space. For example, the
    :meth:`xor` method assumes the sample space consists of 0-like and 1-like
    outcomes.
    """

    def __init__(self, d):
        """
        Initialize the random variable function creator.

        Parameters
        ----------
        d : Distribution
            The distribution used to create the new random variables.

        Examples
        --------
        >>> d = dit.Distribution(['00', '01', '10', '11'], [1/4]*4)
        >>> bf = dit.RVFunctions(d)
        >>> d = dit.insert_rvf(d, bf.xor([0,1]))
        >>> d = dit.insert_rvf(d, bf.xor([1,2]))
        >>> d.outcomes
        ('0000', '0110', '1011', '1101')
        """
        if not isinstance(d, (Distribution,)):
            raise ditException("`d` must be a Distribution instance.")
        if not d.is_joint():
            raise ditException("`d` must be a joint distribution.")
        first = d.outcomes[0]
        if isinstance(first, tuple):
            is_int = len(first) > 0 and isinstance(first[0], (int, float, np.integer, np.floating))
        else:
            try:
                first + ""
                is_int = False
            except TypeError:
                is_int = True

        self.is_int = is_int
        self.L = d.outcome_length()
        self.ctor = d._outcome_ctor
        self.outcome_class = d._outcome_class

    def xor(self, indices):
        """
        Returns a callable which returns the logical XOR of the given indices.

        Outcomes are assumed to be strings of '0' and '1', or tuples of 0 and 1.
        The returned function handles both cases appropriately.

        Parameters
        ----------
        indices : list
            A list of two indices used to take the XOR.

        Returns
        -------
        func : function
            A callable implementing the XOR function. It receives a single
            argument, the outcome and returns an outcome for the calculation.

        Examples
        --------
        >>> d = dit.Distribution(['00', '01', '10', '11'], [1/4]*4)
        >>> bf = dit.RVFunctions(d)
        >>> d = dit.insert_rvf(d, bf.xor([0,1]))
        >>> d.outcomes
        ('000', '011', '101', '110')
        """
        if self.is_int:

            def func(outcome):
                result = outcome[indices[0]] != outcome[indices[1]]
                return (int(result),)
        else:

            def func(outcome):
                result = outcome[indices[0]] != outcome[indices[1]]
                return str(int(result))

        return func

    def from_mapping(self, mapping, force=True):
        """
        Returns a callable implementing a random variable via a mapping.

        Parameters
        ----------
        mapping : dict
            A mapping from outcomes to values of the new random variable.

        force : bool
            Ideally, the values of `mapping` should be satisfy the requirements
            of all outcomes (hashable, ordered sequences), but if `force` is
            `True`, we will attempt to use the distribution's outcome
            constructor and make sure that they are. If they are not, then
            the outcomes will be placed into a 1-tuple. This is strictly
            a convenience for users. As an example, suppose the outcomes are
            strings, the values of `mapping` can also be strings without issue.
            However, if the outcomes are tuples of integers, then the values
            *should* also be tuples. When `force` is `True`, then the values
            can be integers and then they will be transformed into 1-tuples.

        Returns
        -------
        func : function
            A callable implementing the desired function. It receives a single
            argument, the outcome, and returns an outcome for the calculation.

        Examples
        --------
        >>> d = dit.Distribution(['00', '01', '10', '11'], [1/4]*4)
        >>> bf = dit.RVFunctions(d)
        >>> mapping = {'00': '0', '01': '1', '10': '1', '11': '0'}
        >>> d = dit.insert_rvf(d, bf.from_mapping(mapping))
        >>> d.outcomes
        ('000', '011', '101', '110')

        Same example as above but now with tuples.

        >>> d = dit.Distribution([(0,0), (0,1), (1,0), (1,1)], [1/4]*4)
        >>> bf = dit.RVFunctions(d)
        >>> mapping = {(0,0): 0, (0,1): 1, (1,0): 1, (1,1): 0}
        >>> d = dit.insert_rvf(d, bf.from_mapping(mapping, force=True))
        >>> d.outcomes
        ((0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0))

        See Also
        --------
        dit.modify_outcomes
        """
        ctor = self.ctor
        if force:
            try:
                list(map(ctor, mapping.values()))
            except (TypeError, ditException):
                values = [ctor([o]) for o in mapping.values()]
                mapping = dict(zip(mapping.keys(), values, strict=True))

        def func(outcome):
            return mapping[outcome]

        return func

    def from_partition(self, partition):
        """
        Returns a callable implementing a function specified by a partition.

        The partition must divide the sample space of the distribution. The
        number of equivalence classes, n, determines the number of values for
        the random variable. The values are integers from 0 to n-1, but if the
        outcome class of the distribution is string, then this function will
        use the first n letters of
        ``'0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'``
        as values for the random variable. So random variables with more than
        62 outcomes are not supported by this function.

        Parameters
        ----------
        partition : list
            A list of iterables. The outer list is required to determine the
            order of the new random variable

        Returns
        -------
        func : function
            A callable implementing the desired function. It receives a single
            argument, the outcome, and returns an outcome of the new random
            variable that is specified by the partition.

        Examples
        --------
        >>> d = dit.Distribution(['00', '01', '10', '11'], [1/4]*4)
        >>> bf = dit.RVFunctions(d)
        >>> partition = (('00','11'), ('01', '10'))
        >>> d = dit.insert_rvf(d, bf.from_partition(partition))
        >>> d.outcomes
        ('000', '011', '101', '110')
        """
        # Practically, we support the str class. This is bytes in Python
        # versions <3 and unicode >3.
        alphabet = "0123456789"
        letters = "abcdefghijklmnopqrstuvwxyz"
        alphabet += letters
        alphabet += letters.upper()

        n = len(partition)
        vals = range(n)

        mapping = {}
        # Probably could do this more efficiently.
        for i, eqclass in enumerate(partition):
            for outcome in eqclass:
                mapping[self.ctor(outcome)] = vals[i]

        return self.from_mapping(mapping, force=True)

    def from_hexes(self, hexes):
        """
        Returns a callable implementing a boolean function on up to 4-bits.

        Outcomes are assumed to be strings of '0' and '1', or tuples of 0 and 1.
        The returned function handles both cases appropriately.

        The original outcomes are represented in base-16 as one of the letters
        in '0123456789ABCDEF' (not case sensitive). Then, each boolean function
        is a specification of the outcomes for which it be should true. The
        random variable will be false for the complement of this set---so this
        function additional assumes full support. For example, if the random
        variable is a function of 3-bits and should be true only for the
        outcomes 2='010' or 7='111', then `hexes` should be '27'. This nicely
        handles 1- and 2-, and 4-bit inputs in a similar fashion.

        Parameters
        ----------
        hexes : str
            A string of base-16 characters, each element represents an
            (up to) 4-bit outcome for which the random variable should be true.

        Returns
        -------
        func : function
            A callable implementing the desired function. It receives a single
            argument, the outcome, and returns an outcome for the calculation.

        Examples
        --------
        >>> outcomes = ['000', '001', '010', '011', '100', '101', '110', '111']
        >>> pmf = [1/8] * 8
        >>> d = dit.Distribution(outcomes, pmf)
        >>> bf = dit.RVFunctions(d)
        >>> d = dit.insert_rvf(d, bf.from_hexes('27'))
        >>> d.outcomes
        ('0000', '0010', '0101', '0110', '1000', '1010', '1100', '1111')
        """
        base = 16
        template = "{0:0{1}b}"
        outcomes = [template.format(int(h, base), self.L) for h in hexes]
        outcomes = [tuple(map(int, o)) for o in outcomes] if self.is_int else [tuple(o) for o in outcomes]
        outcomes = set(outcomes)

        def func(outcome):
            result = tuple(outcome) in outcomes
            return (int(result),)

        return func


def product_distribution(dist, rvs=None, base=None):
    """
    Returns a new distribution which is the product of marginals.

    Parameters
    ----------
    dist : distribution
        The original distribution.
    rvs : sequence
        A sequence whose elements are also sequences.  Each inner sequence
        defines the marginal distribution used to create the new distribution.
    base : float, 'linear', 'e'
        The desired base for the distribution probabilities.

    Returns
    -------
    d : Distribution
        The product distribution.
    """
    if not dist.is_joint():
        raise Exception("A joint distribution is required.")

    if rvs is None:
        names = dist.get_rv_names()
        if names is None:
            names = range(dist.outcome_length())

        indexes = [[i] for i in names]

    else:
        parse = lambda rv: parse_rvs(dist, rv, unique=True, sort=False)[1]
        indexes = [parse(rv) for rv in rvs]

    all_indexes = [idx for index_list in indexes for idx in index_list]
    if len(all_indexes) != len(set(all_indexes)):
        raise Exception("The elements of `rvs` have nonzero intersection.")

    marginals = [dist.marginal(index_list) for index_list in indexes]
    ctor = dist._outcome_ctor

    outcomes = []
    pmf = []
    for pairs in product(*[marg.zipped() for marg in marginals]):
        outcome = []
        prob = []
        for pair in pairs:
            value = pair[0]
            if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
                outcome.append(value)
            else:
                outcome.extend(value)
            prob.append(pair[1])
        outcomes.append(ctor(outcome))
        # `zipped` yields linear probabilities regardless of the dist's base.
        pmf.append(np.prod(prob))

    d = Distribution(outcomes, pmf, validate=False)
    d.set_base(base if base is not None else dist.get_base())

    return d


def _joint_array(dist, rvs, crvs):
    """
    Coalesce `dist` into a dense array with one axis per element of `rvs`
    followed by a single axis for `crvs` (of size one when unconditioned).
    """
    groups = list(rvs) + ([crvs] if crvs else [])
    d = dist.coalesce(groups)
    d.make_dense()
    pmf = d.pmf.reshape([len(a) for a in d.alphabet])
    if not crvs:
        pmf = pmf[..., np.newaxis]
    return pmf


def _interaction_coefficients(n, order):
    """
    The coefficients c_S = (-1)^{k-|S|} C(n-|S|-1, k-|S|) for |S| <= k of the
    order-k truncated interaction expansion of `n` variables, with k
    defaulting to n - 1.
    """
    k = n - 1 if order is None else order
    if not 0 <= k <= n - 1:
        msg = f"order must satisfy 0 <= order <= {n - 1}, {k} given."
        raise ditException(msg)

    return {S: (-1) ** (k - j) * comb(n - j - 1, k - j) for j in range(k + 1) for S in combinations(range(n), j)}


def _log_linear_product(pmf, exponents):
    """
    Compute log2 prod_S p(x_S | z)^{c_S}, which is -inf wherever a marginal
    used by the product vanishes.

    Parameters
    ----------
    pmf : np.ndarray
        Joint pmf with one axis per variable and a final conditioning axis.
    exponents : dict
        Maps tuples of variable axes S to the exponent c_S.

    Returns
    -------
    log_q : np.ndarray
        The unnormalized log product, with the same shape as `pmf`.
    """
    axes = tuple(range(pmf.ndim - 1))
    p_z = pmf.sum(axis=axes, keepdims=True)
    safe_p_z = np.where(p_z > 0, p_z, 1.0)

    log_q = np.zeros(pmf.shape)
    zero = np.zeros(pmf.shape, dtype=bool)
    for subset, exponent in exponents.items():
        others = tuple(i for i in axes if i not in subset)
        marginal = pmf.sum(axis=others, keepdims=True)
        zero |= np.broadcast_to(marginal == 0, pmf.shape)
        log_q += exponent * np.log2(np.where(marginal > 0, marginal, 1.0) / safe_p_z)

    return np.where(zero, -np.inf, log_q)


def _approximation_distribution(dist, rvs, values):
    """
    Wrap `values`, an array with one axis per element of `rvs`, as a dense,
    unvalidated distribution over the outcomes of `dist` coalesced to `rvs`.
    """
    d = dist.coalesce(rvs)
    d.make_dense()
    q = Distribution(list(d.outcomes), values.ravel(), validate=False)
    q.make_dense()
    return q


def kirkwood_approximation(dist, rvs=None, order=None):
    r"""
    Construct the unnormalized generalized Kirkwood superposition
    approximation of a distribution.

    For variables :math:`X_0, \ldots, X_{n-1}` the order-:math:`k`
    approximation truncates the Möbius (interaction) expansion of
    :math:`\log p` at subsets of size at most :math:`k`
    :cite:`watanabe1960information,killian2007extraction`:

    .. math::
        \tilde{p}_k(x) = \prod_{1 \leq |S| \leq k} p(x_S)^{c_S}, \qquad
        c_S = (-1)^{k-|S|} \binom{n-|S|-1}{k-|S|}.

    The default order :math:`k = n-1` is the standard Kirkwood approximation,
    e.g. :math:`p(x, y) p(x, z) p(y, z) / \left(p(x) p(y) p(z)\right)` for three
    variables. Order 1 is the product of marginals and order 0 is constant
    one. The approximation is not normalized: with :math:`Z` its total mass,
    :func:`~dit.multivariate.kirkwood_mutual_information` is
    :math:`D_{KL}(p \| \tilde{p}_k / Z)`. Wherever a marginal used by the
    product vanishes, the approximation is zero. See
    :func:`lancaster_approximation` for the additive analogue.

    Parameters
    ----------
    dist : Distribution
        The distribution to approximate.
    rvs : list, None
        A list of lists. Each inner list specifies the indexes of the random
        variables forming one variable of the approximation. If None, then
        each random variable is its own variable.
    order : int, None
        The largest marginal size k used by the approximation. Must satisfy
        0 <= k <= n - 1. If None, defaults to n - 1.

    Returns
    -------
    approx : Distribution
        The dense, unnormalized approximation over the outcomes of `dist`
        coalesced to `rvs`.

    Raises
    ------
    ditException
        Raised if `order` is out of range.

    Examples
    --------
    >>> d = dit.example_dists.giant_bit(3, 2)
    >>> dit.kirkwood_approximation(d).pmf
    array([1., 0., 0., 0., 0., 0., 0., 1.])
    """
    rvs, _ = normalize_rvs(dist, rvs, None)
    exponents = _interaction_coefficients(len(rvs), order)
    log_q = _log_linear_product(_joint_array(dist, rvs, []), exponents)
    return _approximation_distribution(dist, rvs, np.exp2(log_q[..., 0]))


def lancaster_approximation(dist, rvs=None, order=None):
    r"""
    Construct the order-k Lancaster approximation of a distribution, the
    additive analogue of the Kirkwood approximation.

    Lancaster's expansion of :math:`X_0, \ldots, X_{n-1}` writes the joint
    distribution relative to the product of its marginals
    :cite:`lancaster1969chi`:

    .. math::
        p(x) = \prod_i p(x_i) \sum_{T} \delta_T(x_T), \qquad
        \delta_T(x_T) = \sum_{S \subseteq T} (-1)^{|T|-|S|}
                        \frac{p(x_S)}{\prod_{i \in S} p(x_i)},

    where :math:`\delta_\emptyset = 1` and :math:`\delta_{\{i\}} = 0`.
    :math:`\prod_i p(x_i) \, \delta_{\{0, \ldots, n-1\}}` is the Lancaster
    interaction measure :math:`\Delta_L p = \prod_i (p^*_{X_i} - p_{X_i})`
    :cite:`sejdinovic2013kernel`. Truncating the expansion at
    :math:`|T| \leq k` gives

    .. math::
        p_k(x) = \sum_{|S| \leq k} c_S \, p(x_S) \prod_{i \notin S} p(x_i),
        \qquad c_S = (-1)^{k-|S|} \binom{n-|S|-1}{k-|S|},

    with the same coefficients as :func:`kirkwood_approximation`, but
    combined arithmetically rather than geometrically. The default order
    :math:`k = n-1` is :math:`p - \Delta_L p`, e.g. for three variables

    .. math::
        p_2(x, y, z) = p(x, y) p(z) + p(x, z) p(y) + p(y, z) p(x) - 2 p(x) p(y) p(z).

    Orders 0 and 1 are both the product of marginals. Every order sums to one,
    and its marginals of size at most k agree with those of :math:`p`, but it
    may be negative. For :math:`n \leq 3`, the default order reproduces
    :math:`p` whenever :math:`p` factors nontrivially, although the converse
    fails. For :math:`n > 3` even that implication fails, which motivates
    Streitberg's correction :cite:`streitberg1990lancaster`. No canonical
    literature source is known for orders below :math:`n-1`.

    Parameters
    ----------
    dist : Distribution
        The distribution to approximate.
    rvs : list, None
        A list of lists. Each inner list specifies the indexes of the random
        variables forming one variable of the approximation. If None, then
        each random variable is its own variable.
    order : int, None
        The largest interaction size k kept by the approximation. Must satisfy
        0 <= k <= n - 1. If None, defaults to n - 1.

    Returns
    -------
    approx : Distribution
        The dense, possibly signed, approximation over the outcomes of `dist`
        coalesced to `rvs`.

    Raises
    ------
    ditException
        Raised if `order` is out of range.

    Examples
    --------
    >>> d = dit.Distribution(['000', '010', '100', '111'], [1/4] * 4)
    >>> dit.lancaster_approximation(d).pmf
    array([ 0.3125, -0.0625,  0.1875,  0.0625,  0.1875,  0.0625,  0.0625,  0.1875])
    """
    rvs, _ = normalize_rvs(dist, rvs, None)
    coefficients = _interaction_coefficients(len(rvs), order)
    pmf = _joint_array(dist, rvs, [])[..., 0]
    axes = tuple(range(pmf.ndim))

    def marginal(S):
        return pmf.sum(axis=tuple(i for i in axes if i not in S), keepdims=True)

    singles = [marginal((i,)) for i in axes]
    approx = np.zeros(pmf.shape)
    for S, c in coefficients.items():
        term = marginal(S)
        for i in axes:
            if i not in S:
                term = term * singles[i]
        approx += c * term

    return _approximation_distribution(dist, rvs, approx)


def fejer_riesz_approximation(dist, rvs=None, order=None):
    r"""
    Construct the order-k Fejér-Riesz approximation of a distribution, a
    nonnegative analogue of the Lancaster approximation.

    The Lancaster approximation truncates the interaction expansion of the
    density ratio :math:`p / \pi`, where :math:`\pi(x) = \prod_i p(x_i)`, and
    so may be negative. In the spirit of the Fejér-Riesz theorem, which
    writes a nonnegative trigonometric polynomial as :math:`|h|^2`
    :cite:`fejer1916trigonometrische`, this approximation instead truncates
    the amplitude :math:`s = \sqrt{p / \pi}` and squares the result:

    .. math::
        s_k(x) = \sum_{|S| \leq k} c_S \, \mathbb{E}_\pi[s \mid x_S], \qquad
        q_k(x) = \frac{\pi(x) \, s_k(x)^2}{Z_k},

    with the coefficients :math:`c_S` of :func:`lancaster_approximation`.
    :math:`s_k` is the :math:`L^2(\pi)` projection of :math:`s` onto
    functions of at most :math:`k` variables :cite:`lancaster1969chi`, so
    :math:`Z_k = \|s_k\|^2_\pi \leq \|s\|^2_\pi = 1`. Order 0 is the product
    of marginals, and :math:`q_k = p` at every order when :math:`p` is
    itself a product of marginals. Unlike the Lancaster approximation,
    :math:`q_k` is a probability distribution, but it need not match the
    marginals of :math:`p`. No canonical literature source is known for
    this construction.

    Parameters
    ----------
    dist : Distribution
        The distribution to approximate.
    rvs : list, None
        A list of lists. Each inner list specifies the indexes of the random
        variables forming one variable of the approximation. If None, then
        each random variable is its own variable.
    order : int, None
        The largest interaction size k kept in the amplitude. Must satisfy
        0 <= k <= n - 1. If None, defaults to n - 1.

    Returns
    -------
    approx : Distribution
        The dense, normalized approximation over the outcomes of `dist`
        coalesced to `rvs`.

    Raises
    ------
    ditException
        Raised if `order` is out of range.

    Examples
    --------
    >>> d = dit.Distribution(['000', '010', '100', '111'], [1/4] * 4)
    >>> dit.fejer_riesz_approximation(d).pmf.round(4)
    array([0.3879, 0.0248, 0.1946, 0.0248, 0.1946, 0.0248, 0.0083, 0.1404])
    """
    rvs, _ = normalize_rvs(dist, rvs, None)
    coefficients = _interaction_coefficients(len(rvs), order)
    pmf = _joint_array(dist, rvs, [])[..., 0]
    axes = tuple(range(pmf.ndim))

    singles = [pmf.sum(axis=tuple(j for j in axes if j != i), keepdims=True) for i in axes]
    product = np.prod(np.broadcast_arrays(*singles), axis=0)
    amplitude = np.sqrt(np.divide(pmf, product, out=np.zeros(pmf.shape), where=product > 0))

    projection = np.zeros(pmf.shape)
    for S, c in coefficients.items():
        others = tuple(i for i in axes if i not in S)
        weighted = amplitude
        for i in others:
            weighted = weighted * singles[i]
        projection = projection + c * weighted.sum(axis=others, keepdims=True)

    approx = product * projection**2
    return _approximation_distribution(dist, rvs, approx / approx.sum())


def all_dist_structures(outcome_length, alphabet_size):
    """
    Return an iterator of distributions over the
    2**(`alphabet_size`**`outcome_length`) possible combinations of joint
    events.

    Parameters
    ----------
    outcome_length : int
        The length of outcomes to consider.
    alphabet_length : int
        The size of the alphabet for each random variable.

    Yields
    ------
    d : Distribution
        A uniform distribution over a subset of the possible joint events.
    """
    alphabet = "".join(str(i) for i in range(alphabet_size))
    words = product(alphabet, repeat=outcome_length)
    topologies = powerset(words)
    next(topologies)  # the first element is the null set
    for t in topologies:
        outcomes = ["".join(_) for _ in t]
        yield uniform(outcomes)


def _int_to_dist(number, outcome_length, alphabet_size):
    """
    Construct the `number`th distribution over `outcome_length` variables each
    with an alphabet of size `alphabet_size`.

    Parameters
    ----------
    number : int
        The index of the distribution to construct.
    outcome_length : int
        The number of random variables in each joint event.
    alphabet_size : int
        The size of the alphabet for each random variable.

    Returns
    -------
    d : Distribution
        A uniform distribution over the joint event specified by the parameters.
    """
    alphabet = "".join(str(i) for i in range(alphabet_size))
    words = ["".join(word) for word in product(alphabet, repeat=outcome_length)]
    events = digits(number, 2, pad=alphabet_size**outcome_length, big_endian=False)
    pmf = [p / sum(events) for p in events]
    return Distribution(words, pmf)


def random_dist_structure(outcome_length, alphabet_size):
    """
    Return a uniform distribution over a random subset of the
    `alphabet_size`**`outcome_length` possible joint events.

    Parameters
    ----------
    outcome_length : int
        The number of random variables in each joint event.
    alphabet_size : int
        The size of the alphabet for each random variable.

    Returns
    -------
    d : Distribution
        A uniform distribution over a random subset of joint events.
    """
    bound = 2 ** (alphabet_size**outcome_length)
    return _int_to_dist(randint(1, bound - 1), outcome_length, alphabet_size)


def _combine_scalar_dists(d1, d2, op):
    """
    Combines `d1` and `d2` according to `op`, as though they are independent.

    Parameters
    ----------
    d1 : Distribution
        The first distribution
    d2 : Distribution
        The second distribution
    op : function
        Function used to combine outcomes

    Returns
    -------
    d : Distribution
        The two distributions combined via `op`
    """
    # `zipped` yields linear probabilities regardless of base.
    dist = defaultdict(float)
    for (o1, p1), (o2, p2) in product(d1.zipped(), d2.zipped()):
        dist[op(o1, o2)] += p1 * p2

    d = Distribution(*zip(*dist.items(), strict=True))
    d.set_base(d1.get_base())
    return d


class DistributionEnumerator:
    """
    This object enumerates all the uniform probability distributions over `n` variables each with cardinality `k`
    """

    def __init__(self, n, k):
        """
        Parameters
        ----------
        n : int
           The number of variables.
        k : int
           The cardinality of each variable.
        """
        self.n = n
        self.k = k

    def __iter__(self):
        """
        Magic function to support usage as an iterator.

        Returns
        -------
        cd : iterator
           Iterator over canonical distributions.
        """
        return self._canonical_distributions()

    def _events_to_int(self, events):
        """
        Convert between a set of events and an associated integer.

        Parameters
        ----------
        events : iterable
           The events.

        Returns
        -------
        i : int
           The associated integer representing `events`.
        """
        return sum(2 ** int("".join(map(str, e)), base=self.k) for e in events)

    def _int_to_events(self, i):
        """
        Convert

        Parameters
        ----------
        i : int
           The integer.

        Returns
        -------
        events : tuple(int)
           The events associated with `i`.
        """
        return [tuple(map(int, e)) for e in _int_to_dist(i, self.n, self.k).outcomes]

    def _all_isomorphisms(self, events):
        """
        Iterate through various isomorphisms of `events`.

        Parameters
        ----------
        events : tuple(int)
           The events to generate isomorphisms of.

        Yields
        ------
        dist : tuple(int)
           Isomorphic forms of `events`.
        """
        cache = set()
        for dist1 in self._variable_permutations(events):
            for dist2 in self._symbol_swaps(dist1):
                tdist2 = tuple(dist2)
                if tdist2 not in cache:
                    cache.add(tdist2)
                    yield dist2

    @staticmethod
    def _variable_permutations(events):
        """
        Permute the order of events in `events`.

        Parameters
        ----------
        events : tuple(int)
           The events to permute.

        Yields
        ------
        permuted : tuple(int)
           One of the permuted forms of `events`.
        """
        for perm in permutations(range(len(events[0]))):
            yield sorted([tuple([e[i] for i in perm]) for e in events])

    @staticmethod
    def _all_mappings(alphabet):
        """
        Iterate through permutations of the alphabet.

        Parameters
        ----------
        alphabet: list[int]
           The distribution's alphabet.

        Yields
        ------
        new_alphabet : dict{int: int}

        """
        for new_alphabet in permutations(alphabet):
            yield {a: b for a, b in zip(alphabet, new_alphabet, strict=True)}

    def _symbol_swaps(self, events):
        """
        Remap the distribution's alphabet.

        Parameters
        ----------
        events : tuple(int)
           The event to construct variants of.

        Yields
        ------
        new_dist : tuple(int)
           An isomorphic distribution with a different alphabet.
        """
        alphabet = list(range(self.k))
        for maps in product(self._all_mappings(alphabet), repeat=len(events[0])):
            yield sorted([tuple([m[var] for m, var in zip(maps, e, strict=True)]) for e in events])

    def _largest_symmetric_subset(self, events):
        """
        Determine how many events are permutations of each other. The purpose of
        this is to prefer highly symmetric forms of a distribution.

        Parameters
        ----------
        events : tuple(int)
           The events to study.

        Returns
        -------
        components : tuple(int)
           Tuple of ordered integers corresponding to how many permuted events
           exist in the distribution.
        """
        isos = {}
        for event in events:
            iso_events = {e[0] for e in self._variable_permutations([event])} & set(events)
            isos[event] = iso_events

        return tuple(sorted(map(len, isos.values()), reverse=True))

    def _canonical(self, isos):
        """
        Determine the canonical form of the equivalent distributions `isos`.
        Among the most symmetric distribution, it is the one lexicographically
        is first.

        Parameters
        ----------
        isos : list[int]
           The int representations of an isomorphic set of distributions.

        Returns
        -------
        cd : tuple(int)
           The canonical element from `isos`.
        """
        ints = {}
        forms = defaultdict(list)
        for iso in isos:
            events = tuple(self._int_to_events(iso))
            ints[events] = iso
            forms[self._largest_symmetric_subset(events)].append(events)

        return ints[tuple(min(forms[max(forms.keys())]))]

    def _isomorphisms(self, i):
        """
        Construct all isomorphisms of the distribution represented by `i`.

        Parameters
        ----------
        i : int
           The integer representation of the event space.

        Returns
        -------
        isos : list[int]
           List of integer representations of all isomorphisms of the `i` distribution.
        """
        events = self._int_to_events(i)
        return [self._events_to_int(new_events) for new_events in self._all_isomorphisms(events)]

    def _canonical_distributions(self):
        """
        Enumerate all the non-isomorphic distributions with `self.n` states and `self.k` alphabet size.

        Yields
        ------
        dist : Distribution
           The non-isomorphic distributions.
        """
        canonicals = {}
        for i in range(1, 2 ** (self.k**self.n)):
            if i in canonicals:
                continue
            isos = self._isomorphisms(i)
            canonical = self._canonical(isos)
            for iso in isos:
                canonicals[iso] = canonical

        for canon in set(canonicals.values()):
            yield _int_to_dist(canon, self.n, self.k)
