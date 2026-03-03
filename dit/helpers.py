"""
Helper non-public API functions for distributions.
"""

import itertools
from operator import itemgetter

import numpy as np

from .exceptions import InvalidOutcome, ditException
from .math.misc import is_number
from .utils import flatten, product_maker

__all__ = (
    "construct_alphabets",
    "copypmf",
    "get_outcome_ctor",
    "get_product_func",
    "normalize_pmfs",
    "normalize_rvs",
    "numerical_test",
    "parse_rvs",
    "reorder",
    "str_outcome_ctor",
)


def str_outcome_ctor(iterable):
    try:
        return "".join(iterable)
    except TypeError as err:
        msg = f"Outcome could not be constructed from {iterable!r}"
        raise ditException(msg) from err


#
# This dictionary is a registry (which could, in principle, be updated
# by users) that maps the outcome class (which is fixed for each distribution)
# to an outcome constructor.  The constructor takes, as input, a tuple
# created by itertools.product and returns an object of the outcome class.
# See get_outcome_constructor() and get_produc_func() for more details.
#
constructor_map = {
    str: str_outcome_ctor,
}



def construct_alphabets(outcomes):
    """
    Construct minimal alphabets for each random variable.

    In the process, it verifies that each outcome is a sequence and that all
    outcomes have the same length.

    Parameters
    ----------
    outcomes : sequence
        A nonempty sequence of outcomes.  Each outcome in `outcomes` should
        be a sequence---these are the elements which determine the alphabet
        for each random variable.

    Returns
    -------
    alphabets : tuple
        The constructed alphabet for each random variable.

    Examples
    --------
    >>> construct_alphabets([(0,1), (1,1)])
    ((0,1), (1,))

    Raises
    ------
    ditException
        When there are no outcomes.
        When not every outcome is a sequence.
        When not all outcomes have the same length.

    """
    # During validation, each outcome is checked to be of the proper class,
    # length, and also a sequence.  However, this function is called before
    # validation and will result in hard to decipher error messages if we
    # don't at least verify that each outcome is a container of the same
    # length.

    # Make sure outcomes is a sequence
    try:
        L = len(outcomes)
    except TypeError as err:
        raise TypeError("`outcomes` must be a sequence.") from err

    if L == 0:
        raise ditException("`outcomes` must not be empty.")

    # Make sure each outcome is sized.  They really should be sequences,
    # but this check is sufficient for now.
    try:
        lengths = list(map(len, outcomes))
    except TypeError as err:
        raise ditException(
            "At least one element in `outcomes` does not implement __len__. "
            "Distribution.from_ndarray may help "
            "resolve this."
        ) from err
    else:
        outcome_length = lengths[0]

    # Make sure each outcome has the same length.
    equal_lengths = np.all(np.equal(lengths, outcome_length))
    if not equal_lengths:
        raise ditException("Not all outcomes have the same length.")

    alphabets = _construct_alphabets(outcomes)
    return alphabets


def _construct_alphabets(outcomes):
    """
    Core construction of alphabets. No sanity checks.

    """
    # Its important that we maintain the order of the sample space.
    # The sample space is given by the Cartesian product of the alphabets.
    # So if the user passes sort=False to the constructor of Distribution,
    # we must make sure to keep the order of the alphabet. So we do not sort
    # the alphabets.
    from dit.utils import OrderedDict

    outcome_length = len(outcomes[0])
    alphabets = [OrderedDict() for i in range(outcome_length)]
    for outcome in outcomes:
        for i, symbol in enumerate(outcome):
            alphabets[i][symbol] = True

    alphabets = tuple(map(tuple, alphabets))
    return alphabets


def get_outcome_ctor(klass):
    """
    Helper function to return an outcome constructor from the outcome class.

    Usually, this will be the outcome class constructor.  However, for some
    classes, such as str, passing in a tuple does not return the desired
    output. For example, str( ('1','0','1') ) yields "('0', '1', '0')" when
    we want "101". The constructor should work with a tuple as input.

    The global `constructor_map` maps classes to their constructor.  If the
    class does not exist in the dict, then we use the class constructor.

    """
    return constructor_map.get(klass, klass)


def get_product_func(klass):
    """
    Helper function to return a product function for the distribution.

    The idea is to return something similar to itertools.product. The
    difference is that the iterables should not be tuples, necessarily.
    Rather, they should match whatever the class of the outcomes is.

    See the docstring for Distribution.

    """
    ctor = get_outcome_ctor(klass)
    product = itertools.product if ctor is tuple else product_maker(ctor)

    return product


def normalize_rvs(dist, rvs, crvs):
    """
    Perform common tasks useful for multivariate information measures.

    Parameters
    ----------
    dist : Distribution
        The distribution that will be operated on.
    rvs : list, None
        List of random variables to use in this measure.
    crvs : list, None
        List of random variables to condition on.

    Returns
    -------
    rvs : list
        The explicit random variables to use.
    crvs : list
        The explicit random variables to condition on.
    """
    if rvs is None:
        rvs = [[i] for i in range(dist.outcome_length())]
    crvs = [] if crvs is None else list(flatten(crvs))

    return rvs, crvs


def parse_rvs(dist, rvs, unique=True, sort=True):
    """
    Resolve random variable specs to (names, indices) tuples.

    Integers are auto-resolved to dimension names via the distribution's
    dims ordering.  Strings are treated as dimension names.

    Parameters
    ----------
    dist : distribution
        The distribution.
    rvs : list
        Random variable identifiers (names or integer indices).
    unique : bool
        If True, require no duplicates.
    sort : bool
        If True, sort output by index.

    Returns
    -------
    rvs : tuple of str
        Dimension names.
    indices : tuple of int
        Corresponding dimension indices.
    """
    if len(rvs) == 0:
        return (), ()

    names = list(dist._resolve_rv_names(list(rvs))) if hasattr(dist, '_resolve_rv_names') else list(rvs)

    if unique and len(set(names)) != len(names):
        msg = "`rvs` contained duplicates."
        raise ditException(msg)

    # Convert names to indices
    if hasattr(dist, 'dims'):
        dim_list = list(dist.dims)
        try:
            indexes = [dim_list.index(n) for n in names]
        except ValueError as err:
            msg = f"`rvs` contains invalid random variables: {names}"
            raise ditException(msg) from err
    else:
        # Legacy path for SampleSpace: names ARE indices
        indexes = list(names)

    all_indexes = set(range(dist.outcome_length()))
    good_indexes = all_indexes.intersection(indexes)
    if len(good_indexes) != len(set(indexes)):
        msg = f"`rvs` contains invalid random variables: {names}"
        raise ditException(msg)

    out = list(zip(names, indexes, strict=True))
    if sort:
        out.sort(key=itemgetter(1))
    names_out, indexes_out = list(zip(*out, strict=True))

    return names_out, indexes_out


def reorder(outcomes, pmf, sample_space, index=None):
    """
    Helper function to reorder outcomes and pmf to match sample_space.

    """
    try:
        order = [(sample_space.index(outcome), i) for i, outcome in enumerate(outcomes)]
    except ValueError:
        # Let's identify which outcomes were not in the sample space.
        bad = []
        for outcome in outcomes:
            try:
                sample_space.index(outcome)
            except ValueError:
                bad.append(outcome)
        raise InvalidOutcome(bad, single=len(bad) == 1) from None

    order.sort()
    _, order = zip(*order, strict=True)

    if index is None:
        index = dict(zip(outcomes, range(len(outcomes)), strict=True))

    outcomes = [outcomes[i] for i in order]
    pmf = [pmf[i] for i in order]
    new_index = dict(zip(outcomes, range(len(outcomes)), strict=True))
    return outcomes, pmf, new_index


def copypmf(d, base=None, mode="asis"):
    """
    Returns a NumPy array of the distribution's pmf.

    Parameters
    ----------
    d : distribution
        The distribution from which the pmf is copied.
    base : float, 'linear', 'e', None
        The desired base of the probabilities. If None, then the probabilities
        maintain their current base.
    mode : ['dense', 'sparse', 'asis']
        A specification of how the pmf should be construted. 'dense' means
        that the pmf should contain the entire sample space. 'sparse' means
        the pmf should only contain nonnull probabilities. 'asis' means to
        make a copy of the pmf, as it exists in the distribution.

    Returns
    -------
    pmf : NumPy array
        The pmf of the distribution.

    """
    from dit.math import get_ops
    from dit.params import validate_base

    # Sanitize inputs, need numerical base for old base.
    base_old = d.get_base(numerical=True)
    base_new = base_old if base is None else validate_base(base)

    # Create ops instances.
    ops_old = d.ops
    ops_new = get_ops(base_new)

    # Build the pmf
    if mode == "asis":
        pmf = np.array(d.pmf, copy=True)
    elif mode == "dense":
        pmf = np.array([d[o] for o in d.sample_space()], dtype=float)
    elif mode == "sparse":
        pmf = np.array([p for p in d.pmf if not ops_old.is_null(p)], dtype=float)

    # Determine the conversion targets.
    islog_old = d.is_log()
    islog_new = base_new != "linear"

    # Do the conversion!
    if islog_old and islog_new:
        # Convert from one log base to another.
        ## log_b(x) = log_b(a) * log_a(x)
        pmf *= ops_new.log(base_old)
    elif not islog_old and not islog_new:
        # No conversion: from linear to linear.
        pass
    elif islog_old and not islog_new:
        # Convert from log to linear.
        ## x = b**log_b(x)
        pmf = base_old**pmf
    else:
        # Convert from linear to log.
        ## x = log_b(x)
        pmf = ops_new.log(pmf)

    return pmf


def normalize_pmfs(dist1, dist2):
    """
    Construct probability vectors with common support.

    Parameters
    ----------
    dist1 : Distribution
        The first distribution.
    dist2 : Distribution
        The second distribution.

    Returns
    -------
    p : np.ndarray
        The pmf of `dist1`.
    q : np.ndarray
        The pmf of `dist2`.
    """
    event_space = list(set().union(dist1.outcomes, dist2.outcomes))
    p = np.array([dist1[e] if e in dist1.outcomes else 0 for e in event_space])
    q = np.array([dist2[e] if e in dist2.outcomes else 0 for e in event_space])
    return p, q


def numerical_test(dist):
    """
    Verifies that all outcomes are numbers.

    Parameters
    ----------
    dist : Distribution
        The distribution whose outcomes are to be checked.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If the outcomes of the `dist` are not numerical.
    """
    if not all(is_number(o) for o in flatten(dist.outcomes)):
        msg = "The outcomes of this distribution are not numerical"
        raise TypeError(msg)
