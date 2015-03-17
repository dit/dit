#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Helper non-public API functions for distributions.

"""

# Standard library
from operator import itemgetter
import itertools
import warnings

# Other
import numpy as np
from six.moves import map, range, zip # pylint: disable=redefined-builtin

# dit
from .exceptions import ditException, InvalidOutcome
from .utils import flatten, product_maker


def str_outcome_ctor(iterable):
    try:
        return ''.join(iterable)
    except TypeError:
        msg = 'Outcome could not be constructed from {0!r}'.format(iterable)
        raise ditException(msg)


#
# This dictionary is a registry (which could, in principle, be updated
# by users) that maps the outcome class (which is fixed for each distribution)
# to an outcome constructor.  The constructor takes, as input, a tuple
# created by itertools.product and returns an object of the outcome class.
# See get_outcome_constructor() and get_produc_func() for more details.
#
constructor_map = {
    str : str_outcome_ctor,
}

class RV_Mode(object):
    """
    Class to manage how rvs and crvs are specified and interpreted.

    """
    INDICES = 0
    NAMES = 1

    _mapping = {
        'indexes': INDICES,
        'indices': INDICES,
        'names': NAMES,
        None: None,
        # Deprecated stuff:
        True: NAMES,
        False: INDICES
    }

    # Temporary until we can convert everything to using: rv_mode
    _deprecated = set([True, False])

    def __getitem__(self, item):
        try:
            mode = self._mapping[item]
        except KeyError:
            raise KeyError('Invalid value for `rv_mode`')

        if item in self._deprecated:
            msg = 'Deprecated value for `rv_mode`: {0!r}.'.format(item)
            msg += ' See docstring for new conventions.'
            warnings.warn(msg, DeprecationWarning)

        return mode

RV_MODES = RV_Mode()

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
    except TypeError:
        raise TypeError('`outcomes` must be a sequence.')

    if L == 0:
        raise ditException('`outcomes` must not be empty.')

    # Make sure each outcome is sized.  They really should be sequences,
    # but this check is sufficient for now.
    try:
        lengths = list(map(len, outcomes))
    except TypeError:
        raise ditException('One or more outcomes is not sized. len() fails.')
    else:
        outcome_length = lengths[0]

    # Make sure each outcome has the same length.
    equal_lengths = np.alltrue(np.equal(lengths, outcome_length))
    if not equal_lengths:
        raise ditException('Not all outcomes have the same length.')

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
    if ctor == tuple:
        # No need to modify the output of itertools.
        product = itertools.product
    else:
        # Assume the sequence-like constructor can handle tuples as input.
        product = product_maker(ctor)

    return product

def normalize_rvs(dist, rvs, crvs, rv_mode):
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
    rv_mode : str, None
        Specifies how to interpret `rvs` and `crvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `crvs` and `rvs` are interpreted as random variable indices. If equal
        to 'names', the the elements are interpreted as random variable names.
        If `None`, then the value of `dist._rv_mode` is consulted, which
        defaults to 'indices'.

    Returns
    -------
    rvs : list
        The explicit random variables to use.
    crvs : list
        The explicit random variables to condition on.
    rv_mode : bool
        The value of rv_mode that should be used.

    Raises
    ------
    ditException
        Raised if `dist` is not a joint distribution.
    """
    if dist.is_joint():
        if rvs is None:
            # Set so that each random variable is its own group.
            rvs = [[i] for i in range(dist.outcome_length())]
            rv_mode = RV_MODES.INDICES
        if crvs is None:
            crvs = []
        else:
            crvs = list(flatten(crvs))
    else:
        msg = "The information measure requires a joint distribution."
        raise ditException(msg)

    return rvs, crvs, rv_mode

def parse_rvs(dist, rvs, rv_mode=None, unique=True, sort=True):
    """
    Returns the indices of the random variables in `rvs`.

    Parameters
    ----------
    dist : joint distribution
        The joint distribution.
    rvs : list
        The list of random variables. This is either a list of random
        variable indexes or a list of random variable names.
    rv_mode : str, None
        Specifies how to interpret the elements of `rvs`. Valid options are:
        {'indices', 'names'}. If equal to 'indices', then the elements of
        `rvs` are interpreted as random variable indices. If equal to 'names',
        the the elements are interpreted as random variable names. If `None`,
        then the value of `dist._rv_mode` is consulted.
    unique : bool
        If `True`, then require that no random variable is repeated in `rvs`.
        If there are any duplicates, an exception is raised. If `False`, random
        variables can be repeated.
    sort : bool
        If `True`, then the output is sorted by the random variable indexes.

    Returns
    -------
    rvs : tuple
        A new tuple of the specified random variables, possibly sorted.
    indices : tuple
        The corresponding indices of the random variables, possibly sorted.

    Raises
    ------
    ditException
        If `rvs` cannot be converted properly into indexes.

    """
    if rv_mode is None:
        rv_mode = dist._rv_mode
    rv_mode = RV_MODES[rv_mode]

    # Quick check for the empty set. Interpretation: no random variables.
    if len(rvs) == 0:
        return (), ()

    # Make sure all random variables are unique.
    if unique and len(set(rvs)) != len(rvs):
        msg = '`rvs` contained duplicates.'
        raise ditException(msg)

    if rv_mode == RV_MODES.NAMES:
        # Then `rvs` contains random variable names.
        # We convert these to indexes.

        if dist._rvs is None:
            raise ditException('There are no random variable names to use.')

        indexes = []
        for rv in rvs:
            if rv in dist._rvs:
                indexes.append(dist._rvs[rv])

        if len(indexes) != len(rvs):
            msg = '`rvs` contains invalid random variable names.'
            raise ditException(msg)
    else:
        # Then `rvs` contained the set of indexes.
        indexes = rvs

    # Make sure all indexes are valid, even if there are duplicates.
    all_indexes = set(range(dist.outcome_length()))
    good_indexes = all_indexes.intersection(indexes)
    if len(good_indexes) != len(set(indexes)):
        msg = '`rvs` contains invalid random variables, {0}, {1} {2}.'
        msg = msg.format(indexes, good_indexes, rv_mode)
        raise ditException(msg)

    # Sort the random variable names (or indexes) by their index.
    out = zip(rvs, indexes)
    if sort:
        out = list(out)
        out.sort(key=itemgetter(1))
    rvs, indexes = list(zip(*out))

    return rvs, indexes

def reorder(outcomes, pmf, sample_space, index=None):
    """
    Helper function to reorder outcomes and pmf to match sample_space.

    """
    try:
        order = [(sample_space.index(outcome), i)
                 for i, outcome in enumerate(outcomes)]
    except ValueError:
        # Let's identify which outcomes were not in the sample space.
        bad = []
        for outcome in outcomes:
            try:
                sample_space.index(outcome)
            except ValueError:
                bad.append(outcome)
        if len(bad) == 1:
            single = True
        else:
            single = False
        raise InvalidOutcome(bad, single=single)

    order.sort()
    _, order = zip(*order)

    if index is None:
        index = dict(zip(outcomes, range(len(outcomes))))

    outcomes = [outcomes[i] for i in order]
    pmf = [pmf[i] for i in order]
    new_index = dict(zip(outcomes, range(len(outcomes))))
    return outcomes, pmf, new_index

def copypmf(d, base=None, mode='asis'):
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
    if base is None:
        base_new = base_old
    else:
        base_new = validate_base(base)

    # Create ops instances.
    ops_old = d.ops
    ops_new = get_ops(base_new)

    # Build the pmf
    if mode == 'asis':
        pmf = np.array(d.pmf, copy=True)
    elif mode == 'dense':
        pmf = np.array([d[o] for o in d.sample_space()], dtype=float)
    elif mode == 'sparse':
        pmf = np.array([p for p in d.pmf if not ops_old.is_null(p)], dtype=float)

    # Determine the conversion targets.
    islog_old = d.is_log()
    if base_new == 'linear':
        islog_new = False
    else:
        islog_new = True

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
