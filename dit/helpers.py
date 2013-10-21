#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Helper non-public API functions for distributions.

"""

# Standard library
from operator import itemgetter
import itertools

# Other
import numpy as np
from six.moves import map, range, zip # pylint: disable=redefined-builtin

# dit
from .exceptions import ditException, InvalidDistribution, InvalidOutcome
from .utils import product_maker



#
# This dictionary is a registry (which could, in principle, be updated
# by users) that maps the outcome class (which is fixed for each distribution)
# to an outcome constructor.  The constructor takes, as input, a tuple
# created by itertools.product and returns an object of the outcome class.
# See get_outcome_constructor() and get_produc_func() for more details.
#
constructor_map = {
    str : ''.join,
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
        When not every outcome is a sequence.
        When not all outcomes have the same length.

    """
    ## Assumption: len(outcomes) > 0

    # During validation, each outcome is checked to be of the proper class,
    # length, and also a sequence.  However, this function is called before
    # validation and will result in hard to decipher error messages if we
    # don't at least verify that each outcome is a container of the same
    # length.

    # Make sure outcomes is a sequence
    try:
        len(outcomes)
    except TypeError:
        raise TypeError('`outcomes` must be a sequence.')

    # Make sure each outcome is sized.  They really should be sequences,
    # but this check is sufficient for now.
    try:
        lengths = list(map(len, outcomes))
    except TypeError:
        raise ditException('One or more outcomes is not sized. len() fails.')
    else:
        outcome_length = lengths[0]

    # Make sure each outcome has the same length.
    equal_lengths = np.alltrue( np.equal( lengths, outcome_length ) )
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
    # we must make sure to keep the order of the alphabet.
    from dit.utils import OrderedDict

    outcome_length = len(outcomes[0])
    alphabets = [OrderedDict() for i in range(outcome_length)]
    for outcome in outcomes:
        for i, symbol in enumerate(outcome):
            alphabets[i][symbol] = True

    alphabets = tuple( map(tuple, alphabets) )
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

def normalize_rvs(dist, rvs, crvs, rv_names):
    """
    For use in multivariate information measures.

    Parameters
    ----------
    rvs : list, None
        List of random variables to use in this measure.

    crvs : list, None
        List of random variables to condition on.

    rv_names : bool, None
        If `True`, then the elements of `rvs` are treated as random variable
        names. If `False`, then the elements of `rvs` are treated as random
        variable indexes.  If `None`, then the value `True` is used if the
        distribution has specified names for its random variables.

    Returns
    -------
    rvs : list
        The explicit random variables to use.

    crvs : list
        The explicit random variables to condition on.

    rv_names : bool


    Raises
    ------
    ditException
        Raised if `dist` is not a joint distribution.
    """
    if dist.is_joint():
        if rvs is None:
            # Set to total correlation of entire distribution
            rvs = [ [i] for i in range(dist.outcome_length()) ]
            rv_names = False
        if crvs is None:
            crvs = []
    else:
        msg = "The information measure requires a joint distribution."
        raise ditException(msg)

    return rvs, crvs, rv_names

def parse_rvs(dist, rvs, rv_names=None, unique=True, sort=True):
    """
    Returns the indexes of the random variables in `rvs`.

    Parameters
    ----------
    dist : joint distribution
        The joint distribution.
    rvs : list
        The list of random variables. This is either a list of random
        variable indexes or a list of random variable names.
    rv_names : bool
        If `True`, then the elements of `rvs` are treated as random variable
        names. If `False`, then the elements of `rvs` are treated as random
        variable indexes.  If `None`, then the value `True` is used if the
        distribution has specified names for its random variables.
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
    indexes : tuple
        The corresponding indexes of the random variables, possibly sorted.

    Raises
    ------
    ditException
        If `rvs` cannot be converted properly into indexes.

    """
    # Quick check for the empty set. Interpretation: no random variables.
    if len(rvs) == 0:
        return (), ()

    # Make sure all random variables are unique.
    if unique and len(set(rvs)) != len(rvs):
        msg = '`rvs` contained duplicates.'
        raise ditException(msg)

    # If `rv_names` is None, then its value depends on whether the distribution
    # has names associated with its random variables.
    if rv_names is None:
        if dist._rvs is None:
            # Interpret `rvs` as listing indexes.
            rv_names = False
        else:
            # Interpret `rvs` as listing random variable names.
            rv_names = True

    if rv_names:
        # Then `rvs` contained random variable names.
        # We convert these to indexes.
        indexes = []
        for rv in rvs:
            if rv in dist._rvs:
                indexes.append( dist._rvs[rv] )

        if len(indexes) != len(rvs):
            msg ='`rvs` contains invalid random variable names.'
            raise ditException(msg)
    else:
        # Then `rvs` contained the set of indexes.
        indexes = rvs

    # Make sure all indexes are valid, even if there are duplicates.
    all_indexes = set(range(dist.outcome_length()))
    good_indexes = all_indexes.intersection(indexes)
    if len(good_indexes) != len(set(indexes)):
        msg = '`rvs` contains invalid random variables'
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
    if index is None:
        index = dict(zip(outcomes, range(len(outcomes))))

    order = [index[outcome] for outcome in sample_space if outcome in index]
    if len(order) != len(outcomes):
        # For example, `outcomes` contains an element not in `sample_space`.
        # For example, `outcomes` contains duplicates.
        msg = 'outcomes and sample_space are not compatible.'
        raise InvalidDistribution(msg)

    outcomes = [outcomes[i] for i in order]
    pmf = [pmf[i] for i in order]
    new_index = dict(zip(outcomes, range(len(outcomes))))
    return outcomes, pmf, new_index

def reorder_cp(pmf, outcomes, alphabet, product, index=None, method=None):
    """
    Helper function to reorder pmf and outcomes so as to match the sample space.

    When the sample space is not stored, explicitly on the distribution, then
    there are two ways to do this:
        1) Determine the order by generating the entire sample space.
        2) Analytically calculate the sort order of each outcome.

    If the sample space is very large and sparsely populated, then method 2)
    is probably faster. However, it must calculate a number using
    (2**(symbol_orders)).sum().  Potentially, this could be costly. If the
    sample space is small, then method 1) is probably fastest. We'll experiment
    and find a good heuristic.

    """
    # A map of the elements in `outcomes` to their index in `outcomes`.
    if index is None:
        index = dict(zip(outcomes, range(len(outcomes))))

    # The number of elements in the sample space?
    sample_space_size = np.prod( list(map(len, alphabet)) )
    if method is None:
        if sample_space_size > 10000 and len(outcomes) < 1000:
            # Large and sparse.
            method = 'analytic'
        else:
            method = 'generate'

    method = 'generate'
    if method == 'generate':
        # Obtain the order from the generated order.
        sample_space = product(*alphabet)
        order = [index[outcome] for outcome in sample_space if outcome in index]
        if len(order) != len(outcomes):
            msg = 'Outcomes and sample_space are not compatible.'
            raise InvalidDistribution(msg)
        outcomes_ = [outcomes[i] for i in order]
        pmf = [pmf[i] for i in order]

        # We get this for free: Check that every outcome was in the sample
        # space. Well, its costs us a bit in memory to keep outcomes and
        # outcomes_.
        if len(outcomes_) != len(outcomes):
            # We lost an outcome.
            bad = set(outcomes) - set(outcomes_)
            L = len(bad)
            if L > 0:
                raise InvalidOutcome(bad, single=(L==1))
        else:
            outcomes = outcomes_

    elif method == 'analytic':
        # Analytically calculate the sort order.
        # Note, this method does not verify that every outcome was in the
        # sample space.

        # Construct a lookup from symbol to order in the alphabet.
        alphabet_size = list(map(len, alphabet))
        alphabet_index = [dict(zip(alph, range(size)))
                          for alph, size in zip(alphabet, alphabet_size)]

        L = len(outcomes[0]) - 1
        codes = []
        for outcome in outcomes:
            idx = 0
            for i, symbol in enumerate(outcome):
                idx += alphabet_index[i][symbol] * (alphabet_size[i])**(L-i)
            codes.append(idx)

        # We need to sort the codes now, keeping track of their indexes.
        order = list(zip(codes, range(len(codes))))
        order.sort()
        _, order = list(zip(*order))
        outcomes = [outcomes[i] for i in order]
        pmf = [pmf[i] for i in order]
    else:
        raise Exception("Method must be 'generate' or 'analytic'")

    new_index = dict(zip(outcomes, range(len(outcomes))))

    return pmf, outcomes, new_index
