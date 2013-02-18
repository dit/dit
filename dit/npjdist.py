#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module defining NumPy array-based distribution classes.

One of the features of joint distributions is that we can marginalize them. This
requires that we are able to construct smaller outcomes from larger outcomes.
For example, an outcome like '10101' might become '010' if the first and last
random variables are marginalized.  Given the alphabet of the joint
distribution, we can construct a tuple ('0','1','0') for any smaller outcome
using itertools.product, but how do we create an outcome of the same type? For
tuples and other similar containers, we just pass the tuple to the outcome
constructor.  This technique does not work for strings, as str(('0','1','0'))
yields "('0', '1', '0')".  So we have to handle strings separately.

Note:
    For dictionaries...
        "k in d" being True means d[k] is a valid operation.
        "k in d" being False means d[k] is a KeyError.
        d[k] describes the underlying data structure.
        __in__ describes the underlying data structure.
        __iter__ describes the underlying data structure.
        __len__ describes the underlying data structure.
    For default dictionaries...
        "k in d" being True means d[k] is a valid operation.
        "k in d" being False means d[k] will modify the data structure to
            make the operation valid. So "k in d" is True afterwards.
        d[k] describes the underlying data structure.
        __in__ describes the underlying data structure.
        __iter__ describes the underlying data structure.
        __len__ describes the underlying data structure.
    For distributions...
        "e in d" being True means d[e] is a valid operation.
        "e in d" being False says nothing about d[e].
            d[e] will be valid if e is in the sample space.
            d[e] will raise an InvalidOutcome if e is not in the sample space.
        d[e] does not describe the underlying data structure.
            It provides a view of the dense data structure.
            With defaultdict, if e not in d, then d[e] will add it.
            With distributions, we don't want the pmf changing size
            just because we queried it.  The size will change only on
            assignment.
        __in__ describes the underlying data structure.
        __iter__ describes the underlying data structure.
        __len__ describes the underlying data structure.

For distributions, the sample space is the alphabet and the alphabet is a single
tuple. For joint distributions, the sample space is the Cartestian product of
the alphabets for each random variable, and the alphabet for the joint
distribution is a tuple of alphabets for each random variable.

"""

import itertools
from collections import defaultdict
import numpy as np
from operator import itemgetter

from .npdist import Distribution
from .exceptions import (
    InvalidDistribution, InvalidOutcome, InvalidProbability, ditException
)
from .math import LinearOperations, LogOperations
from .utils import str_product, product_maker
from .params import ditParams

def get_product_func(outcomes):
    """
    Helper function to return a product function for the distribution.

    See the docstring for JointDistribution.

    """
    # Every outcome must be of the same type.
    if len(outcomes) == 0:
        # Any product will work, since there is nothing to iterate over.
        product = itertools.product
    else:
        outcome = outcomes[0]
        if isinstance(outcome, basestring):
            product = str_product
        elif isinstance(outcome, tuple):
            product = itertools.product
        else:
            # Assume the sequence-like constructor can handle tuples as input.
            product = product_maker(outcome.__class__)

    return product

def parse_rvs(dist, rvs, rv_names=True, unique=True, sort=True):
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
        variable indexes.
    unique : bool
        If `True`, then require that no random variable is repeated in `rvs`.
        If there are any duplicates, an exception is raised. If `False`, random
        variables can be repeated.
    sort : bool
        If `True`, then the output is sorted by the random variable indexes.

    Returns
    -------
    rvs : tuple
        The random variables, possibly sorted.
    indexes : tuple
        The corresponding indexes of the random variables, possibly sorted.

    Raises
    ------
    ditException
        If `rvs` cannot be converted properly into indexes.

    """
    # Make sure all random variables are unique.
    if unique and len(set(rvs)) != len(rvs):
        msg = '`rvs` contained duplicates.'
        raise ditException(msg)

    if rv_names:
        # Convert names to indexes.
        indexes = []
        for rv in rvs:
            if rv in dist._rvs:
                indexes.append( dist._rvs[rv] )

        if len(indexes) != len(rvs):
            msg ='`rvs` contains invalid random variable names.'
            raise ditException(msg)
    else:
        indexes = rvs

    # Make sure all indexes are valid, even if there are duplicates.
    all_indexes = set(range(dist.outcome_length()))
    good_indexes = all_indexes.intersection(indexes)
    if len(good_indexes) != len(set(indexes)):
        msg = '`rvs` contains invalid random variables'
        raise ditException(msg)

    out = zip(rvs, indexes)
    if sort:
        out.sort(key=itemgetter(1))
    rvs, indexes = zip(*out)

    return rvs, indexes

def _make_distribution(pmf, outcomes, alphabet=None, base=None, sparse=True):
    """
    An unsafe, but faster, initialization for distributions.

    If used incorrectly, the data structure will be inconsistent.

    This function can be useful when you are creating many distributions
    in a loop and can guarantee that:

        0) all outcomes are of the same type (eg tuple, str) and length.
        1) the alphabet is in the desired order.
        2) outcomes and pmf are in the same order as the sample space.
           [Thus, `pmf` should not be a dictionary.]

    This function will not order the sample space, nor will it reorder outcomes
    or pmf.  It will not forcibly make outcomes and pmf to be sparse or dense.
    It will simply declare the distribution to be sparse or dense. The
    distribution is not validated either.

    Returns
    -------
    d : Distribution
        The new distribution.

    """
    d = JointDistribution.__new__(JointDistribution)

    # Determine if the pmf represents log probabilities or not.
    if base is None:
        base = ditParams['base']
    if base == 'linear':
        ops = LinearOperations()
    else:
        ops = LogOperations(base)
    d.ops = ops

    ## outcomes
    # Unlike Distribution, we cannot use a default set of outcomes when
    # `outcomes` is `None`.
    if len(pmf) != len(outcomes):
        msg = "Unequal lengths for `values` and `outcomes`"
        raise InvalidDistribution(msg)

    ## alphabets
    # Use outcomes to obtain the alphabets.
    if alphabet is None:

        if len(outcomes) == 0:
            msg = '`outcomes` cannot have zero length if `alphabet` is `None`'
            raise InvalidDistribution(msg)

        # The outcome length.
        outcome_length = len(outcomes[0])
        alphabet = [set([]) for i in range(outcome_length)]
        for outcome in outcomes:
            for i,symbol in enumerate(outcome):
                alphabet[i].add(symbol)
        alphabet = map(tuple, alphabet)

    if len(outcomes):
        d._outcome_class = outcomes[0].__class__

    ## product
    d._product = get_product_func(outcomes)

    # Force the distribution to be numerical and a NumPy array.
    d.pmf = np.asarray(pmf, dtype=float)

    # Tuple outcomes, and an index.
    d.outcomes = tuple(outcomes)
    d._outcomes_index = dict(zip(outcomes, range(len(outcomes))))

    # Tuple sample space and its set.
    d.alphabet = tuple(alphabet)
    d._alphabet_set = map(set, d.alphabet)

    # Set the mask
    d._mask = tuple(False for _ in range(len(alphabet)))

    # Provide a default set of names for the random variables.
    rv_names = range(len(alphabet))
    d._rvs = dict(zip(rv_names, rv_names))

    d._meta['is_sparse'] = sparse

    return d

def reorder(pmf, outcomes, alphabet, product, index=None, method=None):
    """
    Helper function to reorder pmf and outcomes so as to match the sample space.

    The Cartesian product of the alphabets defines the sample space.

    There are two ways to do this:
        1) Determine the order by generating the entire sample space.
        2) Analytically calculate the sort order of each outcome.

    If the sample space is very large and sparsely populated, then method 2)
    is probably faster. However, it must calculate a number using
    (2**(symbol_orders)).sum().  Potentially, this could be costly. If the
    sample space is small, then method 1) is probably fastest. We'll experiment
    and find a good heurestic.

    """
    # A map of the elements in `outcomes` to their index in `outcomes`.
    if index is None:
        index = dict(zip(outcomes, range(len(outcomes))))

    # The number of elements in the sample space?
    sample_space_size = np.prod( map(len, alphabet) )

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
            if L == 1:
                raise InvalidOutcome(bad, single=True)
            elif L:
                raise InvalidOutcome(bad, single=False)
        else:
            outcomes = outcomes_

    elif method == 'analytic':
        # Analytically calculate the sort order.
        # Note, this method does not verify that every outcome was in the
        # sample space.

        # Construct a lookup from symbol to order in the alphabet.
        alphabet_size = map(len, alphabet)
        alphabet_index = [dict(zip(alph, range(size)))
                          for alph, size in zip(alphabet, alphabet_size)]

        L = len(outcomes[0]) - 1
        codes = []
        for outcome in outcomes:
            idx = 0
            for i,symbol in enumerate(outcome):
                idx += alphabet_index[i][symbol] * (alphabet_size[i])**(L-i)
            codes.append(idx)

        # We need to sort the codes now, keeping track of their indexes.
        order = zip(codes, range(len(codes)))
        order.sort()
        sorted_codes, order = zip(*order)
        outcomes = [outcomes[i] for i in order]
        pmf = [pmf[i] for i in order]
    else:
        raise Exception("Method must be 'generate' or 'analytic'")

    new_index = dict(zip(outcomes, range(len(outcomes))))

    return pmf, outcomes, new_index

class JointDistribution(Distribution):
    """
    A numerical joint distribution.

    Meta Properties
    ---------------
    is_joint
        Boolean specifying if the pmf represents a joint distribution.

    is_numerical
        Boolean specifying if the pmf represents numerical values or not.
        The values could be symbolic, for example.

    is_sparse : bool
        `True` if `outcomes` and `pmf` represent a sparse distribution.

    Private Attributes
    ------------------
    _alphabet_set : tuple
        A tuple representing the alphabet of the joint random variable.  The
        elements of the tuple are sets, each of which represents the unordered
        alphabet of a single random variable.

    _outcome_class : class
        The class of all outcomes in the distribution.

    _outcomes_index : dict
        A dictionary mapping outcomes to their index in self.outcomes.

    _mask : tuple
        A tuple of booleans specifying if the corresponding random variable
        has been masked or not.

    _meta : dict
        A dictionary containing the meta information, described above.

    _product : function
        A specialized product function, similar to itertools.product.  The
        primary difference is that instead of yielding tuples, this product
        function will yield objects which are of the same type as the outcomes.

    _rvs : dict
        A dictionary mapping random variable names to their index into the
        outcomes of the distribution.

    Public Attributes
    -----------------
    alphabet : tuple
        A tuple representing the alphabet of the joint random variable.  The
        elements of the tuple are tuples, each of which represents the ordered
        alphabet for a single random variable. The Cartesian product of these
        alphabets defines the sample space.

    outcomes : tuple
        The outcomes of the probability distribution.

    ops : Operations instance
        A class which manages addition and multiplication operations for log
        and linear probabilities.

    pmf : array-like
        The probability mass function for the distribution.  The elements of
        this array are in a one-to-one correspondence with those in `outcomes`.

    prng : RandomState
        A pseudo-random number generator with a `rand` method which can
        generate random numbers. For now, this is assumed to be something
        with an API compatibile to NumPy's RandomState class. This attribute
        is initialized to equal dit.math.prng.

    Public Methods
    --------------
    coalesce
        Returns a new joint distribution after coalescing random variables.

    copy
        Returns a deep copy of the distribution.

    outcome_length
        Returns the length of the outcomes in the distribution.

    sample_space
        Returns an iterator over the outcomes in the sample space.

    get_base
        Returns the base of the distribution.

    get_rv_names
        Returns the names of the random variables.

    has_outcome
        Returns `True` is the distribution has `outcome` in the sample space.

    is_dense
        Returns `True` if the distribution is dense.

    is_homogeneous
        Returns `True` if the alphabet for each random variable is the same.

    is_joint
        Returns `True` if the distribution is a joint distribution.

    is_log
        Returns `True` if the distribution values are log probabilities.

    is_numerical
        Returns `True` if the distribution values are numerical.

    is_sparse
        Returns `True` if the distribution is sparse.

    marginal
        Returns a marginal distribution of the specified random variables.

    marginalize
        Returns a marginal distribution after marginalizing random variables.

    make_dense
        Add all null outcomes to the pmf.

    make_sparse
        Remove all null outcomes from the pmf.

    normalize
        Normalizes the distribution.

    sample
        Returns a sample from the distribution.

    set_base
        Changes the base of the distribution, in-place.

    set_rv_names
        Sets the names of the random variables.

    to_string
        Returns a string representation of the distribution.

    validate
        A method to validate that the distribution is valid.

    zipped
        Returns an iterator over (outcome, probability) tuples. The probability
        could be a log probability or a linear probability.

    Implementation Notes
    --------------------
    The outcomes and pmf of the distribution are stored as a tuple and a NumPy
    array.  The sequences can be either sparse or dense.  By sparse, we do not
    mean that the representation is a NumPy sparse array.  Rather, we mean that
    the sequences need not contain every outcome in the sample space. The order
    of the outcomes and probabilities will always match the order of the sample
    space, even though their length might not equal the length of the sample
    space.

    """
    _alphabet_set = None
    _outcome_class = None
    _outcomes_index = None
    _mask = None # Not initialize-able
    _meta = {
        'is_joint': True,
        'is_numerical': True,
        'is_sparse': None,
        'is_heterogenous': None,
    }
    _product = None
    _rv_map = None # Not initialize-able

    alphabet = None
    outcomes = None
    ops = None
    pmf = None
    prng = None

    def __init__(self, pmf, outcomes=None, alphabet=None, base=None,
                            sort=True, sparse=True, validate=True):
        """
        Initialize the distribution.

        Parameters
        ----------
        pmf : sequence, dict
            The outcome probabilities or log probabilities. If `pmf` is a
            dictionary, then the keys are used as `outcomes`, and the values of
            the dictionary are used as `pmf` instead.  The keys take precedence
            over any specification of them via `outcomes`.

        outcomes : sequence
            The outcomes of the distribution. If specified, then its length
            must equal the length of `pmf`.  If `None`, then the outcomes must
            be obtainable from `pmf`, otherwise an exception will be raised.
            Outcomes must be hashable, orderable, sized, iterable containers
            that are not `None`. The length of an outcome must be the same for
            all outcomes, and every outcome must be of the same type.

        alphabet : sequence
            A sequence representing the alphabet of the joint random variable.
            The elements of the sequence are tuples, each of which represents
            the ordered alphabet for a single random variable. The Cartesian
            product of these alphabets defines the sample space. The order of
            the alphabets and the order within each alphabet is important. If
            `None`, the value of `outcomes` is used to determine the alphabet.

        base : float, None
            If `pmf` specifies log probabilities, then `base` should specify
            the base of the logarithm.  If 'linear', then `pmf` is assumed to
            represent linear probabilities.  If `None`, then the value for
            `base` is taken from ditParams['base'].

        sort : bool
            If `True`, then each random variable's alphabets are sorted.
            Usually, this is desirable, as it normalizes the behavior of
            distributions which have the same sample spaces (when considered as
            a set).  NOte that addition and multiplication of distributions is
            defined only if the sample spaces are equal.

        sparse : bool
            Specifies the form of the pmf.  If `True`, then `outcomes` and `pmf`
            will only contain entries for non-null outcomes and probabilities,
            after initialization.  The order of these entries will always obey
            the order of `sample_space`, even if their number is not equal to
            the size of the sample space.  If `False`, then the pmf will be
            dense and every outcome in the sample space will be represented.

        validate : bool
            If `True`, then validate the distribution.  If `False`, then assume
            the distribution is valid, and perform no checks.

        Raises
        ------
        InvalidDistribution
            If the length of `values` and `outcomes` are unequal.
            If no outcomes can be obtained from `pmf` and `outcomes` is `None`.

        See :meth:`validate` for a list of other potential exceptions.

        """
        # Note, we are not calling Distribution.__init__
        # We want to call BaseDistribution.__init__
        super(Distribution, self).__init__()

        pmf, outcomes, alphabet = self._init(pmf, outcomes, alphabet, base)

        # Sort everything to match the order of the sample space.
        if sort:
            alphabet = map(sorted, alphabet)
            alphabet = map(tuple, alphabet)
            pmf, outcomes, index = reorder(pmf, outcomes,
                                           alphabet, self._product)
        else:
            index = dict(zip(outcomes, range(len(outcomes))))

        # Force the distribution to be numerical and a NumPy array.
        self.pmf = np.asarray(pmf, dtype=float)

        # Tuple outcomes, and an index.
        self.outcomes = tuple(outcomes)
        self._outcomes_index = index

        # Tuple alphabet and its set.
        self.alphabet = tuple(alphabet)
        self._alphabet_set = map(set, self.alphabet)

        # Mask
        self._mask = tuple(False for _ in range(len(alphabet)))

        # Provide a default set of names for the random variables.
        rv_names = range(len(alphabet))
        self._rvs = dict(zip(rv_names, rv_names))

        if sparse:
            self.make_sparse(trim=True)
        else:
            self.make_dense()

        if validate:
            self.validate()

    def _init(self, pmf, outcomes, alphabet, base):
        """
        The barebones initialization.

        """
        def construct_alphabet(evts):
            if len(evts) == 0:
                msg = '`outcomes` cannot have zero length if '
                msg += '`alphabet` is `None`'
                raise InvalidDistribution(msg)

            # The outcome length.
            self._outcome_class = evts[0].__class__
            outcome_length = len(evts[0])
            alpha = [set([]) for i in range(outcome_length)]
            for outcome in evts:
                for i,symbol in enumerate(outcome):
                    alpha[i].add(symbol)
            alpha = map(tuple, alpha)
            return alpha

        if isinstance(pmf, Distribution):
            # Attempt a conversion.
            d = pmf

            outcomes = d.outcomes
            pmf = d.pmf
            if len(outcomes):
                self._outcome_class = outcomes[0].__class__
            if base is None:
                # Allow the user to specify something strange if desired.
                # Otherwise, use the existing base.
                base = d.get_base()
            if alphabet is None:
                if d.is_joint():
                    # This will always work.
                    alphabet = d.alphabet
                else:
                    # We will assume the outcomes are valid joint outcomes.
                    # But we must construct the alphabet.
                    alphabet = construct_alphabet(outcomes)

        else:
            ## pmf
            # Attempt to grab outcomes and pmf from a dictionary
            try:
                outcomes_ = tuple(pmf.keys())
                pmf_ = tuple(pmf.values())
            except AttributeError:
                pass
            else:
                outcomes = outcomes_
                pmf = pmf_

            ## outcomes
            if outcomes is None:
                msg = "`outcomes` must be specified or obtainable from `pmf`."
                raise InvalidDistribution(msg)
            elif len(pmf) != len(outcomes):
                msg = "Unequal lengths for `values` and `outcomes`"
                raise InvalidDistribution(msg)

            ## alphabets
            # Use outcomes to obtain the alphabets.
            if alphabet is None:
                alphabet = construct_alphabet(outcomes)
            elif len(outcomes):
                self._outcome_class = outcomes[0].__class__

        # Determine if the pmf represents log probabilities or not.
        if base is None:
            base = ditParams['base']
        if base == 'linear':
            ops = LinearOperations()
        else:
            ops = LogOperations(base)
        self.ops = ops

        ## product
        self._product = get_product_func(outcomes)

        return pmf, outcomes, alphabet

    def _get_outcome_constructor(self):
        """
        Internal function to return the constructor for outcomes.

        """
        c = self._outcome_class

        # Special cases
        if c == str:
            c = lambda x: ''.join(x)

        return c

    def __setitem__(self, outcome, value):
        """
        Sets the probability associated with `outcome`.

        Parameters
        ----------
        outcome : outcome
            Any hashable and equality comparable object in the sample space.
            If `outcome` does not exist in the sample space, then an
            InvalidOutcome exception is raised.
        value : float
            The probability or log probability of the outcome.

        Returns
        -------
        p : float
            The probability (or log probability) of the outcome.

        Raises
        ------
        InvalidOutcome
            If `outcome` does not exist in the sample space.

        Notes
        -----
        Setting the value of the outcome never deletes the outcome, even if the
        value is equal to the null probabilty. After a setting operation,
        the outcome will always exist in `outcomes` and `pmf`.

        See Also
        --------
        __delitem__

        """
        if self._outcome_class is None:
            # The first __setitem__ call from an empty distribution.
            self._outcome_class = outcome.__class__
            # Reset the product function.
            self._product = get_product_func([outcome])
            self._mask = tuple(False for _ in range(self.outcome_length()))

        if not self.has_outcome(outcome, null=True):
            raise InvalidOutcome(outcome)

        idx = self._outcomes_index.get(outcome, None)
        new_outcome = idx is None

        if not new_outcome:
            # If the distribution is dense, we will be here.
            # We *could* delete if the value was zero, but we will make
            # setting always set, and deleting always deleting (when sparse).
            self.pmf[idx] = value
        else:
            # A new outcome in a sparse distribution.
            # Sticking with the setting always setting...we add even if
            # the value is zero.

            # 1. Add the new outcome and probability
            self.outcomes = self.outcomes + (outcome,)
            self._outcomes_index[outcome] = len(self.outcomes) - 1
            pmf = [p for p in self.pmf] + [value]

            # 2. Reorder  ### This call is different from Distribution
            pmf, outcomes, index = reorder(pmf, self.outcomes, self.alphabet,
                                           self._product,
                                           index=self._outcomes_index)

            # 3. Store
            self.outcomes = tuple(outcomes)
            self._outcomes_index = index
            self.pmf = np.array(pmf, dtype=float)

    def coalesce(self, rvs, rv_names=True, extract=False):
        """
        Returns a new joint distribution after coalescing random variables.

        Given n lists of random variables in the original joint distribution,
        the coalesced distribution is a joint distribution over n random
        variables.  Each random variable is a coalescing of random variables
        in the original joint distribution.

        Parameters
        ----------
        rvs : sequence
            A sequence whose elements are also sequences.  Each inner sequence
            defines a random variable in the new distribution as a combination
            of random variables in the original distribution.  The length of
            `rvs` must be at least one.  The inner sequences need not be
            pairwise mutually exclusive with one another, and each can contain
            repeated random variables.
        extract : bool
            If the length of `rvs` is 1 and `extract` is `True`, then instead
            of the new outcomes being 1-tuples, we extract the sole element to
            create a joint distribution over the random variables in `rvs[0]`.

        Returns
        -------
        d : distribution
            The coalesced distribution.

        Examples
        --------
        If we have a joint distribution over 3 random variables such as:
            Z = (X,Y,Z)
        and would like a new joint distribution over 6 random variables:
            Z = (X,Y,Z,X,Y,Z)
        then this is achieved as:
            d.coalesce([[0,1,2,0,1,2]], extract=True)

        If you want:
            Z = ((X,Y), (Y,Z))
        Then you do:
            d.coalesce([[0,1],[1,2]])

        Notes
        -----
        Generally, the outcomes of the new distribution will be tuples instead
        of matching the outcome class of the original distribution. This is
        because some outcome classes are not recursive containers. For example,
        one cannot have a string of strings where each string consists of more
        than one character. Note however, that it is perfectly valid to have
        a tuple of tuples. Either way, the elements within each tuple of the
        new distribution will still match the outcome class of the original
        distribution.

        See Also
        --------
        marginal, marginalize

        """
        from array import array

        # We don't need the names. We allow repeats and want to keep the order.
        parse = lambda rv : parse_rvs(self, rv, rv_names=rv_names,
                                                unique=False, sort=False)[1]
        indexes = [parse(rv) for rv in rvs]

        # Determine how new outcomes are constructed.
        if len(rvs) == 1 and extract:
            ctor_o = lambda x: x[0]
        else:
            ctor_o = tuple
        # Determine how elements of new outcomes are constructed.
        ctor_i = self._get_outcome_constructor()

        # Build the distribution.
        factory = lambda : array('d')
        d = defaultdict(factory)
        for outcome, p in self.zipped():
            # Build a list of inner outcomes.
            c_outcome = [ctor_i([outcome[i] for i in rv]) for rv in indexes]
            # Build the outer outcome from the inner outcomes.
            c_outcome = ctor_o( c_outcome )
            d[c_outcome].append(p)

        outcomes = tuple(d.keys())
        pmf = map(np.frombuffer, d.values())
        pmf = map(self.ops.add_reduce, pmf)

        if len(rvs) == 1 and extract:
            # The alphabet for each rv is the same as what it was originally.
            alphabet = [self.alphabet[i] for i in indexes[0]]
        else:
            # Each rv is a Cartesian product of original random variables.
            # So we want to use the distribution's customized product function
            # to create all possible outcomes. This will define the alphabet
            # for each random variable.
            alphabet = [tuple(self._product(*[self.alphabet[i] for i in index]))
                        for index in indexes]

        d = JointDistribution(pmf, outcomes,
                              alphabet=alphabet,
                              base=self.get_base(),
                              sort=True,
                              sparse=self.is_sparse(),
                              validate=False)

        # We do not set the rv names, since these are new random variables.

        # Set the mask
        L = len(indexes)
        d._mask = tuple(False for _ in range(L))

        return d

    def copy(self):
        """
        Returns a (deep) copy of the distribution.

        """
        from copy import deepcopy
        d = _make_distribution(pmf=np.array(self.pmf, copy=True),
                               outcomes=deepcopy(self.outcomes),
                               alphabet=deepcopy(self.alphabet),
                               base=self.ops.base,
                               sparse=self._meta['is_sparse'])

        # The following are not initialize-able from the constructor.
        d.set_rv_names(self.get_rv_names())
        d._mask = tuple(self._mask)

        return d

    def outcome_length(self, masked=False):
        """
        Returns the length of outcomes in the joint distribution.

        This is also equal to the number of random variables in the joint
        distribution. This value is fixed once the distribution is initialized.

        Parameters
        ----------
        masked : bool
            If `True`, then the outcome length additionally includes masked
            random variables. If `False`, then the outcome length does not
            include masked random variables. Including the masked random
            variables is not usually helpful since that represents the outcome
            length of a different, unmarginalized distribution.

        """
        if masked:
            return len(self._mask)
        else:
            # Equivalently: sum(self._mask)
            # Recall, self.alphabet contains only the relevant/valid rvs.
            return len(self.alphabet)

    def sample_space(self):
        """
        Returns an iterator over the ordered outcome space.

        """
        return self._product(*self.alphabet)

    def get_rv_names(self):
        """
        Returns the names of the random variables.

        Returns
        -------
        rv_names : tuple
            A tuple with length equal to the outcome length, containing the
            names of the random variables in the distribution.

        """
        rv_names = [x for x in self._rvs.items()]
        rv_names.sort(key=itemgetter(1))
        rv_names = tuple(map(itemgetter(0), rv_names))
        return rv_names

    def has_outcome(self, outcome, null=True):
        """
        Returns `True` if `outcome` exists  in the sample space.

        Whether or not an outcome is in the sample space is a separate question
        from whether or not an outcome currently appears in the pmf.
        See __contains__ for this latter question.

        Parameters
        ----------
        outcome : outcome
            The outcome to be tested.
        null : bool
            Specifies if null outcomes are acceptable.  If `True`, then null
            outcomes are acceptable.  Thus, the only requirement on `outcome`
            is that it exist in the distribution's sample space. If `False`,
            then null outcomes are not acceptable.  Thus, `outcome` must exist
            in the distribution's sample space and be a nonnull outcome.

        Notes
        -----
        This is an O(1) operation.

        """
        # Make sure the outcome exists in the sample space.

        # Note, it is not sufficient to test if each symbol exists in the
        # the alphabet for its corresponding random variable. The reason is
        # that, for example, '111' and ('1', '1', '1') would both be seen
        # as valid.  Thus, we must also verify that the outcome's class
        # matches that of the other outcome's classes.

        # Make sure the outcome class is correct.
        if outcome.__class__ != self._outcome_class:
            # This test works even when the distribution was initialized empty
            # and _outcome_class is None. In that case, we don't know the
            # sample space (since we don't know the outcome class), and we
            # should return False.
            return False

        # Make sure outcome has the correct length.
        if len(outcome) != self.outcome_length(masked=False):
            return False

        if null:
            # The outcome must only be valid.

            # Make sure each symbol exists in its corresponding alphabet.
            z = False
            for symbol, alphabet in zip(outcome, self.alphabet):
                if symbol not in alphabet:
                    break
            else:
                z = True
        else:
            # The outcome must be valid and have positive probability.
            try:
                z = self[outcome] > self.ops.zero
            except InvalidOutcome:
                z = False

        return z

    def is_homogeneous(self):
        """
        Returns `True` if the alphabet for each random variable is the same.

        """
        a1 = self._alphabet_set[0]
        h = False
        for a2 in self._alphabet_set[1:]:
            if a1 != a2:
                break
        else:
            h = True

        return h

    def marginal(self, rvs, rv_names=True):
        """
        Returns a marginal distribution.

        Parameters
        ----------
        rvs : list
            The random variables to keep. All others are marginalized.
        rv_names : bool
            If `True`, then the elements of `rvs` are treated as names of
            random variables. If `False`, then the elements of `rvs` are
            treated as indexes of random variables.

        Returns
        -------
        d : joint distribution
            A new joint distribution with the random variables in `rvs`
            kept and all others marginalized.

        """
        # Sorted names and indexes.
        rvs, indexes = parse_rvs(self, rvs, rv_names, unique=True, sort=True)

        ## Eventually, add in a method specialized for dense distributions.
        ## This one would work only with the pmf, and not the outcomes.

        # Marginalization is a special case of coalescing where there is only
        # one new random variable and it is composed of a strict subset of
        # the orignal random variables, with no duplicates, that maintains
        # the order of the original random variables.
        d = self.coalesce([indexes], extract=True)

        # Handle parts of d that are not settable through initialization.

        # Set the random variable names
        if rv_names:
            names = rvs
        else:
            # We only have the indexes...so reverse lookup to get the names.
            names_, indexes_ = self._rvs.keys(), self._rvs.values()
            rev = dict(zip(indexes_, names_))
            names = [rev[i] for i in indexes]
        d.set_rv_names(names)

        # Set the mask
        L = self.outcome_length()
        d._mask = tuple(False if i in indexes else True for i in range(L))
        return d

    def marginalize(self, rvs, rv_names=True):
        """
        Returns a new distribution after marginalizing random variables.

        Parameters
        ----------
        rvs : list
            The random variables to marginalize. All others are kept.
        rv_names : bool
            If `True`, then the elements of `rvs` are treated as names of
            random variables. If `False`, then the elements of `rvs` are
            treated as indexes of random variables.

        Returns
        -------
        d : joint distribution
            A new joint distribution with the random variables in `rvs`
            marginalized and all others kept.

        """
        rvs, indexes = parse_rvs(self, rvs, rv_names)
        indexes = set(indexes)
        all_indexes = range(self.outcome_length())
        marginal_indexes = [i for i in all_indexes if i not in indexes]
        d = self.marginal(marginal_indexes, rv_names=False)
        return d

    def set_rv_names(self, rv_names):
        """
        Sets the names of the random variables.

        Returns
        -------
        rv_names : tuple
            A tuple with length equal to the outcome length, containing the
            names of the random variables in the distribution.

        """
        L = self.outcome_length()
        if len(set(rv_names)) < L:
            raise ditException('Too few unique random variable names.')
        elif len(set(rv_names)) > L:
            raise ditException('Too many unique random variable names.')
        self._rvs = dict(zip(rv_names, range(L)))

    def to_string(self, digits=None, exact=False, tol=1e-9, show_mask=False,
                        str_outcomes=False):
        """
        Returns a string representation of the distribution.

        Parameters
        ----------
        digits : int or None
            The probabilities will be rounded to the specified number of
            digits, using NumPy's around function. If `None`, then no rounding
            is performed. Note, if the number of digits is greater than the
            precision of the floats, then the resultant number of digits will
            match that smaller precision.
        exact : bool
            If `True`, then linear probabilities will be displayed, even if
            the underlying pmf contains log probabilities.  The closest
            rational fraction within a tolerance specified by `tol` is used
            as the display value.
        tol : float
            If `exact` is `True`, then the probabilities will be displayed
            as the closest rational fraction within `tol`.
        show_mask : bool
            If `True`, show the outcomes in the proper context. Thus, masked
            and unmasked random variables are shown. If `show_mask` is anything
            other than `True` or `False`, it is used as the wildcard symbol.
        str_outcomes
            If `True`, then attempt to convert outcomes which are tuples to
            just strings.  This is just a dislplay technique.

        Returns
        -------
        s : str
            A string representation of the distribution.

        """
        from .distribution import prepare_string

        from itertools import izip
        from StringIO import StringIO

        s = StringIO()

        x = prepare_string(self, digits, exact, tol, show_mask, str_outcomes)
        pmf, outcomes, base, colsep, max_length, pstr = x

        headers = ["Class",
                   "Alphabet",
                   "Base",
                   "Outcome Class",
                   "Outcome Length"]

        vals = []

        # Class
        vals.append(self.__class__.__name__)

        # Alphabet
        if self.is_homogeneous():
            alpha = str(self.alphabet[0]) + " for all rvs"
        else:
            alpha = str(self.alphabet)
        vals.append(alpha)

        # Base
        vals.append(base)

        # Outcome class
        outcome_class = self._outcome_class
        if outcome_class is not None:
            outcome_class = outcome_class.__name__
        vals.append(outcome_class)

        # Outcome length
        if show_mask:
            outcome_length = "{0} (mask: {1})"
            outcome_length = outcome_length.format(self.outcome_length(),
                                                   len(self._mask))
        else:
            outcome_length = str(self.outcome_length())
        vals.append(outcome_length)

        # Info
        L = max(map(len,headers))
        for head, val in zip(headers, vals):
            s.write("{0}{1}\n".format("{0}: ".format(head).ljust(L+2), val))
        s.write("\n")

        # Distribution
        s.write(''.join([ 'x'.ljust(max_length), colsep, pstr, "\n" ]))
        for o,p in izip(outcomes, pmf):
            s.write(''.join( [o.ljust(max_length), colsep, str(p), "\n"] ))
        s.seek(0)

        s = s.read()
        # Remove the last \n
        s = s[:-1]

        return s


