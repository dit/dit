#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module defining NumPy array-based distribution classes.

One of the features of joint distributions is that we can marginalize them. This
requires that we are able to construct smaller outcomes from larger outcomes.
For example, an outcome like '10101' might become '010' if the first and last
random variables are marginalized.  Given the alphabet of the joint
distribution, we can construct a tuple such as ('0','1','0') using
itertools.product, but we really want an outcome which is a string. For
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

from .npscalardist import ScalarDistribution

from .helpers import (
    _construct_alphabets,
    construct_alphabets,
    get_outcome_ctor,
    get_product_func,
    parse_rvs,
    reorder
)

from .exceptions import (
    InvalidDistribution, InvalidOutcome, InvalidProbability, ditException
)
from .math import get_ops, LinearOperations
from .params import ditParams

def _make_distribution(pmf, outcomes, base,
                       alphabet=None, prng=None, sparse=True):
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
    d = Distribution.__new__(Distribution)

    # Call init function of BaseDistribution, not of Distribution.
    # This sets the prng.
    super(ScalarDistribution, d).__init__(prng)

    if base is None:
        # Assume default base.
        base = ditParams['base']
    d.ops = get_ops(base)

    if alphabet is None:
        # Use outcomes to obtain the alphabets.
        alphabet = _construct_alphabets(outcomes)

    ## Set the outcome class, ctor, and product function.
    ## Assumption: the class of each outcome is the same.
    klass = outcomes[0].__class__
    d._outcome_class = klass
    d._outcome_ctor = get_outcome_ctor(klass)
    d._product = get_product_func(klass)

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

    d._meta['is_sparse'] = sparse

    return d

class Distribution(ScalarDistribution):
    """
    A numerical distribution for joint random variables.

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

    _mask : tuple
        A tuple of booleans specifying if the corresponding random variable
        has been masked or not.

    _meta : dict
        A dictionary containing the meta information, described above.

    _outcome_class : class
        The class of all outcomes in the distribution.

    _outcome_ctor : callable
        A callable responsible for converting tuples to outcomes.

    _outcomes_index : dict
        A dictionary mapping outcomes to their index in self.outcomes.

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
    from_distribution
        Alternative constructor from an existing distribution.

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
    array, respectively. The sequences can both be sparse or dense.  By sparse,
    we do not mean that the representation is a NumPy sparse array.  Rather,
    we mean that the sequences need not contain every outcome in the sample
    space. The order of the outcomes and probabilities will always match the
    order of the sample space, even though their length might not equal the
    length of the sample space.

    """
    ## Unadvertised attributes
    _alphabet_set = None
    _mask = None
    _meta = {
        'is_joint': True,
        'is_numerical': True,
        'is_sparse': None,
    }
    _outcome_class = None
    _outcome_ctor = None
    _outcomes_index = None
    _product = None
    _rvs = None

    ## Advertised attributes.
    alphabet = None
    outcomes = None
    ops = None
    pmf = None
    prng = None

    def __init__(self, pmf, outcomes=None, alphabet=None, base=None, prng=None,
                            sort=True, sparse=True, trim=False, validate=True):
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

        prng : RandomState
            A pseudo-random number generator with a `rand` method which can
            generate random numbers. For now, this is assumed to be something
            with an API compatibile to NumPy's RandomState class. This attribute
            is initialized to equal dit.math.prng.

        sort : bool
            If `True`, then each random variable's alphabets are sorted.
            Usually, this is desirable, as it normalizes the behavior of
            distributions which have the same sample spaces (when considered as
            a set).  Note that addition and multiplication of distributions is
            defined only if the sample spaces are compatible.

        sparse : bool
            Specifies the form of the pmf.  If `True`, then `outcomes` and `pmf`
            will only contain entries for non-null outcomes and probabilities,
            after initialization.  The order of these entries will always obey
            the order of `sample_space`, even if their number is not equal to
            the size of the sample space.  If `False`, then the pmf will be
            dense and every outcome in the sample space will be represented.

        trim : bool
            Specifies if null-outcomes should be removed from the sample space
            before it is finalized.  In general, the sample space cannot be
            changed once the distribution has been created.

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
        # Note, we are not calling ScalarDistribution.__init__
        # Instead, we want to call BaseDistribution.__init__.
        # And BaseDistribution is the parent of ScalarDistribution.
        # We do this because we want to init the prng AND ignore everything
        # that ScalarDistribution does.
        super(ScalarDistribution, self).__init__(prng)

        # Do any checks/conversions necessary to get the parameters.
        pmf, outcomes, alphabet = self._init(pmf, outcomes,
                                             alphabet, base, trim)

        # Sort everything to match the order of the sample space.
        ## Question: Using sort=False seems very strange and supporting it
        ##           makes things harder, since we can't assume the outcomes
        ##           and sample space are sorted.  Is there a valid use case
        ##           for an unsorted sample space?
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

        if sparse:
            self.make_sparse(trim=trim)
        else:
            self.make_dense()

        if validate:
            self.validate()

    def _init(self, pmf, outcomes, alphabet, base, trim):
        """
        Pre-initialization with various sanity checks.

        """
        # Attempt to grab outcomes and pmf from a dict-like object.
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
            msg = "`pmf` and `outcomes` do not have the same length."
            raise InvalidDistribution(msg)
        elif len(outcomes) == 0:
            # To initialize a distribution, we must know the sample space.
            # This means we must know the alphabet and the outcome class.
            # From these we can obtain the customized product function
            # (and thus, the sample space). So initialization requires at
            # least one outcome. From there, we can build up a minimal
            # alphabet, if necessary. Note, it might seems that having
            # just an alphabet is sufficient, but then we would not know
            # how to combine symbols to form an outcome (e.g. during
            # marginalization). So we must know the outcome class, and so,
            # we must have at least one outcome.
            msg = 'Neither `pmf` nor `outcomes` can have length zero.'
            raise InvalidDistribution(msg)

        # Determine if the pmf represents log probabilities or not.
        if base is None:
            # Provide help for obvious case of linear probabilities.
            from .validate import is_pmf
            if is_pmf(np.asarray(pmf, dtype=float), LinearOperations()):
                base = 'linear'
            else:
                base = ditParams['base']
        self.ops = get_ops(base)

        if trim:
            # Remove any outcome probability if it is a null probability.
            ops = self.ops
            zipped = zip(pmf, outcomes)
            pairs = [(p, o) for p, o in zipped if not ops.is_null(p)]
            pmf, outcomes = zip(*pairs)

        ## alphabet
        if alphabet is None:
            # Use `outcomes` to obtain the alphabets.
            alphabet = construct_alphabets(outcomes)

        ## Set the outcome class, ctor, and product function.
        ## Assumption: the class of each outcome is the same.
        klass = outcomes[0].__class__
        self._outcome_class = klass
        self._outcome_ctor = get_outcome_ctor(klass)
        self._product = get_product_func(klass)

        return pmf, outcomes, alphabet

    @classmethod
    def from_distribution(cls, dist, base=None, prng=None):
        """
        Returns a new Distribution from an existing distribution.

        Parameters
        ----------
        dist : Distribution, ScalarDistribution
            The existing distribution

        base : 'linear', 'e', or float
            Optionally, change the base of the new distribution. If `None`,
            then the new distribution will have the same base as the existing
            distribution.

        prng : RandomState
            A pseudo-random number generator with a `rand` method which can
            generate random numbers. For now, this is assumed to be something
            with an API compatibile to NumPy's RandomState class. If `None`,
            then we initialize to dit.math.prng. Importantly, we do not
            copy the prng of the existing distribution. For that, see copy().

        Returns
        -------
        d : Distribution
            The new distribution.

        """
        if dist.is_joint():
            if not isinstance(dist, ScalarDistribution):
                raise NotImplementedError
            else:
                # Assume it is a Distribution.
                # Easiest way is to just copy it and then override the prng.
                d = dist.copy(base=base)
        else:
            if not isinstance(dist, ScalarDistribution):
                raise NotImplementedError
            else:
                # Assume it is a ScalarDistribution
                from .convert import SDtoD
                d = SDtoD(dist)
                if base is not None:
                    d.set_base(base)

        if prng is None:
            # Do not use copied prng.
            d.prng = dit.math.prng
        else:
            # Use specified prng.
            d.prng = prng

        return d

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
        if not self.has_outcome(outcome, null=True):
            # Then, the outcome is not in the sample space.
            raise InvalidOutcome(outcome)

        idx = self._outcomes_index.get(outcome, None)
        new_outcome = idx is None

        if not new_outcome:
            # If the distribution is dense, we will always be here.
            # If the distribution is sparse, then we are here for an existing
            # outcome.  In the sparse case, we *could* delete the outcome
            # if the value was zero, but we have choosen to let setting always
            # "set" and deleting always "delete".
            self.pmf[idx] = value
        else:
            # Thus, the outcome is new in a sparse distribution. Even if the
            # value is zero, we still set the value and add it to pmf.

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

    def _validate_outcomes(self):
        """
        Returns `True` if the outcomes are valid.

        Valid means each outcome is in the sample space (and thus of the
        proper class and proper length) and also that the outcome class
        supports the Sequence idiom.

        Returns
        -------
        v : bool
            `True` if the outcomes are valid.

        Raises
        ------
        InvalidOutcome
            When an outcome is not in the sample space.

        """
        from .validate import validate_sequence

        v = super(Distribution, self)._validate_outcomes()
        # If we surived, then all outcomes have the same class since the
        # sample space is created by a product function. Now, we just need
        # to make sure that class is a sequence.
        v &= validate_sequence(self.outcomes[0])
        return v


    def coalesce(self, rvs, rv_names=None, extract=False):
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

        # We allow repeats and want to keep the order. We don't need the names.
        parse = lambda rv : parse_rvs(self, rv, rv_names=rv_names,
                                                unique=False, sort=False)[1]
        indexes = [parse(rv) for rv in rvs]

        # Determine how new outcomes are constructed.
        if len(rvs) == 1 and extract:
            ctor_o = lambda x: x[0]
        else:
            ctor_o = tuple
        # Determine how elements of new outcomes are constructed.
        ctor_i = self._outcome_ctor

        # Build the distribution.
        factory = lambda : array('d')
        d = defaultdict(factory)
        for outcome, p in self.zipped():
            # Build a list of inner outcomes. "c" stands for "constructed".
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

        d = Distribution(pmf, outcomes,
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

    def copy(self, base=None):
        """
        Returns a (deep) copy of the distribution.

        Parameters
        ----------
        base : 'linear', 'e', or float
            Optionally, copy and change the base of the copied distribution.
            If `None`, then the copy will keep the same base.

        """
        from copy import deepcopy

        # Make an exact copy of the PRNG.
        prng = np.random.RandomState()
        prng.set_state( self.prng.get_state() )

        d = _make_distribution(pmf=np.array(self.pmf, copy=True),
                               outcomes=deepcopy(self.outcomes),
                               base=self.ops.base,
                               alphabet=deepcopy(self.alphabet),
                               prng=prng,
                               sparse=self._meta['is_sparse'])

        if base is not None:
            d.set_base(base)

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
            # Equivalently: len(self.outcomes[0])
            # Recall, self.alphabet contains only the unmasked/valid rvs.
            return len(self.alphabet)

    def sample_space(self):
        """
        Returns an iterator over the ordered outcome space.

        """
        # Note, this is a restriction of Distribution.
        # It only deals with sigma-algebras whose sample space is equal to
        # the Cartesian product of each random variable's sample space.
        # We might generalize this in the future.
        return self._product(*self.alphabet)

    def get_rv_names(self):
        """
        Returns the names of the random variables.

        Returns
        -------
        rv_names : tuple or None
            A tuple with length equal to the outcome length, containing the
            names of the random variables in the distribution.  If no random
            variable names have been set, then None is returned.

        """
        if self._rvs is None:
            rv_names = None
        else:
            # _rvs is a dict mapping random variable names to indexes.
            rv_names = [x for x in self._rvs.items()]
            # Sort by index.
            rv_names.sort(key=itemgetter(1))
            # Keep only the sorted names.
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
        This is an O( len(outcome) ) operation.

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
        if len(self.alphabet) == 0:
            # Degenerate case: No random variables, no alphabet.
            return True

        a1 = self._alphabet_set[0]
        h = False
        for a2 in self._alphabet_set[1:]:
            if a1 != a2:
                break
        else:
            h = True

        return h

    def marginal(self, rvs, rv_names=None):
        """
        Returns a marginal distribution.

        Parameters
        ----------
        rvs : list
            The random variables to keep. All others are marginalized.
        rv_names : bool
            If `True`, then the elements of `rvs` are treated as names of
            random variables. If `False`, then the elements of `rvs` are
            treated as indexes of random variables. If `None`, then the value
            `True` is used if the distribution has set names for its random
            variables; otherwise it is set to `False`.

        Returns
        -------
        d : joint distribution
            A new joint distribution with the random variables in `rvs`
            kept and all others marginalized.

        """
        # For marginals, we do must have unique indexes. Additionally, we do
        # not allow the order of the random variables to change. So we sort.
        # We parse the rv_names now, so that we can reassign their names
        # after coalesce has finished.
        rvs, indexes = parse_rvs(self, rvs, rv_names, unique=True, sort=True)

        ## Eventually, add in a method specialized for dense distributions.
        ## This one would work only with the pmf, and not the outcomes.

        # Marginalization is a special case of coalescing where there is only
        # one new random variable and it is composed of a strict subset of
        # the orignal random variables, with no duplicates, that maintains
        # the order of the original random variables.
        d = self.coalesce([indexes], rv_names=False, extract=True)

        # Handle parts of d that are not settable through initialization.

        # Set the random variable names
        if rv_names:
            names = rvs
        else:
            if self._rvs is None:
                # There are no names...
                names = None
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

    def marginalize(self, rvs, rv_names=None):
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
        if rv_names is None:
            # This is an explicit clearing of the rv names.
            rvs = None
        else:
            L = self.outcome_length()
            if len(set(rv_names)) < L:
                raise ditException('Too few unique random variable names.')
            elif len(set(rv_names)) > L:
                raise ditException('Too many unique random variable names.')
            if L > 0:
                rvs = dict(zip(rv_names, range(L)))
            else:
                # This is a corner case of a distribution with 0 rvs.
                # We keep rvs equal to None, instead of an empty dict.
                rvs = None

        self._rvs = rvs

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

        headers = [
            "Class",
            "Alphabet",
            "Base",
            "Outcome Class",
            "Outcome Length",
            "RV Names"
        ]

        vals = []

        # Class
        vals.append(self.__class__.__name__)

        # Alphabet
        if len(self.alphabet) == 0:
            alpha = "()"
        elif self.is_homogeneous():
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

        # Random variable names
        rv_names = self.get_rv_names()
        vals.append(rv_names)

        # Info
        L = max(map(len,headers))
        for head, val in zip(headers, vals):
            s.write("{0}{1}\n".format("{0}: ".format(head).ljust(L+2), val))
        s.write("\n")

        # Distribution
        s.write(''.join([ 'x'.ljust(max_length), colsep, pstr, "\n" ]))
        # Adjust for empty outcomes. Min length should be: len('x') == 1
        max_length = max(1, max_length)
        for o,p in izip(outcomes, pmf):
            s.write(''.join( [o.ljust(max_length), colsep, str(p), "\n"] ))
        s.seek(0)

        s = s.read()
        # Remove the last \n
        s = s[:-1]

        return s


