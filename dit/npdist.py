#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module defining NumPy array-based distribution classes.

One of the features of joint distributions is that we can marginalize them.
This requires that we are able to construct smaller outcomes from larger
outcomes. For example, an outcome like '10101' might become '010' if the first
and last random variables are marginalized.  Given the alphabet of the joint
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

For scalar distributions, the sample space is the alphabet and the alphabet
is a single set. For (joint) distributions, the sample space is provided
at initialization and the alphabet is a tuple of alphabets for each random
variable. The alphabet for each random variable is a tuple.

As of now, dit does not support mixed-type alphabets within a single r.v.
So you can have outcomes like:

    (0, '0'), (1, '1')

but not like:

    (0, '0'), (1, 1)

This has to do with sorting the alphabets. Probably this can be relaxed.

"""

from collections import defaultdict
from operator import itemgetter
import itertools

import numpy as np
from six.moves import map, range, zip # pylint: disable=redefined-builtin

from .npscalardist import ScalarDistribution

from .helpers import (
    construct_alphabets,
    get_outcome_ctor,
    get_product_func,
    parse_rvs,
    reorder,
    RV_MODES,
)

from .samplespace import SampleSpace, CartesianProduct

from .exceptions import (
    InvalidDistribution, InvalidOutcome, ditException
)
from .math import get_ops, LinearOperations
from .params import ditParams

def _make_distribution(outcomes, pmf, base,
                       sample_space=None, prng=None, sparse=True):
    """
    An unsafe, but faster, initialization for distributions.

    If used incorrectly, the data structure will be inconsistent.

    This function can be useful when you are creating many distributions
    in a loop and can guarantee that:

         0) all outcomes are of the same type (eg tuple, str) and length.
         1) the sample space is in the desired order.
         1) outcomes and pmf are in the same order as the sample space.
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

    d._meta['is_joint'] = True
    d._meta['is_numerical'] = True
    d._meta['is_sparse'] = None

    if base is None:
        # Assume default base.
        base = ditParams['base']
    d.ops = get_ops(base)

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

    # Alphabet
    d.alphabet = tuple(construct_alphabets(outcomes))

    # Sample space.
    if sample_space is None:
        d._sample_space = CartesianProduct(d.alphabet, d._product)
    elif isinstance(sample_space, SampleSpace):
        d._sample_space = sample_space
    else:
        d._sample_space = SampleSpace(outcomes)

    # Set the mask
    d._mask = d._new_mask()

    d._meta['is_sparse'] = sparse
    d.rvs = [[i] for i in range(d.outcome_length())]

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

    _sample_space : SampleSpace
        The sample space of the distribution.

    Public Attributes
    -----------------
    alphabet : tuple
        A tuple representing the alphabet of the joint random variable.  The
        elements of the tuple are tuples, each of which represents the ordered
        alphabet for a single random variable.

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

    atoms
        Returns the atoms of the probability space.

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
    _sample_space = None
    _mask = None
    _meta = None
    _outcome_class = None
    _outcome_ctor = None
    _outcomes_index = None
    _product = None
    _rvs = None
    _rv_mode = 'indices'

    ## Advertised attributes.
    alphabet = None
    outcomes = None
    ops = None
    pmf = None
    prng = None

    def __init__(self, outcomes, pmf=None, sample_space=None, base=None,
                            prng=None, sort=True, sparse=True, trim=True,
                            validate=True):
        """
        Initialize the distribution.

        Parameters
        ----------
        outcomes : sequence, dict
            The outcomes of the distribution. If `outcomes` is a dictionary,
            then the keys are used as `outcomes`, and the values of
            the dictionary are used as `pmf` instead.  The values will not be
            used if probabilities are passed in via `pmf`. Outcomes must be
            hashable, orderable, sized, iterable containers. The length of an
            outcome must be the same for all outcomes, and every outcome must
            be of the same type.

        pmf : sequence, None
            The outcome probabilities or log probabilities. `pmf` can be None
            only if `outcomes` is a dict.

        sample_space : sequence, CartesianProduct
            A sequence representing the sample space, and corresponding to the
            complete set of possible outcomes. The order of the sample space
            is important. If `None`, then the outcomes are used to determine
            a Cartesian product sample space instead.

        base : float, str, None
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
            If `True`, then each random variable's alphabets are sorted before
            they are finalized. Usually, this is desirable, as it normalizes
            the behavior of distributions which have the same sample spaces
            (when considered as a set).  Note that addition and multiplication
            of distributions is defined only if the sample spaces are
            compatible.

        sparse : bool
            Specifies the form of the pmf.  If `True`, then `outcomes` and `pmf`
            will only contain entries for non-null outcomes and probabilities,
            after initialization.  The order of these entries will always obey
            the order of `sample_space`, even if their number is not equal to
            the size of the sample space.  If `False`, then the pmf will be
            dense and every outcome in the sample space will be represented.

        trim : bool
            Specifies if null-outcomes should be removed from pmf when
            `make_sparse()` is called (assuming `sparse` is `True`) during
            initialization.

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
        super(ScalarDistribution, self).__init__(prng) # pylint: disable=bad-super-call

        # Set *instance* attributes
        self._meta['is_joint'] = True
        self._meta['is_numerical'] = True
        self._meta['is_sparse'] = None

        # Do any checks/conversions necessary to get the parameters.
        outcomes, pmf = self._init(outcomes, pmf, base)

        if len(outcomes) == 0 and sample_space is None:
            msg = '`outcomes` must be nonempty if no sample space is given'
            raise InvalidDistribution(msg)

        if isinstance(sample_space, SampleSpace):
            if not sample_space._meta['is_joint']:
                msg = '`sample_space` must be a joint sample space.'
                raise InvalidDistribution(msg)

            if sort:
                sample_space.sort()
            self._outcome_class = sample_space._outcome_class
            self._outcome_ctor = sample_space._outcome_ctor
            self._product = sample_space._product
            self._sample_space = sample_space
            if isinstance(sample_space, CartesianProduct):
                alphabets = sample_space.alphabets
            else:
                alphabets = construct_alphabets(sample_space._samplespace)
        else:
            if sample_space is None:
                ss = outcomes
            else:
                ss = sample_space

            alphabets = construct_alphabets(ss)
            if sort:
                alphabets = tuple(map(tuple, map(sorted, alphabets)))

            ## Set the outcome class, ctor, and product function.
            ## Assumption: the class of each outcome is the same.
            klass = ss[0].__class__
            self._outcome_class = klass
            self._outcome_ctor = get_outcome_ctor(klass)
            self._product = get_product_func(klass)

            if sample_space is None:
                self._sample_space = CartesianProduct(alphabets, self._product)
            else:
                self._sample_space = SampleSpace(ss)

        # Sort everything to match the order of the sample space.
        ## Question: Using sort=False seems very strange and supporting it
        ##           makes things harder, since we can't assume the outcomes
        ##           and sample space are sorted.  Is there a valid use case
        ##           for an unsorted sample space?
        if sort and len(outcomes) > 0:
            outcomes, pmf, index = reorder(outcomes, pmf, self._sample_space)
        else:
            index = dict(zip(outcomes, range(len(outcomes))))

        # Force the distribution to be numerical and a NumPy array.
        self.pmf = np.asarray(pmf, dtype=float)


        # Tuple outcomes, and an index.
        self.outcomes = tuple(outcomes)
        self._outcomes_index = index

        self.alphabet = tuple(alphabets)

        self.rvs = [[i] for i in range(self.outcome_length())]

        # Mask
        self._mask = self._new_mask()

        if sparse:
            self.make_sparse(trim=trim)
        else:
            self.make_dense()

        if validate:
            self.validate()

    def _init(self, outcomes, pmf, base):
        """
        Pre-initialization with various sanity checks.

        """
        # Note: We've changed the behavior of _init here.
        # In ScalarDistribution it returns a 3-tuple. Here, a 2-tuple.

        # Attempt to grab outcomes and pmf from a dictionary
        try:
            outcomes_ = tuple(outcomes.keys())
            pmf_ = tuple(outcomes.values())
        except AttributeError:
            pass
        else:
            outcomes = outcomes_
            if pmf is not None:
                msg = '`pmf` must be `None` if `outcomes` is a dict.'
                raise InvalidDistribution(msg)
            pmf = pmf_

        if pmf is None:
            msg = '`pmf` was `None` but `outcomes` was not a dict.'
            raise InvalidDistribution(msg)

        # Make sure pmf and outcomes are sequences
        try:
            len(outcomes)
            len(pmf)
        except TypeError:
            raise TypeError('`outcomes` and `pmf` must be sequences.')

        if len(pmf) != len(outcomes):
            msg = "Unequal lengths for `pmf` and `outcomes`"
            raise InvalidDistribution(msg)

        # reorder() and other functions require that outcomes be indexable. So
        # we make sure it is. We must check for zero length outcomes since, in
        # principle, you can initialize with a 0-length `pmf` and `outcomes`.
        if len(outcomes):
            try:
                outcomes[0]
            except TypeError:
                raise ditException('`outcomes` must be indexable.')

        # Determine if the pmf represents log probabilities or not.
        if base is None:
            # Provide help for obvious case of linear probabilities.
            from .validate import is_pmf
            if is_pmf(np.asarray(pmf, dtype=float), LinearOperations()):
                base = 'linear'
            else:
                base = ditParams['base']
        self.ops = get_ops(base)

        return outcomes, pmf

    def _new_mask(self, from_mask=None, complement=None):
        """
        Creates a new mask for the distribution.

        Parameters
        ----------
        from_mask : iter | None
            Create a mask from an existing mask. If ``None``, then a mask
            will be created which is ``False`` for each random variable.

        complement : bool
            If ``True``, invert the mask that would have been built.
            This includes inverting the mask when ``from_mask=None``.

        Returns
        -------
        mask : tuple
            The newly created mask.

        """
        if from_mask is None:
            L = self.outcome_length(masked=False)
            mask = [False for _ in range(L)]
        else:
            mask = [bool(b) for b in from_mask]

        if complement:
            mask = [not b for b in mask]

        mask = tuple(mask)

        self._mask = mask
        return mask

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
            with an API compatible to NumPy's RandomState class. If `None`,
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
            d.prng = np.random.RandomState()
        else:
            # Use specified prng.
            d.prng = prng

        return d

    @classmethod
    def from_ndarray(cls, ndarray, base=None, prng=None):
        """
        Construct a Distribution from a pmf stored as an ndarray.

        Parameters
        ----------
        ndarray : np.ndarray
            pmf in the form of an ndarray, where each axis is a variable and
            the index along that axis is the variable's value.
        base : 'linear', 'e', or float
            Optionally, specify the base of the new distribution. If `None`,
            then the new distribution will be assumed to have a linear
            distribution.
        prng : RandomState
            A pseudo-random number generator with a `rand` method which can
            generate random numbers. For now, this is assumed to be something
            with an API compatible to NumPy's RandomState class. If `None`,
            then we initialize to dit.math.prng.

        Returns
        -------
        d : Distribution
            The distribution resulting from interpreting `ndarray` as a pmf.

        """
        return cls(*zip(*np.ndenumerate(ndarray)), base=base, prng=prng)

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
            outcomes, pmf, index = reorder(self.outcomes, pmf,
                                           self._sample_space)

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
        # If we survived, then all outcomes have the same class.
        # Now, we just need to make sure that class is a sequence.
        v &= validate_sequence(self.outcomes[0])
        return v


    def coalesce(self, rvs, rv_mode=None, extract=False):
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
        rv_mode : str, None
            Specifies how to interpret the elements of `rvs`. Valid options
            are: {'indices', 'names'}. If equal to 'indices', then the elements
            of `rvs` are interpreted as random variable indices. If equal to
            'names', the the elements are interpreted as random variable names.
            If `None`, then the value of `dist._rv_mode` is consulted.
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
        If we have a joint distribution ``d`` over 3 random variables such as:
            A = (X,Y,Z)
        and would like a new joint distribution over 6 random variables:
            B = (X,Y,Z,X,Y,Z)
        then this is achieved as:
            >>> B = d.coalesce([[0,1,2,0,1,2]], extract=True)

        If you want:
            B = ((X,Y), (Y,Z))
        Then you do:
            >>> B = d.coalesce([[0,1],[1,2]])

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
        parse = lambda rv: parse_rvs(self, rv, rv_mode=rv_mode,
                                     unique=False, sort=False)[1]
        indexes = [parse(rv) for rv in rvs]

        # Determine how new outcomes are constructed.
        if len(rvs) == 1 and extract:
            ctor_o = lambda x: x[0]
        else:
            ctor_o = tuple
            if extract:
                raise Exception('Cannot extract with more than one rv.')

        # Determine how elements of new outcomes are constructed.
        ctor_i = self._outcome_ctor

        # Build the distribution.
        factory = lambda: array('d')
        d = defaultdict(factory)
        for outcome, p in self.zipped():
            # Build a list of inner outcomes. "c" stands for "constructed".
            c_outcome = [ctor_i([outcome[i] for i in rv]) for rv in indexes]
            # Build the outer outcome from the inner outcomes.
            c_outcome = ctor_o(c_outcome)
            d[c_outcome].append(p)

        outcomes = tuple(d.keys())
        pmf = map(np.frombuffer, d.values())
        pmf = map(self.ops.add_reduce, pmf)
        pmf = tuple(pmf)

        # Preserve the sample space during coalescing.
        sample_spaces = [self._sample_space.coalesce([idxes], extract=True)
                         for idxes in indexes]
        if isinstance(self._sample_space, CartesianProduct):
            sample_space = CartesianProduct(sample_spaces,
                                            product=itertools.product)
            if extract:
                sample_space = sample_space.alphabets[0]
        else:
            if extract:
                # There is only one sample space: len(indexes) = 1
                sample_space = sample_spaces[0]
            else:
                sample_space = list(zip(*sample_spaces))

        d = Distribution(outcomes, pmf,
                         base=self.get_base(),
                         sort=True,
                         sample_space=sample_space,
                         sparse=self.is_sparse(),
                         validate=False)

        # We do not set the rv names, since these are new random variables.

        # Set the mask
        L = len(indexes)
        d._mask = tuple(False for _ in range(L))

        return d

    def condition_on(self, crvs, rvs=None, rv_mode=None, extract=False):
        """
        Returns distributions conditioned on random variables ``crvs``.

        Optionally, ``rvs`` specifies which random variables should remain.

        NOTE: Eventually this will return a conditional distribution.

        Parameters
        ----------
        crvs : list
            The random variables to condition on.
        rvs : list, None
            The random variables for the resulting conditional distributions.
            Any random variable not represented in the union of ``crvs`` and
            ``rvs`` will be marginalized. If ``None``, then every random
            variable not appearing in ``crvs`` is used.
        rv_mode : str, None
            Specifies how to interpret ``crvs`` and ``rvs``. Valid options are:
            {'indices', 'names'}. If equal to 'indices', then the elements
            of ``crvs`` and ``rvs`` are interpreted as random variable indices.
            If equal to 'names', the the elements are interpreted as random
            varible names. If ``None``, then the value of ``self._rv_mode`` is
            consulted, which defaults to 'indices'.
        extract : bool
            If the length of either ``crvs`` or ``rvs`` is 1 and ``extract`` is
            ``True``, then instead of the new outcomes being 1-tuples, we
            extract the sole element to create scalar distributions.

        Returns
        -------
        cdist : dist
            The distribution of the conditioned random variables.
        dists : list of distributions
            The conditional distributions for each outcome in ``cdist``.

        Examples
        --------
        First we build a distribution P(X,Y,Z) representing the XOR logic gate.

        >>> pXYZ = dit.example_dists.Xor()
        >>> pXYZ.set_rv_names('XYZ')

        We can obtain the conditional distributions P(X,Z|Y) and the marginal
        of the conditioned variable P(Y) as follows::

        >>> pY, pXZgY = pXYZ.condition_on('Y')

        If we specify ``rvs='Z'``, then only 'Z' is kept and thus, 'X' is
        marginalized out::

        >>> pY, pZgY = pXYZ.condition_on('Y', rvs='Z')

        We can condition on two random variables::

        >>> pXY, pZgXY = pXYZ.condition_on('XY')

        The equivalent call using indexes is:

        >>> pXY, pZgXY = pXYZ.condition_on([0, 1], rv_mode='indexes')

        """
        crvs, cindexes = parse_rvs(self, crvs, rv_mode, unique=True, sort=True)
        if rvs is None:
            indexes = set(range(self.outcome_length())) - set(cindexes)
        else:
            rvs, indexes = parse_rvs(self, rvs, rv_mode, unique=True, sort=True)

        union = set(cindexes).union(indexes)
        if len(union) != len(cindexes) + len(indexes):
            raise ditException('`crvs` and `rvs` must have no intersection.')

        # Marginalize the random variables not in crvs or rvs
        if len(union) < self.outcome_length():
            mapping = dict(zip(sorted(union), range(len(union))))
            d = self.marginal(union, rv_mode=RV_MODES.INDICES)
            # Now we need to shift the indices to their new index values.
            cindexes = [mapping[idx] for idx in cindexes]
            indexes = [mapping[idx] for idx in indexes]
        else:
            # Make a copy so we don't have to worry about changing the input
            # distribution when we make it sparse.
            d = self.copy()

        # It's just easier to not worry about conditioning on zero probs.
        sparse = d.is_sparse()
        d.make_sparse()

        # Note that any previous mask of d from the marginalization will be
        # ignored when we take new marginals. This is desirable here.

        cdist = d.marginal(cindexes, rv_mode=RV_MODES.INDICES)
        dist = d.marginal(indexes, rv_mode=RV_MODES.INDICES)
        sample_space = dist._sample_space
        rv_names = dist.get_rv_names()

        ops = d.ops
        base = ops.get_base()
        ctor = d._outcome_ctor

        # A list of indexes of conditioned outcomes for each joint outcome.
        # These are the indexes of w in the pmf of P(w) for each ws in P(ws).
        cidx = cdist._outcomes_index
        coutcomes = [cidx[ctor([o[i] for i in cindexes])] for o in d.outcomes]

        # A list of indexes of outcomes for each joint outcome.
        # These are the indexes of s in the pmf of P(s) for each ws in P(ws).
        idx = dist._outcomes_index
        outcomes = [idx[ctor([o[i] for i in indexes])] for o in d.outcomes]

        cprobs = np.array([ops.invert(cdist.pmf[i]) for i in coutcomes])
        probs = ops.mult(d.pmf, cprobs)

        # Now build the distributions
        pmfs = np.empty((len(cdist), len(dist)), dtype=float)
        pmfs.fill(ops.zero)
        for i, (coutcome, outcome) in enumerate(zip(coutcomes, outcomes)):
            pmfs[coutcome, outcome] = probs[i]
        dists = [Distribution(dist.outcomes, pmfs[i], sparse=sparse,
                 base=base, sample_space=sample_space, validate=False)
                 for i in range(pmfs.shape[0])]

        # Set the masks and r.v. names for each conditional distribution.
        for dd in dists:
            dd._new_mask(from_mask=dist._mask)
            dd.set_rv_names(rv_names)

        if extract:
            if len(cindexes) == 1:
                cdist = ScalarDistribution.from_distribution(cdist)
            if len(indexes) == 1:
                dists = [ScalarDistribution.from_distribution(d) for d in dists]

        return cdist, dists

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
        prng.set_state(self.prng.get_state())

        d = _make_distribution(outcomes=deepcopy(self.outcomes),
                               pmf=np.array(self.pmf, copy=True),
                               base=self.ops.base,
                               sample_space=deepcopy(self._sample_space),
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
        is_atom = outcome in self._sample_space
        if not is_atom:
            # Outcome does not exist in the sample space.
            return False
        elif null:
            # Outcome exists in the sample space and we don't care about
            # whether it represents a null probability.
            return True
        else:
            idx = self._outcomes_index.get(outcome, None)
            if idx is None:
                # Outcome is not represented in pmf and thus, represents
                # a null probability.
                return False
            else:
                # Outcome is in pmf.  We still need to test if it represents
                # a null probability.
                return self.pmf[idx] > self.ops.zero

    def is_homogeneous(self):
        """
        Returns `True` if the alphabet for each random variable is the same.

        """
        if len(self.alphabet) == 0:
            # Degenerate case: No random variables, no alphabet.
            return True

        a1 = self.alphabet[0]
        h = all(a2 == a1 for a2 in self.alphabet[1:])

        return h

    def marginal(self, rvs, rv_mode=None):
        """
        Returns a marginal distribution.

        Parameters
        ----------
        rvs : list
            The random variables to keep. All others are marginalized.
        rv_mode : str, None
            Specifies how to interpret the elements of `rvs`. Valid options
            are: {'indices', 'names'}. If equal to 'indices', then the elements
            of `rvs` are interpreted as random variable indices. If equal to
            'names', the the elements are interpreted as random variable names.
            If `None`, then the value of `self._rv_mode` is consulted.

        Returns
        -------
        d : joint distribution
            A new joint distribution with the random variables in `rvs`
            kept and all others marginalized.

        """
        # For marginals, we must have unique indexes. Additionally, we do
        # not allow the order of the random variables to change. So we sort.
        # We parse the rv_mode now, so that we can reassign their names
        # after coalesce has finished.
        rvs, indexes = parse_rvs(self, rvs, rv_mode, unique=True, sort=True)

        ## Eventually, add in a method specialized for dense distributions.
        ## This one would work only with the pmf, and not the outcomes.

        # Marginalization is a special case of coalescing where there is only
        # one new random variable and it is composed of a strict subset of
        # the original random variables, with no duplicates, that maintains
        # the order of the original random variables.
        d = self.coalesce([indexes], rv_mode=RV_MODES.INDICES, extract=True)

        # Handle parts of d that are not settable through initialization.

        # Set the random variable names
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

    def marginalize(self, rvs, rv_mode=None):
        """
        Returns a new distribution after marginalizing random variables.

        Parameters
        ----------
        rvs : list
            The random variables to marginalize. All others are kept.
        rv_mode : str, None
            Specifies how to interpret the elements of `rvs`. Valid options
            are: {'indices', 'names'}. If equal to 'indices', then the elements
            of `rvs` are interpreted as random variable indices. If equal to
            'names', the the elements are interpreted as random variable names.
            If `None`, then the value of `self._rv_mode` is consulted.


        Returns
        -------
        d : joint distribution
            A new joint distribution with the random variables in `rvs`
            marginalized and all others kept.

        """
        rvs, indexes = parse_rvs(self, rvs, rv_mode)
        indexes = set(indexes)
        all_indexes = range(self.outcome_length())
        marginal_indexes = [i for i in all_indexes if i not in indexes]
        d = self.marginal(marginal_indexes, rv_mode=RV_MODES.INDICES)
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

        if self._rvs is not None:
            # Unsure if we should change this automatically.
            self._rv_mode = 'names'


    def to_string(self, digits=None, exact=None, tol=1e-9, show_mask=False,
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

        from six import StringIO

        if exact is None:
            exact = ditParams['print.exact']

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
        L = max(map(len, headers))
        for head, val in zip(headers, vals):
            s.write("{0}{1}\n".format("{0}: ".format(head).ljust(L+2), val))
        s.write("\n")

        # Distribution
        s.write(''.join(['x'.ljust(max_length), colsep, pstr, "\n"]))
        # Adjust for empty outcomes. Min length should be: len('x') == 1
        max_length = max(1, max_length)
        for o, p in zip(outcomes, pmf):
            s.write(''.join([o.ljust(max_length), colsep, str(p), "\n"]))
        s.seek(0)

        s = s.read()
        # Remove the last \n
        s = s[:-1]

        return s
