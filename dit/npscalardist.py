#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module defining NumPy array-based distribution classes.

Sparse means the length of outcomes and pmf need not equal the length of the
sample space. There are two important points to keep in mind.
    1) A sparse distribution is not necessarily trim. Recall, a distribution
         is trim if its pmf does not contain null-outcomes.
    2) The length of a sparse distribution's pmf can equal the length of the
       sample space.
If a distribution is dense, then we forcibly make sure the length of the pmf
is always equal to the length of the sample space.

When a distribution is sparse, del d[e] will make the pmf smaller. But d[e] = 0
will simply set the element to zero.

When a distribution is dense, del d[e] will only set the element to zero---the
length of the pmf will still equal the length of the sample space. Also, using
d[e] = 0 still sets the element to zero.

For scalar distributions, the sample space is the alphabet and the alphabet
is a single set. For (joint) distributions, the sample space is provided
at initialization and the alphabet is a tuple of alphabets for each random
variable. The alphabet for each random variable is a set.

"""
import numbers
from collections import defaultdict
from itertools import product

from .distribution import BaseDistribution
from .exceptions import (
    ditException,
    InvalidDistribution,
    InvalidOutcome,
)

from .math import get_ops, LinearOperations, close
from .params import ditParams
from .helpers import flatten, reorder
from .samplespace import BaseSampleSpace, ScalarSampleSpace

import numpy as np

def _make_distribution(outcomes, pmf=None, sample_space=None,
                            base=None, prng=None, sparse=True):
    """
    An unsafe, but faster, initialization for distributions.

    If used incorrectly, the data structure will be inconsistent.

    This function can be useful when you are creating many distributions
    in a loop and can guarantee that:

        0) the sample space is in the desired order.
        1) outcomes and pmf are in the same order as the sample space.
           [Thus, `pmf` should not be a dictionary.]

    This function will not order the sample space, nor will it reorder outcomes
    or pmf.  It will not forcibly make outcomes and pmf to be sparse or dense.
    It will simply declare the distribution to be sparse or dense. The
    distribution is not validated either.

    Returns
    -------
    d : ScalarDistribution
        The new distribution.

    """
    d = ScalarDistribution.__new__(ScalarDistribution)
    super(ScalarDistribution, d).__init__(prng)

    # Determine if the pmf represents log probabilities or not.
    if base is None:
        base = ditParams['base']
    d.ops = get_ops(base)

    ## pmf and outcomes
    if pmf is None:
        ## Then outcomes must be the pmf.
        pmf = outcomes
        outcomes = range(len(pmf))

    # Force the distribution to be numerical and a NumPy array.
    d.pmf = np.asarray(pmf, dtype=float)

    # Tuple outcomes, and an index.
    d.outcomes = tuple(outcomes)
    d._outcomes_index = dict(zip(outcomes, range(len(outcomes))))


    if isinstance(sample_space, BaseSampleSpace):
        if sample_space._meta['is_joint']:
            msg = '`sample_space` must be a scalar sample space.'
            raise InvalidDistribution(msg)
        d._sample_space = sample_space
    else:
        if sample_space is None:
            sample_space = outcomes
        d._sample_space = ScalarSampleSpace(sample_space)

    # For scalar dists, the alphabet is the sample space.
    d.alphabet = tuple(d._sample_space)

    d._meta['is_sparse'] = sparse

    return d

class ScalarDistribution(BaseDistribution):
    """
    A numerical distribution.

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
    _sample_space : tuple
        A tuple representing the sample space of the probability space.

    _outcomes_index : dict
        A dictionary mapping outcomes to their index in self.outcomes.

    _meta : dict
        A dictionary containing the meta information, described above.

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

    copy
        Returns a deep copy of the distribution.

    sample_space
        Returns an iterator over the outcomes in the sample space.

    get_base
        Returns the base of the distribution.

    has_outcome
        Returns `True` is the distribution has `outcome` in the sample space.

    is_dense
        Returns `True` if the distribution is dense.

    is_joint
        Returns `True` if the distribution is a joint distribution.

    is_log
        Returns `True` if the distribution values are log probabilities.

    is_numerical
        Returns `True` if the distribution values are numerical.

    is_sparse
        Returns `True` if the distribution is sparse.

    make_dense
        Add all null outcomes to the pmf.

    make_sparse
        Remove all null outcomes from the pmf.

    normalize
        Normalizes the distribution.

    rand
        Returns a random draw from the distribution.

    set_base
        Changes the base of the distribution, in-place.

    to_string
        Returns a string representation of the distribution.

    validate
        A method to validate that the distribution is valid.

    zipped
        Returns an iterator over (outcome, probability) tuples.  The
        probability could be a log probability or a linear probability.

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

    _sample_space = None
    _outcomes_index = None
    _meta = None

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
            the dictionary are used as `pmf` instead.  Note: an outcome is
            any hashable object (except `None`) which is equality comparable.
            If `sort` is `True`, then outcomes must also be orderable.

        pmf : sequence
            The outcome probabilities or log probabilities.  If `None`, then
            `outcomes` is treated as the probability mass function and the
            outcomes are consecutive integers beginning from zero.

        sample_space : sequence
            A sequence representing the sample space, and corresponding to the
            complete set of possible outcomes. The order of the sample space
            is important. If `None`, then the outcomes are used to determine
            the sample space instead.

        base : float, None
            If `pmf` specifies log probabilities, then `base` should specify
            the base of the logarithm.  If 'linear', then `pmf` is assumed to
            represent linear probabilities.  If `None`, then the value for
            `base` is taken from ditParams['base'].

        prng : RandomState
            A pseudo-random number generator with a `rand` method which can
            generate random numbers. For now, this is assumed to be something
            with an API compatible to NumPy's RandomState class. This attribute
            is initialized to equal dit.math.prng.

        sort : bool
            If `True`, then the sample space is sorted before finalizing it.
            Usually, this is desirable, as it normalizes the behavior of
            distributions which have the same sample space (when considered
            as a set).  Note that addition and multiplication of distributions
            is defined only if the sample spaces (as tuples) are equal.

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

        See :meth:`validate` for a list of other potential exceptions.

        """
        super(ScalarDistribution, self).__init__(prng)

        # Set *instance* attributes.
        self._meta['is_joint'] = False
        self._meta['is_numerical'] = True
        self._meta['is_sparse'] = None

        if pmf is None and not isinstance(outcomes, dict):
            # If we make it through the checks, the outcomes will be integers.
            sort = False

        outcomes, pmf, skip_sort = self._init(outcomes, pmf, base)

        ## alphabets
        if len(outcomes) == 0 and sample_space is None:
            msg = '`outcomes` must be nonempty if no sample space is given'
            raise InvalidDistribution(msg)

        if isinstance(sample_space, BaseSampleSpace):
            if sample_space._meta['is_joint']:
                msg = '`sample_space` must be a scalar sample space.'
                raise InvalidDistribution(msg)

            if sort:
                sample_space.sort()
            self._sample_space = sample_space

        else:
            if sample_space is None:
                sample_space = outcomes

            if sort:
                sample_space = list(sorted(sample_space))
            self._sample_space = ScalarSampleSpace(sample_space)

        ## Question: Using sort=False seems very strange and supporting it
        ##           makes things harder, since we can't assume the outcomes
        ##           and sample space are sorted.  Is there a valid use case
        ##           for an unsorted sample space?
        if sort and len(outcomes) > 0 and not skip_sort:
            outcomes, pmf, index = reorder(outcomes, pmf, self._sample_space)
        else:
            index = dict(zip(outcomes, range(len(outcomes))))

        # Force the distribution to be numerical and a NumPy array.
        self.pmf = np.asarray(pmf, dtype=float)

        # Tuple outcomes, and an index.
        self.outcomes = tuple(outcomes)
        self._outcomes_index = index

        # For scalar dists, the alphabet is the sample space.
        self.alphabet = tuple(self._sample_space)

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
        # If we generate integer outcomes, then we can skip the sort.
        skip_sort = False

        ## pmf
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
                raise ditException(msg)
            pmf = pmf_

        if pmf is None:
            # Use the outcomes as the pmf and generate integers as outcomes.
            pmf = outcomes
            try:
                np.asarray(pmf, dtype=float)
            except ValueError:
                msg = 'Failed to convert `outcomes` to an array.'
                raise ditException(msg)
            outcomes = range(len(pmf))
            skip_sort = True

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

        return outcomes, pmf, skip_sort

    @classmethod
    def from_distribution(cls, dist, base=None, prng=None, extract=True):
        """
        Returns a new ScalarDistribution from an existing distribution.

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

        extract : bool
            If `True`, then when converting from a Distribution whose outcome
            length is 1, we extract the sole element from each length-1
            sequence and use it as a scalar outcome.  If the outcome length
            is not equal to one, then no additional change is made. If `False`,
            then no extraction is attempted.

        Returns
        -------
        d : ScalarDistribution
            The new distribution.

        """
        from .npdist import Distribution # pylint: disable=cyclic-import
        if isinstance(dist, Distribution):
            from .convert import DtoSD
            d = DtoSD(dist, extract)
            if base is not None:
                d.set_base(base)
        else:
            # Easiest way is to just copy it and then override the prng.
            d = dist.copy(base=base)

        if prng is None:
            # Do not use copied prng.
            d.prng = np.random.RandomState()
        else:
            # Use specified prng.
            d.prng = prng

        return d

    @classmethod
    def from_rv_discrete(cls, ssrv, prng=None):
        """
        Create a ScalarDistribution from a scipy.states.rv_discrete instance.

        Parameters
        ----------
        ssrv : scipy.stats.rv_discrete
            The random variable to convert to a dit.ScalarDistribution.
        prng : RandomState
            A pseudo-random number generator with a `rand` method which can
            generate random numbers. For now, this is assumed to be something
            with an API compatibile to NumPy's RandomState class. If `None`,
            then we initialize to dit.math.prng.

        Returns
        -------
        d : ScalarDistribution
            A ScalarDistribution representation of `ssrv`.
        """
        outcomes = ssrv.xk
        pmf = ssrv.pk
        prng = prng if (prng is not None) else ssrv.random_state
        return cls(outcomes, pmf, prng=prng)

    def __meta_magic(self, other, op, msg):
        """
        Private method to implement generic comparisons or operations
        outcome-wise between ScalarDistributions and either scalars or another
        ScalarDistribution.

        Parameters
        ----------
        other : ScalarDistribution or scalar
            The object to compare or operate against.
        op : function
            The comparison or operator to implement.
        msg : str
            The error message to raise if the objects are not comparible.
        """
        if self.is_joint():
            msg = "This operation is not supported for joint distribibutions."
            raise NotImplementedError(msg)

        if isinstance(other, ScalarDistribution):
            # pylint: disable=cyclic-import
            from .distconst import _combine_scalar_dists
            return _combine_scalar_dists(self, other, op)

        elif isinstance(other, numbers.Number):
            # pylint: disable=cyclic-import
            from .distconst import modify_outcomes
            d = modify_outcomes(self, lambda x: op(x, other))
            return d

        else:
            raise NotImplementedError(msg.format(type(self), type(other)))


    def __add__(self, other):
        """
        Addition of ScalarDistribution with either a scalar or another
        Scalardistribution.

        Example
        -------
        >>> d6 + 3
        Class:    ScalarDistribution
        Alphabet: (4, 5, 6, 7, 8, 9)
        Base:     linear

        x   p(x)
        4   1/6
        5   1/6
        6   1/6
        7   1/6
        8   1/6
        9   1/6

        >>> d6 + d6
        Class:    ScalarDistribution
        Alphabet: (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
        Base:     linear

        x    p(x)
        2    1/36
        3    1/18
        4    1/12
        5    1/9
        6    5/36
        7    1/6
        8    5/36
        9    1/9
        10   1/12
        11   1/18
        12   1/36
        """
        op = lambda x, y: x + y
        msg = "Cannot add types {0} and {1}"
        return self.__meta_magic(other, op, msg)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        """
        Subtraction of ScalarDistribution with either a scalar or another
        Scalardistribution.

        Example
        -------
        >>> d6 - 3
        Class:    ScalarDistribution
        Alphabet: (-2, -1, 0, 1, 2, 3)
        Base:     linear

        x    p(x)
        -2   1/6
        -1   1/6
        0    1/6
        1    1/6
        2    1/6
        3    1/6

        >>> d6 - d6
        Class:    ScalarDistribution
        Alphabet: (-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5)
        Base:     linear

        x    p(x)
        -5   1/36
        -4   1/18
        -3   1/12
        -2   1/9
        -1   5/36
        0    1/6
        1    5/36
        2    1/9
        3    1/12
        4    1/18
        5    1/36
        """
        op = lambda x, y: x - y
        msg = "Cannot subtract types {0} and {1}"
        return self.__meta_magic(other, op, msg)

    def __rsub__(self, other):
        """
        Subtraction of ScalarDistribution from either a scalar or another
        Scalardistribution.

        Example
        -------
        >>> 10 - d6
        Class:    ScalarDistribution
        Alphabet: (4, 5, 6, 7, 8, 9)
        Base:     linear

        x   p(x)
        4   1/6
        5   1/6
        6   1/6
        7   1/6
        8   1/6
        9   1/6

        >>> d6 - d6
        Class:    ScalarDistribution
        Alphabet: (-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5)
        Base:     linear

        x    p(x)
        -5   1/36
        -4   1/18
        -3   1/12
        -2   1/9
        -1   5/36
        0    1/6
        1    5/36
        2    1/9
        3    1/12
        4    1/18
        5    1/36
        """
        op = lambda x, y: y - x
        msg = "Cannot subtract types {1} and {0}"
        return self.__meta_magic(other, op, msg)

    def __mul__(self, other):
        """
        Multiplication of ScalarDistribution with either a scalar or another
        ScalarDistribution.

        Example
        -------
        >>> 2 * d6
        Class:    ScalarDistribution
        Alphabet: (2, 4, 6, 8, 10, 12)
        Base:     linear

        x    p(x)
        2    1/6
        4    1/6
        6    1/6
        8    1/6
        10   1/6
        12   1/6

        >>> d6 * d6
        Class:    ScalarDistribution
        Alphabet: (1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 30, 36)
        Base:     linear

        x    p(x)
        1    1/36
        2    1/18
        3    1/18
        4    1/12
        5    1/18
        6    1/9
        8    1/18
        9    1/36
        10   1/18
        12   1/9
        15   1/18
        16   1/36
        18   1/18
        20   1/18
        24   1/18
        25   1/36
        30   1/18
        36   1/36
        """
        op = lambda x, y: x * y
        msg = "Cannot multiply types {0} and {1}"
        return self.__meta_magic(other, op, msg)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __div__(self, other):
        """
        Division of ScalarDistribution with either a scalar or another
        Scalardistribution.

        Example
        -------
        >>> d6 / 3
        Class:    ScalarDistribution
        Alphabet: (0.3333333333333333, 0.6666666666666666, 1.0, 1.3333333333333333, 1.6666666666666667, 2.0)
        Base:     linear

        x                p(x)
        0.333333333333   1/6
        0.666666666667   1/6
        1.0              1/6
        1.33333333333    1/6
        1.66666666667    1/6
        2.0              1/6

        >>> d6 / d6
        Class:    ScalarDistribution
        Alphabet: (0.16666666666666666, 0.2, 0.25, 0.3333333333333333, 0.4, 0.5, 0.6, 0.6666666666666666, 0.75, 0.8, 0.8333333333333334, 1.0, 1.2, 1.25, 1.3333333333333333, 1.5, 1.6666666666666667, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0)
        Base:     linear

        x                p(x)
        0.166666666667   1/36
        0.2              1/36
        0.25             1/36
        0.333333333333   1/18
        0.4              1/36
        0.5              1/12
        0.6              1/36
        0.666666666667   1/18
        0.75             1/36
        0.8              1/36
        0.833333333333   1/36
        1.0              1/6
        1.2              1/36
        1.25             1/36
        1.33333333333    1/36
        1.5              1/18
        1.66666666667    1/36
        2.0              1/12
        2.5              1/36
        3.0              1/18
        4.0              1/36
        5.0              1/36
        6.0              1/36
        """
        op = lambda x, y: x / y
        msg = "Cannot divide types {0} by {1}"
        return self.__meta_magic(other, op, msg)

    def __rdiv__(self, other):
        """
        Division of ScalarDistribution from either a scalar or another
        Scalardistribution.

        Example
        -------
        >>> 6 / d6
        Class:    ScalarDistribution
        Alphabet: (1.0, 1.2, 1.5, 2.0, 3.0, 6.0)
        Base:     linear

        x     p(x)
        1.0   1/6
        1.2   1/6
        1.5   1/6
        2.0   1/6
        3.0   1/6
        6.0   1/6

        >>> d6 / d6
        Class:    ScalarDistribution
        Alphabet: (0.16666666666666666, 0.2, 0.25, 0.3333333333333333, 0.4, 0.5, 0.6, 0.6666666666666666, 0.75, 0.8, 0.8333333333333334, 1.0, 1.2, 1.25, 1.3333333333333333, 1.5, 1.6666666666666667, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0)
        Base:     linear

        x                p(x)
        0.166666666667   1/36
        0.2              1/36
        0.25             1/36
        0.333333333333   1/18
        0.4              1/36
        0.5              1/12
        0.6              1/36
        0.666666666667   1/18
        0.75             1/36
        0.8              1/36
        0.833333333333   1/36
        1.0              1/6
        1.2              1/36
        1.25             1/36
        1.33333333333    1/36
        1.5              1/18
        1.66666666667    1/36
        2.0              1/12
        2.5              1/36
        3.0              1/18
        4.0              1/36
        5.0              1/36
        6.0              1/36
        """
        op = lambda x, y: y / x
        msg = "Cannot divide types {1} by {0}"
        return self.__meta_magic(other, op, msg)

    def __truediv__(self, other):
        """
        Division of ScalarDistribution with either a scalar or another
        Scalardistribution.

        Example
        -------
        >>> d6 / 3
        Class:    ScalarDistribution
        Alphabet: (0.3333333333333333, 0.6666666666666666, 1.0, 1.3333333333333333, 1.6666666666666667, 2.0)
        Base:     linear

        x                p(x)
        0.333333333333   1/6
        0.666666666667   1/6
        1.0              1/6
        1.33333333333    1/6
        1.66666666667    1/6
        2.0              1/6

        >>> d6 / d6
        Class:    ScalarDistribution
        Alphabet: (0.16666666666666666, 0.2, 0.25, 0.3333333333333333, 0.4, 0.5, 0.6, 0.6666666666666666, 0.75, 0.8, 0.8333333333333334, 1.0, 1.2, 1.25, 1.3333333333333333, 1.5, 1.6666666666666667, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0)
        Base:     linear

        x                p(x)
        0.166666666667   1/36
        0.2              1/36
        0.25             1/36
        0.333333333333   1/18
        0.4              1/36
        0.5              1/12
        0.6              1/36
        0.666666666667   1/18
        0.75             1/36
        0.8              1/36
        0.833333333333   1/36
        1.0              1/6
        1.2              1/36
        1.25             1/36
        1.33333333333    1/36
        1.5              1/18
        1.66666666667    1/36
        2.0              1/12
        2.5              1/36
        3.0              1/18
        4.0              1/36
        5.0              1/36
        6.0              1/36
        """
        op = lambda x, y: (1.0*x) / y
        msg = "Cannot divide types {0} by {1}"
        return self.__meta_magic(other, op, msg)

    def __rtruediv__(self, other):
        """
        Division of ScalarDistribution from either a scalar or another
        Scalardistribution.

        Example
        -------
        >>> 6 / d6
        Class:    ScalarDistribution
        Alphabet: (1.0, 1.2, 1.5, 2.0, 3.0, 6.0)
        Base:     linear

        x     p(x)
        1.0   1/6
        1.2   1/6
        1.5   1/6
        2.0   1/6
        3.0   1/6
        6.0   1/6

        >>> d6 / d6
        Class:    ScalarDistribution
        Alphabet: (0.16666666666666666, 0.2, 0.25, 0.3333333333333333, 0.4, 0.5, 0.6, 0.6666666666666666, 0.75, 0.8, 0.8333333333333334, 1.0, 1.2, 1.25, 1.3333333333333333, 1.5, 1.6666666666666667, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0)
        Base:     linear

        x                p(x)
        0.166666666667   1/36
        0.2              1/36
        0.25             1/36
        0.333333333333   1/18
        0.4              1/36
        0.5              1/12
        0.6              1/36
        0.666666666667   1/18
        0.75             1/36
        0.8              1/36
        0.833333333333   1/36
        1.0              1/6
        1.2              1/36
        1.25             1/36
        1.33333333333    1/36
        1.5              1/18
        1.66666666667    1/36
        2.0              1/12
        2.5              1/36
        3.0              1/18
        4.0              1/36
        5.0              1/36
        6.0              1/36
        """
        op = lambda x, y: (1.0*y) / x
        msg = "Cannot divide types {1} by {0}"
        return self.__meta_magic(other, op, msg)

    def __floordiv__(self, other):
        """
        Integer division of ScalarDistribution with either a scalar or another
        Scalardistribution.

        Example
        -------
        >>> d6 // 3
        Class:    ScalarDistribution
        Alphabet: (0, 1, 2)
        Base:     linear

        x   p(x)
        0   1/3
        1   1/2
        2   1/6

        >>> d6 // d6
        Class:    ScalarDistribution
        Alphabet: (0, 1, 2, 3, 4, 5, 6)
        Base:     linear

        x   p(x)
        0   5/12
        1   1/3
        2   1/9
        3   1/18
        4   1/36
        5   1/36
        6   1/36
        """
        op = lambda x, y: x // y
        msg = "Cannot divide types {0} by {1}"
        return self.__meta_magic(other, op, msg)

    def __rfloordiv__(self, other):
        """
        Integer division of ScalarDistribution from either a scalar or another
        Scalardistribution.

        Example
        -------
        >>> 3 // d6
        Class:    ScalarDistribution
        Alphabet: (0, 1, 3)
        Base:     linear

        x   p(x)
        0   1/2
        1   1/3
        3   1/6

        >>> d6 // d6
        Class:    ScalarDistribution
        Alphabet: (0, 1, 2, 3, 4, 5, 6)
        Base:     linear

        x   p(x)
        0   5/12
        1   1/3
        2   1/9
        3   1/18
        4   1/36
        5   1/36
        6   1/36
        """
        op = lambda x, y: y // x
        msg = "Cannot divide types {1} by {0}"
        return self.__meta_magic(other, op, msg)

    def __mod__(self, other):
        """
        Modulo of ScalarDistribution with either a scalar or another
        Scalardistribution.

        Example
        -------
        >>> d6 % 2
        Class:    ScalarDistribution
        Alphabet: (0, 1)
        Base:     linear

        x   p(x)
        0   1/2
        1   1/2

        >>> d6 % d6
        Class:    ScalarDistribution
        Alphabet: (0, 1, 2, 3, 4, 5)
        Base:     linear

        x   p(x)
        0   7/18
        1   5/18
        2   1/6
        3   1/12
        4   1/18
        5   1/36
        """
        op = lambda x, y: x % y
        msg = "Cannot mod types {0} by {1}"
        return self.__meta_magic(other, op, msg)

    def __rmod__(self, other):
        """
        Modulo of ScalarDistribution from either a scalar or another
        Scalardistribution.

        Example
        -------
        >>> 3 % d6
        Class:    ScalarDistribution
        Alphabet: (0, 1, 3)
        Base:     linear

        x   p(x)
        0   1/3
        1   1/6
        3   1/2

        >>> d6 % d6
        Class:    ScalarDistribution
        Alphabet: (0, 1, 2, 3, 4, 5)
        Base:     linear

        x   p(x)
        0   7/18
        1   5/18
        2   1/6
        3   1/12
        4   1/18
        5   1/36
        """
        op = lambda x, y: y % x
        msg = "Cannot mod types {1} by {0}"
        return self.__meta_magic(other, op, msg)

    def __lt__(self, other):
        """
        Determine the distribution of outcomes that are less than either a
        scalar or another ScalarDistribution.

        Example
        -------
        >>> d6 < 4
        Class:    ScalarDistribution
        Alphabet: (False, True)
        Base:     linear

        x       p(x)
        False   1/2
        True    1/2

        >>> d6 < d6
        Class:    ScalarDistribution
        Alphabet: (False, True)
        Base:     linear

        x       p(x)
        False   7/12
        True    5/12
        """
        op = lambda x, y: x < y
        msg = "Cannot compare types {0} and {1}"
        return self.__meta_magic(other, op, msg)

    def __le__(self, other):
        """
        Determine the distribution of outcomes that are less than or equal to
        either a scalar or another ScalarDistribution.

        Example
        -------
        >>> d6 <= 4
        Class:    ScalarDistribution
        Alphabet: (False, True)
        Base:     linear

        x       p(x)
        False   1/3
        True    2/3

        >>> d6 <= d6
        Class:    ScalarDistribution
        Alphabet: (False, True)
        Base:     linear

        x       p(x)
        False   5/12
        True    7/12
        """
        op = lambda x, y: x <= y
        msg = "Cannot compare types {0} and {1}"
        return self.__meta_magic(other, op, msg)

    def __eq__(self, other):
        """
        Determine the distribution of outcomes that are equal to either a scalar
        or another ScalarDistribution.

        Example
        -------
        >>> d6 == 6
        Class:    ScalarDistribution
        Alphabet: (False, True)
        Base:     linear

        x       p(x)
        False   5/6
        True    1/6

        >>> d6 == d6
        Class:    ScalarDistribution
        Alphabet: (False, True)
        Base:     linear

        x       p(x)
        False   5/6
        True    1/6
        """
        op = lambda x, y: x == y
        msg = "Cannot test equality of types {0} and {1}"
        return self.__meta_magic(other, op, msg)

    def __ne__(self, other):
        """
        Determine the distribution of outcomes that are not equal to either a
        scalar or another ScalarDistribution.

        Example
        -------
        >>> d6 != 6
        Class:    ScalarDistribution
        Alphabet: (False, True)
        Base:     linear

        x       p(x)
        False   1/6
        True    5/6

        >>> d6 != d6
        Class:    ScalarDistribution
        Alphabet: (False, True)
        Base:     linear

        x       p(x)
        False   1/6
        True    5/6
        """
        op = lambda x, y: x != y
        msg = "Cannot test equality of types {0} and {1}"
        return self.__meta_magic(other, op, msg)

    def __gt__(self, other):
        """
        Determine the distribution of outcomes that are greater than either a
        scalar or another ScalarDistribution.

        Example
        -------
        >>> d6 > 3
        Class:    ScalarDistribution
        Alphabet: (False, True)
        Base:     linear

        x       p(x)
        False   1/2
        True    1/2

        >>> d6 > d6
        Class:    ScalarDistribution
        Alphabet: (False, True)
        Base:     linear

        x       p(x)
        False   7/12
        True    5/12
        """
        op = lambda x, y: x > y
        msg = "Cannot compare types {0} and {1}"
        return self.__meta_magic(other, op, msg)

    def __ge__(self, other):
        """
        Determine the distribution of outcomes that are greater than or equal to
        either a scalar or another ScalarDistribution.

        Example
        -------
        >>> d6 >= 4
        Class:    ScalarDistribution
        Alphabet: (False, True)
        Base:     linear

        x       p(x)
        False   1/2
        True    1/2

        >>> d6 >= d6
        Class:    ScalarDistribution
        Alphabet: (False, True)
        Base:     linear

        x       p(x)
        False   5/12
        True    7/12
        """
        op = lambda x, y: x >= y
        msg = "Cannot compare types {0} and {1}"
        return self.__meta_magic(other, op, msg)

    def __matmul__(self, other):
        """
        Construct the cartesian product of `self` and `other`, as though they
        were independent.

        Example
        -------
        >>> d6 @ d6
        Class:          Distribution
        Alphabet:       (1, 2, 3, 4, 5, 6) for all rvs
        Base:           linear
        Outcome Class:  tuple
        Outcome Length: 2
        RV Names:       None

        x        p(x)
        (1, 1)   1/36
        (1, 2)   1/36
        (1, 3)   1/36
        (1, 4)   1/36
        (1, 5)   1/36
        (1, 6)   1/36
        (2, 1)   1/36
        (2, 2)   1/36
        (2, 3)   1/36
        (2, 4)   1/36
        (2, 5)   1/36
        (2, 6)   1/36
        (3, 1)   1/36
        (3, 2)   1/36
        (3, 3)   1/36
        (3, 4)   1/36
        (3, 5)   1/36
        (3, 6)   1/36
        (4, 1)   1/36
        (4, 2)   1/36
        (4, 3)   1/36
        (4, 4)   1/36
        (4, 5)   1/36
        (4, 6)   1/36
        (5, 1)   1/36
        (5, 2)   1/36
        (5, 3)   1/36
        (5, 4)   1/36
        (5, 5)   1/36
        (5, 6)   1/36
        (6, 1)   1/36
        (6, 2)   1/36
        (6, 3)   1/36
        (6, 4)   1/36
        (6, 5)   1/36
        (6, 6)   1/36
        """
        if isinstance(other, ScalarDistribution):
            from .npdist import Distribution # pylint: disable=cyclic-import
            # Copy to make sure we don't lose precision when converting.
            d2 = other.copy(base=self.get_base())

            dist = defaultdict(float)
            for (o1, p1), (o2, p2) in product(self.zipped(), d2.zipped()):
                dist[tuple(flatten((o1, o2)))] += self.ops.mult(p1, p2)

            return Distribution(*zip(*dist.items()), base=self.get_base())
        else:
            msg = "Cannot construct a joint from types {0} and {1}"
            raise NotImplementedError(msg.format(type(self), type(other)))

    def __rmatmul__(self, other): # pragma: no cover
        return other.__matmul__(self)

    def __hash__(self):
        """
        ToDo
        ----
        Create a real hash function here.
        """
        return id(self)

    def __contains__(self, outcome):
        """
        Returns `True` if `outcome` is in self.outcomes.

        Note, the outcome could correspond to a null-outcome if the pmf
        explicitly contains null-probabilities. Also, if `outcome` is not in
        the sample space, then an exception is not raised. Instead, `False`
        is returned.

        """
        # The behavior of this must always match self.zipped(). So since
        # zipped() defaults to mode='pmf', then this checks only if the
        # outcome is in the pmf.
        return outcome in self._outcomes_index

    def __delitem__(self, outcome):
        """
        Deletes `outcome` from the distribution.

        Parameters
        ----------
        outcome : outcome
            Any hashable and equality comparable object. If `outcome` exists
            in the sample space, then it is removed from the pmf---if the
            outcome did not already exist in the pmf, then no exception is
            raised. If `outcome` does not exist in the sample space, then an
            InvalidOutcome exception is raised.

        Raises
        ------
        InvalidOutcome
            If `outcome` does not exist in the sample space.

        Notes
        -----
        If the distribution is dense, then the outcome's value is set to zero,
        and the length of the pmf is left unchanged.

        If the outcome was a non-null outcome, then the resulting distribution
        will no longer be normalized (assuming it was in the first place).

        """
        if not self.has_outcome(outcome, null=True):
            raise InvalidOutcome(outcome)

        outcomes = self.outcomes
        outcomes_index = self._outcomes_index
        if self.is_dense():
            # Dense distribution, just set it to zero.
            idx = outcomes_index[outcome]
            self.pmf[idx] = self.ops.zero
        elif outcome in outcomes_index:
            # Remove the outcome from the sparse distribution.
            # Since the pmf was already ordered, no need to reorder.
            # Update the outcomes and the outcomes index.
            idx = outcomes_index[outcome]
            new_indexes = [i for i in range(len(outcomes)) if i != idx]
            new_outcomes = tuple([outcomes[i] for i in new_indexes])
            self.outcomes = new_outcomes
            self._outcomes_index = dict(zip(new_outcomes,
                                        range(len(new_outcomes))))

            # Update the probabilities.
            self.pmf = self.pmf[new_indexes]

    def __getitem__(self, outcome):
        """
        Returns the probability associated with `outcome`.

        Parameters
        ----------
        outcome : outcome
            Any hashable and equality comparable object in the sample space.
            If `outcome` does not exist in the sample space, then an
            InvalidOutcome exception is raised.

        Returns
        -------
        p : float
            The probability (or log probability) of the outcome.

        Raises
        ------
        InvalidOutcome
            If `outcome` does not exist in the sample space.

        """
        if not self.has_outcome(outcome, null=True):
            raise InvalidOutcome(outcome)

        idx = self._outcomes_index.get(outcome, None)
        if idx is None:
            p = self.ops.zero
        else:
            p = self.pmf[idx]
        return p

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

        Setting a new outcome in a sparse distribution is costly. It is better
        to know the non-null outcomes before creating the distribution.

        """
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
            # We add the outcome and its value, regardless if the value is zero.

            # 1. Add the new outcome and probability
            self.outcomes = self.outcomes + (outcome,)
            self._outcomes_index[outcome] = len(self.outcomes) - 1
            pmf = [p for p in self.pmf] + [value]

            # 2. Reorder
            outcomes, pmf, index = reorder(self.outcomes, pmf,
                                           self._sample_space,
                                           index=self._outcomes_index)

            # 3. Store
            self.outcomes = tuple(outcomes)
            self._outcomes_index = index
            self.pmf = np.array(pmf, dtype=float)

    def copy(self, base=None):
        """
        Returns a (deep) copy of the distribution.

        Parameters
        ----------
        base : 'linear', 'e', or float
            Optionally, copy and change the base of the copied distribution.
            If `None`, then the copy will keep the same base.

        """
        # For some reason, we can't just return a deepcopy of self.
        # It works for linear distributions but not for log distributions.

        from copy import deepcopy

        # Make an exact copy of the PRNG.
        prng = np.random.RandomState()
        prng.set_state(self.prng.get_state())

        d = _make_distribution(outcomes=deepcopy(self.outcomes),
                               pmf=np.array(self.pmf, copy=True),
                               sample_space=deepcopy(self._sample_space),
                               base=self.ops.base,
                               prng=prng,
                               sparse=self._meta['is_sparse'])
        if base is not None:
            d.set_base(base)

        return d

    def sample_space(self):
        """
        Returns an iterator over the ordered outcome space.

        """
        return iter(self._sample_space)

    def has_outcome(self, outcome, null=True):
        """
        Returns `True` if `outcome` exists in the sample space.

        Parameters
        ----------
        outcome : outcome
            The outcome to be tested.
        null : bool
            Specifies if null outcomes are acceptable.  If `True`, then null
            outcomes are acceptable.  Thus, the only requirement on `outcome`
            is that it exist in the distribution's sample space. If `False`,
            then null outcomes are not acceptable.  Thus, `outcome` must exist
            in the distribution's sample space and also have a nonnull
            probability in order to return `True`.

        Notes
        -----
        This is an O(1) operation.

        """
        if null:
            # Make sure the outcome exists in the sample space, which equals
            # the alphabet for distributions.
            z = outcome in self._sample_space
        else:
            # Must be valid and have positive probability.
            try:
                z = self[outcome] > self.ops.zero
            except InvalidOutcome:
                z = False

        return z

    def is_approx_equal(self, other, rtol=None, atol=None):
        """
        Returns `True` is `other` is approximately equal to this distribution.

        For two distributions to be equal, they must have the same sample space
        and must also agree on the probabilities of each outcome.

        Parameters
        ----------
        other : distribution
            The distribution to compare against.
        rtol : float
            The relative tolerance to use when comparing probabilities.
            See :func:`dit.math.close` for more information.
        atol : float
            The absolute tolerance to use when comparing probabilities.
            See :func:`dit.math.close` for more information.

        Notes
        -----
        The distributions need not have the length, but they must have the
        same base.

        """
        if rtol is None:
            rtol = ditParams['rtol']
        if atol is None:
            atol = ditParams['atol']

        # We assume the distributions are properly normalized.

        # Potentially nonzero probabilities from self must equal those from
        # others. No need to check the other way around since we will verify
        # that the sample spaces are equal.
        for outcome in self.outcomes:
            if not close(self[outcome], other[outcome], rtol=rtol, atol=atol):
                return False

        # Outcome spaces must be equal.
        if self._sample_space != other._sample_space:
            return False

        return True

    def is_dense(self):
        """
        Returns `True` if the distribution is dense and `False` otherwise.

        """
        return not self.is_sparse()

    def is_sparse(self):
        """
        Returns `True` if the distribution is sparse and `False` otherwise.

        """
        return self._meta['is_sparse']

    def normalize(self):
        """
        Normalize the distribution, in-place.

        Returns
        -------
        z : float
            The previous normalization constant.  This will be negative if
            the distribution represents log probabilities.

        """
        ops = self.ops
        pmf = self.pmf
        z = ops.add_reduce(pmf)
        ops.mult_inplace(pmf, ops.invert(z))
        return z

    def set_base(self, base):
        """
        Changes the base of the distribution.

        If the distribution is a linear distribution and the base is changed,
        the the distribution will be subsequently represent log distributions.
        If the distribution is a log distribution, then it can be converted to
        another base by passing in some other base. Alternatively, one can
        convert to a linear distribution by passing 'linear'.

        Generally, it is dangerous to change the base in-place, as numerical
        errors can be introduced (especially when converting from very negative
        log probabilities to linear probabilities). Additionally, functions or
        classes that hold references to the distribution may not expect a
        change in base. For this reason, one should prefer to use self.copy()
        along with the `base` parameter.

        Parameters
        ----------
        base : float or string
            The desired base for the distribution.  If 'linear', then the
            distribution's pmf will represent linear probabilities. If any
            positive float (other than 1) or 'e', then the pmf will represent
            log probabilities with the specified base.

        See Also
        --------
        copy

        """
        from .math import LinearOperations, LogOperations
        from .params import validate_base

        # Sanitize inputs
        base = validate_base(base)

        # Determine the conversion targets.
        from_log = self.is_log()
        if base == 'linear':
            to_log = False
            new_ops = LinearOperations()
        else:
            to_log = True
            new_ops = LogOperations(base)

        # If self.ops is None, then we are initializing the distribution.
        # self.pmf will be set by the __init__ function.

        if self.ops is not None:
            # Then we are converting.
            old_ops = self.ops

            # In order to do conversions, we need a numerical value for base.
            old_base = old_ops.get_base(numerical=True)

            # Caution: The in-place multiplication ( *= ) below will work only
            # if pmf has a float dtype.  If not (e.g., dtype=int), then the
            # multiplication gives incorrect results due to coercion. The
            # __init__ function is responsible for guaranteeing the dtype.
            # So we proceed assuming that in-place multiplication works for us.

            if from_log and to_log:
                # Convert from one log base to another.
                ## log_b(x) = log_b(a) * log_a(x)
                self.pmf *= new_ops.log(old_base)
            elif not from_log and not to_log:
                # No conversion: from linear to linear.
                pass
            elif from_log and not to_log:
                # Convert from log to linear.
                ## x = b**log_b(x)
                self.pmf = old_base**self.pmf
            else:
                # Convert from linear to log.
                ## x = log_b(x)
                self.pmf = new_ops.log(self.pmf)

        self.ops = new_ops

    def make_dense(self):
        """
        Make pmf contain all outcomes in the sample space.

        This does not change the sample space.

        Returns
        -------
        n : int
            The number of null outcomes added.

        """
        L = len(self)
        # Recall, __getitem__ is a view to the dense distribution.
        outcomes = tuple(self.sample_space())
        pmf = [self[o] for o in outcomes]
        self.pmf = np.array(pmf, dtype=float)
        self.outcomes = outcomes
        self._outcomes_index = dict(zip(outcomes, range(len(outcomes))))

        self._meta['is_sparse'] = False
        n = len(self) - L

        return n

    def make_sparse(self, trim=True):
        """
        Allow the pmf to omit null outcomes.

        This does not change the sample space.

        Parameters
        ----------
        trim : bool
            If `True`, then remove all null outcomes from the pmf.

        Notes
        -----
        Sparse distributions need not be trim.  One can add a null outcome to
        the pmf and the distribution could still be sparse.  A sparse
        distribution can even appear dense.  Essentially, sparse means that
        the shape of the pmf can grow and shrink.

        Returns
        -------
        n : int
            The number of null outcomes removed.

        """
        L = len(self)

        if trim:
            ### TODO: Use np.isclose() when it is available (NumPy 1.7)
            zero = self.ops.zero
            outcomes = []
            pmf = []
            for outcome, prob in self.zipped():
                if not close(prob, zero):
                    outcomes.append(outcome)
                    pmf.append(prob)

            # Update the outcomes and the outcomes index.
            self.outcomes = tuple(outcomes)
            self._outcomes_index = dict(zip(outcomes, range(len(outcomes))))

            # Update the probabilities.
            self.pmf = np.array(pmf)

        self._meta['is_sparse'] = True
        n = L - len(self)
        return n

    def to_rv_discrete(self):
        """
        Convert this distribution to an instance of scipy.stats.rv_discrete

        Returns
        -------
        ssrv : scipy.stats.rv_discrete
            A copy of this distribution as a rv_discrete.
        """
        from scipy.stats import rv_discrete
        d = rv_discrete(values=(self.outcomes, self.pmf), seed=self.prng)
        return d

    def validate(self, **kwargs):
        """
        Returns True if the distribution is valid, otherwise raise an exception.

        The default value of every parameter is `True`.

        Parameters
        ----------
        outcomes : bool
            If `True` verify that every outcome exists in the outcome space.
            This is a sanity check on the data structure.
        norm : bool
            If `True`, verify that the distribution is properly normalized.
        probs : bool
            If `True`, verify that each probability is valid. For linear
            probabilities, this means between 0 and 1, inclusive. For log
            probabilities, this means between -inf and 0, inclusive.

        Returns
        -------
        valid : bool
            True if the distribution is valid.

        Raises
        ------
        InvalidOutcome
            Raised if an outcome is not in the sample space.
        InvalidNormalization
            Raised if the distribution is improperly normalized.

        """
        mapping = {
            'outcomes': '_validate_outcomes',
            'norm': '_validate_normalization',
            'probs': '_validate_probabilities'
        }
        for kw, method in mapping.items():
            test = kwargs.get(kw, True)
            if test:
                getattr(self, method)()

        return True

    def _validate_probabilities(self):
        """
        Returns `True` if the values are probabilities.

        Raises
        ------
        InvalidProbability
            When a value is not between 0 and 1, inclusive.

        """
        from .validate import validate_probabilities
        v = validate_probabilities(self.pmf, self.ops)
        return v
