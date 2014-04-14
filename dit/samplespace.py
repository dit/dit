# -*- coding: utf-8 -*-
"""
Classes for representing sample spaces.

Consider the following distribution:

    d = dit.Distribution(['00', '01'], [.5, .5],
                         sample_space=['00', '01', '10', '11'])

When we marginalize, we only iterate over the elements explicitly stored in
the pmf. Before this module was implemented, whenever the sample space of a
distribution was not explicitly specified, we took the outcomes to be the
sample space. This had undesirable side effects. For example, d.marginal([0])
would only have a sample space of ['0']. What we really would like is to be
able to marginalize the sample space as well, so that we get a marginal
sample space of ['0', '1'].

However, the only way to achieve that is the iterate over the entire sample
space.  But while coalescing, we chose to iterate through the explicitly stored
elements for efficiency reasons---as only those elements could affect the
subsequent pmf. Since sample spaces can get quite large, especially if your
sample space is a Cartesian product, there would be a fairly large penalty
to doing this.

We usually like to work with Cartesian product sample spaces, and if the
product is abstractly represented, then we can marginalize sample spaces very
quickly just by combining the alphabets for each random variable. Since this
is our typical use case, we optimize for it.

With this module implemented, we now assume the user wants a Cartesian product
sample space and build it whenever one is not explicitly passed in. This gives
us desirable behavior when marginalizing, etc. If the user does pass in a
custom sample space, then we use it, but this requires that we iterate through
the whole sample space during marginalization. So this particular use case will
experience the penalty discussed above.

"""
from .helpers import (
    parse_rvs, get_outcome_ctor, construct_alphabets, get_product_func
)
from .utils import OrderedDict

try:
    from collections.abc import Set
except ImportError:
    # Py 2.x and < 3.3
    from collections import Set

from operator import mul
from itertools import product

import numpy as np

class BaseSampleSpace(Set):
    """
    An abstract representation of a sample space.

    A sized, iterable, container.

    """
    _meta = {}
    def __init__(self, samplespace):
        self._samplespace = list(samplespace)
        self._length = len(samplespace)

        # Store a set for O(1) lookup.
        self._set = set(samplespace)

    def __len__(self):
        return self._length

    def __contains__(self, item):
        return item in self._set

    def __iter__(self):
        return iter(self._samplespace)

    def index(self, item):
        """
        Returns a key for sorting items in the sample space.

        """
        return self._samplespace.index(item)

    def sort(self):
        self._samplespace.sort()

class ScalarSampleSpace(BaseSampleSpace):
    _meta = {
        'is_joint': False,
    }

class SampleSpace(ScalarSampleSpace):
    """
    An abstract representation of a sample space.

    A sized, iterable, container.

    """
    _meta = {
        'is_joint': True,
    }

    def __init__(self, samplespace, product=None):
        super(SampleSpace, self).__init__(samplespace)

        self._outcome_length = len(samplespace[0])
        self._outcome_class = samplespace[0].__class__
        self._outcome_ctor = get_outcome_ctor(self._outcome_class)
        # Since we have access to an outcome, we determine a product from it.
        if product is None:
            self._product = get_product_func(self._outcome_class)
        else:
            self._product = product

    def coalesce(self, rvs, extract=False):
        """
        Returns a new sample space after coalescing the specified indexes.

        Given n lists, each consisting of indexes from the original sample
        space, the coalesced sample space has n indexes, where the alphabet
        for each index is a tuple of the indexes that defined the ith list.

        Parameters
        ----------
        rvs : sequence
            A sequence whose elements are also sequences.  Each inner sequence
            defines an index in the new distribution as a combination
            of indexes in the original distribution.  The length of `rvs` must
            be at least one. The inner sequences need not be pairwise mutually
            exclusive with one another, and each can contain repeated indexes.
        extract : bool
            If the length of `rvs` is 1 and `extract` is `True`, then instead
            of the new outcomes being 1-tuples, we extract the sole element to
            create a sample space over the indexes in `rvs[0]`.

        Returns
        -------
        ss : SampleSpace
            The coalesced sample space.

        Examples
        --------
        If we have a joint distribution over 3 random variables such as:
            Z = (X,Y,Z)
        and would like a new sample space over 6 random variables:
            Z = (X,Y,Z,X,Y,Z)
        then this is achieved as:
            coalesce([[0,1,2,0,1,2]], extract=True)

        If you want:
            Z = ((X,Y), (Y,Z))
        Then you do:
            coalesce([[0,1],[1,2]])

        Notes
        -----
        Generally, the outcomes of the new sample space will be tuples instead
        of matching the class of the original sample space. This is because some
        classes are not recursive containers. For example, one cannot have a
        string of strings where each string consists of more than one
        character. Note however, that it is perfectly valid to have
        a tuple of tuples. Either way, the elements within each tuple of the
        sample space will still match the class of the original sample space.

        See Also
        --------
        marginal, marginalize

        """
        # We allow repeats and want to keep the order. We don't need the names.
        parse = lambda rv: parse_rvs(self, rv, rv_names=False,
                                               unique=False, sort=False)[1]
        indexes = [parse(rv) for rv in rvs]

        # Determine how new outcomes are constructed.
        if len(rvs) == 1 and extract:
            ctor_o = lambda x: x[0]
        else:
            ctor_o = tuple
        # Determine how elements of new outcomes are constructed.
        ctor_i = self._outcome_ctor

        # Build the new sample space.
        outcomes = OrderedDict()
        for outcome in self._samplespace:
            # Build a list of inner outcomes. "c" stands for "constructed".
            c_outcome = [ctor_i([outcome[i] for i in rv]) for rv in indexes]
            # Build the outer outcome from the inner outcomes.
            c_outcome = ctor_o(c_outcome)
            outcomes[c_outcome] = True

        return SampleSpace(outcomes.keys())

    def marginal(self, rvs):
        """
        Returns a marginal distribution.

        Parameters
        ----------
        rvs : list of int
            The indexes to keep. All others are marginalized.

        Returns
        -------
        ss : SampleSpace
            A new sample space with only indexes in `rvs`.

        """
        # For marginals, we must have unique indexes. Additionally, we do
        # not allow the order of the random variables to change. So we sort.
        rv_names = False
        rvs, indexes = parse_rvs(self, rvs, rv_names, unique=True, sort=True)

        # Marginalization is a special case of coalescing where there is only
        # one new random variable and it is composed of a strict subset of
        # the original random variables, with no duplicates, that maintains
        # the order of the original random variables.
        ss = self.coalesce([indexes], extract=True)
        return ss

    def marginalize(self, rvs):
        """
        Returns a new sample space after marginalizing the specified indexes.

        Parameters
        ----------
        rvs : list of int
            The indexes to marginalize. All others are kept.

        Returns
        -------
        ss : SampleSpace
            A new sample space with indexes in `rvs` marginalized away.

        """
        rvs, indexes = parse_rvs(self, rvs, rv_names=False)
        indexes = set(indexes)
        all_indexes = range(self.outcome_length())
        marginal_indexes = [i for i in all_indexes if i not in indexes]
        ss = self.marginal(marginal_indexes)
        return ss

    def outcome_length(self):
        return self._outcome_length

class CartesianProduct(SampleSpace):
    """
    An abstract representation of a Cartesian product sample space.

    """
    def __init__(self, alphabets, product=product):
        self.alphabets = tuple(alphabet if isinstance(alphabet, SampleSpace)
                              else tuple(alphabet) for alphabet in alphabets)
        self._alphabet_sets = [alphabet if isinstance(alphabet, SampleSpace)
                              else set(alphabet) for alphabet in alphabets]

        self.alphabet_sizes = tuple(len(alphabet) for alphabet in alphabets)
        # Here, the user MUST specify how we take products.
        # We infer the class from the specified product.
        self._product = product
        # Set initial value, in case there are no alphabets.
        self._length = reduce(mul, self.alphabet_sizes, 1)
        self._outcome_length = len(self.alphabet_sizes)
        self._outcome_class = next(self._product(*self.alphabets)).__class__
        self._outcome_ctor = get_outcome_ctor(self._outcome_class)

        # Used for calculating indexes
        shifts = np.cumprod(self.alphabet_sizes[::-1])
        shifts = [1] + list(shifts[:-1])
        shifts.reverse()
        self._shifts = np.array(shifts)

    def __contains__(self, item):
        return all([x in self._alphabet_sets[i] for i, x in enumerate(item)])

    def __iter__(self):
        return self._product(*self.alphabets)

    @classmethod
    def from_outcomes(cls, outcomes, product=None):
        alphabets = construct_alphabets(outcomes)
        klass = outcomes[0].__class__
        if product is None:
            product = get_product_func(klass)
        return cls(alphabets, product=product)

    def index(self, item):
        """
        Returns a key for sorting items in the sample space.

        """
        # This works even if alphabets[i] is itself a sample space.
        try:
            indexes = [self.alphabets[i].index(symbol)
                       for i, symbol in enumerate(item)]
        except (ValueError, IndexError):
            msg = '{0!r} is not in the sample space'.format(item)
            raise ValueError(msg)

        idx = np.sum(np.array(indexes) * self._shifts)
        return idx

    def coalesce(self, rvs, extract=False):
        """
        Returns a new sample space after coalescing the specified indexes.

        Given n lists, each consisting of indexes from the original sample
        space, the coalesced sample space has n indexes, where the alphabet
        for each index is a tuple of the indexes that defined the ith list.

        Parameters
        ----------
        rvs : sequence
            A sequence whose elements are also sequences.  Each inner sequence
            defines an index in the new distribution as a combination
            of indexes in the original distribution.  The length of `rvs` must
            be at least one. The inner sequences need not be pairwise mutually
            exclusive with one another, and each can contain repeated indexes.
        extract : bool
            If the length of `rvs` is 1 and `extract` is `True`, then instead
            of the new outcomes being 1-tuples, we extract the sole element to
            create a sample space over the indexes in `rvs[0]`.

        Returns
        -------
        ss : SampleSpace
            The coalesced sample space.

        Examples
        --------
        If we have a joint distribution over 3 random variables such as:
            Z = (X,Y,Z)
        and would like a new sample space over 6 random variables:
            Z = (X,Y,Z,X,Y,Z)
        then this is achieved as:
            coalesce([[0,1,2,0,1,2]], extract=True)

        If you want:
            Z = ((X,Y), (Y,Z))
        Then you do:
            coalesce([[0,1],[1,2]])

        Notes
        -----
        Generally, the outcomes of the new sample space will be tuples instead
        of matching the class of the original sample space. This is because some
        classes are not recursive containers. For example, one cannot have a
        string of strings where each string consists of more than one
        character. Note however, that it is perfectly valid to have
        a tuple of tuples. Either way, the elements within each tuple of the
        sample space will still match the class of the original sample space.

        See Also
        --------
        marginal, marginalize

        """
        # We allow repeats and want to keep the order. We don't need the names.
        parse = lambda rv: parse_rvs(self, rv, rv_names=False,
                                               unique=False, sort=False)[1]
        indexes = [parse(rv) for rv in rvs]

        alphabets_i = [[self.alphabets[i] for i in idx] for idx in indexes]
        sample_spaces = [CartesianProduct(alphabets, self._product)
                         for alphabets in alphabets_i]

        if len(rvs) == 1 and extract:
            ss = sample_spaces[0]
        else:
            ss = CartesianProduct(sample_spaces, product)
        return ss

    def sort(self):
        alphabets = []
        for i, alphabet in enumerate(self.alphabets):
            if isinstance(alphabet, SampleSpace):
                alphabet.sort()
            else:
                alphabet = tuple(sorted(alphabet))
            alphabets.append(alphabet)
        self.alphabets = tuple(alphabets)
