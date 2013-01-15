#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module defining NumPy array-based distribution classes.

Definitions
-----------
See http://docs.python.org/3/library/collections.abc.html#collections-abstract-base-classes

A sequence is a sized, iterable container.

`None` is not a valid event, as it is used by dit to signify that an event
was not found in the list of events.  This is not enforced in the data
structures---things will simply not work as expected.

The most basic type of event must be hashable and equality comparable.
Thus, the events should be immutable.  If the distribution's eventspace is
to be sorted, then the events must also be orderable.  Regular events need
not be sequences.

The joint event type must be a hashable, orderable, and a sequence. Thus, joint
events must be hashable, orderable, sized, iterable containers.

One of the features of joint distributions is that we can marginalize them.
This requires that we are able to construct smaller events from larger events.
For example, an event like '10101' might become '010' if the first and last
random variables are marginalized.  Given the alphabet of the joint
distribution, we can construct a tuple ('0','1','0') for any smaller event
using itertools.product, but how do we create an event of the same type?
For tuples and other similar containers, we just pass the tuple to the
event constructor.  This technique does not work for strings, as
str(('0','1','0')) yields "('0', '1', '0')".  So we have to handle strings
separately.

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
            d[e] will be valid if e is in the eventspace.
            d[e] will raise an InvalidEvent if e is not in the eventspace.
        d[e] does not describe the underlying data structure.
            It provides a view of the dense data structure.
            With defaultdict, if e not in d, then d[e] will add it.
            With distributions, we don't want the pmf changing size
            just because we queried it.  The size will change only on
            assignment.
        __in__ describes the underlying data structure.
        __iter__ describes the underlying data structure.
        __len__ describes the underlying data structure.


Sparse means the length of events and pmf need not equal the length of the
eventspace. There are two important points to keep in mind.
    1) A sparse distribution is not necessarily trim. Recall, a distribution
         is trim if its pmf does not contain null-events.
    2) The length of a sparse distribution's pmf can equal the length of the
       eventspace.
If a distribution is dense, then we forcibly make sure the length of the pmf
is always equal to the length of the eventspace.

When a distribution is sparse, del d[e] will make the pmf smaller.  But d[e] = 0
will simply set the element to zero.

When a distribution is dense, del d[e] will set the element to zero, and
d[e] = 0 still sets the element to zero.

"""

import itertools
import numpy as np

from .npdist import Distribution
from .exceptions import InvalidDistribution, InvalidEvent, InvalidProbability
from .math import LinearOperations, LogOperations
from .utils import str_product, product_maker
from .params import ditParams

def get_product_func(events):
    """
    Helper function to return a product function for the distribution.

    See the docstring for JointDistribution.

    """
    # Every event must be of the same type.
    if len(events) == 0:
        # Any product will work, since there is nothing to iterate over.
        product = itertools.product
    else:
        event = events[0]
        if isinstance(event, basestring):
            product = str_product
        elif isinstance(event, tuple):
            product = itertools.product
        else:
            # Assume the sequence-like constructor can handle tuples as input.
            product = product_maker(event.__class__)

    return product

def _make_distribution(pmf, events, alphabet=None, base=None, sparse=True):
    """
    An unsafe, but faster, initialization for distributions.

    If used incorrectly, the data structure will be inconsistent.

    This function can be useful when you are creating many distributions
    in a loop and can guarantee that:

        0) all events are of the same type (eg tuple, str) and length.
        1) the alphabet is in the desired order.
        2) events and pmf are in the same order as the eventspace.
           [Thus, `pmf` should not be a dictionary.]

    This function will not order the eventspace, nor will it reorder events
    or pmf.  It will not forcibly make events and pmf to be sparse or dense.
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

    ## events
    # Unlike Distribution, we cannot use a default set of events when
    # `events` is `None`.
    if len(pmf) != len(events):
        msg = "Unequal lengths for `values` and `events`"
        raise InvalidDistribution(msg)

    ## alphabets
    # Use events to obtain the alphabets.
    if alphabet is None:

        if len(events) == 0:
            msg = '`events` cannot have zero length if `alphabet` is `None`'
            raise InvalidDistribution(msg)

        # The event length.
        event_length = len(events[0])
        alphabet = [set([]) for i in range(event_length)]
        for event in events:
            for i,symbol in enumerate(event):
                alphabet[i].add(symbol)
        alphabet = map(tuple, alphabet)

    if len(events):
        d._event_class = events[0].__class__

    ## product
    d._product = get_product_func(events)

    # Force the distribution to be numerical and a NumPy array.
    d.pmf = np.asarray(pmf, dtype=float)

    # Tuple events, and an index.
    d.events = tuple(events)
    d._events_index = dict(zip(events, range(len(events))))

    # Tuple eventspace and its set.
    d.alphabet = tuple(alphabet)
    d._alphabet_set = map(set, d.alphabet)

    d._meta['is_sparse'] = sparse

    return d

def reorder(pmf, events, alphabet, product, index=None, method=None):
    """
    Helper function to reorder pmf and events so as to match the eventspace.

    The Cartesian product of the alphabets defines the eventspace.

    There are two ways to do this:
        1) Determine the order by generating the entire eventspace.
        2) Analytically calculate the sort order of each event.

    If the eventspace is very large and sparsely populated, then method 2)
    is probably faster. However, it must calculate a number using
    (2**(symbol_orders)).sum().  Potentially, this could be costly. If the
    eventspace is small, then method 1) is probably fastest. We'll experiment
    and find a good heurestic.

    """
    # A map of the elements in `events` to their index in `events`.
    if index is None:
        index = dict(zip(events, range(len(events))))

    # The number of elements in the eventspace?
    eventspace_size = np.prod( map(len, alphabet) )

    if method is None:
        if eventspace_size > 10000 and len(events) < 1000:
            # Large and sparse.
            method = 'analytic'
        else:
            method = 'generate'

    method = 'generate'
    if method == 'generate':
        # Obtain the order from the generated order.
        eventspace = product(*alphabet)
        order = [index[event] for event in eventspace if event in index]
        if len(order) != len(events):
            raise InvalidDistribution('Events and eventspace are not compatible.')
        events_ = [events[i] for i in order]
        pmf = [pmf[i] for i in order]

        # We get this for free: Check that every event was in the eventspace.
        # (Well, its costs us a bit in memory to keep events and events_.)
        if len(events_) != len(events):
            # We lost an event.
            bad = set(events) - set(events_)
            L = len(bad)
            if L == 1:
                raise InvalidEvent(bad, single=True)
            elif L:
                raise InvalidEvent(bad, single=False)
        else:
            events = events_

    elif method == 'analytic':
        # Analytically calculate the sort order.
        # Note, this method does not verify that every event was in the
        # eventspace.

        # Construct a lookup from symbol to order in the alphabet.
        alphabet_size = map(len, alphabet)
        alphabet_index = [dict(zip(alph, range(size)))
                          for alph, size in zip(alphabet, alphabet_size)]

        L = len(events[0]) - 1
        codes = []
        for event in events:
            idx = 0
            for i,symbol in enumerate(event):
                idx += alphabet_index[i][symbol] * (alphabet_size[i])**(L-i)
            codes.append(idx)

        # We need to sort the codes now, keeping track of their indexes.
        order = zip(codes, range(len(codes)))
        order.sort()
        sorted_codes, order = zip(*order)
        events = [events[i] for i in order]
        pmf = [pmf[i] for i in order]
    else:
        raise Exception("Method must be 'generate' or 'analytic'")

    new_index = dict(zip(events, range(len(events))))

    return pmf, events, new_index

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
        `True` if `events` and `pmf` represent a sparse distribution.

    Private Attributes
    ------------------
    _alphabet_set : tuple
        A tuple representing the alphabet of the joint random variable.  The
        elements of the tuple are sets, each of which represents the unordered
        alphabet of a single random variable.

    _event_class : class
        The class of all events in the distribution.

    _events_index : dict
        A dictionary mapping events to their index in self.events.

    _meta : dict
        A dictionary containing the meta information, described above.

    _product : function
        A specialized product function, similar to itertools.product.  The
        primary difference is that instead of yielding tuples, this product
        function will yield objects which are of the same type as the events.

    Public Attributes
    -----------------
    alphabet : tuple
        A tuple representing the alphabet of the joint random variable.  The
        elements of the tuple are tuples, each of which represents the ordered
        alphabet for a single random variable. The Cartesian product of these
        alphabets defines the eventspace.

    events : tuple
        The events of the probability distribution.

    ops : Operations instance
        A class which manages addition and multiplication operations for log
        and linear probabilities.

    pmf : array-like
        The probability mass function for the distribution.  The elements of
        this array are in a one-to-one correspondence with those in `events`.

    prng : RandomState
        A pseudo-random number generator with a `rand` method which can
        generate random numbers. For now, this is assumed to be something
        with an API compatibile to NumPy's RandomState class. This attribute
        is initialized to equal dit.math.prng.

    Public Methods
    --------------
    copy
        Returns a deep copy of the distribution.

    event_length
        Returns the length of the events in the distribution.

    eventprobs
        Returns an iterator over (event, probability) tuples.  The probability
        could be a log probability or a linear probability.

    eventspace
        Returns an iterator over the events in the eventspace.

    get_base
        Returns the base of the distribution.

    has_event
        Returns `True` is the distribution has `event` in the eventspace.

    is_dense
        Returns `True` if the distribution is dense.

    is_heterogeneous
        Returns `True` if the alphabet for each random variable is the same.

    is_joint
        Returns `True` if the distribution is a joint distribution.

    is_log
        Returns `True` if the distribution values are log probabilities.

    is_numerical
        Returns `True` if the distribution values are numerical.

    is_sparse
        Returns `True` if the distribution is sparse.

    make_dense
        Add all null events to the pmf.

    make_sparse
        Remove all null events from the pmf.

    normalize
        Normalizes the distribution.

    sample
        Returns a sample from the distribution.

    set_base
        Changes the base of the distribution, in-place.

    to_string
        Returns a string representation of the distribution.

    validate
        A method to validate that the distribution is valid.

    Implementation Notes
    --------------------
    The events and pmf of the distribution are stored as a tuple and a NumPy
    array.  The sequences can be either sparse or dense.  By sparse, we do not
    mean that the representation is a NumPy sparse array.  Rather, we mean that
    the sequences need not contain every event in the eventspace. The order of
    the events and probabilities will always match the order of the eventspace,
    even though their length might not equal the length of the eventspace.

    """


    _alphabet_set = None
    _event_class = None
    _events_index = None
    _meta = {
        'is_joint': True,
        'is_numerical': True,
        'is_sparse': None,
        'is_heterogenous': None,
    }
    _product = None

    alphabet = None
    events = None
    ops = None
    pmf = None
    prng = None

    def __init__(self, pmf, events=None, alphabet=None, base=None,
                            validate=True, sort=True, sparse=True):
        """
        Initialize the distribution.

        Parameters
        ----------
        pmf : sequence, dict
            The event probabilities or log probabilities. If `pmf` is a
            dictionary, then the keys are used as `events`, and the values of
            the dictionary are used as `pmf` instead.  The keys take precedence
            over any specification of them via `events`.

        events : sequence
            The events of the distribution. If specified, then its length must
            equal the length of `pmf`.  If `None`, then the events must be
            obtainable from `pmf`, otherwise an exception will be raised.
            Events must be hashable, orderable, sized, iterable containers.
            The length of an event must be the same for all events, and every
            event must be of the same type.

        alphabet : sequence
            A sequence representing the alphabet of the joint random variable.
            The elements of the sequence are tuples, each of which represents
            the ordered alphabet for a single random variable. The Cartesian
            product of these alphabets defines the eventspace. The order of the
            alphabets and the order within each alphabet is important. If
            `None`, the value of `events` is used to determine the alphabet.

        base : float, None
            If `pmf` specifies log probabilities, then `base` should specify
            the base of the logarithm.  If 'linear', then `pmf` is assumed to
            represent linear probabilities.  If `None`, then the value for
            `base` is taken from ditParams['base'].

        validate : bool
            If `True`, then validate the distribution.  If `False`, then assume
            the distribution is valid, and perform no checks.

        sort : bool
            If `True`, then each random variable's alphabets are sorted.
            Usually, this is desirable, as it normalizes the behavior of
            distributions which have the same eventspaces (when considered as
            a set).  NOte that addition and multiplication of distributions is
            defined only if the eventspaces are equal.

        sparse : bool
            Specifies the form of the pmf.  If `True`, then `events` and `pmf`
            will only contain entries for non-null events and probabilities,
            after initialization.  The order of these entries will always obey
            the order of `eventspace`, even if their number is not equal to the
            size of the eventspace.  If `False`, then the pmf will be dense and
            every event in the eventspace will be represented.

        Raises
        ------
        InvalidDistribution
            If the length of `values` and `events` are unequal.
            If no events can be obtained from `pmf` and `events` is `None`.

        See :meth:`validate` for a list of other potential exceptions.

        """
        # Note, we are not calling Distribution.__init__
        # We want to call BaseDistribution.__init__
        super(Distribution, self).__init__()

        pmf, events, alphabet = self._init(pmf, events, alphabet, base)

        # Sort everything to match the order of the eventspace.
        if sort:
            alphabet = map(sorted, alphabet)
            alphabet = map(tuple, alphabet)
            pmf, events, index = reorder(pmf, events, alphabet, self._product)
        else:
            index = dict(zip(events, range(len(events))))

        # Force the distribution to be numerical and a NumPy array.
        self.pmf = np.asarray(pmf, dtype=float)

        # Tuple events, and an index.
        self.events = tuple(events)
        self._events_index = index

        # Tuple alphabet and its set.
        self.alphabet = tuple(alphabet)
        self._alphabet_set = map(set, self.alphabet)

        if sparse:
            self.make_sparse(trim=True)
        else:
            self.make_dense()

        if validate:
            self.validate()

    def _init(self, pmf, events, alphabet, base):
        """
        The barebones initialization.

        """
        def construct_alphabet(evts):
            if len(evts) == 0:
                msg = '`events` cannot have zero length if `alphabet` is `None`'
                raise InvalidDistribution(msg)

            # The event length.
            self._event_class = evts[0].__class__
            event_length = len(evts[0])
            alpha = [set([]) for i in range(event_length)]
            for event in evts:
                for i,symbol in enumerate(event):
                    alpha[i].add(symbol)
            alpha = map(tuple, alpha)
            return alpha

        if isinstance(pmf, Distribution):
            # Attempt a conversion.
            d = pmf

            events = d.events
            pmf = d.pmf
            if len(events):
                self.event_class = events[0].__class__
            if base is None:
                # Allow the user to specify something strange if desired.
                # Otherwise, use the existing base.
                base = d.get_base()
            if alphabet is None:
                if d.is_joint():
                    # This will always work.
                    alphabet = d.alphabet
                else:
                    # We will assume the events are valid joint events.
                    # But we must construct the alphabet.
                    alphabet = construct_alphabet(events)

        else:
            ## pmf
            # Attempt to grab events and pmf from a dictionary
            try:
                events_ = tuple(pmf.keys())
                pmf_ = tuple(pmf.values())
            except AttributeError:
                pass
            else:
                events = events_
                pmf = pmf_

            ## events
            if events is None:
                msg = "`events` must be specified or obtainable from `pmf`."
                raise InvalidDistribution(msg)
            elif len(pmf) != len(events):
                msg = "Unequal lengths for `values` and `events`"
                raise InvalidDistribution(msg)

            ## alphabets
            # Use events to obtain the alphabets.
            if alphabet is None:
                alphabet = construct_alphabet(events)

        # Determine if the pmf represents log probabilities or not.
        if base is None:
            base = ditParams['base']
        if base == 'linear':
            ops = LinearOperations()
        else:
            ops = LogOperations(base)
        self.ops = ops

        ## product
        self._product = get_product_func(events)

        return pmf, events, alphabet

    def __setitem__(self, event, value):
        """
        Sets the probability associated with `event`.

        Parameters
        ----------
        event : event
            Any hashable and equality comparable object in the eventspace.
            If `event` does not exist in the eventspace, then an InvalidEvent
            exception is raised.
        value : float
            The probability or log probability of the event.

        Returns
        -------
        p : float
            The probability (or log probability) of the event.

        Raises
        ------
        InvalidEvent
            If `event` does not exist in the eventspace.

        Notes
        -----
        Setting the value of the event never deletes the event, even if the
        value is equal to the null probabilty. After a setting operation,
        the event will always exist in `events` and `pmf`.

        See Also
        --------
        __delitem__

        """
        if self._event_class is None:
            # The first __setitem__ call from an empty distribution.
            # If there is no event class, make one.
            self._event_class = event.__class__
            # Reset the product function.
            self._product = get_product_func([event])

        if not self.has_event(event, null=True):
            raise InvalidEvent(event)

        idx = self._events_index.get(event, None)
        new_event = idx is None

        if not new_event:
            # If the distribution is dense, we will be here.
            # We *could* delete if the value was zero, but we will make
            # setting always set, and deleting always deleting (when sparse).
            self.pmf[idx] = value
        else:
            # A new event in a sparse distribution.
            # Sticking with the setting always setting...we add even if
            # the value is zero.

            # 1. Add the new event and probability
            self.events = self.events + (event,)
            self._events_index[event] = len(self.events) - 1
            pmf = [p for p in self.pmf] + [value]

            # 2. Reorder  ### This call is different from Distribution
            pmf, events, index = reorder(pmf, self.events, self.alphabet,
                                         self._product,
                                         index=self._events_index)

            # 3. Store
            self.events = tuple(events)
            self._events_index = index
            self.pmf = np.array(pmf, dtype=float)

    def event_length(self, masked=False):
        """
        Returns the length of events in the joint distribution.

        This is also equal to the number of random variables in the joint
        distribution. This value is fixed once the distribution is initialized.

        Parameters
        ----------
        masked : bool
            If `True`, then the event length additionally includes masked
            random variables. If `False`, then the event length does not
            include masked random variables. Including the masked random
            variables is not usually helpful since that represents the event
            length of a different, unmarginalized distribution.

        """
        if masked:
            return len(self.mask)
        else:
            return len(self.alphabet)

    def eventspace(self):
        """
        Returns an iterator over the ordered event space.

        """
        return self._product(*self.alphabet)

    def has_event(self, event, null=True):
        """
        Returns `True` if `event` exists  in the eventspace.

        Whether or not an event is in the eventspace is a separate question
        from whether or not an event currently appears in the pmf.
        See __contains__ for this latter question.

        Parameters
        ----------
        event : event
            The event to be tested.
        null : bool
            Specifies if null events are acceptable.  If `True`, then null
            events are acceptable.  Thus, the only requirement on `event` is
            that it exist in the distribution's eventspace. If `False`, then
            null events are not acceptable.  Thus, `event` must exist in the
            distribution's eventspace and also correspond to be nonnull.

        Notes
        -----
        This is an O(1) operation.

        """
        # Make sure the event exists in the eventspace.

        # Note, it is not sufficient to test if each symbol exists in the
        # the alphabet for its corresponding random variable. The reason is
        # that, for example, '111' and ('1', '1', '1') would both be seen
        # as valid.  Thus, we must also verify that the event's class
        # matches that of the other event's classes.

        # Make sure the event class is correct.
        if event.__class__ != self._event_class:
            # This test works even when the distribution was initialized empty
            # and _event_class is None. In that case, we don't know the event
            # space (since we don't know the event class), and we should return
            # False.
            return False

        # Make sure event has the correct length.
        if len(event) != self.event_length(masked=False):
            return False

        if null:
            # The event must only be valid.

            # Make sure each symbol exists in its corresponding alphabet.
            z = False
            for symbol, alphabet in zip(event, self.alphabet):
                if symbol not in alphabet:
                    break
            else:
                z = True
        else:
            # The event must be valid and have positive probability.
            try:
                z = self[event] > self.ops.zero
            except InvalidEvent:
                z = False

        return z

    def is_heterogenous(self):
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

    def to_string(self, digits=None, exact=False, tol=1e-9):
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

        Returns
        -------
        s : str
            A string representation of the distribution.

        """
        from .distribution import prepare_string

        from itertools import izip
        from StringIO import StringIO

        s = StringIO()

        x = prepare_string(self, digits, exact, tol)
        pmf, events, base, colsep, max_length, pstr = x

        s.write("Class: {}\n".format(self.__class__.__name__))
        if self.is_heterogenous():
            alpha = str(self.alphabet[0]) + " for all rvs"
        else:
            alpha = str(self.alphabet)
        s.write("Alphabet: {}\n".format(alpha))
        s.write("Base: {}\n".format(base))
        s.write("Event Class: {}\n".format(self._event_class.__name__))
        s.write("Event Length: {}\n\n".format(self.event_length()))
        s.write(''.join([ 'x'.ljust(max_length), colsep, pstr, "\n" ]))

        for e,p in izip(events, pmf):
            s.write(''.join( [e.ljust(max_length), colsep, str(p), "\n"] ))
        s.seek(0)
        s = s.read()
        # Remove the last \n
        s = s[:-1]
        return s


