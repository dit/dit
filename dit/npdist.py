#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module defining NumPy array-based distribution classes.


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

from .distribution import BaseDistribution
from .exceptions import InvalidEvent, InvalidProbability
from .math import LinearOperations, LogOperations, close
from .params import ditParams

import numpy as np

def _make_distribution(pmf, events=None, eventspace=None, base=None, sparse=True):
    """
    An unsafe, but faster, initialization for distributions.

    If used incorrectly, the data structure will be inconsistent.

    This function can be useful when you are creating many distributions
    and in loop and can guarantee that:

        1) the event space is in the desired order.
        2) events and pmf are in the same order as the eventspace.
           [Thus, `pmf` should not be a dictionary.]
        3) events and pmf are either sparse xor dense (and not something else).

    This function will not order the eventspace, nor will it reorder events
    or pmf.  It will not forcibly make events and pmf to be sparse or dense.
    It will simply declare the distribution to be sparse or dense. The
    distribution is not validated either.

    Returns
    -------
    d : Distribution
        The new distribution.

    """
    d = Distribution.__new__(Distribution)

    # Determine if the pmf represents log probabilities or not.
    if base is None:
        base = ditParams['base']
    if base == 'linear':
        ops = LinearOperations()
    else:
        ops = LogOperations(base)
    d.ops = ops

    if events is None:
        events = range(len(pmf))

    if eventspace is None:
        eventspace = events

    # Force the distribution to be numerical and a NumPy array.
    d.pmf = np.asarray(pmf, dtype=float)

    # Tuple events, and an index.
    d.events = tuple(events)
    d._events_index = dict(zip(events, range(len(events))))

    # Tuple eventspace and its set.
    d._eventspace = tuple(eventspace)
    d._eventspace_set = set(eventspace)

    d._meta['is_sparse'] = sparse

    return d

def reorder(events, pmf, eventspace, index=None):
    """
    Helper function to reorder events and pmf to match eventspace.

    """
    if index is None:
        index = dict(zip(events, range(len(events))))

    evts = set(events)
    order = [index[event] for event in eventspace if event in evts]
    events = [events[i] for i in order]
    pmf = [pmf[i] for i in order]
    new_index = dict(zip(events, range(len(events))))
    return events, pmf, new_index

class Distribution(BaseDistribution):
    """
    A numerical distribution.

    Meta Properties
    ---------------
    is_numerical
        Boolean specifying if the pmf represents numerical values or not.
        The values could be symbolic, for example.

    is_sparse : bool
        `True` if `events` and `pmf` represent a sparse distribution.

    Private Attributes
    ------------------
    _events_index : dict
        A dictionary mapping events to their index in self.events.

    _eventspace : tuple
        The ordered event space.

    _eventspace_set : set
        The set of the event space.

    _meta : dict
        A dictionary containing the meta information, described above.

    Public Attributes
    -----------------
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

    _meta = {
        'is_numerical': True,
        'is_sparse': None
    }

    def __init__(self, pmf, events=None, eventspace=None, base=None,
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
            equal the length of `pmf`.  If `None`, then consecutive integers
            beginning from 0 are used as the events. An event is any hashable
            object which is equality comparable.  If `sort` is `True`, then
            events must also be orderable.

        eventspace : sequence
            The complete listing of possible events.  If `None`, the value of
            `events` is used to specify all possible events.  The order of the
            events is important, so sort beforehand if necessary.

        base : float, None
            If `pmf` specifies log probabilities, then `base` should specify
            the base of the logarithm.  If 'linear', then `pmf` is assumed to
            represent linear probabilities.  If `None`, then the value for
            `base` is taken from ditParams['base'].

        validate : bool
            If `True`, then validate the distribution.  If `False`, then assume
            the distribution is valid, and perform no checks.

        sort : bool
            If `True`, then the eventspace is sorted first. Usually, this is
            desirable, as it normalizes the behavior of distributions which
            have the same eventspace (when considered as a set).  Addition and
            multiplication of distributions is defined only if the eventspace
            (as a tuple) is equal.

        sparse : bool
            Specifies the form of the pmf.  If `True`, then `events` and `pmf`
            will only contain entries for non-null events and probabilities.
            The order of these entries will always obey the order of
            `eventspace`, even if their number is not equal to the size of
            the eventspace.  If `False`, then the pmf will be dense and every
            event in the eventspace will be represented.

        Raises
        ------
        InvalidDistribution
            If the length of `values` and `events` are unequal.

        See :meth:`validate` for a list of other potential exceptions.

        """
        super(Distribution, self).__init__()

        pmf, events, eventspace = self._init(pmf, events, eventspace, base)

        # Sort everything to match the order of the eventspace.
        if sort:
            eventspace = tuple(sorted(eventspace))
            events, pmf, index = reorder(events, pmf, eventspace)
        else:
            index = dict(zip(events, range(len(events))))

        # Force the distribution to be numerical and a NumPy array.
        self.pmf = np.asarray(pmf, dtype=float)

        # Tuple events, and an index.
        self.events = tuple(events)
        self._events_index = index

        # Tuple eventspace and its set.
        self._eventspace = tuple(eventspace)
        self._eventspace_set = set(eventspace)

        if sparse:
            self.make_sparse(trim=True)
        else:
            self.make_dense()

        if validate:
            self.validate()

    def _init(self, pmf, events, eventspace, base):
        """
        The barebones initialization.

        """
        # Determine if the pmf represents log probabilities or not.
        if base is None:
            base = ditParams['base']
        if base == 'linear':
            ops = LinearOperations()
        else:
            ops = LogOperations(base)
        self.ops = ops

        ## pmf
        # Attempt to grab events and pmf from a dictionary
        try:
            events = tuple(pmf.keys())
            pmf = tuple(pmf.values())
        except AttributeError:
            pass

        ## events
        # Make sure events and values have the same length.
        if events is None:
            events = range(len(pmf))
        elif len(pmf) != len(events):
            msg = "Unequal lengths for `values` and `events`"
            raise InvalidDistribution(msg)

        ## eventspace
        # Use events as default eventspace.
        if eventspace is None:
            eventspace = events

        return pmf, events, eventspace

    def __add__(self, other):
        """
        Addition of distributions of the same kind.

        The other distribution must have the same meta information and the
        same eventspace.  If not, raise an exception.

        """
        if self._eventspace != other._eventspace:
            raise IncompatibleDistribution()

        # Copy to make sure we don't lose precision when converting.
        d2 = other.copy()
        d2.set_base(self.get_base())

        # If self is dense, the result will be dense.
        # If self is sparse, the result will be sparse.
        d = self.copy()
        for event, prob in other.eventprobs():
            d[event] = d.ops.add(d[event], prob)

        return d

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        """
        Scalar multiplication on distributions.

        Note, we do not implement distribution-to-distribution multiplication.

        """
        d = self.copy()
        d.pmf *= other
        return d

    def __rmul__(self, other):
        return self.__mul__(other)

    def __contains__(self, event):
        """
        Returns `True` if `event` is in self.events.

        Note, the event could correspond to a null-event. Also, if `event` is
        not in the eventspace, then an exception is not raised. Instead,
        `False` is returned.

        """
        return event in self._events_index

    def __delitem__(self, event):
        """
        Deletes `event` from the distribution.

        Parameters
        ----------
        event : event
            Any hashable and equality comparable object. If `event` exists
            in the eventspace, then it is removed from the pmf.  If the event
            did not already exist in the pmf, then no exception is raised. If
            `event` does not exist in the eventspace, then an InvalidEvent
            exception is raised.

        Raises
        ------
        InvalidEvent
            If `event` does not exist in the eventspace.

        Notes
        -----
        If the distribution is dense, then the event's value is set to zero,
        and the length of the pmf is left unchanged.

        If the event was a non-null event, then the resulting distribution
        will no longer be normalized (assuming it was in the first place).

        See Also
        --------
        normalize, __setitem__

        """
        events = self.events
        events_index = self._events_index

        ## Note, the event stays in the eventspace.

        if event in self._eventspace_set:
            if self.is_dense():
                # Dense distribution, just set it to zero.
                idx = events_index[event]
                self.pmf[idx] = self.ops.zero
            elif event in events_index:
                # Remove the event from the sparse distribution.

                # Update the events and the events index.
                idx = events_index[event]
                new_indexes = [i for i in range(len(events)) if i != idx]
                new_events = tuple([ events[i] for i in new_indexes])
                self.events = new_events
                self._events_index = dict(zip(new_events, range(len(new_events))))

                # Update the probabilities.
                self.pmf = self.pmf[new_indexes]
        else:
            raise InvalidEvent(event)

    def __getitem__(self, event):
        """
        Returns the probability associated with `event`.

        Parameters
        ----------
        event : event
            Any hashable and equality comparable object in the eventspace.
            If `event` does not exist in the eventspace, then an InvalidEvent
            exception is raised.

        Returns
        -------
        p : float
            The probability (or log probability) of the event.

        Raises
        ------
        InvalidEvent
            If `event` does not exist in the eventspace.

        """
        if event not in self._eventspace_set:
            raise InvalidEvent(event)
        else:
            idx = self._events_index.get(event, None)
            if idx is None:
                p = self.ops.zero
            else:
                p = self.pmf[idx]
            return p

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
        if event not in self._eventspace_set:
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

            # 2. Reorder
            events, pmf, index = reorder(self.events, pmf, self._eventspace,
                                         index=self._events_index)

            # 3. Store
            self.events = tuple(events)
            self._events_index = index
            self.pmf = np.array(pmf, dtype=float)

    def copy(self):
        """
        Returns a (deep) copy of the distribution.

        """
        # For some reason, we can't just return a deepcopy of self.
        # It works for linear distributions but not for log distributions.

        from copy import deepcopy
        d = _make_distribution(pmf=np.array(self.pmf, copy=True),
                               events=deepcopy(self.events),
                               eventspace=deepcopy(self._eventspace),
                               base=self.ops.base,
                               sparse=self._meta['is_sparse'])
        return d

    def eventspace(self):
        """
        Returns an iterator over the ordered event space.

        """
        return iter(self._eventspace)

    def is_approx_equal(self, other):
        """
        Returns `True` is `other` is approximately equal to this distribution.

        For two distributions to be equal, they must have the same eventspace
        and must also agree on the probabilities of each event.

        Parameters
        ----------
        other : distribution
            The distribution to compare against.

        Notes
        -----
        The distributions need not have the same base or even same length.

        """
        # Event spaces must be equal.
        es1, es2 = tuple(self.eventspace()), tuple(other.eventspace())
        if  es1 != es2:
            return False

        # The set of all specified events (some may be null events).
        if self.is_dense() or other.is_dense():
            events = es1
        else:
            events = set(self.events)
            events.update(other.events)

        # Potentially nonzero probabilities must be equal.
        for event in events:
            if not close(self[event], other[event]):
                return False
        else:
            return True

    def has_event(self, event, null=True):
        """
        Returns `True` if `event` is a valid event (exists in the eventspace).

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
        if null:
            # No additional restrictions.
            z = event in self._eventspace_set
        else:
            # Must be valid and have positive probability.
            try:
                z = not close(self[event], self.ops.zero)
            except InvalidEvent:
                z = False

        return z

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
        convert to a linear distribution by passing 'linear'. Note, conversions
        introduce errors, especially when converting from very negative log
        probabilities to linear probabilities (underflow is an issue).

        Parameters
        ----------
        base : float or string
            The desired base for the distribution.  If 'linear', then the
            distribution's pmf will represent linear probabilities. If any
            positive float (other than 1) or 'e', then the pmf will represent
            log probabilities with the specified base.

        """
        from dit.math import LinearOperations, LogOperations
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

            # Use numerical value for base.
            if old_ops.base == 'e':
                old_base = np.e
            else:
                old_base = old_ops.base

            # Caution: The in-place multiplication ( *= ) below will work only
            # if pmf has a float dtype.  If not (e.g., dtype=int), then the
            # multiplication gives incorrect results due to coercion. The
            # __init__ function is responsible for guaranteeing the dtype.

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
        Make pmf contain all events in the eventspace.

        This does not change the eventspace.

        Returns
        -------
        n : int
            The number of null events added.

        """
        L = len(self)

        # Recall, __getitem__ is a view to the dense distribution.
        pmf = [ self[e] for e in self._eventspace ]
        self.pmf = np.array(pmf)
        self.events = self._eventspace
        self._events_index = dict(zip(self.events, range(len(self.events))))

        self._meta['is_sparse'] = False
        n = len(self) - L
        return n

    def make_sparse(self, trim=True):
        """
        Allow the pmf to omit null events.

        This does not change the eventspace.

        Parameters
        ----------
        trim : bool
            If `True`, then remove all null events from the pmf.

        Notes
        -----
        Sparse distributions need not be trim.  One can add a null event to
        the pmf and the distribution could still be sparse.  A sparse
        distribution can even appear dense.  Essentially, sparse means that
        the shape of the pmf can grow and shrink.

        Returns
        -------
        n : int
            The number of null events removed.

        """
        L = len(self)

        if trim:
            ### Use np.isclose() when it is available (NumPy 1.7)
            zero = self.ops.zero
            events = []
            pmf = []
            for event, prob in self.eventprobs():
                if not close(prob, zero):
                    events.append(event)
                    pmf.append(prob)

            # Update the events and the events index.
            self.events = tuple(events)
            self._events_index = dict(zip(events, range(len(events))))

            # Update the probabilities.
            self.pmf = np.array(pmf)

        self._meta['is_sparse'] = True
        n = L - len(self)
        return n

    def validate(self, **kwargs):
        """
        Returns True if the distribution is valid, otherwise raise an exception.

        The default value of every parameter is `True`.

        Parameters
        ----------
        events : bool
            If `True` verify that every event exists in the event space.
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
        InvalidEvent
            Raised if an event is not in the event space.
        InvalidNormalization
            Raised if the distribution is improperly normalized.

        """
        mapping = {
            'events': '_validate_events',
            'norm': '_validate_normalization',
            'probs': '_validate_probabilities'
        }
        for kw, method in mapping.iteritems():
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
        one = self.ops.one
        zero = self.ops.zero
        pmf = self.pmf

        # Make sure the values are in the correct range.
        too_low = pmf < zero
        too_high = pmf > one
        if too_low.any() or too_high.any():
            bad = pmf[ np.logical_or(too_low, too_high) ]
            raise InvalidProbability( bad )

        return True


