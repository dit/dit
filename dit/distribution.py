#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module defining base distribution class.


Definitions
-----------
See http://docs.python.org/3/library/collections.abc.html#collections-abstract-base-classes

A sequence is a sized, iterable container.

`None` is not a valid event, as it is used by dit to signify that an event
was not found in the list of events.  This is not enforced in the data
structures---things will simply not work as expected.

The most basic type of event must be: hashable and equality comparable. If the
distribution's eventspace is to be ordered, then the events must also be
orderable.

The joint event type must be: hashable and orderable.

"""

from itertools import izip

import numpy as np
from cmpy.math import close

from .exceptions import (
    InvalidBase,
    InvalidEvent,
    InvalidNormalization
)

class BaseDistribution(object):
    """
    The base class for all "distribution" classes.

    Generally, distributions are mutuable in that the events and probabilities
    can be changed from zero to nonzero and back again. However, the eventspace
    is not mutable and must remain fixed over the lifetime of the distribution.

    Meta Properties
    ---------------
    is_numeric
        Boolean specifying if the pmf represents numerical values or not.
        The values could be symbolic, for example.

    Private Attributes
    ------------------
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

    Methods
    -------
    copy
        Returns a deep copy of the distribution.

    eventprobs
        Returns an iterator over (event, probability) tuples.  The probability
        could be a log probability or a linear probability.

    eventspace
        Returns an iterator over the events in the eventspace.

    get_base
        Returns the base of the distribution.

    is_event
        Returns `True` is the event exists in the event space.

    is_log
        Returns `True` if the distribution values are log probabilities.

    is_numeric
        Returns `True` if the distribution values are numerical.

    normalize
        Normalizes the distribution.

    set_base
        Changes the base of the distribution, in-place.

    validate
        A method to validate that the distribution is valid.

    """
    # Subclasses should update these meta attributes *before* calling the base
    # distribution's __init__ function.
    _meta = {
        'is_numeric': None,
    }

    # These should be set in the subclass's init function.
    events = None
    ops = None
    pmf = None

    def __init__(self):
        raise NotImplementedError

    def __contains__(self, event):
        raise NotImplementedError

    def __delitem__(self, event):
        raise NotImplementedError

    def __getitem__(self, event):
        raise NotImplementedError

    def __iter__(self):
        """
        Returns an iterator over the non-null events in the distribution.

        """
        return iter(self.events)

    def __len__(self):
        """
        Returns the number of non-null events in the distribution.

        """
        return len(self.events)

    def __reversed__(self):
        """
        Returns a reverse iterator over the non-null events.

        """
        return reversed(self.events)

    def __setitem__(self, key, value):
        raise NotImplementedError

    def _validate_events(self):
        """
        Returns `True` if the events are in the event space.

        Returns
        -------
        v : bool
            `True` if the events are in the event space.

        Raises
        ------
        InvalidEvent
            When an event is not in the event space.

        """
        # Make sure the events are in the event space.
        events = set(self.events)
        eventspace = set(self.eventspace())
        bad = events.difference(eventspace)
        if len(bad) > 0:
            raise InvalidEvent(bad)

        return True

    def _validate_normalization(self):
        """
        Returns `True` if the distribution is properly normalized.

        Returns
        -------
        v : bool
            `True` if the distribution is properly normalized.

        Raises
        ------
        InvalidNormalization
            When the distribution is not properly normalized.

        """
        # log_func is the identity function for non-log distributions.
        log = self.ops.log
        one = self.ops.one

        # Make sure the distribution is normalized properly.
        total = self.ops.add_reduce( self.pmf )
        if not close(total, one):
            raise InvalidNormalization(total)

        return True

    def copy(self):
        """
        Returns a deep copy of the distribution.

        """
        raise NotImplementedError

    def eventprobs(self):
        """
        Returns an iterator over (event, probability) tuples.

        """
        return izip(self.events, self.pmf)

    def eventspace(self):
        """
        Returns an iterator over the ordered event space.

        """
        raise NotImplementedError

    def is_event(self, event):
        """
        Returns `True` if `event` is a valid event in the distribution.

        The event may be a null-probability event.

        """
        raise NotImplementedError

    def is_log(self):
        """
        Returns `True` if the distribution values are log probabilities.

        """
        return self.ops.base != 'linear'

    def is_numeric(self):
        """
        Returns `True` if the distribution values are numerical.

        """
        return self._meta['is_numeric']

    def normalize(self):
        """
        Normalizes the distribution.

        """
        raise NotImplementedError

    def get_base(self):
        """
        Returns the base of the distribution.

        If the distribution represents linear probabilities, then the string
        'linear' will be returned.  If the base of log probabilities is e,
        then the returned base could be the string 'e' or its numerical value.

        """
        return self.ops.base

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
        raise NotImplementedError

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
        }
        for kw, method in mapping.iteritems():
            test = kwargs.get(kw, True)
            if test:
                getattr(self, method)()

        return True

    ### We choose to implement only scalar multiplication and distribution
    ### addition.  While other operations could be defined, their usage is
    ### likely uncommon and the implementation slower as well.

    def __add__(self, other):
        """
        Addition of distributions of the same kind.

        The other distribution must have the same meta information and the
        same eventspace.  If not, raise an exception.

        """
        raise NotImplementedError

    def __mul__(self, other):
        """
        Scalar multiplication on distributions.

        Note, we do not implement distribution-to-distribution multiplication.

        """
        raise NotImplementedError

    def __radd__(self, other):
        raise NotImplementedError

    def __rmul__(self, other):
        raise NotImplementedError
