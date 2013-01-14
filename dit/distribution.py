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

The joint event type must be: hashable, orderable, and a sequence.

"""
from __future__ import print_function

from itertools import izip

import numpy as np

from .math import close, prng, approximate_fraction

from .exceptions import (
    InvalidBase,
    InvalidEvent,
    InvalidNormalization
)

def prepare_string(dist, digits=None, exact=False, tol=1e-9):
    """
    Returns a string representation of the distribution.

    Parameters
    ----------
    dist : distribution
        The distribution to be stringified.
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
    pmf : sequence
        The formatted pmf.  This could be a NumPy array (possibly rounded)
        or a list of Fraction instances.
    event : sequence
        The formated events.
    base : str or float
        The base of the formatted pmf.
    colsep : str
        The column separation for printing.
    max_length : int
        The maximum length of the events, as strings.
    pstr : str
        A string representing the probabilit of an event: 'p(x)' or 'log p(x)'.

    """
    colsep = '   '
    events = map(str, dist.events)

    if  len(events):
        max_length = max(map(len, events))
    else:
        max_length = 0

    # 1) Convert to linear probabilities, if necessary.
    if exact:
        # Copy to avoid precision loss
        d = dist.copy()
        d.set_base('linear')
    else:
        d = dist

    # 2) Round, if necessary, possibly after converting to linear probabilities.
    if digits is not None and digits is not False:
        pmf = d.pmf.round(digits)
    else:
        pmf = d.pmf

    # 3) Construct fractions, in necessary.
    if exact:
        pmf = [approximate_fraction(x, tol) for x in pmf]

    if d.is_log():
        pstr = 'log p(x)'
    else:
        pstr = 'p(x)'

    base = d.get_base()

    return pmf, events, base, colsep, max_length, pstr


class BaseDistribution(object):
    """
    The base class for all "distribution" classes.

    Generally, distributions are mutuable in that the events and probabilities
    can be changed from zero to nonzero and back again. However, the eventspace
    is not mutable and must remain fixed over the lifetime of the distribution.

    Meta Properties
    ---------------
    is_joint
        Boolean specifying if the pmf represents a joint distribution.

    is_numerical
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

    prng : RandomState
        A pseudo-random number generator with a `rand` method which can
        generate random numbers. For now, this is assumed to be something
        with an API compatibile to NumPy's RandomState class. This attribute
        is initialized to equal dit.math.prng.

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

    has_event
        Returns `True` is the distribution has `event` in the eventspace.

    is_approx_equal
        Returns `True` if the distribution is approximately equal to another
        distribution.

    is_joint
        Returns `True` if the distribution is a joint distribution.

    is_log
        Returns `True` if the distribution values are log probabilities.

    is_numerical
        Returns `True` if the distribution values are numerical.

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

    """
    # Subclasses should update these meta attributes *before* calling the base
    # distribution's __init__ function.
    _meta = {
        'is_joint': False,
        'is_numerical': None,
    }

    # These should be set in the subclass's init function.
    events = None
    ops = None
    pmf = None
    prng = None

    def __init__(self):
        """
        Common initialization for all distribution types.

        """
        # We set the prng to match the global dit.math prng.
        # Usually, this should be good enough.  If something more exotic
        # is desired, the user can change the prng manually.
        self.prng = prng

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
        Returns the number of events in the distribution's pmf.

        """
        return len(self.events)

    def __reversed__(self):
        """
        Returns a reverse iterator over the non-null events.

        """
        return reversed(self.events)

    def __setitem__(self, key, value):
        raise NotImplementedError

    def __str__(self):
        """
        Returns a string representation of the distribution.

        """
        return self.to_string()

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
        L = len(bad)
        if L == 1:
            raise InvalidEvent(bad, single=True)
        elif L:
            raise InvalidEvent(bad, single=False)

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

    def get_base(self):
        """
        Returns the base of the distribution.

        If the distribution represents linear probabilities, then the string
        'linear' will be returned.  If the base of log probabilities is e,
        then the returned base could be the string 'e' or its numerical value.

        """
        return self.ops.base

    def has_event(self, event, null=True):
        """
        Returns `True` if `event` is a valid event (exists in the eventspace).

        """
        raise NotImplementedError

    def is_approx_equal(self, other):
        """
        Returns `True` is `other` is approximately equal to this distribution.

        Parameters
        ----------
        other : distribution
            The distribution to compare against.

        Notes
        -----
        The distributions need not have the same base or even same length.

        """
        raise NotImplementedError

    def is_joint(self):
        """
        Returns `True` if the distribution is a joint distribution.

        """
        return self._meta["is_joint"]

    def is_log(self):
        """
        Returns `True` if the distribution values are log probabilities.

        """
        return self.ops.base != 'linear'

    def is_numerical(self):
        """
        Returns `True` if the distribution values are numerical.

        """
        return self._meta['is_numerical']

    def normalize(self):
        """
        Normalizes the distribution.

        """
        raise NotImplementedError

    def sample(self, size=None, rand=None, prng=None):
        """
        Returns a sample from a discrete distribution.

        Parameters
        ----------
        size : int or None
            The number of samples to draw from the distribution. If `None`,
            then a single sample is returned.  Otherwise, a list of samples is
            returned.
        rand : float or NumPy array or None
            When `size` is `None`, `rand` should be a random number from the
            interval [0,1]. When `size` is not `None`, then `rand` should be
            a NumPy array of random numbers.  In either situation, if `rand` is
            `None`, then the random number will be drawn from a pseudo random
            number generator.
        prng : random number generator
            A random number generator with a `rand' method that returns a
            random number between 0 and 1 when called with no arguments. If
            unspecified, then we use the random number generator on the
            distribution.

        Returns
        -------
        s : sample or list
            The sample(s) drawn from the distribution.

        """
        import dit.math
        s = dit.math.sample(self, size, rand, prng)
        return s

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
        from itertools import izip
        from StringIO import StringIO
        s = StringIO()

        x = prepare_string(self, digits, exact, tol)
        pmf, events, base, colsep, max_length, pstr = x

        s.write("Class: {}\n".format(self.__class__.__name__))
        s.write("Alphabet: {}\n".format(self.alphabet))
        s.write("Base: {}\n\n".format(base))
        s.write(''.join([ 'x'.ljust(max_length), colsep, pstr, "\n" ]))

        for e,p in izip(events, pmf):
            s.write(''.join( [e.ljust(max_length), colsep, str(p), "\n"] ))
        s.seek(0)
        s = s.read()
        # Remove the last \n
        s = s[:-1]
        return s

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
    ### addition, as they will be useful for constructing convex combinations.
    ### While other operations could be defined, their usage is likely uncommon
    ### and the implementation slower as well.

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
