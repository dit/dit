#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module defining base distribution class.

The set of all possible outcomes is known as the sample space. An element of
the sample space is known as an outcome or sample. Ambiguously, "sample" is
also used to refer to a sequence of outcomes. For example, we often speak
about the "sample size".  In general, `dit` will tend to use the term "outcome"
over "sample".  The main exception will be that we still refer to the "sample
space", instead of the "outcome space", as this is mostly a universal standard.

Recall that an event is a subset of outcomes from the sample space. In `dit`
distributions are specified by assigning probabilities to each outcome
of the sample space.  This corresponds to assigning probabilities to each of
the atoms in the underlying sigma-algebra that defines the probability space.
Queries to the distribution using the [] operator return the probability of an
outcome---it is not necessary to pass the outcome in as a singleton event.
Event probabilities are obtained through the `event_probability` method.
There is a corresponding `outcome_probability` method as well (which just uses
the [] operator).

`None` is not an allowable outcome.  `dit` uses `None` to signify that an
outcome does not exist in the sample space.  We do not enforce this rule.
Rather, things will probably just break.

The most basic type of outcome must be: hashable and equality comparable.
If the distribution's sample space is to be ordered, then the outcomes must
also be orderable (which implies equality comparable).

Joint outcomes must be: hashable, orderable, and also a sequence.
Recall, a sequence is a sized, iterable container. See:
http://docs.python.org/3/library/collections.abc.html#collections-abstract-base-classes

TODO: Examine thread-safety issues.  Most of the methods will not function
properly if interrupted midway through (and the distribution is modified).

"""
from __future__ import print_function, division

import numpy as np

from .math import approximate_fraction

from .exceptions import ditException

from .params import ditParams

from six.moves import map, zip # pylint: disable=redefined-builtin

def prepare_string(dist, digits=None, exact=False, tol=1e-9,
                   show_mask=False, str_outcomes=False):
    """
    Prepares a distribution for a string representation.

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
    show_mask : bool
        If `True`, show the mask for marginal distributions.
    str_outcomes : bool
        If `True`, then attempt to convert outcomes which are tuples to just
        strings.  This is only a dislplay technique.

    Returns
    -------
    pmf : sequence
        The formatted pmf.  This could be a NumPy array (possibly rounded)
        or a list of Fraction instances.
    outcomes : sequence
        The formated outcomes.
    base : str or float
        The base of the formatted pmf.
    colsep : str
        The column separation for printing.
    max_length : int
        The length of the largest outcome, as a string.
    pstr : str
        A informative string representing the probability of an outcome.
        This will be 'p(x)' xor 'log p(x)'.

    """
    colsep = '   '

    # Create outcomes with wildcards, if desired and possible.
    if show_mask:
        if not dist.is_joint():
            msg = '`show_mask` can be `True` only for joint distributions'
            raise ditException(msg)

        if show_mask != True and show_mask != False:
            # The user is specifying what the mask should look like.
            wc = show_mask
        else:
            wc = '*'

        ctor = dist._outcome_ctor
        def outcome_wc(outcome):
            """
            Builds the wildcarded outcome.

            """
            i = 0
            e = []
            for is_masked in dist._mask:
                if is_masked:
                    symbol = wc
                else:
                    symbol = outcome[i]
                    i += 1
                e.append(symbol)

            e = ctor(e)
            return e
        outcomes = map(outcome_wc, dist.outcomes)
    else:
        outcomes = dist.outcomes

    # Convert outcomes to strings, if desired and possible.
    if str_outcomes:
        if not dist.is_joint():
            msg = '`str_outcomes` can be `True` only for joint distributions'
            raise ditException(msg)

        try:
            # First, convert the elements of the outcome to strings.
            outcomes_ = [map(str, outcome) for outcome in outcomes]
            # Now convert the entire outcome to a string
            outcomes_ = map(lambda o: ''.join(o), outcomes_)
            # Force the iterators to expand in case there are exceptions.
            outcomes = list(outcomes_)
        except:
            outcomes = map(str, outcomes)
    else:
        outcomes = map(str, outcomes)

    outcomes = list(outcomes)
    if len(outcomes):
        max_length = max(map(len, outcomes))
    else:
        max_length = 0

    # 1) Convert to linear probabilities, if necessary.
    if exact:
        # Copy to avoid precision loss
        d = dist.copy(base='linear')
    else:
        d = dist

    # 2) Round, if necessary, possibly after converting to linear probabilities.
    if digits is not None and digits is not False:
        pmf = d.pmf.round(digits)
    else:
        pmf = d.pmf

    # 3) Construct fractions, if necessary.
    if exact:
        pmf = [approximate_fraction(x, tol) for x in pmf]

    if d.is_log():
        pstr = 'log p(x)'
    else:
        pstr = 'p(x)'

    base = d.get_base()

    return pmf, outcomes, base, colsep, max_length, pstr


class BaseDistribution(object):
    """
    The base class for all "distribution" classes.

    Generally, distributions are mutable in that the outcomes and probabilities
    can be changed from zero to nonzero and back again. However, the sample
    space is not mutable and must remain fixed over the lifetime of the
    distribution.

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

    Methods
    -------
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

    rand
        Returns a random draw from the distribution.

    set_base
        Changes the base of the distribution, possibly in-place.

    to_string
        Returns a string representation of the distribution.

    validate
        A method to validate that the distribution is valid.

    zipped
        Returns an iterator over (outcome, probability) tuples. The probability
        could be a log probability or a linear probability.

    """
    _meta = None

    # These should be set in the subclass's init function.
    outcomes = None
    ops = None
    pmf = None
    prng = None

    def __init__(self, prng=None):
        """
        Common initialization for all distribution types.

        """
        # We set the prng to match the global dit.math prng.
        # Usually, this should be good enough.  If something more exotic
        # is desired, the user can change the prng manually.
        if prng is None:
            import dit.math
            prng = dit.math.prng
        self.prng = prng

        self._meta = {
            'is_joint': None,
            'is_numerical': None,
        }


    def __contains__(self, outcome):
        raise NotImplementedError

    def __delitem__(self, outcome):
        raise NotImplementedError

    def __getitem__(self, outcome):
        raise NotImplementedError

    def __iter__(self):
        """
        Returns an iterator over the outcomes in the distribution.

        """
        return iter(self.outcomes)

    def __len__(self):
        """
        Returns the number of outcomes in the distribution.

        """
        return len(self.outcomes)

    def __repr__(self):
        """
        """
        if ditParams['repr.print']:
            return self.to_string()
        else:
            return super(BaseDistribution, self).__repr__()

    def __reversed__(self):
        """
        Returns a reverse iterator over the outcomes.

        """
        return reversed(self.outcomes)

    def __setitem__(self, key, value):
        raise NotImplementedError

    def __str__(self):
        """
        Returns a string representation of the distribution.

        """
        return self.to_string()

    def _validate_outcomes(self):
        """
        Returns `True` if the outcomes are in the sample space.

        Returns
        -------
        v : bool
            `True` if the outcomes are in the sample space.

        Raises
        ------
        InvalidOutcome
            When an outcome is not in the sample space.

        """
        from .validate import validate_outcomes
        # Provide the actual sample space, not an iterator over it.
        v = validate_outcomes(self.outcomes, self._sample_space) # pylint: disable=no-member
        return v

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
        from .validate import validate_normalization
        v = validate_normalization(self.pmf, self.ops)
        return v

    def atoms(self, patoms=False):
        """
        Returns the atoms of the probability space.

        Parameters
        ----------
        patoms : bool
            If `True`, then return only the P-atoms, those atoms with positive
            probability according to the probability measure P. If `False`,
            then all atoms of the underlying measurable space are returned.
            In this case, its behavior is synonymous with `sample_space()`.

        """
        mode = 'atoms'
        if patoms:
            mode = 'patoms'

        for outcome, _ in self.zipped(mode):
            yield outcome

    def copy(self):
        """
        Returns a deep copy of the distribution.

        """
        raise NotImplementedError


    def event_probability(self, event):
        """
        Returns the probability of an event.

        Parameters
        ----------
        event : iterable
            Any subset of outcomes from the sample space.

        Returns
        -------
        p : float
            The probability of the event.

        """
        pvals = np.array([self[o] for o in event], dtype=float)
        p = self.ops.add_reduce(pvals)
        return p

    def event_space(self):
        """
        Returns a generator over the event space.

        The event space is the powerset of the sample space.

        """
        from dit.utils import powerset
        return powerset(list(self.sample_space()))

    def get_base(self, numerical=False):
        """
        Returns the base of the distribution.

        If the distribution represents linear probabilities, then the string
        'linear' will be returned.  If the base of log probabilities is e,
        then the returned base could be the string 'e' or its numerical value.

        Parameters
        ----------
        numerical : bool
            If `True`, then if the base is 'e', it is returned as a float.

        """
        return self.ops.get_base(numerical=numerical)

    def has_outcome(self, outcome, null=True):
        """
        Returns `True` if `outcome` exists in the sample space).

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

    def rand(self, size=None, rand=None, prng=None):
        """
        Returns a random sample from the distribution.

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

    def sample_space(self):
        """
        Returns an iterator over the ordered sample space.

        """
        raise NotImplementedError

    def set_base(self, base, copy=True):
        """
        Changes the base of the distribution.

        If the distribution is a linear distribution and the base is changed,
        the the distribution will be subsequently represent log distributions.
        If the distribution is a log distribution, then it can be converted to
        another base by passing in some other base. Alternatively, one can
        convert to a linear distribution by passing 'linear'. Note, conversions
        introduce errors, especially when converting from very negative log
        probabilities to linear probabilities (underflow is an issue).
        For this reason, `copy` defaults to True.

        Parameters
        ----------
        base : float or string
            The desired base for the distribution.  If 'linear', then the
            distribution's pmf will represent linear probabilities. If any
            positive float (other than 1) or 'e', then the pmf will represent
            log probabilities with the specified base.
        copy : bool
            If `True`, a copy is made of the distribution first. Then the base
            of the copied distribution is modified. Otherwise, the base is
            changed in-place. Generally, it is dangerous to change the base
            in-place, as numerical errors can be introduced.  Additionally,
            functions or classes that hold references to the distribution may
            not expect a change in base.

        Returns
        -------
        dist : distribution
            The distribution with the new base. If `inplace` was True, the
            same distribution is returned, just modified.

        """
        raise NotImplementedError

    def to_dict(self):
        """
        Returns the distribution as a standard Python dictionary.

        """
        return dict(self.zipped())

    def to_string(self, digits=None, exact=None, tol=1e-9):
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
        from six import StringIO
        s = StringIO()

        if exact is None:
            exact = ditParams['print.exact']

        x = prepare_string(self, digits, exact, tol)
        pmf, outcomes, base, colsep, max_length, pstr = x

        headers = ["Class: ",
                   "Alphabet: ",
                   "Base: "]
        vals = [self.__class__.__name__,
                self.alphabet, # pylint: disable=no-member
                base]

        L = max(map(len, headers))
        for head, val in zip(headers, vals):
            s.write("{0}{1}\n".format(head.ljust(L), val))
        s.write("\n")

        s.write(''.join(['x'.ljust(max_length), colsep, pstr, "\n"]))
        for o, p in zip(outcomes, pmf):
            s.write(''.join([o.ljust(max_length), colsep, str(p), "\n"]))

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
        outcomes : bool
            If `True` verify that every outcome exists in the outcome space.
            This is a sanity check on the data structure.

        norm : bool
            If `True`, verify that the distribution is properly normalized.

        Returns
        -------
        valid : bool
            True if the distribution is valid.

        Raises
        ------
        InvalidOutcome
            Raised if an outcome is not in the outcome space.
        InvalidNormalization
            Raised if the distribution is improperly normalized.

        """
        mapping = {
            'outcomes': '_validate_outcomes',
            'norm': '_validate_normalization',
        }
        for kw, method in mapping.items():
            test = kwargs.get(kw, True)
            if test:
                getattr(self, method)()

        return True

    def zipped(self, mode='pmf'):
        """
        Returns an iterator over (outcome, probability) tuples.

        Parameters
        ----------
        mode : str
            A string specifying the iteration mode. Three options are valid:
            'pmf', 'atoms', 'patoms'.  With 'pmf', we iterate over every
            outcome in and probability represented by the pmf.  Some of these
            outcomes could be null probabilities or not.  Only if the
            distribution is dense will every atom in the sample space be
            yielded.  With 'atoms', we iterate over the entire sample space.
            With 'patoms', we iterate over the entire sample space but yield
            only on those atoms which are P-atoms, where P is the probability
            measure on the probability space. That is, only atoms such that
            P(atom) > 0 are yielded.

        """
        modes = ['pmf', 'atoms', 'patoms']
        if mode not in modes:
            raise ditException('Invalid mode.')

        if mode == 'pmf':
            for x in zip(self.outcomes, self.pmf):
                yield x
        elif mode == 'atoms':
            # Then we want to iterate over the sample space.
            #
            # There are many ways to do this, but we need to do it in a way
            # that does not rely on the particular implementation's storage
            # mechanism for the sample space.
            for outcome in self.sample_space():
                yield outcome, self[outcome]
        else:
            is_null = self.ops.is_null
            # Here we can shortcut and go through outcomes and pmf,
            # while pruning null outcomes.
            for outcome, prob in zip(self.outcomes, self.pmf):
                if not is_null(prob):
                    yield outcome, prob

    ### We choose to implement only scalar multiplication and distribution
    ### addition, as they will be useful for constructing convex combinations.
    ### While other operations could be defined, their usage is likely uncommon
    ### and the implementation slower as well.

    def __add__(self, other):
        """
        Addition of distributions of the same kind.

        The other distribution must have the same meta information and the
        same sample space.  If not, raise an exception.

        """
        raise NotImplementedError

    def __mul__(self, other):
        """
        Scalar multiplication on distributions.

        Note, we do not implement distribution-to-distribution multiplication.
        Perhaps we should?

        """
        raise NotImplementedError

    def __radd__(self, other):
        raise NotImplementedError

    def __rmul__(self, other):
        raise NotImplementedError
