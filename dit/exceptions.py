#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Information Theory Exceptions
=============================

Exceptions related to information theory.
"""

__all__ = ['ditException',
           'IncompatibleDistribution',
           'InvalidBase',
           'InvalidDistribution',
           'InvalidNormalization',
           'InvalidOutcome',
           'InvalidProbability']

class ditException(Exception):
    """
    Base class for all `dit` exceptions.

    """
    def __init__(self, *args, **kwargs):
        if 'msg' in kwargs:
            # Override the message in the first argument.
            self.msg = kwargs['msg']
        elif args:
            self.msg = args[0]
        else:
            self.msg = ''
        self.args = args
        self.kwargs = kwargs

    def __str__(self):
        return self.msg

    def __repr__(self):
        return "{0}{1}".format(self.__class__.__name__, repr(self.args))

class IncompatibleDistribution(ditException):
    """
    Exception for an incompatible distribution.

    """
    def __init__(self, *args, **kwargs):
        """
        Initialize the exception.

        """
        msg = "The distribution is not compatible."
        args = (msg,) + args
        ditException.__init__(self, *args, **kwargs)

class InvalidBase(ditException):
    """
    Exception for an invalid logarithm base.

    """
    def __init__(self, *args, **kwargs):
        """
        Initialize the exception.

        Parameters
        ----------
        base : float
            The invalid base.

        """
        if args:
            msg = "{0} is not a valid logarithm base.".format(args[0])
            args = (msg,) + args
        ditException.__init__(self, *args, **kwargs)

class InvalidDistribution(ditException):
    """
    Exception thrown for an invalid distribution.

    """
    pass

class InvalidOutcome(ditException):
    """
    Exception for an invalid outcome.

    """
    def __init__(self, *args, **kwargs):
        """
        Initialize the exception.

        Parameters
        ----------
        outcome : sequence
            The invalid outcomes.
        single : bool
            Specifies whether `outcome` represents a single outcome or not.

        """
        single = kwargs.get('single', True)
        try:
            bad = args[0]
        except IndexError:
            # Demand a custom message.
            if 'msg' in kwargs:
                msg = kwargs['msg']
            else:
                msg = ''
        else:
            if single:
                msg = "Outcome {0!r} is not in the sample space.".format(bad)
            else:
                msg = "Outcomes {0} are not in the sample space.".format(bad)
        args = (msg,) + args
        ditException.__init__(self, *args, **kwargs)

class InvalidNormalization(ditException):
    """
    Exception thrown when a distribution is not normalized.

    """
    def __init__(self, *args, **kwargs):
        """
        Initializes the exception.

        The sole argument should be the summation of the probabilities.

        """
        msg = "Bad normalization: {0!r}".format(args[0])
        self.summation = args[0]
        args = (msg,) + args
        ditException.__init__(self, *args, **kwargs)

class InvalidProbability(ditException):
    """
    Exception thrown when a probability is not in [0,1].

    """
    def __init__(self, *args, **kwargs):
        """
        Initialize the exception.

        Parameters
        ----------
        p : float | sequence
            The invalid probability.
        ops : operations
            The operation handler for the incoming probabilities.

        """
        ops = kwargs['ops']
        bounds = "[{0}, {1}]".format(ops.zero, ops.one)
        prob = args[0]
        if len(args[0]) == 1:
            msg = "Probability {0} is not in {1} (base: {2!r})."
        else:
            prob = list(prob)
            msg = "Probabilities {0} are not in {1} (base: {2!r})."
        msg = msg.format(prob, bounds, ops.base)
        args = (msg,) + args
        ditException.__init__(self, *args, **kwargs)
