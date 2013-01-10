"""
Information Theory Exceptions
=============================

Exceptions related to information theory.
"""

__all__ = ['ditException',
           'IncompatibleDistribution',
           'InvalidBase',
           'InvalidDistribution',
           'InvalidEvent',
           'InvalidNormalization',
           'InvalidProbability']

class ditException(Exception):
    """
    Base class for all `dit` exceptions.

    """
    def __init__(self, *args):
        if args:
            self.msg = args[0]
        else:
            self.msg = ''
        self.args = args

    def __str__(self):
        return self.msg

    def __repr__(self):
        return "{0}{1}".format(self.__class__.__name__, repr(self.args))

class IncompatibleDistribution(ditException):
    """
    Exception for an incompatible distribution.

    """
    def __init__(self, *args):
        """
        Initialize the exception.

        """
        msg = "The distribution is not compatible."
        args = (msg,) + args
        ditException.__init__(self, *args)

class InvalidBase(ditException):
    """
    Exception for an invalid logarithm base.

    """
    def __init__(self, *args):
        """
        Initialize the exception.

        Parameters
        ----------
        base : float
            The invalid base.

        """
        msg = "{0} is not a valid logarithm base.".format(args[0])
        args = (msg,) + args
        ditException.__init__(self, *args)

class InvalidDistribution(ditException):
    """
    Exception thrown for an invalid distribution.

    """
    pass

class InvalidEvent(ditException):
    """
    Exception for an invalid event.

    """
    def __init__(self, *args):
        """
        Initialize the exception.

        Parameters
        ----------
        event: sequence
            The invalid events.
        eventspace : sequence
            The event space.

        """
        if len(args) == 1:
            msg = "Event {0} is not in the event space.".format(args[0])
        else:
            msg = "Events {0} are not in the event space.".format(args)
        args = (msg,) + args
        ditException.__init__(self, *args)

class InvalidNormalization(ditException):
    """
    Exception thrown when a distribution is not normalized.

    """
    def __init__(self, *args):
        """
        Initializes the exception.

        The sole argument should be the summation of the probabilities.

        """
        msg = "Bad normalization: {0}".format(args[0])
        self.summation = args[0]
        args = (msg,) + args
        ditException.__init__(self, *args)

class InvalidProbability(ditException):
    """
    Exception thrown when a probability is not in [0,1].

    """
    def __init__(self, *args):
        """
        Initialize the exception.

        Parameters
        ----------
        p : float | sequence
            The invalid probability.

        """
        if len(args[0]) == 1:
            msg = "Probability {0} is not in [0,1].".format(args[0])
        else:
            msg = "Probabilities {0} are not in [0,1].".format(args[0])
        args = (msg,) + args
        ditException.__init__(self, *args)

