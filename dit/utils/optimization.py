"""
Helpful utilities for performing optimization.
"""

from collections import namedtuple

from operator import itemgetter

from string import digits, ascii_letters

import numpy as np

class BasinHoppingInnerCallBack(object):
    """
    A callback to track the optimization vectors as the optimization is performed.

    Attributes
    ----------
    positions : [ndarray]
        The path of the optimization.
    jumps : [int]
        The times at which a basin hop was performed.
    """

    def __init__(self):
        """
        Initialize the object.
        """
        self.positions = []
        self.jumps = []

    def __call__(self, x):
        """
        Record the current position.

        Parameters
        ----------
        x : ndarray
            The current optimization vector.
        """
        self.positions.append(x.copy())


Candidate = namedtuple('Candidate', ['position', 'value', 'constraints'])


class BasinHoppingCallBack(object):
    """
    scipy's basinhopping return status often will return an optimization vector which does not
    satisfy the constraints if it has a lower objective value and ran in to some sort of error
    status rather than detecting that it is in a local minima. This object tracks the minima found
    for each basin hop, potentially keeping track of a global optima which would be discarded.

    Attributes
    ----------
    constraints : [functions]
        The constraints used.
    icb : BasinHoppingInnerCallBack, None
        A callback object for recording the full path.
    candidates : [ndarray]
        The minima of each basin.
    """

    def __init__(self, constraints, icb=None):
        """
        Parameters
        ----------
        optimizer : MarkovVarOptimizer
            The optimizer to track the optimization of.
        """
        self.constraints = [ c['fun'] for c in constraints ]
        self.icb = icb
        self.candidates = []

    def __call__(self, x, f, accept):
        """
        Parameters
        ----------
        x : ndarray
            Optimization vector.
        f : float
            Current value of the objective.
        accept : bool
        """
        x = x.copy()

        constraints = [ float(c(x)) for c in self.constraints ]
        self.candidates.append(Candidate(x, f, constraints))

        if self.icb:
            self.icb.jumps.append(len(self.icb.positions))

    def minimum(self, cutoff=1e-7):
        """
        Return the position of the smallest basin.

        Parameters
        ----------
        cutoff : float
            The cutoff for constraint satisfaction.

        Returns
        -------
        min : ndarray
            The minimum basin.
        """
        possible = [(x, f) for x, f, cs in self.candidates if max(cs) < cutoff]
        possible = [(x, f) for x, f in possible if accept_test(x_new=x)]
        return min(possible, key=itemgetter(1))[0] if possible else None


class Uniquifier(object):
    """
    Given a stream of catagorical symbols, provide a mapping to unique consecutive integers.

    Attributes
    ----------
    mapping : dict
    inverse : dict
    """

    def __init__(self):
        """
        """
        self.chars = digits+ascii_letters
        self.mapping = {}
        self.inverse = {}

    def __call__(self, item, string=False):
        """
        """
        if item not in self.mapping:
            n = len(self.mapping)
            self.mapping[item] = n
            self.inverse[n] = item

        if string:
            return self.chars[self.mapping[item]]
        else:
            return self.mapping[item]


def accept_test(**kwargs):
    """
    Reject basin jumps that move outside of [0,1].

    Returns
    -------
    accept : bool
        Whether to accept a jump or not.
    """
    x = kwargs['x_new']
    tmax = bool(np.all(x <= 1))
    tmin = bool(np.all(x >= 0))
    return tmin and tmax


def basinhop_status(res):
    """
    Determine whether an optimization result was successful or not, working
    around differences in scipy < 0.17.0 and scipy >= 0.17.0.

    Parameters
    ----------
    res : OptimizeResult
        The result to parse

    Returns
    -------
    success : bool
        Whether the optimization was successful or not.
    msg : str
        The result's message.
    """
    try:
        success = res.lowest_optimization_result.success
        msg = res.lowest_optimization_result.message
    except AttributeError:
        success = 'success' in res.message[0]
        msg = res.message[0]
    return success, msg
