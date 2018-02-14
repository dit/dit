"""
Helpful utilities for performing optimization.
"""

from collections import defaultdict, namedtuple

from functools import wraps

from operator import itemgetter

from string import digits, ascii_letters

import numpy as np

from scipy.optimize import OptimizeResult


__all__ = [
    'BasinHoppingCallBack',
    'BasinHoppingInnerCallBack',
    'Uniquifier',
    'accept_test',
    'basinhop_status',
    'colon',
    'memoize_optvec',
]


colon = slice(None, None, None)


class BasinHoppingInnerCallBack(object):  # pragma: no cover
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

    def jumped(self, time):
        """
        Add jump time to `self.jumps`

        Parameters
        ----------
        time : int
            The time of a jump
        """
        self.jumps.append(time)


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

    Notes
    -----
    This will be unneccessary once this PR if complete: https://github.com/scipy/scipy/pull/7819
    """

    def __init__(self, constraints, icb=None):
        """
        Parameters
        ----------
        optimizer : MarkovVarOptimizer
            The optimizer to track the optimization of.
        """
        self.eq_constraints = [c['fun'] for c in constraints if c['type'] == 'eq']
        self.ineq_constraints = [c['fun'] for c in constraints if c['type'] == 'ineq']
        self.icb = icb
        self.eq_candidates = []
        self.ineq_candidates = []

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

        eq_constraints = [float(c(x)) for c in self.eq_constraints]
        ineq_constraints = [float(c(x)) for c in self.ineq_constraints]
        self.eq_candidates.append(Candidate(x, f, eq_constraints))
        self.ineq_candidates.append(Candidate(x, f, ineq_constraints))

        if self.icb:  # pragma: no cover
            self.icb.jumped(len(self.icb.positions))

    def minimum(self, cutoff=1e-7):
        """
        Return the position of the smallest basin.

        Parameters
        ----------
        cutoff : float
            The cutoff for constraint satisfaction.

        Returns
        -------
        min : np.ndarray
            The minimum basin.
        """
        eq_possible = [(x, f) for x, f, cs in self.eq_candidates if all(c < cutoff for c in cs)]
        ineq_possible = [(x, f) for x, f, cs in self.ineq_candidates if all(c > -cutoff for c in cs)]
        possible = [(x, f) for x, f in eq_possible if any(np.allclose(x, x2) for x2, _ in ineq_possible)]
        possible = [(x, f) for x, f in possible if accept_test(x_new=x)]
        try:
            x, val = min(possible, key=itemgetter(1))
            opt_res = OptimizeResult({'x': x,
                                      'success': True,
                                      })
            return opt_res
        except ValueError:  # pragma: no cover
            return None


class Uniquifier(object):
    """
    Given a stream of categorical symbols, provide a mapping to unique consecutive integers.

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
    except AttributeError:  # pragma: no cover
        success = 'success' in res.message[0]
        msg = res.message[0]
    return success, msg


def memoize_optvec(f):  # pragma: no cover
    """
    Make a memoized version of methods which take a single argument;
    an optimization vector.

    Parameters
    ----------
    f : func
        A method which takes as a single argument an np.ndarray.

    Returns
    -------
    wrapper : func
        A memoized version of `f`.
    """
    @wraps(f)
    def wrapper(self, x):
        tx = tuple(x)
        try:
            cache = self.__cache
        except AttributeError:
            self.__cache = defaultdict(dict)
            cache = self.__cache

        if tx in cache:
            if f in cache[tx]:
                return cache[tx][f]

        value = f(self, x)

        cache[tx][f] = value

        return value

    return wrapper
