"""
The I_wedge measure, as proposed by Griffith et al.
"""

from collections import defaultdict

from ...distribution import Distribution
from ...multivariate import coinformation
from ..pid import BasePID

__all__ = ("PID_GK",)


def _meet_labels(d, groups):
    """
    The component of the Gács-Körner meet of `groups` holding each outcome of `d`.

    Outcomes are joined when some group takes the same value in both. Union-find
    over the support finds the components without building sigma-algebras. The
    meet of a single group is the group itself.

    Parameters
    ----------
    d : Distribution
        The distribution.
    groups : iterable of iterables
        The groups of variables.

    Returns
    -------
    labels : list
        A label of the component of each outcome of positive probability, in order.
    """
    parent = {}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    firsts = []
    for outcome, p in zip(d.outcomes, d.pmf, strict=True):
        if p == 0:
            continue
        keys = [(i, tuple(outcome[c] for c in group)) for i, group in enumerate(groups)]
        for key in keys:
            parent.setdefault(key, key)
        root = find(keys[0])
        for key in keys[1:]:
            other = find(key)
            if other != root:
                parent[other] = root
        firsts.append(keys[0])
    return [find(key) for key in firsts]


def _meets(d, *sides):
    """
    The joint distribution of the Gács-Körner meets of `sides`, one variable per side.

    Parameters
    ----------
    d : Distribution
        The distribution.
    *sides : iterables of iterables
        Each a collection of groups of variables.

    Returns
    -------
    meets : Distribution
        The distribution of the components of the meets.
    """
    labels = [_meet_labels(d, groups) for groups in sides]
    ids = [{} for _ in sides]
    joint = defaultdict(int)
    probs = [p for p in d.pmf if p != 0]
    for p, *roots in zip(probs, *labels, strict=True):
        joint[tuple(i.setdefault(r, len(i)) for i, r in zip(ids, roots, strict=True))] += p
    return Distribution(list(joint), list(joint.values()))


def _meet_information(d, sources, target):
    """
    Compute I(meet(sources) : target).

    Parameters
    ----------
    d : Distribution
        The distribution.
    sources : iterable of iterables
        The source variables.
    target : iterable
        The target variable.

    Returns
    -------
    info : float
        The mutual information between the meet and the target.
    """
    return coinformation(_meets(d, sources, [target]), [[0], [1]])


class PID_GK(BasePID):
    """
    The Griffith et al partial information decomposition.

    This PID is known to produce negative partial information values.
    """

    _name = "I_∧"

    @staticmethod
    def _measure(d, sources, target):
        """
        Compute I_wedge(sources : target) = I(meet(sources) : target)

        Parameters
        ----------
        d : Distribution
            The distribution to compute i_wedge for.
        sources : iterable of iterables
            The source variables.
        target : iterable
            The target variable.

        Returns
        -------
        iwedge : float
            The value of I_wedge.
        """
        return _meet_information(d, sources, target)
