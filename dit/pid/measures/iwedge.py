"""
The I_wedge measure, as proposed by Griffith et al.
"""

from collections import defaultdict

from ...distribution import Distribution
from ...multivariate import coinformation
from ..pid import BasePID

__all__ = ("PID_GK",)


def _meet_information(d, sources, target):
    """
    Compute I(meet(sources) : target).

    The Gács-Körner meet of the sources labels each outcome by its connected
    component, outcomes being joined when some source takes the same value in
    both. Union-find over the support finds the components without building
    sigma-algebras.

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
    parent = {}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    rows = []
    for outcome, p in zip(d.outcomes, d.pmf, strict=True):
        if p == 0:
            continue
        keys = [(i, tuple(outcome[c] for c in source)) for i, source in enumerate(sources)]
        for key in keys:
            parent.setdefault(key, key)
        root = find(keys[0])
        for key in keys[1:]:
            other = find(key)
            if other != root:
                parent[other] = root
        rows.append((keys[0], tuple(outcome[c] for c in target), p))

    labels = {}
    joint = defaultdict(int)
    for key, t, p in rows:
        joint[(labels.setdefault(find(key), len(labels)), *t)] += p
    meet = Distribution(list(joint), list(joint.values()))
    return coinformation(meet, [[0], list(range(1, len(target) + 1))])


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
