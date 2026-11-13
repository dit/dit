"""
The path-based redundancy measure of Sigtermans, based on causal tensors.
"""

from itertools import permutations

import numpy as np

from ..pid import BasePID

__all__ = ("PID_CT",)


def path_redundancy(d, sources, target, atol=1e-6):
    """
    Compute the path-based redundancy of Sigtermans :cite:`sigtermans2020path`
    (Definition 2): the least path-based mutual information over the paths
    through all the sources that end at the target, leaving out the paths whose
    link to the target is not direct.

    A path is an ordering of the sources followed by the target. Each of its
    links is the transition tensor :math:`p(b \\mid a)` between consecutive
    variables, and its tensor is their product. Its path-based mutual information
    is that of its first source and the target when the source passes through
    that tensor, the mutual information of the Markov chain along the path.

    The link from a source to the target is not direct when an indirect path from
    that source to the target, through some of the other sources, has the same
    tensor (Proposition 1). The paths that end with such a link are left out,
    unless every path does.

    Parameters
    ----------
    d : Distribution
        The joint distribution.
    sources : iterable of iterables
        The source variables.
    target : iterable
        The target variable.
    atol : float
        Two tensors are the same if their entries differ by at most this much in
        total.

    Returns
    -------
    red : float
        The redundancy, in bits.
    """
    sources = [list(source) for source in sources]
    d = d.coalesce([*sources, list(target)])
    d.make_dense()
    p = d.pmf.reshape([len(a) for a in d.alphabet])
    n = len(sources)
    axes = range(n + 1)
    marginals = [p.sum(axis=tuple(k for k in axes if k != i)) for i in axes]

    def link(i, j):
        pair = p.sum(axis=tuple(k for k in axes if k not in (i, j)))
        pair = pair if i < j else pair.T
        return pair / marginals[i][:, np.newaxis]

    def tensor(path):
        product = link(path[0], path[1])
        for a, b in zip(path[1:-1], path[2:], strict=True):
            product = product @ link(a, b)
        return product

    def information(path):
        product = tensor(path)
        joint = marginals[path[0]][:, np.newaxis] * product
        mask = joint > 0
        return float(np.sum(joint[mask] * np.log2((product / marginals[n][np.newaxis, :])[mask])))

    if n == 1:
        return information((0, n))

    def direct(i):
        others = [j for j in range(n) if j != i]
        return not any(
            np.abs(tensor((i, *via, n)) - link(i, n)).sum() <= atol
            for k in range(1, n)
            for via in permutations(others, k)
        )

    is_direct = [direct(i) for i in range(n)]
    orders = list(permutations(range(n)))
    paths = [order for order in orders if is_direct[order[-1]]] or orders
    return min(information((*order, n)) for order in paths)


def i_triangle(d, source_0, source_1, target):
    """
    Compute the path-based redundancy of Sigtermans for two sources: the least
    path-based mutual information of the paths through the other source whose
    last link is direct.

    Parameters
    ----------
    d : Distribution
        The joint distribution.
    source_0, source_1 : iterable
        The source variables.
    target : iterable
        The target variable.

    Returns
    -------
    red : float
        The redundancy, in bits.
    """
    return path_redundancy(d, [source_0, source_1], target)


class PID_CT(BasePID):
    """
    The partial information decomposition of Sigtermans, from the path-based
    redundancy of causal tensors, for any number of sources.
    """

    _name = "I_△"

    @staticmethod
    def _measure(d, sources, target):
        """
        The path-based redundancy of Sigtermans.

        Parameters
        ----------
        d : Distribution
            The distribution to compute the redundancy of.
        sources : iterable of iterables
            The source variables.
        target : iterable
            The target variable.

        Returns
        -------
        red : float
            The redundancy.
        """
        return path_redundancy(d, sources, target)
