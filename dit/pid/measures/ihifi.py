"""
The information-theoretic Hi-Fi decomposition, the zero-overlap end of the segment of
nonnegative two-source decompositions.
"""

from ...exceptions import ditException
from ...multivariate import coinformation
from .._utils import _group_node, _source_pairs
from ..pid import BaseBivariatePID, BaseIncompletePID

__all__ = ("PID_HiFi",)


def _pair_nodes(sources):
    """
    The lattice nodes formed by two disjoint groups of sources.

    Parameters
    ----------
    sources : tuple of tuples
        The sources.

    Returns
    -------
    nodes : frozenset
        The nodes, labelled like those of the redundancy lattice: each group of sources
        becomes the sorted tuple of its variables.
    """
    return frozenset(_group_node(sources, pair) for pair in _source_pairs(len(sources)))


class PID_HiFi(BaseBivariatePID):
    r"""
    The information-theoretic Hi-Fi decomposition :cite:`lazic2026information`.

    Hi-Fi :cite:`ontiveroortega2025assessing` decomposes the dependence of a target
    :math:`Y` on a driver :math:`X_i`, given subsets :math:`W` of the other variables:
    the unique part is :math:`U_i = \min_W I(X_i : Y | W)`, the redundant part
    :math:`I(X_i : Y) - U_i`, and the synergistic part
    :math:`\max_W I(X_i : Y | W) - I(X_i : Y)`. :math:`U_i` is the optimally conditioned
    transfer entropy of :cite:`korenek2025higher`. For two sources this is a
    decomposition with

    .. math::

        R = \max\{0, I(X_0 : X_1 : Y)\}, \qquad S = \max\{0, -I(X_0 : X_1 : Y)\}, \qquad
        U_i = \min\{I(X_i : Y), I(X_i : Y | X_j)\}.

    It is the zero-overlap end of the segment of nonnegative two-source decompositions,
    whose ends were given by :cite:`goodwell2017temporal`. It has the smallest redundancy
    and synergy, and the largest unique informations, of any nonnegative decomposition
    :cite:`vansanten2026between`. It is symmetric, satisfies self-redundancy, monotonicity
    and the identity property, and vanishes for independent sources. It depends on more
    than the source-target marginals and lacks the Blackwell property.

    Notes
    -----
    With more than two sources Hi-Fi is not a decomposition. As for other two-source
    measures, the redundancy of every pair of disjoint groups of sources is computed, so
    that every such pair has zero overlap, and the rest of the lattice is filled in from
    those values. Zero overlap on every pair at once is often impossible
    :cite:`vansanten2026between`. The decomposition then has negative atoms, as for the
    distribution of :math:`X_0 = (a, b)`, :math:`X_1 = a`, :math:`X_2 = c` and
    :math:`Y = (a, b \oplus c)`, which forces one bit of overlap. Atoms the pairs do not
    determine are ``nan``. :func:`~dit.pid.hifi` gives Hi-Fi's bounds for each source.
    """

    _name = "I_HF"

    # only pairs of disjoint groups of sources have a Hi-Fi redundancy; a node of fewer
    # elements than sources must not borrow one from a decomposition of its own elements.
    REDUCED_PID = False

    @staticmethod
    def _measure(d, sources, target):
        """
        The redundancy of two (groups of) sources: the positive part of their co-information
        with the target.

        Parameters
        ----------
        d : Distribution
            The distribution.
        sources : iterable of iterables
            The two source variables.
        target : iterable
            The target variable.

        Returns
        -------
        red : float
            The Hi-Fi redundancy.

        Raises
        ------
        ditException
            If there are not two sources.
        """
        if len(sources) != 2:
            msg = f"I_HF is only defined for two sources, {len(sources)} given."
            raise ditException(msg)
        return max(0.0, float(coinformation(d, [*sources, target])))

    def _redundancy(self, node):
        """
        Pairs of disjoint groups of sources have a Hi-Fi redundancy; every other node is
        found by filling in the lattice.
        """
        if node in _pair_nodes(self._sources):
            return self._scaled(self._measure(self._dist, node, self._target, **self._kwargs))
        return BaseIncompletePID._redundancy(self, node)

    def _compute(self):
        """
        Compute the redundancy of every pair of disjoint groups of sources, then fill in
        what those values determine.
        """
        for node in _pair_nodes(self._sources):
            if node not in self._reds:
                self._reds[node] = self._scaled(self._measure(self._dist, node, self._target, **self._kwargs))
        BaseIncompletePID._compute(self)
