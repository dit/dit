"""
The I_rdr measure (reachable decision regions) of Mages & Rohner.

For each target outcome t, a source S defines a binary-input channel for the event
T = t, and the decisions it allows form a convex region bounded by its ROC curve.
The union information of a collection of sources is the pointwise information of
the convex hull of their regions (their join), averaged over t, and the redundancy
of an antichain follows from the union information of its sub-collections by
inclusion-exclusion.

References
----------
.. [1] T. Mages and C. Rohner, "Decomposing and Tracing Mutual Information by
       Quantifying Reachable Decision Regions", Entropy 25, 1014, 2023.
.. [2] T. Mages, E. Anastasiadi and C. Rohner, "Non-Negative Decomposition of
       Multivariate Information: From Minimum to Blackwell-Specific Information",
       Entropy 26, 424, 2024.
"""

from itertools import combinations

import numpy as np

from ..pid import BasePID, _pid_coerce

__all__ = ("PID_RDR",)


# Points of ROC curves closer than this in both coordinates are taken to be equal.
_TOL = 1e-12


def _roc_curves(dist, source, target, p_t):
    """
    ROC curves of the binary-input channels from the events T = t to a source.

    Parameters
    ----------
    dist : Distribution
        The joint distribution.
    source : iterable
        The source variables.
    target : iterable
        The target variables.
    p_t : dict
        The probability of each target value t with 0 < p(t) < 1.

    Returns
    -------
    curves : dict
        For each t in `p_t`, the vertices (false positive rate, true positive rate)
        of the ROC curve of the source for the event T = t, from (0, 0) to (1, 1).
    """
    pairs = dist.coalesce([list(source), list(target)]).to_dict()
    symbols = {s: i for i, s in enumerate(dict.fromkeys(s for s, _ in pairs))}
    targets = {t: j for j, t in enumerate(p_t)}
    joint = np.zeros((len(symbols), len(targets)))
    for (s, t), p in pairs.items():
        if t in targets:
            joint[symbols[s], targets[t]] += p

    curves = {}
    for t, j in targets.items():
        hit = joint[:, j]
        miss = np.delete(joint, j, axis=1).sum(axis=1)
        ratio = np.divide(hit, miss, out=np.full_like(hit, np.inf), where=miss > 0)
        order = np.argsort(-ratio, kind="stable")
        # Dividing the cumulative sums by their totals ends every curve exactly at (1, 1).
        fpr = np.cumsum(miss[order])
        tpr = np.cumsum(hit[order])
        curves[t] = np.vstack([[0.0, 0.0], np.column_stack([fpr / fpr[-1], tpr / tpr[-1]])])
    return curves


def _upper_envelope(points):
    """
    Vertices of the upper concave envelope of points in the unit square that include
    (0, 0) and (1, 1), from (0, 0) to (1, 1), by the monotone chain algorithm.

    Points which are numerically equal are merged, so that the envelope has no
    spurious edges of rounding-error size.
    """
    points = points[np.lexsort((points[:, 1], points[:, 0]))]
    hull = []
    for p in points:
        while hull and np.all(np.abs(p - hull[-1]) <= _TOL):
            p = np.maximum(p, hull.pop())
        while len(hull) >= 2:
            (x0, y0), (x1, y1) = hull[-2], hull[-1]
            if (x1 - x0) * (p[1] - y0) - (y1 - y0) * (p[0] - x0) < 0:
                break
            hull.pop()
        hull.append(p)
    return np.array(hull)


def _pointwise_information(p, envelope):
    """
    Pointwise information about the event T = t, of probability `p`, of the channel
    whose ROC curve is `envelope`: sum_k x_k log2(x_k / (p x_k + (1 - p) y_k)), over
    the edges (y_k, x_k) of the envelope (Equation 28a of Mages & Rohner).
    """
    y, x = np.clip(np.diff(envelope, axis=0), 0.0, None).T
    x, y = x[x > 0], y[x > 0]
    return float(np.sum(x * np.log2(x / (p * x + (1 - p) * y))))


class _UnionInformation:
    """
    The I_rdr union information of collections of sources, caching the ROC curve of
    each source and the union information of each collection.
    """

    def __init__(self, dist, target):
        self._dist = dist
        self._target = tuple(target)
        p_t = dist.coalesce([list(target)]).to_dict()
        self._p_t = {t: p for (t,), p in p_t.items() if 0 < p < 1}
        self._curves = {}
        self._values = {}

    def _source_curves(self, source):
        if source not in self._curves:
            self._curves[source] = _roc_curves(self._dist, source, self._target, self._p_t)
        return self._curves[source]

    def __call__(self, collection):
        """
        The union information of a collection of sources, sum_t p(t) i(t; join).
        """
        key = tuple(sorted(tuple(source) for source in collection))
        if key not in self._values:
            curves = [self._source_curves(source) for source in key]
            self._values[key] = sum(
                p * _pointwise_information(p, _upper_envelope(np.vstack([c[t] for c in curves])))
                for t, p in self._p_t.items()
            )
        return self._values[key]


def _redundancy(union, sources):
    """
    The redundancy of an antichain of sources, by inclusion-exclusion over the union
    information of its sub-collections.
    """
    sources = [tuple(source) for source in sources]
    return sum((-1) ** (k + 1) * union(sub) for k in range(1, len(sources) + 1) for sub in combinations(sources, k))


class PID_RDR(BasePID):
    """
    The reachable decision regions partial information decomposition of Mages & Rohner.

    For each target outcome, the union information of a collection of sources is the
    pointwise information of the join of their reachable decision regions, the convex
    hull of their ROC curves. Redundancies follow by inclusion-exclusion, and the
    union information of each collection is computed once for the whole lattice.
    """

    _name = "I_rdr"

    @staticmethod
    def _measure(dist, sources, target):
        """
        Compute I_rdr(sources : target) by inclusion-exclusion over union information.

        Parameters
        ----------
        dist : Distribution
            The distribution to compute i_rdr for.
        sources : iterable of iterables
            The source variables.
        target : iterable
            The target variable.

        Returns
        -------
        irdr : float
            The value of I_rdr.
        """
        return _redundancy(_UnionInformation(dist, target), sources)

    def _compute(self):
        """
        Compute every redundancy from union information shared across the lattice.
        """
        missing = [node for node in self._lattice if node not in self._reds]
        if missing:
            union = _UnionInformation(self._dist, self._target)
            for node in missing:
                self._reds[node] = _pid_coerce(self._scaled(_redundancy(union, node)))
        super()._compute()
