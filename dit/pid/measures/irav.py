"""
The I_rav measure, defining a 'redundancy' auxiliary variable to capture the
redundant information between sources
"""

import itertools

import numpy as np

from ..pid import BasePID

__all__ = ("PID_RAV",)


def _partition_batches(n, batch=20000):
    """
    All set partitions of n items, in batches, as restricted growth strings.

    A row L assigns item c to block L[c], with L[0] = 0 and L[c] <= 1 + max(L[:c]),
    so every partition of the items appears exactly once.

    Parameters
    ----------
    n : int
        The number of items, at least 1.
    batch : int
        Rows are split into batches of about this size before they grow.

    Yields
    ------
    rows : ndarray
        An integer array of shape (number of rows, n).
    """
    stack = [(np.zeros((1, 1), dtype=np.int16), np.zeros(1, dtype=np.int16))]
    while stack:
        rows, top = stack.pop()
        if rows.shape[1] == n:
            yield rows
            continue
        if len(rows) > batch:
            half = len(rows) // 2
            stack.append((rows[half:], top[half:]))
            stack.append((rows[:half], top[:half]))
            continue
        # every row continues with the labels 0, ..., top + 1
        labels = np.arange(int(top.max()) + 2, dtype=np.int16)
        r_idx, lab = np.nonzero(labels[None, :] <= top[:, None] + 1)
        lab = lab.astype(np.int16)
        stack.append((np.concatenate([rows[r_idx], lab[:, None]], axis=1), np.maximum(top[r_idx], lab)))


def _entropy_rows(m):
    """
    The entropy, in bits, of each row of `m` (the first axis indexes the rows).
    """
    m = m.reshape(len(m), -1)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(m > 0, m * np.log2(np.where(m > 0, m, 1.0)), 0.0)
    return -terms.sum(axis=1)


def _entropy(p):
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _i_rav(d, sources, target):
    """
    The largest coinformation I(S_1; ...; S_n; T; F) over the functions F of the sources.

    F only matters on the joint outcomes of the sources with positive probability (the
    cells), so the maximum is over the partitions of the cells. The coinformation is the
    sum, over the nonempty subsets A of {S_1, ..., S_n, T, F}, of (-1)^(|A| + 1) H(A).
    The terms without F do not depend on F, nor do those whose subset holds F and every
    source, since F is then determined by the rest. The remaining terms are
    (-1)^|B| H(F, B), for the subsets B of {S_1, ..., S_n, T} that leave out some source;
    they are computed for batches of partitions at once.
    """
    d = d.coalesce(tuple(sources) + (target,))
    d = d.copy(base="linear")
    outs = [(o, p) for o, p in zip(d.outcomes, d.pmf, strict=True) if p > 0]
    probs = np.array([p for _, p in outs], dtype=float)
    n = len(sources)

    cell_index = {}
    cell_of = np.array([cell_index.setdefault(o[:-1], len(cell_index)) for o, _ in outs])
    n_cells = len(cell_index)

    constant = 0.0
    varying = []  # (sign, M) with M[c, v] the probability of cell c and value v of B
    for size in range(n + 2):
        for B in itertools.combinations(range(n + 1), size):
            values = {}
            v_of = np.array([values.setdefault(tuple(o[i] for i in B), len(values)) for o, _ in outs])
            h = _entropy(np.bincount(v_of, weights=probs)) if B else 0.0  # H(B), 0 for B empty
            if B:
                constant += (-1) ** (size + 1) * h
            if set(range(n)) <= set(B):
                constant += (-1) ** size * h  # H(F, B) = H(B)
            else:
                M = np.zeros((n_cells, len(values)))
                np.add.at(M, (cell_of, v_of), probs)
                varying.append(((-1) ** size, M))

    best = -np.inf
    for L in _partition_batches(n_cells):
        F = np.eye(n_cells)[L].transpose(0, 2, 1)  # F[row, block, cell]
        total = np.zeros(len(L))
        for sign, M in varying:
            total += sign * _entropy_rows(F @ M)  # the probabilities of (F, B)
        best = max(best, float(total.max()))
    return constant + best


class PID_RAV(BasePID):
    """
    The maximum coinformation auxiliary random variable method.
    """

    _name = "I_RAV"

    @staticmethod
    def _measure(d, sources, target):
        """
        I_RAV is maximum coinformation between all sources, targets, and an
        arbitrary function of the sources.

        The maximum is over all partitions of the joint outcomes of the sources with
        positive probability (their number is the Bell number of the count of those
        outcomes), which are enumerated in batches.

        Parameters
        ----------
        d : Distribution
            The distribution to compute i_rav for.
        sources : iterable of iterables
            The source variables.
        target : iterable
            The target variable.

        Returns
        -------
        i_rav : float
            The value of I_RAV.
        """
        return _i_rav(d, sources, target)
