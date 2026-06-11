"""
Minimal GF(2) linear algebra for the channel-coding constructions.
"""

import numpy as np

from ..exceptions import ditException

__all__ = (
    "inverse",
    "matvec",
    "nullspace",
    "rank",
    "rref",
)


def matvec(A, x):
    """Matrix-vector product over GF(2)."""
    return (np.asarray(A, dtype=int) @ np.asarray(x, dtype=int)) % 2


def rref(matrix):
    """
    Reduced row-echelon form over GF(2).

    Parameters
    ----------
    matrix : array-like
        A binary matrix.

    Returns
    -------
    R : ndarray
        The reduced row-echelon form.
    pivots : list of int
        The pivot column indices.
    """
    M = np.asarray(matrix, dtype=int).copy() % 2
    rows, cols = M.shape
    pivots = []
    r = 0
    for c in range(cols):
        pivot = None
        for i in range(r, rows):
            if M[i, c]:
                pivot = i
                break
        if pivot is None:
            continue
        M[[r, pivot]] = M[[pivot, r]]
        for i in range(rows):
            if i != r and M[i, c]:
                M[i] = (M[i] + M[r]) % 2
        pivots.append(c)
        r += 1
        if r == rows:
            break
    return M, pivots


def rank(matrix):
    """The rank of a binary matrix over GF(2)."""
    _, pivots = rref(matrix)
    return len(pivots)


def nullspace(matrix):
    """
    A basis for the null space ``{x : matrix @ x = 0}`` over GF(2).

    Parameters
    ----------
    matrix : array-like
        A binary matrix with ``n`` columns.

    Returns
    -------
    basis : ndarray
        An array whose rows span the null space.
    """
    M = np.asarray(matrix, dtype=int) % 2
    n = M.shape[1]
    R, pivots = rref(M)
    pivot_set = set(pivots)
    free = [c for c in range(n) if c not in pivot_set]
    basis = []
    for f in free:
        v = np.zeros(n, dtype=int)
        v[f] = 1
        for ri, pc in enumerate(pivots):
            v[pc] = R[ri, f]
        basis.append(v % 2)
    return np.array(basis, dtype=int).reshape(len(basis), n)


def inverse(matrix):
    """
    The inverse of a square binary matrix over GF(2).

    Parameters
    ----------
    matrix : array-like
        A square, invertible binary matrix.

    Returns
    -------
    inv : ndarray
    """
    M = np.asarray(matrix, dtype=int).copy() % 2
    k = M.shape[0]
    if M.shape[0] != M.shape[1]:
        raise ditException("Matrix must be square to invert.")
    # Gauss-Jordan elimination on [M | I]; the right block becomes the inverse.
    A = np.hstack([M, np.eye(k, dtype=int)])
    for c in range(k):
        pivot = None
        for i in range(c, k):
            if A[i, c]:
                pivot = i
                break
        if pivot is None:
            raise ditException("Matrix is singular over GF(2).")
        A[[c, pivot]] = A[[pivot, c]]
        for i in range(k):
            if i != c and A[i, c]:
                A[i] = (A[i] + A[c]) % 2
    return A[:, k:]
