"""
Submodular function minimization via the Fujishige--Wolfe minimum-norm-base algorithm.

The greedy algorithm implements linear optimization over the base polyhedron of a
normalized submodular function. Wolfe's minimum-norm-point procedure on that base
polyhedron yields the minimum-norm base used by Chan--Liu agglomerative
info-clustering :cite:`ChanLiu2017agglomerative`; see also
:cite:`Chakrabarty2014wolfe` and Fujishige's connection between minimum-norm
points and submodular minimization.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np

__all__ = (
    "greedy_base_vertex",
    "minimum_norm_base",
)

SubmodularFn = Callable[[frozenset[int]], float]


def _minimum_norm_on_segment(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Euclidean minimum-norm point in ``conv({p, q})``."""
    d = q - p
    denom = float(np.dot(d, d))
    if denom <= 0:
        return p.copy()
    t = -float(np.dot(p, d)) / denom
    t = min(1.0, max(0.0, t))
    return p + t * d


def greedy_base_vertex(
    f: SubmodularFn,
    ground: Sequence[int],
    weights: Mapping[int, float],
) -> dict[int, float]:
    """
    Return the vertex of ``B(f)`` minimizing ``sum_i weights[i] * x[i]``.

    Parameters
    ----------
    f : callable
        Normalized submodular function on subsets of ``ground``; ``f(frozenset())`` must be 0.
    ground : sequence of int
        Ground set elements (distinct).
    weights : mapping
        Linear objective coefficients indexed by ground elements.

    Returns
    -------
    x : dict[int, float]
        The greedy optimum ``x`` with ``x[i]`` for each ``i`` in ``ground``.
    """
    order = sorted(ground, key=lambda e: weights[e])
    x: dict[int, float] = {}
    prefix: frozenset[int] = frozenset()
    for element in order:
        new_prefix = frozenset(set(prefix) | {element})
        x[element] = f(new_prefix) - f(prefix)
        prefix = new_prefix
    return x


def minimum_norm_base(
    f: SubmodularFn,
    ground: Sequence[int],
    *,
    tol: float = 1e-9,
    max_iters: int = 10_000,
    max_minor_iters: int = 10_000,
) -> dict[int, float]:
    """
    Compute the minimum Euclidean-norm point in the base polyhedron ``B(f)``.

    Parameters
    ----------
    f : callable
        Normalized submodular function on subsets of ``ground``.
    ground : sequence of int
        Ground-set elements.
    tol : float
        Wolfe termination tolerance.
    max_iters : int
        Maximum major-cycle iterations.

    Returns
    -------
    x : dict[int, float]
        Minimum-norm base coordinates indexed by ground elements.
    """
    ground_tuple = tuple(ground)
    if not ground_tuple:
        return {}

    def to_dict(vector: np.ndarray) -> dict[int, float]:
        return {ground_tuple[i]: float(vector[i]) for i in range(len(ground_tuple))}

    def to_vec(mapping: Mapping[int, float]) -> np.ndarray:
        return np.array([mapping[i] for i in ground_tuple], dtype=float)

    if len(ground_tuple) == 1:
        element = ground_tuple[0]
        return {element: f(frozenset({element}))}

    if len(ground_tuple) == 2:
        a, b = ground_tuple
        v1 = to_vec(greedy_base_vertex(f, ground_tuple, {a: 0.0, b: 1.0}))
        v2 = to_vec(greedy_base_vertex(f, ground_tuple, {a: 1.0, b: 0.0}))
        return to_dict(_minimum_norm_on_segment(v1, v2))

    weights = dict.fromkeys(ground_tuple, 0.0)
    x = to_vec(greedy_base_vertex(f, ground_tuple, weights))

    # Corral ``S``: vertices whose convex hull contains the current iterate.
    vertices: list[np.ndarray] = [x.copy()]
    lambdas = np.array([1.0])

    for _ in range(max_iters):
        q = to_vec(greedy_base_vertex(f, ground_tuple, to_dict(x)))

        if float(np.dot(x, x)) <= float(np.dot(x, q)) + tol:
            break

        vertices.append(q)
        lambdas = np.append(lambdas, 0.0)

        for _minor in range(max_minor_iters):
            basis = np.column_stack(vertices)
            gram = basis.T @ basis
            ones = np.ones(len(vertices))
            try:
                alpha = np.linalg.solve(gram, ones)
            except np.linalg.LinAlgError:
                alpha = np.linalg.lstsq(gram, ones, rcond=None)[0]
            alpha = alpha / alpha.sum()
            y = basis @ alpha

            if np.all(alpha >= -tol):
                x = y.copy()
                lambdas = alpha.copy()
                break

            negative = alpha < -tol
            ratios = lambdas[negative] / (lambdas[negative] - alpha[negative])
            theta = float(np.min(ratios))

            lambdas = theta * alpha + (1.0 - theta) * lambdas
            x = basis @ lambdas

            keep = lambdas > tol
            vertices = [vertices[i] for i in range(len(vertices)) if keep[i]]
            lambdas = lambdas[keep]
        else:
            msg = f"minimum_norm_base minor cycle did not converge within {max_minor_iters} iterations"
            raise RuntimeError(msg)
    else:
        msg = f"minimum_norm_base did not converge within {max_iters} iterations"
        raise RuntimeError(msg)

    return to_dict(x)
