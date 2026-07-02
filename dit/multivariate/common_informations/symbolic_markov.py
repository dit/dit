"""
Symbolic feasible-set parametrization for the common-information optimizers.

The Wyner and Exact common informations both minimise a functional over
auxiliary variables ``V`` that render the ``rvs`` conditionally independent.
That feasible set is exactly the set of nonnegative rank-``k`` mixture
decompositions of the joint tensor::

    p(x_1, ..., x_n) = sum_v pi_v * prod_i r_i(x_i | v)

This module builds that decomposition symbolically (sympy unknowns for the
mixture weights ``pi_v`` and the per-variable channels ``r_i(x_i | v)``) together
with the equations a feasible point must satisfy: the marginal-match equations
(the mixture reproduces the given joint) and the simplex normalisations.

Only the ``numpy``/linear representation and single-dimension ``rvs`` groups are
supported (v1); this covers the giant bit, doubly-symmetric binary source,
products, and XOR-type sources.
"""

from collections import namedtuple
from itertools import product

import numpy as np

from ...helpers import normalize_rvs, parse_rvs

__all__ = (
    "MixtureModel",
    "build_mixture_model",
    "min_feasible_cardinality",
)


class SymbolicOptimizationError(Exception):
    """Raised when a symbolic common-information solve cannot close."""


# ``pi``    : tuple of ``k`` mixture-weight symbols.
# ``r``     : list (one per grouped rv) of ``(size, k)`` object arrays of
#             channel symbols ``r_i[a, v] = r_i(x_i = a | v)``.
# ``equations`` : list of sympy expressions constrained to ``== 0`` (marginal
#             match + simplex normalisations).
# ``symbols``   : flat tuple of all unknown symbols.
# ``joint``     : the given joint tensor (object array), dims = grouped rvs.
# ``sizes``     : per-grouped-rv alphabet sizes.
# ``k``         : the auxiliary cardinality.
MixtureModel = namedtuple(
    "MixtureModel",
    ["pi", "r", "equations", "symbols", "joint", "sizes", "k"],
)


def _grouped_joint(dist, rvs):
    """
    Return the joint tensor over the grouped ``rvs`` as an object array.

    Each entry of ``rvs`` must select exactly one distribution dimension (v1
    restriction). The result has one axis per group, in ``rvs`` order, with
    that group's alphabet along the axis.

    Returns
    -------
    joint : np.ndarray (dtype object)
        The marginal joint over the grouped variables.
    sizes : list of int
        The alphabet size of each grouped variable.
    """
    import sympy

    dims = list(dist.data.dims)
    group_axes = []
    for group in rvs:
        names, indices = parse_rvs(dist, group, unique=True, sort=True)
        if len(indices) != 1:
            raise NotImplementedError(
                "Symbolic common information supports only single-variable rvs "
                f"groups in v1; got group {group!r} spanning {len(indices)} variables."
            )
        group_axes.append(indices[0])

    # Marginalise out any dimension not selected by rvs.
    keep = set(group_axes)
    drop_axes = tuple(i for i in range(len(dims)) if i not in keep)
    data = dist._linear_data().values
    marg = data.sum(axis=drop_axes) if drop_axes else data

    # After summing, the surviving axes keep their original relative order; map
    # them onto the requested rvs order.
    surviving = [i for i in range(len(dims)) if i in keep]
    perm = [surviving.index(ax) for ax in group_axes]
    joint = np.transpose(marg, perm)

    joint = np.array([sympy.sympify(v) for v in joint.ravel()], dtype=object).reshape(joint.shape)
    sizes = list(joint.shape)
    return joint, sizes


def min_feasible_cardinality(sizes):
    """
    Smallest auxiliary cardinality ``k`` for which the mixture has at least as
    many free parameters as marginal-match constraints.

    unknowns(k) = (k - 1) + k * sum_i (size_i - 1)
    constraints = prod_i size_i - 1

    Returns
    -------
    k : int
        The smallest ``k >= 1`` satisfying ``unknowns(k) >= constraints``.
    """
    constraints = int(np.prod(sizes)) - 1
    per_k = sum(s - 1 for s in sizes)
    k = 1
    while (k - 1) + k * per_k < constraints:
        k += 1
    return k


def build_mixture_model(dist, rvs=None, crvs=None, k=None):
    """
    Build the symbolic rank-``k`` mixture model for ``dist``.

    Parameters
    ----------
    dist : Distribution
        A symbolic (or numeric) distribution.
    rvs : list of lists, None
        The variables to render conditionally independent. If None, every
        single variable is used.
    crvs : list, None
        Conditioning variables. Not supported in v1 (must be falsy).
    k : int, None
        The auxiliary cardinality. If None, the smallest feasible ``k`` from
        :func:`min_feasible_cardinality` is used.

    Returns
    -------
    model : MixtureModel
        The unknowns and feasibility equations.
    """
    import sympy

    rvs, crvs = normalize_rvs(dist, rvs, crvs)
    if crvs:
        raise NotImplementedError("Symbolic common information does not support crvs (v1).")

    joint, sizes = _grouped_joint(dist, rvs)
    n = len(sizes)
    if k is None:
        k = min_feasible_cardinality(sizes)

    pi = tuple(sympy.Symbol(f"pi_{v}", nonnegative=True) for v in range(k))

    r = []
    for i, size in enumerate(sizes):
        arr = np.empty((size, k), dtype=object)
        for a in range(size):
            for v in range(k):
                arr[a, v] = sympy.Symbol(f"r{i}_{a}_{v}", nonnegative=True)
        r.append(arr)

    equations = []

    # Simplex: mixture weights sum to one.
    equations.append(sympy.Add(*pi) - 1)

    # Simplex: each channel column (a distribution over the group's alphabet
    # given v) sums to one.
    for i, size in enumerate(sizes):
        for v in range(k):
            equations.append(sympy.Add(*[r[i][a, v] for a in range(size)]) - 1)

    # Marginal match: the mixture reproduces every joint cell.
    for cell in product(*[range(s) for s in sizes]):
        terms = []
        for v in range(k):
            prod_term = pi[v]
            for i in range(n):
                prod_term = prod_term * r[i][cell[i], v]
            terms.append(prod_term)
        equations.append(sympy.Add(*terms) - joint[cell])

    flat_symbols = list(pi)
    for arr in r:
        flat_symbols.extend(arr.ravel().tolist())

    return MixtureModel(
        pi=pi,
        r=r,
        equations=equations,
        symbols=tuple(flat_symbols),
        joint=joint,
        sizes=sizes,
        k=k,
    )
