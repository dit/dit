r"""
Nonnegative decompositions in closed form, from minima of a set function.

For a nondecreasing set function :math:`g` with :math:`g(\emptyset) = 0`, the redundancy
:math:`I_\cap(\beta) = \min_{B \in \beta} g(B)` has nonnegative atoms, all on one chain of
nodes: cut :math:`[0, g([n]))` at the values of :math:`g`, and give each layer to the node
formed by the minimal groups reaching it (the layer cake, Lemma 2 of
:cite:`vansanten2026between`). With :math:`g(A) = I(X_A : Y)` it is :math:`\Immi{\bullet}`.
Only the mutual information of each group of sources is needed, not the redundancy
lattice, so these functions work with more sources than :class:`PIDBounds`.
"""

import numpy as np

from ._utils import _group_node
from .bounds import _informations
from .diagnostics import _linear_and_scale, _sources_and_target

__all__ = (
    "max_unique_atoms",
    "mmi_atoms",
)


def _layer_cake(values, n, atol):
    """
    The nonzero atoms of the redundancy min_{B in beta} g(B) (Lemma 2).

    Parameters
    ----------
    values : np.ndarray
        The values of g, in bits, indexed by the bitmasks of groups of `n` sources.
    n : int
        The number of sources.
    atol : float
        Values of g closer than this, in bits, count as one.

    Returns
    -------
    atoms : list of tuples
        Pairs of a node, as a tuple of bitmasks, and its atom, along the chain from the
        bottom.
    """
    g = np.maximum(np.asarray(values, dtype=float), 0.0)
    # the least nondecreasing majorant, which only absorbs rounding errors
    for group in range(1, 1 << n):
        for i in range(n):
            if group >> i & 1:
                g[group] = max(g[group], g[group & ~(1 << i)])
    levels = []
    for value in sorted(set(g[1:].tolist())):
        if value <= atol:
            continue
        if levels and value - levels[-1][1] <= atol:
            levels[-1][1] = value
        else:
            levels.append([value, value])
    atoms, previous = [], 0.0
    for threshold, value in levels:
        reaching = {group for group in range(1, 1 << n) if g[group] >= threshold}
        minimal = tuple(
            sorted(
                group
                for group in reaching
                if not any(group & ~(1 << i) in reaching for i in range(n) if group >> i & 1)
            )
        )
        atoms.append((minimal, value - previous))
        previous = value
    return atoms


def _prepare(dist, sources, target):
    sources, target = _sources_and_target(dist, sources, target)
    linear, scale = _linear_and_scale(dist)
    return sources, _informations(linear, sources, target), float(scale)


def mmi_atoms(dist, sources=None, target=None, atol=1e-9):
    r"""
    The nonzero atoms of the minimum mutual information decomposition, in closed form.

    The redundancy of :math:`\Immi{\bullet}` is :math:`\min_{B \in \beta} I(X_B : Y)`. Its
    nonzero atoms lie on one chain: with :math:`0 = w_0 < w_1 < \cdots < w_m` the distinct
    values of :math:`I(X_A : Y)`, the node formed by the minimal groups :math:`A` with
    :math:`I(X_A : Y) \geq w_k` has the atom :math:`w_k - w_{k-1}`
    :cite:`vansanten2026between`, Lemma 2. The top atom is
    :math:`\min_i I(X_i : Y | X_{\text{rest}})`, the largest synergy of any nonnegative
    decomposition. Only the :math:`2^n - 1` mutual informations are needed, not the
    redundancy lattice.

    Parameters
    ----------
    dist : Distribution
        The distribution.
    sources : iterable of iterables, None
        The sources. If None, every variable but the target.
    target : iterable, None
        The target. If None, the last variable.
    atol : float
        Mutual informations closer than this count as equal.

    Returns
    -------
    atoms : dict
        Maps each node with a nonzero atom to the atom, in the units of `dist`, along the
        chain from the bottom. Every other atom is zero.

    Examples
    --------
    >>> from dit.pid import mmi_atoms
    >>> from dit.pid.distributions import bivariates
    >>> {node: round(atom, 4) for node, atom in mmi_atoms(bivariates['and']).items()}
    {((0,), (1,)): 0.3113, ((0, 1),): 0.5}
    """
    sources, informations, scale = _prepare(dist, sources, target)
    return {
        _group_node(sources, node): atom * scale for node, atom in _layer_cake(informations, len(sources), atol / scale)
    }


def max_unique_atoms(dist, sources=None, target=None, atol=1e-9):
    r"""
    The nonzero atoms of a nonnegative decomposition in which every source has its largest
    unique information.

    No nonnegative decomposition gives source :math:`X_i` more unique information than
    Hi-Fi's :math:`U_i = \min_Z I(X_i : Y | X_Z)` (see :func:`hifi`), and this one attains
    every bound at once: it places :math:`U_i` on the unique atom of each source and fills
    in the rest with the layer cake of :math:`g(A) = I(X_A : Y) - \sum_{i \in A} U_i`
    :cite:`vansanten2026between`, Theorem 1(ii). For two sources it is :class:`PID_HiFi`.
    Other decompositions attain the same unique informations, so this one is not canonical.
    Only the :math:`2^n - 1` mutual informations are needed, not the redundancy lattice.

    Parameters
    ----------
    dist : Distribution
        The distribution.
    sources : iterable of iterables, None
        The sources. If None, every variable but the target.
    target : iterable, None
        The target. If None, the last variable.
    atol : float
        Values closer than this count as equal.

    Returns
    -------
    atoms : dict
        Maps each node with a nonzero atom to the atom, in the units of `dist`: the unique
        atoms first, then the chain of the layer cake from the bottom. Every other atom is
        zero.

    Examples
    --------
    >>> from dit.pid import max_unique_atoms
    >>> from dit.pid.distributions import bivariates
    >>> {node: round(atom, 4) for node, atom in max_unique_atoms(bivariates['and']).items()}
    {((0,),): 0.3113, ((1,),): 0.3113, ((0, 1),): 0.1887}
    """
    sources, informations, scale = _prepare(dist, sources, target)
    n = len(sources)
    full = (1 << n) - 1
    uniques = []
    for i in range(n):
        rest = full & ~(1 << i)
        context = (z for z in range(1 << n) if not z & ~rest)
        uniques.append(max(0.0, min(informations[z | 1 << i] - informations[z] for z in context)))
    g = informations.copy()
    for group in range(1, 1 << n):
        g[group] -= sum(u for i, u in enumerate(uniques) if group >> i & 1)
    atoms = {}
    for i, unique in enumerate(uniques):
        if unique > atol / scale:
            atoms[_group_node(sources, (1 << i,))] = unique
    for node, atom in _layer_cake(g, n, atol / scale):
        label = _group_node(sources, node)
        atoms[label] = atoms.get(label, 0.0) + atom
    return {node: atom * scale for node, atom in atoms.items()}
