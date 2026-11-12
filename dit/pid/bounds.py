r"""
Bounds on the atoms of every nonnegative partial information decomposition.

A decomposition of sources :math:`X_0, \ldots, X_{n-1}` and a target :math:`Y` assigns an
atom :math:`\pi(\alpha)` to every node :math:`\alpha` of the redundancy lattice, so that

.. math::

    I(X_A : Y) = \sum_{\alpha : A \in \mathcal{U}(\alpha)} \pi(\alpha)

for every nonempty group :math:`A` of sources, where :math:`\mathcal{U}(\alpha)` is the
set of groups containing an element of :math:`\alpha`. These equations and
:math:`\pi \geq 0` define a polytope, the nonnegative decompositions of the distribution
:cite:`vansanten2026between`. Its projection onto each atom, and onto each redundancy
:math:`I_\cap(\beta) = \sum_{\alpha \preceq \beta} \pi(\alpha)`, is an interval, found by
linear programming.
"""

import warnings
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from scipy.optimize import linprog

from ..exceptions import OptimizationException, ditException
from ..multivariate import coinformation
from ..utils import build_table
from ._utils import _canonical, _group_node, _group_variables, _label, _source_pairs
from .diagnostics import _linear_and_scale, _sources_and_target

__all__ = (
    "ForcedOverlap",
    "Interval",
    "OverlapCondition",
    "PIDBounds",
    "forced_overlap",
    "zero_overlap_conditions",
)

# five sources already give 7579 nodes; six would give 7828352
_MAX_SOURCES = 5

# the HiGHS defaults (1e-7) leave residuals of about 4e-8 bits
_LP_OPTIONS = {"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10}

# a value at most this large, in bits, in a feasible decomposition certifies a lower bound of 0
_ZERO = 1e-12


def _informations(dist, sources, target):
    """
    The mutual information, in bits, of the target and each group of sources.

    Parameters
    ----------
    dist : Distribution
        A distribution with linear probabilities.
    sources : tuple of tuples
        The sources.
    target : tuple
        The target.

    Returns
    -------
    informations : np.ndarray
        :math:`I(X_A : Y)`, indexed by the bitmask of :math:`A`; entry 0 is 0.
    """
    informations = np.zeros(1 << len(sources))
    for group in range(1, 1 << len(sources)):
        informations[group] = float(coinformation(dist, [list(_group_variables(sources, group)), list(target)]))
    return informations


def _antichains(n):
    """
    The nodes of the redundancy lattice of `n` sources: the antichains of nonempty groups of
    sources, each group a bitmask and each antichain a tuple of them.
    """
    groups = range(1, 1 << n)
    nodes = []

    def extend(start, chosen):
        if chosen:
            nodes.append(tuple(chosen))
        for k in range(start, len(groups)):
            group = groups[k]
            if all(group & other not in (group, other) for other in chosen):
                chosen.append(group)
                extend(k + 1, chosen)
                chosen.pop()

    extend(0, [])
    return nodes


def _up_set(node, n):
    """
    The groups containing an element of `node`, as a bitmask with bit ``g - 1`` for group ``g``.
    """
    up_set = 0
    for group in range(1, 1 << n):
        if any(element & group == element for element in node):
            up_set |= 1 << (group - 1)
    return up_set


class _Lattice:
    """
    The redundancy lattice of some sources, built from bitmasks so that five sources are
    feasible, with nodes labelled as in :class:`~dit.pid.pid.BasePID`.
    """

    def __init__(self, sources):
        self.sources = sources
        self.n = n = len(sources)
        nodes = _antichains(n)
        up_sets = [_up_set(node, n) for node in nodes]
        labels = [_group_node(sources, node) for node in nodes]
        if len(set(labels)) != len(labels):
            msg = f"The sources {sources} do not give distinct groups of variables."
            raise ditException(msg)
        # the top node first, as in the tables of BasePID
        order = sorted(range(len(nodes)), key=lambda j: (up_sets[j].bit_count(), -len(nodes[j]), labels[j]))
        self.nodes = [nodes[j] for j in order]
        self.labels = [labels[j] for j in order]
        self.up_sets = np.array([up_sets[j] for j in order], dtype=np.int64)
        self.index = {label: j for j, label in enumerate(self.labels)}
        groups = np.arange(1, 1 << n, dtype=np.int64)
        # row g - 1 adds up the atoms counted in I(X_g : Y)
        self.incidence = ((self.up_sets[None, :] >> (groups[:, None] - 1)) & 1).astype(float)

    def below(self, j):
        """
        The indicator of the nodes at or below node `j`, whose up-sets contain its up-set.
        """
        return ((self.up_sets[j] & ~self.up_sets) == 0).astype(float)

    def find(self, node):
        """
        The index of `node`, given as an iterable of iterables of variables.
        """
        try:
            return self.index[_canonical(node)]
        except (KeyError, TypeError):
            msg = f'"{node}" is not a node of the lattice of {self.sources}.'
            raise ditException(msg) from None

    def pairs(self, pairs):
        """
        Pairs of disjoint groups of sources, as pairs of bitmasks.

        Parameters
        ----------
        pairs : str or iterable
            ``'hifi'``, ``'all'``, or the lattice nodes of the pairs.

        Returns
        -------
        pairs : list of tuples of int
        """
        if isinstance(pairs, str):
            return _source_pairs(self.n, pairs)
        groups = {_group_variables(self.sources, group): group for group in range(1, 1 << self.n)}
        resolved = []
        for pair in pairs:
            msg = f'"{pair}" is not a pair of disjoint groups of the sources {self.sources}.'
            try:
                first, second = (groups[tuple(sorted(element))] for element in pair)
            except (KeyError, TypeError, ValueError):
                raise ditException(msg) from None
            if first & second:
                raise ditException(msg)
            resolved.append((first, second))
        return resolved


class Interval(NamedTuple):
    """
    The range of an atom or a redundancy over the nonnegative decompositions.
    """

    lower: float
    upper: float

    @property
    def width(self):
        """
        The length of the interval.
        """
        return self.upper - self.lower


@dataclass(frozen=True)
class ForcedOverlap:
    """
    The least total overlap that a nonnegative decomposition can have on some pairs of
    groups of sources, returned by :func:`forced_overlap`.

    Attributes
    ----------
    total : float
        The least sum, over the nonnegative decompositions, of the overlaps of the pairs.
        It is zero exactly when one decomposition has zero overlap on every pair.
    overlaps : dict
        Maps each pair, as the lattice node of its two groups, to its overlap in `witness`.
    witness : dict
        The nonzero atoms of one decomposition attaining `total`. It is generally not unique.
    """

    total: float
    overlaps: dict
    witness: dict


class OverlapCondition(NamedTuple):
    """
    A violated condition of Proposition 8 of :cite:`vansanten2026between`, returned by
    :func:`zero_overlap_conditions`: in every nonnegative decomposition the overlaps of
    `pairs` add up to at least `amount`.

    Attributes
    ----------
    condition : str
        ``'M'`` or ``'S'``.
    pairs : tuple
        The pairs, as lattice nodes of two groups of sources.
    amount : float
        The overlap the condition forces.
    """

    condition: str
    pairs: tuple
    amount: float


class PIDBounds:
    r"""
    The range of each atom and redundancy over every nonnegative partial information
    decomposition of a distribution.

    By default the only constraints are the mutual informations :math:`I(X_A : Y)` of the
    target with each group of sources, which every decomposition reproduces, and local
    positivity, :math:`\pi \geq 0`; no redundancy measure is chosen. The nonnegative decompositions form
    a polytope containing, for instance, that of :math:`\Immi{\bullet}`
    :cite:`vansanten2026between`. Each interval is a projection of the polytope: the ends
    of different intervals are generally attained by different decompositions, so not
    every combination of values within the intervals is possible.

    Two sources leave one free parameter, the overlap of redundancy and synergy, so every
    interval is a segment from Hi-Fi to MMI (see :func:`pid_overlap`). The largest
    redundancy of every node is that of MMI, and the largest unique information of every
    source is Hi-Fi's :math:`U_i` (:func:`hifi`); both are attained by one decomposition
    for all nodes or sources at once :cite:`vansanten2026between`, Theorem 1.

    Parameters
    ----------
    dist : Distribution
        The distribution.
    sources : iterable of iterables, None
        The sources, at most five. If None, every variable but the target.
    target : iterable, None
        The target. If None, the last variable.
    fixed : dict, None
        Redundancies to fix, mapping lattice nodes to values, e.g. the pairwise redundancies
        of a two-source measure.
    zero_overlap : str, iterable, None
        Require zero overlap, :math:`I_\cap(\{A\}\{B\}) = [I(X_A : X_B : Y)]_+`, on pairs of
        disjoint groups of sources: ``'hifi'`` for the pairs with a single source on one
        side, ``'all'`` for every pair, or an iterable of the lattice nodes of the pairs.
    atol : float
        Atoms whose interval is at most this wide count as identified.

    Raises
    ------
    ValueError
        If there are more than five sources, or the log base is below one.
    NotImplementedError
        If `dist` is symbolic.

    Notes
    -----
    Every interval needs up to two linear programs. Each takes milliseconds for three or
    four sources, and about 30 ms for five, whose full table solves some 15,000 of them;
    intervals are therefore computed only when requested. For more sources use
    :func:`mmi_atoms` and :func:`max_unique_atoms`, which need only the mutual information
    of each group of sources.

    Examples
    --------
    >>> from dit.pid import PIDBounds
    >>> from dit.pid.distributions import bivariates
    >>> bounds = PIDBounds(bivariates['and'])
    >>> [round(value, 4) for value in bounds[((0,), (1,))]]
    [0.0, 0.3113]
    """

    def __init__(self, dist, sources=None, target=None, *, fixed=None, zero_overlap=None, atol=1e-9):
        sources, target = _sources_and_target(dist, sources, target)
        if len(sources) > _MAX_SOURCES:
            msg = (
                f"PIDBounds supports at most {_MAX_SOURCES} sources, {len(sources)} given; "
                "use mmi_atoms or max_unique_atoms for more."
            )
            raise ValueError(msg)
        linear, scale = _linear_and_scale(dist)
        self._setup(_informations(linear, sources, target), sources, scale, fixed, zero_overlap, atol)

    @classmethod
    def _from_informations(cls, informations, sources, fixed=None, zero_overlap=None, atol=1e-9):
        """
        Bounds for a nondecreasing set function in place of the mutual informations, in bits.
        """
        bounds = cls.__new__(cls)
        bounds._setup(np.asarray(informations, dtype=float), tuple(map(tuple, sources)), 1.0, fixed, zero_overlap, atol)
        return bounds

    def _setup(self, informations, sources, scale, fixed, zero_overlap, atol):
        self._informations = informations
        self._scale = float(scale)
        self._atol = atol
        self._lattice = lattice = _Lattice(sources)
        rows = [lattice.incidence]
        values = [informations[1:]]
        for node, value in (fixed or {}).items():
            rows.append(lattice.below(lattice.find(node))[None, :])
            values.append([value / scale])
        for first, second in lattice.pairs(zero_overlap) if zero_overlap is not None else []:
            rows.append(lattice.below(lattice.index[_group_node(sources, (first, second))])[None, :])
            values.append([max(0.0, self._coinformation(first, second))])
        self._constrained = len(rows) > 1
        self._a_eq = np.vstack(rows)
        self._b_eq = np.concatenate([np.asarray(value, dtype=float) for value in values])
        size = len(lattice.nodes)
        self._atom_zero = np.zeros(size, dtype=bool)
        self._red_zero = np.zeros(size, dtype=bool)
        self._atoms, self._reds = {}, {}
        self.feasible = self._solve(np.zeros(size)) is not None

    def _coinformation(self, first, second):
        """
        The co-information of two groups of sources and the target, in bits.
        """
        f = self._informations
        return f[first] + f[second] - f[first | second]

    def _solve(self, objective):
        """
        A decomposition minimizing ``objective @ pi``, or None if the constraints are infeasible.
        """
        result = linprog(
            objective, A_eq=self._a_eq, b_eq=self._b_eq, bounds=(0, None), method="highs", options=_LP_OPTIONS
        )
        if result.status == 2:
            return None
        if result.status != 0:  # pragma: no cover
            msg = f"The linear program did not solve: {result.message}"
            raise OptimizationException(msg)
        self._certify(result.x)
        return result.x

    def _certify(self, point):
        """
        Record which atoms and redundancies vanish in the decomposition `point`: their lower
        bounds are zero.
        """
        self._atom_zero |= point <= _ZERO
        positive = np.zeros(len(point), dtype=bool)
        for j in np.flatnonzero(point > _ZERO):
            # the nodes above an atom count it in their redundancy
            positive |= (self._lattice.up_sets & ~self._lattice.up_sets[j]) == 0
        self._red_zero |= ~positive

    def _extremes(self, objective, known_zero):
        """
        The least and largest values of ``objective @ pi``, in bits.
        """
        lower = 0.0 if known_zero else max(0.0, float(objective @ self._solve(objective)))
        upper = float(objective @ self._solve(-objective))
        return lower, max(lower, upper)

    def _check_feasible(self):
        if not self.feasible:
            raise ditException("No nonnegative decomposition satisfies the constraints.")

    @property
    def nodes(self):
        """
        The nodes of the lattice, top first.
        """
        return list(self._lattice.labels)

    def get_pi(self, node):
        """
        The range of the atom of `node`.

        Parameters
        ----------
        node : iterable of iterables
            The node.

        Returns
        -------
        interval : Interval
            The least and largest atom, in the units of the distribution.
        """
        self._check_feasible()
        j = self._lattice.find(node)
        if j not in self._atoms:
            objective = np.zeros(len(self._lattice.nodes))
            objective[j] = 1.0
            lower, upper = self._extremes(objective, self._atom_zero[j])
            self._atoms[j] = Interval(lower * self._scale, upper * self._scale)
        return self._atoms[j]

    __getitem__ = get_pi

    def get_red(self, node):
        """
        The range of the redundancy of `node`.

        Parameters
        ----------
        node : iterable of iterables
            The node.

        Returns
        -------
        interval : Interval
            The least and largest redundancy, in the units of the distribution.
        """
        self._check_feasible()
        j = self._lattice.find(node)
        if j not in self._reds:
            groups = self._lattice.nodes[j]
            if len(groups) == 1 and not self._constrained:
                value = float(self._informations[groups[0]])
                lower, upper = value, value
            elif not self._constrained:
                # the redundancy of MMI is the largest (Theorem 1(i))
                lower = 0.0 if self._red_zero[j] else max(0.0, float(self._solve_red(j)))
                upper = float(min(self._informations[group] for group in groups))
            else:
                lower, upper = self._extremes(self._lattice.below(j), self._red_zero[j])
            self._reds[j] = Interval(lower * self._scale, max(lower, upper) * self._scale)
        return self._reds[j]

    def _solve_red(self, j):
        below = self._lattice.below(j)
        return below @ self._solve(below)

    def _warn_full_table(self):
        if self._lattice.n == _MAX_SOURCES:
            warnings.warn(
                "A full table for five sources solves about 15,000 linear programs, which takes minutes.",
                stacklevel=3,
            )

    @property
    def identified(self):
        """
        The nodes whose atom is the same in every nonnegative decomposition, within `atol`.
        """
        self._warn_full_table()
        return frozenset(label for label in self._lattice.labels if self.get_pi(label).width <= self._atol)

    def positions(self, pid):
        """
        Where the atoms of a decomposition lie within their intervals.

        Parameters
        ----------
        pid : BasePID
            A decomposition of the same distribution, sources and target.

        Returns
        -------
        positions : dict
            Maps each node to 0 at the lower end of its interval and 1 at the upper end, or
            nan if the interval is no wider than `atol`. Values outside [0, 1] mean that the
            decomposition is not nonnegative.
        """
        positions = {}
        for label in self._lattice.labels:
            interval = self.get_pi(label)
            value = float(pid.get_pi(label))
            positions[label] = (
                (value - interval.lower) / interval.width if interval.width > self._atol else float("nan")
            )
        return positions

    def to_string(self, digits=4):
        """
        A table of the intervals.

        Parameters
        ----------
        digits : int
            The number of decimal places.

        Returns
        -------
        table : str
        """
        self._warn_full_table()
        if not self.feasible:
            return "No nonnegative decomposition satisfies the constraints."
        columns = ["node", "I_r min", "I_r max", "pi min", "pi max"]
        table = build_table(columns, title="nonnegative PIDs")
        for column in columns[1:]:
            table.float_format[column] = f"{digits + 2}.{digits}"
        for label in self._lattice.labels:
            table.add_row([_label(label), *self.get_red(label), *self.get_pi(label)])
        return table.get_string()

    def __str__(self):
        return self.to_string()

    def _forced_overlap(self, pairs):
        """
        The least total overlap on `pairs`; see :func:`forced_overlap`.
        """
        self._check_feasible()
        lattice = self._lattice
        masks = lattice.pairs(pairs)
        nodes = [lattice.index[_group_node(lattice.sources, pair)] for pair in masks]
        floors = [max(0.0, self._coinformation(*pair)) for pair in masks]
        point = self._solve(sum(lattice.below(j) for j in nodes))
        overlaps = {
            lattice.labels[j]: max(0.0, float(lattice.below(j) @ point) - floor)
            for j, floor in zip(nodes, floors, strict=True)
        }
        return ForcedOverlap(
            total=sum(overlaps.values()) * self._scale,
            overlaps={label: value * self._scale for label, value in overlaps.items()},
            witness={lattice.labels[j]: float(point[j]) * self._scale for j in np.flatnonzero(point > _ZERO)},
        )


def forced_overlap(dist, sources=None, target=None, pairs="hifi"):
    r"""
    The least total overlap of redundancy and synergy that a nonnegative decomposition can
    have on pairs of groups of sources.

    Merging the sources of two disjoint groups :math:`A` and :math:`B` gives a two-source
    decomposition, with redundancy :math:`R_{A,B} = I_\cap(\{A\}\{B\})` and overlap
    :math:`\delta_{A,B} = R_{A,B} - [I(X_A : X_B : Y)]_+`. Hi-Fi takes every overlap to be
    zero, but with three or more sources that is often impossible: the pairs share atoms
    :cite:`vansanten2026between`. The least total overlap is a linear program; it is zero
    exactly when one nonnegative decomposition has zero overlap on every pair. Its
    minimizers, minimal-overlap decompositions, generally are not unique, so the witness is
    one of them.

    Parameters
    ----------
    dist : Distribution
        The distribution.
    sources : iterable of iterables, None
        The sources, at most five. If None, every variable but the target.
    target : iterable, None
        The target. If None, the last variable.
    pairs : str, iterable
        ``'hifi'`` for the pairs that Hi-Fi examines, which have a single source on one
        side (each pair of single sources once), ``'all'`` for every pair of disjoint groups,
        or an iterable of the lattice nodes of the pairs.

    Returns
    -------
    result : ForcedOverlap
        In the units of the distribution.

    Examples
    --------
    For a copy and a partner, every nonnegative decomposition has a bit of overlap between
    the first source and the other two:

    >>> from dit.pid import forced_overlap
    >>> from dit.pid.distributions import trivariates
    >>> result = forced_overlap(trivariates['copy partner'])
    >>> round(result.total, 6), {pair: round(value, 6) for pair, value in result.overlaps.items() if value > 1e-9}
    (1.0, {((0,), (1, 2)): 1.0})
    """
    return PIDBounds(dist, sources, target)._forced_overlap(pairs)


def zero_overlap_conditions(dist, sources=None, target=None, atol=1e-9):
    r"""
    The conditions of :cite:`vansanten2026between`, Proposition 8, that rule out zero overlap
    on every pair Hi-Fi examines.

    For disjoint groups :math:`A`, :math:`B` and :math:`B'` of sources, with
    :math:`C_{A,B} = I(X_A : X_B : Y)`, every nonnegative decomposition has

    .. math::

        \text{(M)} \quad \delta_{A,B \cup B'} &\geq [C_{A,B}]_+ - [C_{A,B \cup B'}]_+ , \\
        \text{(S)} \quad \delta_{A,B} + \delta_{A,B'} + \delta_{B,B'} &\geq
            [C_{A,B \cup B'}]_+ - [C_{A,B}]_+ - [C_{A,B'}]_+ - [-C_{B,B'}]_+ .

    (M) holds because redundancy with a group cannot shrink as the group grows, and (S)
    because what :math:`X_A` shares with :math:`(X_B, X_{B'})` but with neither alone is
    synergy of :math:`X_B` and :math:`X_{B'}`. The conditions are checked for every single
    source :math:`A` and disjoint :math:`B`, :math:`B'` of the others. For three sources no
    violation means that a decomposition with zero overlap on all six pairs exists (Theorem
    2); for more sources the conditions are necessary but not sufficient.

    Parameters
    ----------
    dist : Distribution
        The distribution.
    sources : iterable of iterables, None
        The sources. If None, every variable but the target.
    target : iterable, None
        The target. If None, the last variable.
    atol : float
        Conditions forcing at most this much overlap are not reported.

    Returns
    -------
    violations : tuple of OverlapCondition
        The violated conditions, with the overlap each forces, in the units of `dist`.

    Examples
    --------
    >>> from dit.pid import zero_overlap_conditions
    >>> from dit.pid.distributions import trivariates
    >>> [(c.condition, c.pairs, round(c.amount, 6)) for c in zero_overlap_conditions(trivariates['copy partner'])]
    [('M', (((0,), (1, 2)),), 1.0)]
    """
    sources, target = _sources_and_target(dist, sources, target)
    linear, scale = _linear_and_scale(dist)
    return _conditions(_informations(linear, sources, target), sources, scale, atol)


def _conditions(informations, sources, scale, atol):
    """
    The violated conditions (M) and (S) for a set function in place of the mutual
    informations, in bits; see :func:`zero_overlap_conditions`.
    """
    n = len(sources)
    f = informations
    scale = float(scale)

    def pos(value):
        return max(0.0, float(value))

    def c(first, second):
        return f[first] + f[second] - f[first | second]

    def pair(first, second):
        return _group_node(sources, (first, second))

    violations = []
    for i in range(n):
        a = 1 << i
        rest = ((1 << n) - 1) & ~a
        for b in range(1, 1 << n):
            if b & ~rest:
                continue
            for b_prime in range(1, 1 << n):
                if b_prime & ~rest or b_prime & b:
                    continue
                amount = (pos(c(a, b)) - pos(c(a, b | b_prime))) * scale
                if amount > atol:
                    violations.append(OverlapCondition("M", (pair(a, b | b_prime),), amount))
                if b < b_prime:
                    amount = (pos(c(a, b | b_prime)) - pos(c(a, b)) - pos(c(a, b_prime)) - pos(-c(b, b_prime))) * scale
                    if amount > atol:
                        violations.append(
                            OverlapCondition("S", (pair(a, b), pair(a, b_prime), pair(b, b_prime)), amount)
                        )
    return tuple(violations)
