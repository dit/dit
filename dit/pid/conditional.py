"""
Decompositions of the information that some sources provide about a target beyond what is
already known: the conditional and the contextual decompositions of
:cite:`gutknecht2026partial`.

Given a context, either variables outside the decomposition or some of its sources, there
are two ways to decompose the information that the remaining sources provide about the
target beyond the context:

* the **conditional** decomposition (:class:`ConditionalPID`) decomposes the remaining
  sources under each conditional distribution given the context, and averages;
* the **contextual** decomposition (:class:`ContextualPID`) groups the atoms of the
  decomposition of all the sources that the context does not provide by the remaining
  sources that give access to them given the context.

Both decompose the same conditional mutual informations, and they agree atom by atom
exactly when the redundancy measure satisfies the source chain rule (Theorem 3.9 of
:cite:`gutknecht2026partial`), checked by ``pid.check('SCR')``.
"""

from itertools import chain

import numpy as np
from lattices.lattice import Lattice

from ..distribution import Distribution
from ..exceptions import ditException
from ..helpers import as_linear
from ..multivariate import coinformation
from ..params import ditParams
from ..utils import build_table
from ._utils import _canonical, _conditionals, _indices, _label, _node, _positions
from .ped import BasePED
from .pid import BasePID, _descendants, _lattice_of, _unit_scale, _validate_sources, sort_key

__all__ = (
    "ConditionalPID",
    "ContextualPID",
)


def _context_positions(pid, context):
    """
    Positions in the sources of `pid` of the context, given as sources (or single variable
    indices).
    """
    keys = [tuple(sorted(c)) if isinstance(c, (list, tuple, set, frozenset)) else (c,) for c in context]
    positions = set()
    for key in keys:
        matches = [j for j, source in enumerate(pid._sources) if tuple(sorted(source)) == key]
        if not matches:
            msg = f"{key} is not one of the sources {pid._sources}"
            raise ValueError(msg)
        positions.add(matches[0])
    return tuple(sorted(positions))


class _DerivedPID:
    """
    What the conditional and contextual decompositions share: they are indexed by the
    lattice of their sources, and their atoms follow from those of other decompositions.
    """

    _red_string = "I_r"
    _pi_string = "pi"

    _dist: Distribution
    _lattice: Lattice
    _name: str
    _sources: tuple

    def get_red(self, node):  # pragma: no cover
        raise NotImplementedError

    def get_pi(self, node):  # pragma: no cover
        raise NotImplementedError

    def _check_node(self, node, what):
        if node not in self._lattice:
            msg = f"Cannot get the {what} of {node}: it is not in the lattice of {self._sources}."
            raise ditException(msg)

    def __getitem__(self, node):
        """
        The partial information of `node`.
        """
        return self.get_pi(node)

    @property
    def nonnegative(self):
        """
        Whether every (known) atom is nonnegative.
        """
        return all(np.round(pi, 4) >= 0 for pi in map(self.get_pi, self._lattice) if not np.isnan(pi))

    @property
    def complete(self):
        """
        Whether every atom is known.
        """
        return not any(np.isnan(self.get_pi(node)) for node in self._lattice)

    def __eq__(self, other):
        """
        Whether `other` has the same lattice, redundancies and atoms.
        """
        if not isinstance(other, _DerivedPID) or set(self._lattice) != set(other._lattice):
            return NotImplemented
        return all(
            np.isclose(self.get_red(node), other.get_red(node), atol=1e-5, rtol=1e-5, equal_nan=True)
            and np.isclose(self.get_pi(node), other.get_pi(node), atol=1e-5, rtol=1e-5, equal_nan=True)
            for node in self._lattice
        )

    __hash__ = None

    def to_string(self, digits=4):
        """
        A table of the redundancy and partial information of every node.

        Parameters
        ----------
        digits : int
            The number of digits of precision to display.

        Returns
        -------
        table : str
            The table of values.
        """
        dist_name = getattr(self._dist, "name", "")
        title = f"{self._name} | {dist_name}" if dist_name else self._name
        table = build_table([self._name, self._red_string, self._pi_string], title=title)
        table.float_format[self._red_string] = f"{digits + 2}.{digits}"
        table.float_format[self._pi_string] = f"{digits + 2}.{digits}"
        for node in sorted(self._lattice, key=sort_key(self._lattice)):
            red, pi = self.get_red(node), self.get_pi(node)
            tolerance = 10 ** -(digits - 1)
            red = 0.0 if np.isclose(0, red, atol=tolerance, rtol=tolerance) else red
            pi = 0.0 if np.isclose(0, pi, atol=tolerance, rtol=tolerance) else pi
            table.add_row([_label(node), red, pi])
        return table.get_string()

    def __str__(self):
        return self.to_string()

    def __repr__(self):  # pragma: no cover
        if ditParams["repr.print"]:
            return self.to_string()
        return super().__repr__()


class ConditionalPID(_DerivedPID):
    """
    The conditional partial information decomposition (Definition 3.4 of
    :cite:`gutknecht2026partial`): the decomposition of the information the sources provide
    about the target given the variables `crvs`, by decomposing the sources under each
    conditional distribution given `crvs` and averaging,

    .. math::

        \\Pi(\\alpha; T \\mid Z) = \\sum_z p(z) \\, \\Pi^{p(\\cdot \\mid Z = z)}(\\alpha; T).

    Redundancies average the same way, and are related to the atoms by Möbius inversion;
    the atoms decompose the conditional mutual informations :math:`I(a; T \\mid Z)`.

    Examples
    --------
    Given one of two independent bits, their maximum and minimum each determine the other
    bit for one of its values, and nothing else:

    >>> from dit.pid import ConditionalPID, PID_SX
    >>> from dit.pid.distributions import trivariates
    >>> d = trivariates['max min']  # max(C, T), min(C, T), C, T
    >>> pid = ConditionalPID(PID_SX, d, [[0], [1]], [3], crvs=[2])
    >>> pid.get_red(((0,), (1,))), pid.get_pi(((0,),)), pid.get_pi(((1,),))
    (0.0, 0.5, 0.5)
    """

    def __init__(self, measure, dist, sources=None, target=None, crvs=None, **kwargs):
        """
        Parameters
        ----------
        measure : type
            The partial information decomposition to condition, e.g. ``PID_SX``.
        dist : Distribution
            The distribution.
        sources : iter of iters, None
            The sources. If None, every variable in neither the target nor `crvs`.
        target : iter, None
            The target. If None, the last variable.
        crvs : iter
            The variables to condition on, distinct from the sources and the target.
        kwargs : dict
            Options of `measure`.

        Raises
        ------
        TypeError
            If `measure` is not a partial information decomposition, or does not take
            an option.
        ditException
            If `crvs` is empty or overlaps the sources or the target, or the sources are
            not valid sources.
        NotImplementedError
            If `dist` is symbolic.
        """
        if not (isinstance(measure, type) and issubclass(measure, BasePID)) or issubclass(measure, BasePED):
            msg = f"measure must be a partial information decomposition, such as PID_SX, not {measure!r}."
            raise TypeError(msg)
        if not crvs:
            msg = "A conditional decomposition needs variables to condition on (crvs)."
            raise ditException(msg)
        self._unit_scale = _unit_scale(dist)
        self._dist = dist = as_linear(dist)
        if dist.is_symbolic():
            msg = "Conditional decompositions of symbolic distributions are not supported."
            raise NotImplementedError(msg)

        crvs = tuple(crvs)
        if target is None:
            target = dist.rvs[-1]
        if sources is None:
            sources = [var for var in dist.rvs if var[0] not in target and var[0] not in crvs]
        self._sources = tuple(map(tuple, sources))
        self._target = tuple(target)
        self._crvs = crvs
        _validate_sources(dist, self._sources, self._target)
        given = _indices(dist, crvs)
        for rvs in (*self._sources, self._target):
            if set(_indices(dist, rvs)) & set(given):
                msg = f"The variables conditioned on, {list(crvs)}, overlap {list(rvs)}."
                raise ditException(msg)
        self._measure = measure
        self._kwargs = kwargs
        self._name = f"{measure._name}|{_label([crvs])}"

        groups = [_indices(dist, source) for source in self._sources]
        conditionals, layout = _conditionals(dist, given, [*groups, _indices(dist, self._target)])
        *source_layout, target_layout = layout
        self._pids = [(z, pz, measure(d, source_layout, target_layout, **kwargs)) for z, pz, d in conditionals]

        self._lattice = _lattice_of(self._sources)
        sub_lattice = _lattice_of(tuple(source_layout))
        self._sub_node = {_node(_positions(node, source_layout), self._sources): node for node in sub_lattice}
        self._total = self._scaled(sum(pz * pid._total for _, pz, pid in self._pids))

    def _scaled(self, value):
        """
        Express `value` (in bits) in the units of the distribution.
        """
        return value * self._unit_scale

    def _average(self, node, what):
        sub = self._sub_node[node]
        return float(self._scaled(sum(float(pz) * float(getattr(pid, what)(sub)) for _, pz, pid in self._pids)))

    @property
    def pids(self):
        """
        The decompositions under each conditional distribution.

        Returns
        -------
        pids : list of (tuple, float, BasePID)
            Each outcome `z` of `crvs`, its probability, and the decomposition under the
            distribution given ``crvs = z``, whose sources and target are numbered
            consecutively from 0, in order.
        """
        return list(self._pids)

    def get_red(self, node):
        """
        The conditional redundancy of `node`, the average of its redundancies under the
        conditional distributions.

        Parameters
        ----------
        node : iterable of iterables
            The node, as in the lattice of the sources.

        Returns
        -------
        red : float
            The conditional redundancy of `node`.
        """
        node = _canonical(node)
        self._check_node(node, "conditional redundancy")
        return self._average(node, "get_red")

    def get_pi(self, node):
        """
        The conditional partial information of `node`, the average of its partial
        informations under the conditional distributions.

        Parameters
        ----------
        node : iterable of iterables
            The node, as in the lattice of the sources.

        Returns
        -------
        pi : float
            The conditional partial information of `node`.
        """
        node = _canonical(node)
        self._check_node(node, "conditional partial information")
        return self._average(node, "get_pi")

    @property
    def consistent(self):
        """
        Whether the decomposition under every conditional distribution is consistent.
        """
        return all(pid.consistent for _, _, pid in self._pids)


class ContextualPID(_DerivedPID):
    """
    The contextual decomposition (Definition 3.1 of :cite:`gutknecht2026partial`) of the
    information the sources of a decomposition provide about its target beyond a context,
    some of its sources.

    The atoms that the context provides itself, those with an element made of context
    sources only, are dropped. Erasing the context sources from the elements of every
    other atom's label, and keeping the minimal elements, gives the remaining sources
    that give access to the atom once the context is known: the contextual atom of each
    such label collects the atoms that map to it. The contextual redundancies
    (Proposition 3.6) are

    .. math::

        I_\\cap(a_1, \\ldots, a_m; T \\setminus c) =
        I_\\cap(a_1 \\cup c, \\ldots, a_m \\cup c; T) - I(c; T).

    The contextual decomposition agrees with the conditional one,
    ``ConditionalPID(type(pid), dist, remaining sources, target, crvs=context)``, exactly
    when the measure satisfies the source chain rule.

    Examples
    --------
    For :math:`I_{min}`, which does not satisfy the source chain rule, the two
    decompositions differ (Section 7 of :cite:`gutknecht2026partial`):

    >>> from dit.pid import ContextualPID, PID_WB
    >>> from dit.pid.distributions import trivariates
    >>> pid = ContextualPID(PID_WB(trivariates['max min']), context=[2])
    >>> pid.get_red(((0,), (1,)))
    0.5
    """

    def __init__(self, pid, context):
        """
        Parameters
        ----------
        pid : BasePID
            The decomposition of all the sources.
        context : iter
            The sources making up the context, each given as a source (a list of
            variables) or, for single-variable sources, a variable.

        Raises
        ------
        ValueError
            If the context is not made of sources of `pid`, or leaves none.
        """
        if not isinstance(pid, BasePID) or isinstance(pid, BasePED):
            msg = f"pid must be a partial information decomposition, not {pid!r}."
            raise TypeError(msg)
        positions = _context_positions(pid, context)
        if not positions or len(positions) == len(pid._sources):
            msg = "The context must be some, but not all, of the sources."
            raise ValueError(msg)
        self._pid = pid
        self._dist = pid._dist
        self._target = pid._target
        self._context = positions
        self._sources = tuple(source for j, source in enumerate(pid._sources) if j not in positions)
        self._crvs = tuple(chain(*(pid._sources[j] for j in positions)))
        self._name = f"{pid._name}\\{_label([self._crvs])}"
        self._lattice = _lattice_of(self._sources)
        self._mi = float(pid._scaled(coinformation(pid._dist, [list(self._crvs), list(self._target)])))
        self._total = float(pid._total) - self._mi

        context_set = set(positions)
        self._atoms = {node: [] for node in self._lattice}
        for node in pid._lattice:
            groups = _positions(node, pid._sources)
            if any(group <= context_set for group in groups):
                continue  # the context gives access to this atom by itself
            self._atoms[_node([group - context_set for group in groups], pid._sources)].append(node)

    def get_red(self, node):
        """
        The contextual redundancy of `node`, :math:`I_\\cap(a_1 \\cup c, \\ldots; T) - I(c; T)`.

        Parameters
        ----------
        node : iterable of iterables
            The node, as in the lattice of the remaining sources.

        Returns
        -------
        red : float
            The contextual redundancy of `node`.
        """
        node = _canonical(node)
        self._check_node(node, "contextual redundancy")
        joined = _canonical(tuple(chain(element, self._crvs)) for element in node)
        return float(self._pid.get_red(joined)) - self._mi

    def get_pi(self, node):
        """
        The contextual partial information of `node`, the sum of the atoms of the
        decomposition of all the sources that map to it.

        Parameters
        ----------
        node : iterable of iterables
            The node, as in the lattice of the remaining sources.

        Returns
        -------
        pi : float
            The contextual partial information of `node`.
        """
        node = _canonical(node)
        self._check_node(node, "contextual partial information")
        return float(sum(self._pid.get_pi(atom) for atom in self._atoms[node]))

    def atoms(self, node):
        """
        The atoms of the decomposition of all the sources that make up the contextual
        atom of `node`.

        Parameters
        ----------
        node : iterable of iterables
            The node, as in the lattice of the remaining sources.

        Returns
        -------
        atoms : list of tuples
            The nodes of the decomposition of all the sources.
        """
        node = _canonical(node)
        self._check_node(node, "atoms")
        return list(self._atoms[node])

    @property
    def consistent(self):
        """
        Whether the contextual redundancies and atoms satisfy the consistency equations,
        as they do when the decomposition of all the sources does.
        """
        for node in self._lattice:
            red = self.get_red(node)
            parts = sum(self.get_pi(n) for n in _descendants(self._lattice, node, include=True))
            if not np.isnan(red) and not np.isnan(parts) and not np.isclose(red, parts, atol=1e-5, rtol=1e-5):
                return False
        return True
