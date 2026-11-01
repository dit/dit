"""
Classes implementing the partial information decomposition.
"""

import inspect
from abc import ABCMeta, abstractmethod
from copy import copy
from itertools import chain, product

import networkx as nx
import numpy as np
from lattices.lattices import free_distributive_lattice

from ..exceptions import ditException
from ..helpers import as_linear
from ..multivariate import coinformation
from ..params import ditParams
from ..utils import build_table, flatten, powerset

__all__ = (
    "BaseBivariatePID",
    "BaseIncompletePID",
    "BasePID",
    "BasePointwisePID",
    "BaseUniquePID",
    "sort_key",
)


def _pid_coerce(value):
    """Coerce a measure/PI value to float, preserving symbolic (sympy) values.

    Numeric distributions yield floats (unchanged behaviour); symbolic
    distributions yield sympy expressions that must not be cast to float.
    """
    try:
        from sympy import Basic
    except ImportError:  # pragma: no cover
        return float(value)
    if isinstance(value, Basic):
        return value
    return float(value)


def sort_key(lattice):
    """
    A key for sorting the nodes of a PID lattice.

    Parameters
    ----------
    lattice : nx.DiGraph
        The lattice to sort.

    Returns
    -------
    key : function
        A function on nodes which returns the properties from which the lattice should be ordered.
    """
    pls = nx.shortest_path_length(lattice._lattice, source=lattice.top)

    def key(node):
        depth = pls[node]
        size = len(node)
        return depth, -size, node

    return key


def _transform(lattice):
    """
    Transform a free distributive lattice from being frozensets of frozensets
    of sources (tuples of variables) to being tuples of tuples of variables:
    each collection of sources becomes the sorted tuple of their variables.

    Parameters
    ----------
    lattice : Lattice
        The lattice to transform.

    Returns
    -------
    tupled_lattice : Lattice
        The lattice, but with tuples in place of frozensets.

    Raises
    ------
    ditException
        If two nodes would become the same tuple, as happens when a source is
        the union of other sources.
    """

    def tuplefy(n):
        return tuple(sorted((tuple(sorted(set(chain(*_)))) for _ in n), key=lambda tup: (len(tup), tup)))

    mapping = {}
    inverse = {}
    for node in lattice._lattice:
        image = tuplefy(node)
        if image in inverse:
            first, second = ([sorted(map(list, c)) for c in n] for n in (inverse[image], node))
            msg = (
                f"The collections of sources {first} and {second} both have the variables {image}: "
                "no source may contain another source or be the union of other sources."
            )
            raise ditException(msg)
        mapping[node] = image
        inverse[image] = node

    tuple_lattice = copy(lattice)
    tuple_lattice._lattice = nx.relabel_nodes(lattice._lattice, mapping, copy=True)
    tuple_lattice._relationship = lambda a, b: lattice._relationship(inverse[a], inverse[b])
    tuple_lattice.top = mapping[lattice.top]
    tuple_lattice.bottom = mapping[lattice.bottom]
    tuple_lattice._ts = [mapping[n] for n in lattice._ts]
    tuple_lattice._meet_table = None
    tuple_lattice._join_table = None
    tuple_lattice._mobius_cache = None

    return tuple_lattice


def _validate_sources(dist, sources, target=None):
    """
    Check that the sources (and target) of a decomposition are well formed.

    Parameters
    ----------
    dist : Distribution
        The distribution of the decomposition.
    sources : tuple of tuples
        The sources.
    target : tuple, None
        The target, if there is one.

    Raises
    ------
    ditException
        If there are no sources, a source or the target is empty, two sources
        are the same, or a source shares variables with the target.
    """
    if not sources:
        raise ditException("A decomposition needs at least one source.")
    names = [frozenset(dist._resolve_rv_names(list(source))) for source in sources]
    if not all(names):
        raise ditException(f"Sources must be nonempty: {sources}.")
    if len(set(names)) < len(names):
        raise ditException(f"The sources {sources} contain the same source twice.")
    if target is not None:
        target_names = frozenset(dist._resolve_rv_names(list(target)))
        if not target_names:
            raise ditException("The target must be nonempty.")
        for source, name in zip(sources, names, strict=True):
            if name & target_names:
                raise ditException(f"The source {source} shares variables with the target {target}.")


def _check_kwargs(cls, kwargs):
    """
    Check that the keyword arguments of a decomposition are accepted by its measure.

    Parameters
    ----------
    cls : type
        The decomposition.
    kwargs : dict
        The keyword arguments passed on to `cls._measure`.

    Raises
    ------
    TypeError
        If `cls._measure` does not accept `kwargs`.
    """
    try:
        signature = inspect.signature(cls._measure, follow_wrapped=False)
    except (TypeError, ValueError):  # pragma: no cover
        return
    try:
        signature.bind(None, None, None, **kwargs)
    except TypeError as exc:
        raise TypeError(f"{cls.__name__}: {exc}") from None


def _unit_scale(dist):
    """
    Factor converting bits into the units of `dist`'s log base (1 if linear).
    """
    return 1 / np.log2(dist.get_base(numerical=True)) if dist.is_log() else 1


def _expectation(dist, pointwise):
    """
    The expectation of pointwise values over the outcomes of `dist` of positive probability.

    Parameters
    ----------
    dist : Distribution
        The distribution.
    pointwise : dict
        Maps outcomes to values.

    Returns
    -------
    expectation : float
        The probability-weighted sum of the values, ignoring outcomes of probability 0.
    """
    linear = dist.copy(base="linear") if dist.is_log() else dist
    total = 0.0
    for outcome, value in pointwise.items():
        p = linear[outcome]
        if p != 0:
            total += p * value
    return total


class BasePID(metaclass=ABCMeta):
    """
    This implements the basic Williams & Beer Partial Information Decomposition.
    """

    _red_string = "I_r"
    _pi_string = "pi"

    def __init__(self, dist, sources=None, target=None, reds=None, pis=None, compute=True, **kwargs):
        """
        Parameters
        ----------
        dist : Distribution
            The distribution to compute the decomposition on.
        sources : iter of iters, None
            The set of input variables. If None, `dist.rvs` less indices
            in `target` is used.
        target : iter, None
            The target variable. If None, `dist.rvs[-1]` is used.
        reds : dict, None
            Redundancy values pre-assessed.
        pis : dict, None
            Partial information values pre-assessed.
        compute : bool
            If the all the partial information values should be automatically
            computed.
        """
        self._unit_scale = _unit_scale(dist)
        self._dist = dist = as_linear(dist)

        if target is None:
            target = dist.rvs[-1]
        if sources is None:
            sources = [var for var in dist.rvs if var[0] not in target]

        self._sources = tuple(map(tuple, sources))
        self._target = tuple(target)
        _validate_sources(dist, self._sources, self._target)
        _check_kwargs(type(self), kwargs)
        self._kwargs = kwargs

        self._lattice = _transform(free_distributive_lattice(self._sources))
        self._total = self._scaled(coinformation(self._dist, [list(flatten(self._sources)), self._target]))

        self._reds = {} if reds is None else dict(reds)
        self._pis = {} if pis is None else dict(pis)
        self._computed = False

        if compute:
            self._ensure_computed()

    def _scaled(self, value):
        """
        Express `value` (computed in bits) in the units of the distribution.
        """
        if self._unit_scale == 1:
            return value
        if isinstance(value, dict):
            return {k: v * self._unit_scale for k, v in value.items()}
        return value * self._unit_scale

    @staticmethod
    @abstractmethod
    def _measure(dist, sources, target, **kwargs):
        """
        Compute a redundancy value for `sources`.

        Parameters
        ----------
        dist : Distribution
            The (linear) distribution of the decomposition.
        sources : tuple(tuples)
            The lattice node to compute the redundancy of.
        target : tuple
            The indices to consider the target of the PID.
        kwargs : dict
            The options of the measure, as passed to the decomposition.

        Returns
        -------
        red : float
            The redundancy value, in bits.
        """
        pass

    @property
    @classmethod
    @abstractmethod
    def _name(cls):
        """
        The name of the PID.

        Returns
        -------
        name : str
            The name.
        """
        pass

    def __eq__(self, other):
        """
        Test if this and `other` are equal partial information decompositions.

        Parameters
        ----------
        other : BasePID

        Returns
        -------
        eq : bool
            If `self` and `other` are the same partial information decomposition.
        """
        return all(np.isclose(self[node], other[node], atol=1e-5, rtol=1e-5, equal_nan=True) for node in self._lattice)

    def __ne__(self, other):
        """
        Test if this and `other` are not equal.

        Parameters
        ----------
        other : BasePID

        Returns
        -------
        eq : bool
            If `self` and `other` are different partial information decomposition.
        """
        return not (self == other)

    def __getitem__(self, node):
        """
        Get the partial information value associated with `node`.

        Parameters
        ----------
        node : iterable of iterables
            The node to get the partial information of.

        Returns
        -------
        pi : float
            The partial information associated with `node`.
        """
        return self.get_pi(node)

    def __repr__(self):  # pragma: no cover
        """
        Returns a representation of the PID.

        Returns
        -------
        repr : str
            A representation of this object.
        """
        if ditParams["repr.print"]:
            return self.to_string()
        else:
            return super().__repr__()

    def __str__(self):
        """
        Return a string representation of the PID.

        Returns
        -------
        pid : str
            The PID as a string.
        """
        return self.to_string()

    def _ensure_computed(self):
        """
        Compute the lattice of partial information values, once.
        """
        if not self._computed:
            self._computed = True
            try:
                self._compute()
            except BaseException:
                self._computed = False
                raise

    def _compute(self):
        """
        Method for precomputing elements of partial information lattice,
        if this is necessary.
        """
        for node in self._lattice:
            self.get_red(node)

        self._compute_mobius_inversion()

    def _compute_mobius_inversion(self):
        """
        Perform the Mobius inversion.
        """
        for node in reversed(list(self._lattice)):
            self.get_pi(node)

    def _redundancy(self, node):
        """
        Compute the redundancy of `node`, in the units of the distribution.

        Parameters
        ----------
        node : tuple of tuples
            A node of the lattice.

        Returns
        -------
        red : float
            The redundancy of `node`.
        """
        return _pid_coerce(self._scaled(self._measure(self._dist, node, self._target, **self._kwargs)))

    def _atom(self, node):
        """
        Compute the partial information of `node` by Mobius inversion.

        Parameters
        ----------
        node : tuple of tuples
            A node of the lattice.

        Returns
        -------
        pi : float
            The partial information of `node`.
        """
        return _pid_coerce(self.get_red(node) - sum(self.get_pi(n) for n in self._lattice.descendants(node)))

    def _check_node(self, node, what):
        """
        Raise an informative error if `node` is not in the lattice.
        """
        if node not in self._lattice:
            msg = f'Cannot get the {what} of "{node}" because it is not in the lattice of {self._sources}.'
            raise ditException(msg)

    def get_red(self, node):
        """
        Get the redundancy value associated with `node`.

        Parameters
        ----------
        node : iterable of iterables
            The node to get the partial information of.

        Returns
        -------
        pi : float
            The partial information associated with `node`.
        """
        if node not in self._reds:
            self._check_node(node, "redundancy")
            self._reds[node] = self._redundancy(node)

        return self._reds[node]

    def get_pi(self, node):
        """
        Get the partial information value associated with `node`.

        Parameters
        ----------
        node : iterable of iterables
            The node to get the partial information of.

        Returns
        -------
        pi : float
            The partial information associated with `node`.
        """
        if node not in self._pis:
            self._check_node(node, "partial information")
            self._pis[node] = self._atom(node)

        return self._pis[node]

    def to_string(self, digits=4):
        """
        Create a table representing the redundancy and PI lattices.

        Parameters
        ----------
        digits : int
            The number of digits of precision to display.

        Returns
        -------
        table : str
            The table of values.
        """
        red_string = self._red_string
        pi_string = self._pi_string

        dist_name = getattr(self._dist, "name", "")
        title = f"{self._name} | {dist_name}" if dist_name else self._name
        table = build_table([self.name, red_string, pi_string], title=title)

        table.float_format[red_string] = f"{digits + 2}.{digits}"
        table.float_format[pi_string] = f"{digits + 2}.{digits}"

        for node in sorted(self._lattice, key=sort_key(self._lattice)):
            node_label = "".join("{{{}}}".format(":".join(map(str, n))) for n in node)
            red_value = self.get_red(node)
            pi_value = self.get_pi(node)
            if np.isclose(0, red_value, atol=10 ** -(digits - 1), rtol=10 ** -(digits - 1)):  # pragma: no cover
                red_value = 0.0
            if np.isclose(0, pi_value, atol=10 ** -(digits - 1), rtol=10 ** -(digits - 1)):  # pragma: no cover
                pi_value = 0.0
            table.add_row([node_label, red_value, pi_value])

        return table.get_string()

    @property
    def name(self):  # pragma: no cover
        """
        Get the name of the decomposition. If colorama is available, the name will be styled
        according to its properties.

        Returns
        -------
        name : str
            The name of the decomposition.
        """
        try:
            from colorama import Fore, Style

            inconsistent_style = lambda x: Fore.RED + x + Style.RESET_ALL
            negative_style = lambda x: Fore.GREEN + x + Style.RESET_ALL
            incomplete_style = lambda x: Fore.BLUE + x + Style.RESET_ALL
        except:
            inconsistent_style = lambda x: x
            negative_style = lambda x: x
            incomplete_style = lambda x: x

        if not self.consistent:
            return inconsistent_style(self._name)
        elif not self.nonnegative:
            return negative_style(self._name)
        elif not self.complete:
            return incomplete_style(self._name)
        else:
            return self._name

    @property
    def consistent(self):
        """
        Determine if the assignment of values to the lattice is self-consistent.

        Returns
        -------
        valid : bool
            True if the lattice is self-consistent, False otherwise.
        """
        return True

    @property
    def nonnegative(self):
        """
        Determine if the partial information values are all non-negative.

        Returns
        -------
        nonnegative : bool
            True if all pi values are non-negative, False otherwise.
        """
        self._ensure_computed()

        nonnegative = all(np.round(pi, 4) >= 0 for pi in self._pis.values() if not np.isnan(pi))
        return nonnegative

    @property
    def complete(self):
        """
        Determine if all partial information values are assigned.

        Returns
        -------
        valid : bool
            True if the lattice is self-consistant, False otherwise.
        """
        return True

    def check(self, prop, atol=1e-5, **kwargs):
        """
        Check whether this decomposition satisfies a property.

        Parameters
        ----------
        prop : str
            The property, by key (e.g. 'SCR') or name (e.g. 'source_chain_rule').
            See `dit.pid.properties.PROPERTIES`.
        atol : float
            The absolute tolerance of each comparison.
        kwargs : dict
            Options of the check: `context` (SCR), `split` (TC), `orders` (S0),
            and `other` (AD).

        Returns
        -------
        check : PropertyCheck
            The result, with any violations; truthy if the property holds.
        """
        from .properties import check_property

        return check_property(self, prop, atol=atol, **kwargs)

    def check_properties(self, props=None, atol=1e-5):
        """
        Check several properties of this decomposition.

        Parameters
        ----------
        props : iterable of str, None
            The properties to check. If None, all of them.
        atol : float
            The absolute tolerance of each comparison.

        Returns
        -------
        report : PropertyReport
            The results, next to the properties claimed for this measure.
        """
        from .properties import check_properties

        return check_properties(self, props=props, atol=atol)

    @classmethod
    def claims(cls):
        """
        The properties claimed for this measure in the literature.

        Returns
        -------
        claims : dict
            Maps property keys to Claims; empty if the measure is not covered.
        """
        from .properties import claims_for

        return claims_for(cls)


class BasePointwisePID(BasePID):
    """
    Extension of BasePID for measures that support pointwise (per-outcome) computation.

    Subclasses must implement ``_pointwise_measure`` which returns per-outcome
    redundancy values.  Optionally they may implement ``_pointwise_measure_parts``
    to expose the informative (+) and misinformative (-) decomposition.

    When *pointwise* is True the constructor computes, for every lattice node
    and every joint outcome:

    * pointwise redundancies  (``_pw_reds``)
    * pointwise PI atoms via per-outcome Möbius inversion (``_pw_pis``)
    * (optional) plus / minus components of each
    """

    def __init__(self, dist, sources=None, target=None, reds=None, pis=None, compute=True, pointwise=False, **kwargs):
        """
        Parameters
        ----------
        dist : Distribution
            The distribution to compute the decomposition on.
        sources : iter of iters, None
            The set of input variables. If None, ``dist.rvs`` less indices
            in *target* is used.
        target : iter, None
            The target variable. If None, ``dist.rvs[-1]`` is used.
        reds : dict, None
            Redundancy values pre-assessed.
        pis : dict, None
            Partial information values pre-assessed.
        compute : bool
            If the partial information values should be automatically computed.
        pointwise : bool
            If True, also compute pointwise (per-outcome) redundancies and
            PI atoms.
        """
        self._pointwise = pointwise
        self._pw_reds = {}
        self._pw_pis = {}
        self._pw_reds_plus = {}
        self._pw_reds_minus = {}
        self._pw_pis_plus = {}
        self._pw_pis_minus = {}
        super().__init__(dist, sources=sources, target=target, reds=reds, pis=pis, compute=compute, **kwargs)

    # ------------------------------------------------------------------
    # Abstract / optional hooks for subclasses
    # ------------------------------------------------------------------

    @staticmethod
    @abstractmethod
    def _pointwise_measure(dist, sources, target):
        """
        Compute per-outcome redundancy values.

        Parameters
        ----------
        dist : Distribution
        sources : tuple of tuples
        target : tuple

        Returns
        -------
        pw : dict
            ``{outcome: float}`` mapping each joint outcome to its
            pointwise redundancy value.
        """

    @staticmethod
    def _pointwise_measure_parts(dist, sources, target):
        """
        Compute per-outcome informative (+) and misinformative (-) parts.

        Returns ``None`` by default; subclasses may override.

        Returns
        -------
        parts : tuple of (dict, dict) or None
            ``(plus_dict, minus_dict)`` where each maps outcomes to floats,
            or None if the decomposition is not available.
        """
        return None

    # ------------------------------------------------------------------
    # Default _measure derived from _pointwise_measure
    # ------------------------------------------------------------------

    @classmethod
    def _measure(cls, dist, sources, target, **kwargs):
        """
        Averaged redundancy: expectation of the pointwise redundancy over the
        joint distribution.
        """
        pw = cls._pointwise_measure(dist, sources, target, **kwargs)
        return _expectation(dist, pw)

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def _compute(self):
        if self._pointwise:
            self._compute_pointwise()
        super()._compute()
        if self._pointwise:
            any_node = next(iter(self._pw_reds))
            outcomes = list(self._pw_reds[any_node].keys())
            self._compute_pointwise_mobius(outcomes)

    def _compute_pointwise(self):
        """
        Compute the pointwise redundancies of every lattice node, and the
        redundancies they average to.
        """
        for node in self._lattice:
            pw = self._scaled(self._pointwise_measure(self._dist, node, self._target, **self._kwargs))
            self._pw_reds[node] = pw
            if node not in self._reds:
                self._reds[node] = _pid_coerce(_expectation(self._dist, pw))

            parts = self._pointwise_measure_parts(self._dist, node, self._target, **self._kwargs)
            if parts is not None:
                self._pw_reds_plus[node], self._pw_reds_minus[node] = (self._scaled(part) for part in parts)

    def _compute_pointwise_mobius(self, outcomes):
        """Möbius inversion per outcome to obtain pointwise PI atoms."""
        for node in reversed(list(self._lattice)):
            pw_pi = {}
            for o in outcomes:
                desc_sum = sum(self._pw_pis[n][o] for n in self._lattice.descendants(node))
                pw_pi[o] = self._pw_reds[node].get(o, 0.0) - desc_sum
            self._pw_pis[node] = pw_pi

        if self._pw_reds_plus:
            for node in reversed(list(self._lattice)):
                pw_pi_plus = {}
                pw_pi_minus = {}
                for o in outcomes:
                    desc_sum_p = sum(self._pw_pis_plus[n][o] for n in self._lattice.descendants(node))
                    desc_sum_m = sum(self._pw_pis_minus[n][o] for n in self._lattice.descendants(node))
                    pw_pi_plus[o] = self._pw_reds_plus[node].get(o, 0.0) - desc_sum_p
                    pw_pi_minus[o] = self._pw_reds_minus[node].get(o, 0.0) - desc_sum_m
                self._pw_pis_plus[node] = pw_pi_plus
                self._pw_pis_minus[node] = pw_pi_minus

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_pw_red(self, node):
        """
        Pointwise redundancy values for *node*.

        Returns
        -------
        pw_red : dict
            ``{outcome: float}``
        """
        return self._pw_reds[node]

    def get_pw_pi(self, node):
        """
        Pointwise PI atom values for *node*.

        Returns
        -------
        pw_pi : dict
            ``{outcome: float}``
        """
        return self._pw_pis[node]

    def get_pw_red_plus(self, node):
        """Informative (+) component of pointwise redundancy."""
        return self._pw_reds_plus[node]

    def get_pw_red_minus(self, node):
        """Misinformative (-) component of pointwise redundancy."""
        return self._pw_reds_minus[node]

    def get_pw_pi_plus(self, node):
        """Informative (+) component of pointwise PI atom."""
        return self._pw_pis_plus[node]

    def get_pw_pi_minus(self, node):
        """Misinformative (-) component of pointwise PI atom."""
        return self._pw_pis_minus[node]

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def pw_to_string(self, outcome=None, digits=4):
        """
        Create a table of pointwise PID values.

        Parameters
        ----------
        outcome : tuple, optional
            If given, show values for this single outcome.  Otherwise show
            all outcomes.
        digits : int
            Precision for display.

        Returns
        -------
        table : str
        """
        if not self._pw_pis:
            return "(pointwise PID not computed — pass pointwise=True)"

        any_node = next(iter(self._pw_pis))
        all_outcomes = list(self._pw_pis[any_node].keys())
        outcomes = [outcome] if outcome is not None else all_outcomes
        red_string = "i_r"
        pi_string = "pi"
        parts = bool(self._pw_reds_plus)

        columns = [self._name, "outcome", red_string, pi_string]
        if parts:
            columns += ["pi+", "pi-"]
        dist_name = getattr(self._dist, "name", "")
        base_title = f"{self._name} (pointwise)"
        title = f"{base_title} | {dist_name}" if dist_name else base_title
        table = build_table(columns, title=title)
        table.float_format[red_string] = f"{digits + 2}.{digits}"
        table.float_format[pi_string] = f"{digits + 2}.{digits}"
        if parts:
            table.float_format["pi+"] = f"{digits + 2}.{digits}"
            table.float_format["pi-"] = f"{digits + 2}.{digits}"

        for node in sorted(self._lattice, key=sort_key(self._lattice)):
            node_label = "".join("{{{}}}".format(":".join(map(str, n))) for n in node)
            for o in outcomes:
                red_val = self._pw_reds[node].get(o, 0.0)
                pi_val = self._pw_pis[node].get(o, 0.0)
                row = [node_label, str(o), red_val, pi_val]
                if parts:
                    row.append(self._pw_pis_plus[node].get(o, 0.0))
                    row.append(self._pw_pis_minus[node].get(o, 0.0))
                table.add_row(row)

        return table.get_string()


def _is_nan(value):
    """
    Whether `value` is a floating point nan (symbolic values never are).
    """
    return isinstance(value, (float, np.floating)) and np.isnan(value)


class BaseIncompletePID(BasePID):
    """
    A special PID class for measures which do not compute the redundancy of an arbitrary antichain.
    Here we use various techniques to extend computed redundancy values to the rest of the lattice.

    Every atom that the known redundancies and atoms determine through the Mobius inversion is
    computed exactly. With `LATTICE_MONOTONICITY`, values are also filled in by assuming that the
    measure is monotone on the lattice and that its atoms are nonnegative. Whatever remains unknown
    is nan.

    Properties
    ----------
    LATTICE_MONOTONICITY : bool
        Fill in values assuming monotone redundancies and nonnegative atoms.
    REDUCED_PID : bool
        Take the redundancy of a node with fewer elements than there are sources from the
        decomposition of its elements.
    SELF_REDUNDANCY : bool
        The redundancy of a single source is its mutual information with the target.
    """

    LATTICE_MONOTONICITY = True
    REDUCED_PID = True
    SELF_REDUNDANCY = True

    def __eq__(self, other):
        """
        Test if this and `other` are equal partial information decompositions.

        Parameters
        ----------
        other : BasePID

        Returns
        -------
        eq : bool
            If `self` and `other` are the same partial information decomposition.
        """
        equal_pi = super().__eq__(other)
        equal_red = (
            np.isclose(self.get_red(node), other.get_red(node), atol=1e-5, rtol=1e-5, equal_nan=True)
            for node in self._lattice
        )
        return equal_pi and all(equal_red)

    def _redundancy(self, node):
        """
        The redundancy of a single source is its mutual information with the target; that of
        any other node is found by filling in the whole lattice.
        """
        if len(node) == 1 and self.SELF_REDUNDANCY:
            return self._scaled(coinformation(self._dist, [node[0], self._target]))
        self._ensure_computed()
        return self._reds[node]

    def _atom(self, node):
        """
        Atoms are found by filling in the whole lattice.
        """
        self._ensure_computed()
        return self._pis[node]

    def _assume(self, values, kind, node, value):
        """
        Fill in a value which rests on the assumptions of `LATTICE_MONOTONICITY`.
        """
        if node not in values:
            values[node] = value
            self._assumed.add((kind, node))

    def _compute_lattice_monotonicity(self, below, above):
        """
        Infer the redundancy and partial information of lattice elements via lattice monotonicity.

        These values assume that the measure is monotone on the lattice and that its atoms
        are nonnegative.

        Parameters
        ----------
        below : dict
            The nodes at or below each node.
        above : dict
            The nodes at or above each node.
        """
        # everything below a redundancy of 0 is a redundancy of 0
        for node in self._lattice:
            if node in self._reds and np.isclose(0, self._reds[node]):
                for n in below[node]:
                    self._assume(self._reds, "red", n, 0)

        # everything above a redundancy of I(sources, target) is I(sources, target)
        for node in self._lattice:
            if node in self._reds and np.isclose(self._reds[node], self._total):
                for n in above[node]:
                    self._assume(self._reds, "red", n, self._total)

        # if redundancy of A == redundancy of B, then for all A -> C -> B, redundancy of C = redundancy of A, B
        graph = self._lattice._lattice
        tops = [node for node in self._lattice if node in self._reds and any(n not in self._reds for n in graph[node])]
        bottoms = [
            node
            for node in self._lattice
            if node in self._reds and any(n not in self._reds for n in graph.predecessors(node))
        ]
        for top, bottom in product(tops, bottoms):
            if np.isclose(self._reds[top], self._reds[bottom], atol=1e-5, rtol=1e-5):
                for node in below[top] & above[bottom]:
                    self._assume(self._reds, "red", node, self._reds[top])

        # once every redundancy is known the Mobius inversion is exact; the
        # heuristic below assumes nonnegative atoms, so it would override it.
        if all(node in self._reds for node in self._lattice):
            return

        # if redundancy of A is equal to the redundancy any of A's children, then pi(A) = 0
        for node in self._lattice:
            if (
                node not in self._pis
                and node in self._reds
                and any(
                    np.isclose(self._reds[n], self._reds[node], atol=1e-5, rtol=1e-5)
                    for n in self._lattice.covers(node)
                    if n in self._reds
                )
            ):
                self._assume(self._pis, "pi", node, 0)

    def _compute_known_atoms(self, below):
        """
        Mobius inversion where it applies directly: the atom of a node whose redundancy and whose
        descendants' atoms are known, the redundancy of a node whose atoms are all known, and the
        redundancy of the single child of a node whose redundancy and atom are known.

        A value derived from an assumed value is itself assumed.

        Parameters
        ----------
        below : dict
            The nodes at or below each node.
        """
        assumed = self._assumed
        changed = True
        while changed:
            changed = False
            for node in reversed(self._lattice._ts):
                if node in self._pis or node not in self._reds:
                    continue
                descendants = below[node] - {node}
                if all(n in self._pis for n in descendants):
                    self._pis[node] = self._reds[node] - sum(self._pis[n] for n in descendants)
                    if ("red", node) in assumed or any(("pi", n) in assumed for n in descendants):
                        assumed.add(("pi", node))
                    changed = True
            for node in self._lattice:
                if node not in self._reds and all(n in self._pis for n in below[node]):
                    self._reds[node] = sum(self._pis[n] for n in below[node])
                    if any(("pi", n) in assumed for n in below[node]):
                        assumed.add(("red", node))
                    changed = True
                covers = self._lattice._lattice[node]
                if len(covers) == 1 and node in self._reds and node in self._pis:
                    (child,) = covers
                    if child not in self._reds:
                        self._reds[child] = self._reds[node] - self._pis[node]
                        if ("red", node) in assumed or ("pi", node) in assumed:
                            assumed.add(("red", child))
                        changed = True

    def _solve_atoms(self, below, exact):
        """
        Solve the linear equations that the known values impose on the atoms: the redundancy of a
        node is the sum of the atoms at and below it.

        Parameters
        ----------
        below : dict
            The nodes at or below each node.
        exact : bool
            Use only computed values, leaving out those resting on `LATTICE_MONOTONICITY`.

        Returns
        -------
        solution : dict, None
            The atoms that the equations determine, or None if the equations contradict each other.
        """
        nodes = self._lattice._ts
        index = {node: i for i, node in enumerate(nodes)}
        rows, rhs = [], []
        for kind, values in (("red", self._reds), ("pi", self._pis)):
            for node, value in values.items():
                if exact and (kind, node) in self._assumed:
                    continue
                row = np.zeros(len(nodes))
                row[[index[n] for n in below[node]] if kind == "red" else index[node]] = 1.0
                rows.append(row)
                rhs.append(value)
        if not rows:
            return {}
        a, b = np.array(rows), np.array(rhs, dtype=float)
        solution = np.linalg.lstsq(a, b, rcond=None)[0]
        if np.abs(a @ solution - b).max() > 1e-5 * max(1.0, np.abs(b).max()):
            return None
        _, singular, vt = np.linalg.svd(a)
        rank = int(np.sum(singular > 1e-9 * singular[0]))
        null = vt[rank:]
        identified = np.all(np.abs(null) < 1e-8, axis=0) if len(null) else np.ones(len(nodes), dtype=bool)
        return {nodes[i]: float(solution[i]) for i in np.flatnonzero(identified)}

    def _compute_identifiable(self, below):
        """
        Compute every atom that the known redundancies and atoms determine.

        An atom is determined exactly when the null space of the equations the known values
        impose vanishes in its coordinate. Atoms determined by computed values alone are exact,
        whatever their sign. If the computed values contradict each other the decomposition is
        inconsistent. Atoms that are determined only with the help of values assumed under
        `LATTICE_MONOTONICITY` are kept only if they are all nonnegative, as those assumptions
        require.

        Parameters
        ----------
        below : dict
            The nodes at or below each node.
        """
        values = chain(self._reds.values(), self._pis.values())
        if not all(isinstance(value, (int, float, np.integer, np.floating)) for value in values):
            return  # symbolic values: only the exact inversion of `_compute_known_atoms` applies
        if all(node in self._pis for node in self._lattice):
            return
        exact = self._solve_atoms(below, exact=True)
        if exact is None:
            self._inconsistent = True
            return
        for node, value in exact.items():
            self._pis.setdefault(node, value)
        if self._assumed:
            assumed = self._solve_atoms(below, exact=False)
            new = {node: value for node, value in (assumed or {}).items() if node not in self._pis}
            if new and all(value > -1e-6 for value in new.values()):
                for node, value in new.items():
                    self._assume(self._pis, "pi", node, value)
        self._compute_known_atoms(below)

    def _compute(self):
        """
        Precompute the lattice of partial information values.
        Use a variety of methods to fill out as much of the lattice as possible.
        """
        self._inconsistent = False
        self._assumed = set()
        below = {node: self._lattice.descendants(node, include=True) for node in self._lattice}
        above = {node: self._lattice.ascendants(node, include=True) for node in self._lattice}

        # set redundancies of single input sets to I(input, target)
        if self.SELF_REDUNDANCY:  # pragma: no branch
            for node in self._lattice:
                if len(node) == 1:
                    self._reds[node] = self._scaled(coinformation(self._dist, [node[0], self._target]))

        if self.LATTICE_MONOTONICITY:  # pragma: no branch
            self._compute_lattice_monotonicity(below, above)

        # if a node exists in a smaller PID, use that to compute redundancy (if possible)
        if self.REDUCED_PID:  # pragma: no branch
            for node in self._lattice:
                if node not in self._reds and len(node) < len(self._sources):
                    try:
                        red = self.__class__(self._dist, node, self._target, **self._kwargs).get_red(node)
                    except ditException:
                        # the elements overlap so much that their own lattice is degenerate
                        continue
                    if not _is_nan(red):
                        self._reds[node] = self._scaled(red)

        while True:
            num_values = len(self._reds) + len(self._pis)

            self._compute_known_atoms(below)
            self._compute_identifiable(below)

            # if the lattice is monotonic, then everything below a zero is zero, and everything above a max is max
            if self.LATTICE_MONOTONICITY:  # pragma: no branch
                self._compute_lattice_monotonicity(below, above)

            if len(self._reds) + len(self._pis) == num_values:
                break

        # nonnegative atoms summing to I(sources, target) leave nothing for the others
        missing = [node for node in self._lattice if node not in self._pis]
        if self.LATTICE_MONOTONICITY and missing and np.isclose(sum(self._pis.values()), self._total):
            for node in missing:
                self._assume(self._pis, "pi", node, 0)
            self._compute_known_atoms(below)

        # plug in nan for all unknown values
        for node in self._lattice:
            self._reds.setdefault(node, np.nan)
            self._pis.setdefault(node, np.nan)

    @BasePID.consistent.getter
    def consistent(self):
        """
        Determine if the assignment of values to the lattice is self-consistant.

        Returns
        -------
        valid : bool
            True if the lattice is self-consistent, False otherwise.
        """
        self._ensure_computed()

        if self._inconsistent:
            return False

        if self.SELF_REDUNDANCY:  # pragma: no branch
            for node in self._lattice:
                if len(node) == 1:
                    red = self.get_red(node)
                    mi = self._scaled(coinformation(self._dist, [node[0], self._target]))
                    if not np.isclose(red, mi, atol=1e-5, rtol=1e-5):  # pragma: no cover
                        return False

        # ensure that the mobius inversion holds
        for node in self._lattice:
            red = self.get_red(node)
            parts = sum(self.get_pi(n) for n in self._lattice.descendants(node, include=True))
            if not np.isnan(red) and not np.isnan(parts) and not np.isclose(red, parts, atol=1e-5, rtol=1e-5):
                return False

        return True

    @BasePID.complete.getter
    def complete(self):
        """
        Determine if all partial information values are assigned.

        Returns
        -------
        valid : bool
            True if the lattice is self-consistant, False otherwise.
        """
        self._ensure_computed()

        return not any(np.isnan(pi) for pi in self._pis.values())


class BaseUniquePID(BaseIncompletePID):
    """
    PID class for measures which define only unique informations.
    """

    def _compute(self):
        """ """
        # a single source has no other sources to share with, so its unique information is all of it
        if len(self._sources) > 1:
            uniques = self._scaled(self._measure(self._dist, self._sources, self._target, **self._kwargs))
            for node in self._lattice:
                if len(node) == 1 and node[0] in uniques and node not in self._pis and not _is_nan(uniques[node[0]]):
                    self._pis[node] = uniques[node[0]]

        super()._compute()


class BaseBivariatePID(BaseIncompletePID):
    """
    PID class for measures which define only a bivariate measure of redundancy.
    """

    def _redundancy(self, node):
        """
        The measure gives the redundancy of nodes with two elements directly.
        """
        if len(node) == 2:
            return self._scaled(self._measure(self._dist, node, self._target, **self._kwargs))
        return super()._redundancy(node)

    def _compute(self):
        """ """
        for node in self._lattice:
            if len(node) == 2 and node not in self._reds:
                self._reds[node] = self._scaled(self._measure(self._dist, node, self._target, **self._kwargs))

        super()._compute()


class BaseSynergyPID(BaseIncompletePID):
    """
    PID class for measures which define a (weak) synergy, or equivalently a union information.

    Subclasses implement `_measure` to return the synergy of a collection of sources `alpha`
    relative to its own union A = A_1 u ... u A_m,

        S(alpha -> T) = I(A; T) - I_union(alpha -> T).

    Redundancies are then obtained by inclusion-exclusion over sub-collections of each lattice
    node,

        I_cap(alpha) = sum_{beta subset alpha, beta nonempty} (-1)^(|beta| + 1) I_union(beta),

    which holds exactly in the parthood formulation of the PID :cite:`gutknecht2021bits`.
    """

    LATTICE_MONOTONICITY = False
    REDUCED_PID = False

    def __init__(self, dist, sources=None, target=None, reds=None, pis=None, compute=True, **kwargs):
        """ """
        self._unions = {}
        super().__init__(dist, sources, target, reds=reds, pis=pis, compute=compute, **kwargs)

    def get_union(self, node):
        """
        Get the union information associated with the collection of sources `node`.

        Parameters
        ----------
        node : iterable of iterables
            A collection of sources.

        Returns
        -------
        union : float
            The union information I_union(node -> target).
        """
        key = tuple(sorted(tuple(source) for source in node))
        if key not in self._unions:
            whole = coinformation(self._dist, [list(flatten(key)), self._target])
            syn = self._measure(self._dist, key, self._target, **self._kwargs)
            self._unions[key] = _pid_coerce(self._scaled(whole - syn))
        return self._unions[key]

    def get_syn(self, node):
        """
        Get the synergy associated with the collection of sources `node`, relative to all sources.

        Parameters
        ----------
        node : iterable of iterables
            A collection of sources.

        Returns
        -------
        syn : float
            The synergy S(node -> target) = I(sources; target) - I_union(node -> target).
        """
        return self._total - self.get_union(node)

    def _redundancy(self, node):
        """
        The redundancy of a node, by inclusion-exclusion over the union information of its
        sub-collections.
        """
        if len(node) > 1:
            return _pid_coerce(sum((-1) ** (len(sub) + 1) * self.get_union(sub) for sub in powerset(node) if sub))
        return super()._redundancy(node)

    def _compute(self):
        """ """
        for node in self._lattice:
            if len(node) > 1 and node not in self._reds:
                self._reds[node] = self._redundancy(node)

        super()._compute()
