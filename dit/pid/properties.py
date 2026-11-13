"""
Properties of partial information decompositions.

A property can be checked on a computed decomposition (does this PID, on this
distribution, satisfy it?) or claimed for a measure. The checks cover every
property catalogued by :cite:`liardi2026landscape` that can be tested on a single
distribution, plus the source chain rule of :cite:`gutknecht2026partial`. The
claims transcribe Table 5 of :cite:`liardi2026landscape`, with corrections, and
the source chain rule results of :cite:`gutknecht2026partial`. They also record
properties that the literature leaves open, determined here: each failure rests
on a counterexample that `check_property` reproduces, and each success on the
argument stated with it.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from itertools import chain, combinations, permutations, product

import numpy as np

from ..distribution import Distribution
from ..exceptions import ditException
from ..helpers import as_linear
from ..multivariate import coinformation
from ..params import ditParams
from ..shannon import conditional_entropy
from ..utils import build_table
from ._utils import _canonical, _conditionals, _from_table, _label, _node, _positions, _restricted, _support
from .conditional import ConditionalPID, ContextualPID, _context_positions
from .pid import BaseBivariatePID, BaseIncompletePID, BaseUniquePID

__all__ = (
    "PROPERTIES",
    "Claim",
    "Property",
    "PropertyCheck",
    "PropertyReport",
    "Violation",
    "check_properties",
    "check_property",
    "claims_for",
)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Violation:
    """
    One (in)equality required by a property that fails on a decomposition.

    Attributes
    ----------
    where : str
        What was compared.
    lhs, rhs : float
        The two sides of the comparison.
    relation : str
        The relation the property requires: '=', '<=', '>=', or '!='.
    """

    where: str
    lhs: float
    rhs: float
    relation: str

    @property
    def gap(self):
        """
        How far the comparison is from holding (0 for '!=').
        """
        if self.relation == "=":
            return abs(self.lhs - self.rhs)
        if self.relation == "<=":
            return self.lhs - self.rhs
        if self.relation == ">=":
            return self.rhs - self.lhs
        return 0.0

    def __str__(self):
        if self.relation == "!=":
            return f"{self.where}: got {self.lhs:.6g}"
        return f"{self.where}: needs {self.lhs:.6g} {self.relation} {self.rhs:.6g}"


@dataclass
class PropertyCheck:
    """
    The result of checking one property on one decomposition.

    A check is truthy only if the property holds. `holds` is None when the
    property does not apply, or could not be evaluated.

    Attributes
    ----------
    key : str
        The property's key in `PROPERTIES`.
    holds : bool, None
        Whether the property holds.
    tested : int
        The number of (in)equalities that were compared.
    violations : list of Violation
        The comparisons that failed.
    note : str
        Context for the result.
    """

    key: str
    holds: bool | None
    tested: int = 0
    violations: list = field(default_factory=list)
    note: str = ""

    def __bool__(self):
        return self.holds is True

    @property
    def name(self):
        """
        The name of the property.
        """
        return PROPERTIES[self.key].name

    @property
    def max_violation(self):
        """
        The largest gap among the violations (0 if there are none).
        """
        return max((v.gap for v in self.violations), default=0.0)

    def __str__(self):
        status = {True: "holds", False: "fails", None: "undetermined"}[self.holds]
        lines = [f"{self.key} ({self.name}): {status}, {self.tested} tested"]
        if self.note:
            lines.append(f"  {self.note}")
        lines.extend(f"  {v}" for v in self.violations[:5])
        if len(self.violations) > 5:
            lines.append(f"  ... and {len(self.violations) - 5} more")
        return "\n".join(lines)

    def __repr__(self):
        return (
            f"<PropertyCheck {self.key}: holds={self.holds}, tested={self.tested}, violations={len(self.violations)}>"
        )


_STATUS = {True: "✓", False: "✗", None: "?"}


class PropertyReport:
    """
    The results of checking several properties on one decomposition, next to
    the properties claimed for its measure.
    """

    def __init__(self, title, checks, claims):
        """
        Parameters
        ----------
        title : str
            The title of the report.
        checks : dict
            Maps property keys to PropertyCheck results.
        claims : dict
            Maps property keys to the Claim made for the measure.
        """
        self.title = title
        self.checks = checks
        self.claims = claims

    def __getitem__(self, prop):
        return self.checks[_resolve(prop)]

    def __iter__(self):
        return iter(self.checks.values())

    def __len__(self):
        return len(self.checks)

    def to_string(self, digits=4):
        """
        A table of claims and check results, one row per property.

        Parameters
        ----------
        digits : int
            The number of significant digits for the largest violation.

        Returns
        -------
        table : str
            The table.
        """
        table = build_table(["property", "claim", "check", "tested", "max gap", "note"], title=self.title)
        for key, check in self.checks.items():
            claim = self.claims.get(key)
            gap = f"{check.max_violation:.{digits}g}" if check.violations else ""
            table.add_row([key, str(claim) if claim else "", _STATUS[check.holds], check.tested, gap, check.note])
        return table.get_string()

    def __str__(self):
        return self.to_string()

    def __repr__(self):  # pragma: no cover
        if ditParams["repr.print"]:
            return self.to_string()
        return super().__repr__()


class _Tally:
    """
    Accumulates the comparisons made while checking one property.
    """

    def __init__(self, atol):
        self.atol = atol
        self.tested = 0
        self.skipped = 0
        self.violations = []

    def _compare(self, where, lhs, rhs, relation, ok):
        lhs, rhs = float(lhs), float(rhs)
        if np.isnan(lhs) or np.isnan(rhs):
            self.skipped += 1
            return
        self.tested += 1
        if not ok(lhs, rhs):
            self.violations.append(Violation(where, lhs, rhs, relation))

    def equal(self, where, lhs, rhs):
        self._compare(where, lhs, rhs, "=", lambda a, b: abs(a - b) <= self.atol)

    def at_most(self, where, lhs, rhs):
        self._compare(where, lhs, rhs, "<=", lambda a, b: a <= b + self.atol)

    def at_least(self, where, lhs, rhs):
        self._compare(where, lhs, rhs, ">=", lambda a, b: a >= b - self.atol)

    def result(self, key, note=""):
        notes = [note] if note else []
        if self.violations:
            holds = False
        elif self.tested:
            holds = True
        elif self.skipped:
            holds = None
            notes.append("only undetermined (nan) values to compare")
        else:
            holds = True
            notes.append("vacuous: nothing to compare")
        return PropertyCheck(key, holds, self.tested, self.violations, "; ".join(notes))


def _combine(key, checks, note=""):
    """
    A check that holds if every one of `checks` does.
    """
    holds = [check.holds for check in checks]
    combined = False if False in holds else (None if None in holds else True)
    notes = [note] + [check.note for check in checks if check.note]
    return PropertyCheck(
        key,
        combined,
        sum(check.tested for check in checks),
        [v for check in checks for v in check.violations],
        "; ".join(n for n in notes if n),
    )


# ---------------------------------------------------------------------------
# Lattice nodes and distributions
# ---------------------------------------------------------------------------


def _with_copy(dist, rvs):
    """
    `dist` with a copy of the variables `rvs` appended, and the indices of the copy.
    """
    table = defaultdict(float)
    for outcome, p in _support(dist):
        table[tuple(outcome) + tuple(outcome[c] for c in rvs)] += p
    offset = dist.outcome_length()
    return _from_table(table), tuple(range(offset, offset + len(rvs)))


def _relabeled(dist):
    """
    `dist` with the alphabet of every variable reversed.
    """
    maps = [dict(zip(alphabet, reversed(alphabet), strict=True)) for alphabet in dist.alphabet]
    table = defaultdict(float)
    for outcome, p in _support(dist):
        table[tuple(m[x] for m, x in zip(maps, outcome, strict=True))] += p
    return _from_table(table)


def _independent_joint(first, second):
    """
    The joint of two independent distributions, the variables of `second` after those of `first`.
    """
    return _from_table(
        {tuple(a) + tuple(b): pa * pb for (a, pa), (b, pb) in product(_support(first), _support(second))}
    )


def _conditionally_independent(dist, sources, target):
    """
    p(t) prod_i p(x_i | t): the same source-target marginals as `dist`, with the
    sources independent given the target.
    """
    marginal, layout = _restricted(dist, [*sources, target])
    *source_layout, target_layout = layout
    p_t = defaultdict(float)
    p_xt = [defaultdict(float) for _ in sources]
    for outcome, p in _support(marginal):
        t = tuple(outcome[c] for c in target_layout)
        p_t[t] += p
        for joint, columns in zip(p_xt, source_layout, strict=True):
            joint[tuple(outcome[c] for c in columns), t] += p
    table = {}
    for t, pt in p_t.items():
        options = [[(x, q / pt) for (x, s), q in joint.items() if s == t] for joint in p_xt]
        for combo in product(*options):
            table[tuple(chain(*(x for x, _ in combo))) + t] = pt * np.prod([q for _, q in combo])
    return _from_table(table), source_layout, target_layout


# ---------------------------------------------------------------------------
# Recomputation with the same measure
# ---------------------------------------------------------------------------


def _same_measure(pid, dist, sources, target):
    """
    The decomposition of `dist` by the measure, and options, of `pid`.
    """
    return type(pid)(dist, [list(s) for s in sources], list(target), **pid._kwargs)


def _red(pid, other, node):
    """
    A redundancy from `other`, a decomposition of a linear distribution, in the units of `pid`.
    """
    return pid._scaled(float(other.get_red(node)))


def _mi(pid, dist, rvs, target):
    """
    I[rvs : target] in the units of `pid`.
    """
    return pid._scaled(float(coinformation(dist, [list(rvs), list(target)])))


def _redundancy(pid, dist, node, target):
    """
    The redundancy of one antichain of `dist`, by the measure of `pid`, in the units of `pid`.

    Measures that compute any antichain directly are asked for just that one;
    the others decompose the antichain's elements as sources, as
    `BaseIncompletePID` does for nodes of smaller decompositions. Two-source
    measures give nan for antichains of three or more elements, which they do
    not define, and so does any measure for an antichain whose elements overlap
    so much that their own lattice is degenerate.
    """
    target = tuple(target)
    if not isinstance(pid, BaseIncompletePID):
        return pid._scaled(float(type(pid)._measure(dist, node, target, **pid._kwargs)))
    if len(node) == 1:
        return _mi(pid, dist, node[0], target)
    if len(node) > 2 and isinstance(pid, (BaseBivariatePID, BaseUniquePID)):
        return np.nan
    try:
        sub = _same_measure(pid, dist, node, target)
    except ditException:
        # the elements overlap so much that their own lattice is degenerate
        return np.nan
    return _red(pid, sub, node)


# ---------------------------------------------------------------------------
# Checks computed from the lattice of the decomposition
# ---------------------------------------------------------------------------


def _check_self_redundancy(pid, atol):
    tally = _Tally(atol)
    for node in pid._lattice:
        if len(node) == 1:
            mi = _mi(pid, pid._dist, node[0], pid._target)
            tally.equal(f"I_cap({_label(node)}) = I({_label(node)} : T)", pid.get_red(node), mi)
    return tally.result("SR")


def _check_global_positivity(pid, atol):
    tally = _Tally(atol)
    for node in pid._lattice:
        tally.at_least(f"I_cap({_label(node)}) >= 0", pid.get_red(node), 0.0)
    return tally.result("GP")


def _check_local_positivity(pid, atol, key="LP"):
    n = len(pid._sources)
    if key == "LP0" and n != 2:
        return PropertyCheck(key, None, note="concerns two sources")
    if key == "LP1" and n < 3:
        return PropertyCheck(key, None, note="concerns three or more sources")
    tally = _Tally(atol)
    for node in pid._lattice:
        tally.at_least(f"pi({_label(node)}) >= 0", pid.get_pi(node), 0.0)
    return tally.result(key)


def _monotone(pid, tally):
    """
    Redundancy does not decrease up the lattice.
    """
    for node in pid._lattice:
        for child in pid._lattice.covers(node):
            where = f"I_cap({_label(child)}) <= I_cap({_label(node)})"
            tally.at_most(where, pid.get_red(child), pid.get_red(node))


def _subset_equal(pid, tally):
    """
    Dropping an element that determines another element leaves redundancy unchanged.
    """
    determined = {}
    for node in pid._lattice:
        for a, b in permutations(node, 2):
            if (a, b) not in determined:
                determined[a, b] = conditional_entropy(pid._dist, list(a), list(b)) < 1e-9
            if determined[a, b]:
                smaller = tuple(element for element in node if element != b)
                where = (
                    f"I_cap({_label(node)}) = I_cap({_label(smaller)}), {_label([a])} being a function of {_label([b])}"
                )
                tally.equal(where, pid.get_red(node), pid.get_red(smaller))


def _target_determined_equal(pid, tally):
    """
    Dropping an element that determines the target leaves redundancy unchanged.
    """
    determines = {}
    for node in pid._lattice:
        if len(node) < 2:
            continue
        for b in node:
            if b not in determines:
                determines[b] = conditional_entropy(pid._dist, list(pid._target), list(b)) < 1e-9
            if determines[b]:
                smaller = tuple(element for element in node if element != b)
                where = f"I_cap({_label(node)}) = I_cap({_label(smaller)}), T being a function of {_label([b])}"
                tally.equal(where, pid.get_red(node), pid.get_red(smaller))


def _check_weak_monotonicity(pid, atol):
    tally = _Tally(atol)
    _monotone(pid, tally)
    _subset_equal(pid, tally)
    return tally.result("M0")


def _check_subset_equality(pid, atol):
    tally = _Tally(atol)
    _subset_equal(pid, tally)
    return tally.result("SE")


def _check_lower_bound(pid, atol):
    from .measures.iwedge import _meet_information

    tally = _Tally(atol)
    for node in pid._lattice:
        meet = pid._scaled(float(_meet_information(pid._dist, node, pid._target)))
        tally.at_least(f"I_cap({_label(node)}) >= I(meet : T)", pid.get_red(node), meet)
    return tally.result("LB", "Q ranges up to the Gacs-Korner meet")


def _check_blackwell(pid, atol):
    from ..channelorder import blackwell_order_joint

    sources, target = pid._sources, pid._target
    if len(sources) != 2:
        return PropertyCheck("BP", None, note="concerns two sources")
    tested, violations = 0, []
    for source, other in ((sources[0], sources[1]), (sources[1], sources[0])):
        unique = float(pid.get_pi(_canonical([source])))
        if np.isnan(unique):
            continue
        tested += 1
        inferior = bool(blackwell_order_joint(pid._dist, list(target), list(source), list(other)))
        vanishes = abs(unique) <= atol
        s, o = _label([source]), _label([other])
        if inferior and not vanishes:
            violations.append(Violation(f"pi({s}) must vanish, as {s} is Blackwell inferior to {o}", unique, 0.0, "="))
        elif vanishes and not inferior:
            where = f"pi({s}) must not vanish, as {s} is not Blackwell inferior to {o}"
            violations.append(Violation(where, unique, 0.0, "!="))
    holds = (not violations) if tested else None
    return PropertyCheck("BP", holds, tested, violations)


# ---------------------------------------------------------------------------
# Checks that recompute the decomposition
# ---------------------------------------------------------------------------


def _check_source_chain_rule(pid, atol, context=None):
    """
    Compare the contextual and the conditional decompositions given each context: by
    Theorem 3.9 of :cite:`gutknecht2026partial` they agree exactly when the source chain
    rule holds, and their redundancies are its two sides, less I(c : T).
    """
    sources, target, n = pid._sources, pid._target, len(pid._sources)
    if context is None:
        contexts = [c for k in range(1, n - 1) for c in combinations(range(n), k)]
    else:
        contexts = [_context_positions(pid, context)]
    tally = _Tally(atol)
    if not contexts:
        return tally.result("SCR", "has content only for three or more sources")
    for c in contexts:
        contextual = ContextualPID(pid, [sources[j] for j in c])
        conditional = ConditionalPID(
            type(pid), pid._dist, contextual._sources, target, crvs=contextual._crvs, **pid._kwargs
        )
        mi_c, c_rvs = contextual._mi, contextual._crvs
        for node in contextual._lattice:
            groups = _positions(node, sources)
            joined = _node(groups, sources, c_rvs)
            where = f"I_cap({_label(joined)}) = I(c : T) + I_cap({_label(node)} | c), c = {_label([c_rvs])}"
            tally.equal(where, mi_c + contextual.get_red(node), mi_c + pid._scaled(conditional.get_red(node)))
    return tally.result("SCR")


def _target_split(target, split):
    """
    The two parts of the target for the chain rule (first variable, and the rest, by default).
    """
    if split is None:
        return target[:1], target[1:]
    first, second = (tuple(part) for part in split)
    if sorted(first + second) != sorted(target) or not first or not second:
        msg = f"split {split} does not partition the target {target}"
        raise ValueError(msg)
    return first, second


def _check_target_chain_rule(pid, atol, split=None):
    sources, target = pid._sources, pid._target
    if len(target) < 2:
        return PropertyCheck("TC", None, note="needs a target of two or more variables")
    first, second = _target_split(target, split)
    by_first = _same_measure(pid, pid._dist, sources, first)
    conditional = defaultdict(float)
    conditionals, layout = _conditionals(pid._dist, first, [*sources, second])
    *source_layout, second_layout = layout
    for _, pz, dist in conditionals:
        sub = _same_measure(pid, dist, source_layout, second_layout)
        for node in sub._lattice:
            conditional[_positions(node, source_layout)] += pz * _red(pid, sub, node)
    tally = _Tally(atol)
    a, b = _label([first]), _label([second])
    for node in pid._lattice:
        label = _label(node)
        rhs = _red(pid, by_first, node) + conditional[_positions(node, sources)]
        tally.equal(f"I_cap({label} : T) = I_cap({label} : {a}) + I_cap({label} : {b} | {a})", pid.get_red(node), rhs)
    return tally.result("TC")


def _check_target_monotonicity(pid, atol):
    sources, target = pid._sources, pid._target
    if len(target) < 2:
        return PropertyCheck("TM", None, note="needs a target of two or more variables")
    tally = _Tally(atol)
    for k, rv in enumerate(target):
        sub = _same_measure(pid, pid._dist, sources, target[:k] + target[k + 1 :])
        for node in pid._lattice:
            where = f"I_cap({_label(node)} : T) >= I_cap({_label(node)} : T without {rv})"
            tally.at_least(where, pid.get_red(node), _red(pid, sub, node))
    return tally.result("TM")


def _check_target_equality(pid, atol):
    # The added source is a copy of the target, so that sources and target stay disjoint.
    dist, copy = _with_copy(pid._dist, pid._target)
    tally = _Tally(atol)
    for node in pid._lattice:
        where = f"I_cap({_label(node)}) = I_cap({_label(node)}{{T}})"
        tally.equal(where, pid.get_red(node), _redundancy(pid, dist, _canonical([*node, copy]), pid._target))
    return tally.result("TE")


def _check_strong_monotonicity(pid, atol):
    tally = _Tally(atol)
    _monotone(pid, tally)
    _subset_equal(pid, tally)
    _target_determined_equal(pid, tally)
    lattice = tally.result("M1")
    return _combine("M1", [lattice, _check_target_equality(pid, atol)], "M0 and TE (Liardi et al., Theo. 6)")


def _check_weak_symmetry(pid, atol, orders=None):
    sources, target, n = pid._sources, pid._target, len(pid._sources)
    if orders is None:
        orders = [tuple(reversed(range(n)))] + ([tuple(range(1, n)) + (0,)] if n > 2 else [])
    elif orders == "all":
        orders = list(permutations(range(n)))
    tally = _Tally(atol)
    for order in orders:
        order = tuple(order)
        if sorted(order) != list(range(n)):
            msg = f"{order} is not an ordering of the {n} sources"
            raise ValueError(msg)
        dist, layout = _restricted(pid._dist, [*(sources[j] for j in order), target])
        *source_layout, target_layout = layout
        sub = _same_measure(pid, dist, source_layout, target_layout)
        for node in pid._lattice:
            moved = [[order.index(k) for k in group] for group in _positions(node, sources)]
            where = f"I_cap({_label(node)}) with the sources in order {order}"
            tally.equal(where, pid.get_red(node), _red(pid, sub, _node(moved, source_layout)))
    return tally.result("S0")


def _check_strong_symmetry(pid, atol):
    sources, target = pid._sources, pid._target
    bottom = _canonical(sources)
    redundancy = pid.get_red(bottom)
    tally = _Tally(atol)
    for i, source in enumerate(sources):
        swapped = _canonical([target, *(s for j, s in enumerate(sources) if j != i)])
        where = f"I_cap({_label(bottom)} : T) = I_cap({_label(swapped)} : {_label([source])})"
        tally.equal(where, redundancy, _redundancy(pid, pid._dist, swapped, source))
    return tally.result("S1")


def _check_equivalence_class_invariance(pid, atol):
    sub = _same_measure(pid, _relabeled(pid._dist), pid._sources, pid._target)
    tally = _Tally(atol)
    for node in pid._lattice:
        tally.equal(f"I_cap({_label(node)}) after reversing every alphabet", pid.get_red(node), _red(pid, sub, node))
    return tally.result("EI")


def _check_assumption_star(pid, atol):
    dist, source_layout, target_layout = _conditionally_independent(pid._dist, pid._sources, pid._target)
    bottom = _canonical(pid._sources)
    tally = _Tally(atol)
    where = f"I_cap({_label(bottom)}) under p(t) prod_i p(x_i | t)"
    tally.equal(where, pid.get_red(bottom), _redundancy(pid, dist, _canonical(source_layout), target_layout))
    return tally.result("*", "same source-target marginals, sources independent given T")


def _check_additivity(pid, atol, other=None):
    sources, target, dist = pid._sources, pid._target, pid._dist
    second = dist if other is None else as_linear(other)
    if second.outcome_length() != dist.outcome_length():
        msg = "the second system must have the same variables as the decomposed distribution"
        raise ValueError(msg)
    offset = dist.outcome_length()
    joint_sources = [tuple(s) + tuple(i + offset for i in s) for s in sources]
    joint_target = tuple(target) + tuple(i + offset for i in target)
    joint = _same_measure(pid, _independent_joint(dist, second), joint_sources, joint_target)
    second_pid = None if other is None else _same_measure(pid, second, sources, target)
    tally = _Tally(atol)
    for node in pid._lattice:
        expected = pid.get_red(node) + (pid.get_red(node) if second_pid is None else _red(pid, second_pid, node))
        joined = _node(_positions(node, sources), joint_sources)
        where = f"I_cap({_label(node)}) of the joint of two independent systems"
        tally.equal(where, _red(pid, joint, joined), expected)
    return tally.result("AD", "an independent copy" if other is None else "")


# ---------------------------------------------------------------------------
# Checks of the measure on copy distributions
# ---------------------------------------------------------------------------


def _copy(joint):
    """
    Sources X0, X1 with the given joint distribution, and the target (X0, X1).
    """
    return Distribution([(a, b, a, b) for a, b in joint], list(joint.values()))


_INDEPENDENT_COPIES = (
    ("independent fair bits", {(a, b): 1 / 4 for a in "01" for b in "01"}),
    ("an independent biased bit and trit", {(a, b): pa / 3 for a, pa in (("0", 0.3), ("1", 0.7)) for b in "012"}),
)
_CORRELATED_COPIES = (("correlated bits", {("0", "0"): 0.4, ("0", "1"): 0.1, ("1", "0"): 0.1, ("1", "1"): 0.4}),)


def _check_identity(pid, atol, key="ID"):
    cases = _INDEPENDENT_COPIES + (_CORRELATED_COPIES if key == "ID" else ())
    tally = _Tally(atol)
    for name, joint in cases:
        dist = _copy(joint)
        expected = _mi(pid, dist, (0,), (1,))
        tally.equal(
            f"{name}: I_cap({{0}}{{1}} : {{2:3}}) = I(0 : 1)", _redundancy(pid, dist, ((0,), (1,)), (2, 3)), expected
        )
    return tally.result(key, "measure-level: on copy distributions")


# ---------------------------------------------------------------------------
# The properties
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Property:
    """
    A property a partial information decomposition may satisfy.

    Attributes
    ----------
    key : str
        The abbreviation of :cite:`liardi2026landscape` (SCR for the source chain rule).
    name : str
        The name of the property.
    description : str
        A one-line statement of the property.
    check : callable, None
        `check(pid, atol, **kwargs)`, returning a PropertyCheck; None if the
        property cannot be checked on a single distribution.
    """

    key: str
    name: str
    description: str
    check: object = None


PROPERTIES = {
    p.key: p
    for p in (
        Property(
            "SR",
            "self-redundancy",
            "The redundancy of a single source is its mutual information with the target.",
            _check_self_redundancy,
        ),
        Property(
            "S0",
            "weak symmetry",
            "Redundancy does not depend on the order of the sources.",
            _check_weak_symmetry,
        ),
        Property(
            "M0",
            "weak monotonicity",
            "Adding a source does not increase redundancy, and adding one that determines another changes nothing.",
            _check_weak_monotonicity,
        ),
        Property(
            "GP",
            "global positivity",
            "Redundancy is nonnegative.",
            _check_global_positivity,
        ),
        Property(
            "LP0",
            "weak local positivity",
            "The atoms of a two-source decomposition are nonnegative.",
            partial(_check_local_positivity, key="LP0"),
        ),
        Property(
            "LP1",
            "strong local positivity",
            "The atoms of a decomposition of three or more sources are nonnegative.",
            partial(_check_local_positivity, key="LP1"),
        ),
        Property(
            "LP",
            "local positivity",
            "All atoms are nonnegative.",
            partial(_check_local_positivity, key="LP"),
        ),
        Property(
            "IID",
            "independent identity",
            "Independent sources share nothing about a target that copies them.",
            partial(_check_identity, key="IID"),
        ),
        Property(
            "ID",
            "identity",
            "Two sources share exactly their mutual information about a target that copies them.",
            partial(_check_identity, key="ID"),
        ),
        Property(
            "TM",
            "target monotonicity",
            "Adding a target variable does not decrease redundancy.",
            _check_target_monotonicity,
        ),
        Property(
            "TC",
            "target chain rule",
            "Redundancy obeys the chain rule over the target variables.",
            _check_target_chain_rule,
        ),
        Property(
            "SE",
            "subset equality",
            "Removing a source that determines another source leaves redundancy unchanged.",
            _check_subset_equality,
        ),
        Property(
            "LB",
            "lower bound",
            "A variable that is a function of every source lower-bounds redundancy.",
            _check_lower_bound,
        ),
        Property(
            "TE",
            "target equality",
            "Adding the target to the sources does not change redundancy.",
            _check_target_equality,
        ),
        Property(
            "M1",
            "strong monotonicity",
            "Weak monotonicity, and removing a source that determines the target changes nothing.",
            _check_strong_monotonicity,
        ),
        Property(
            "S1",
            "strong symmetry",
            "Redundancy is unchanged when the target is exchanged with a source.",
            _check_strong_symmetry,
        ),
        Property(
            "*",
            "assumption (*)",
            "Redundancy depends only on the source-target marginals.",
            _check_assumption_star,
        ),
        Property(
            "BP",
            "Blackwell property",
            "A source's unique information vanishes iff it is Blackwell inferior to the other source.",
            _check_blackwell,
        ),
        Property(
            "AD",
            "additivity",
            "The redundancy of two independent systems is the sum of their redundancies.",
            _check_additivity,
        ),
        Property(
            "CO",
            "continuity",
            "Redundancy is continuous in the distribution.",
        ),
        Property(
            "DI",
            "differentiability",
            "Redundancy is differentiable in the distribution.",
        ),
        Property(
            "EI",
            "equivalence-class invariance",
            "Redundancy is unchanged by relabeling the outcomes of any variable.",
            _check_equivalence_class_invariance,
        ),
        Property(
            "SCR",
            "source chain rule",
            "Collections sharing a context share its information, plus their redundancy given the context.",
            _check_source_chain_rule,
        ),
    )
}

_ALIASES = {
    **{key.lower(): key for key in PROPERTIES},
    **{p.name.replace(" ", "_").replace("-", "_").lower(): key for key, p in PROPERTIES.items()},
    "symmetry": "S0",
    "monotonicity": "M0",
    "star": "*",
    "assumption_star": "*",
    "blackwell": "BP",
}


def _resolve(prop):
    """
    The key of a property given by key (e.g. 'SCR') or name (e.g. 'source_chain_rule').
    """
    if prop in PROPERTIES:
        return prop
    key = _ALIASES.get(str(prop).replace(" ", "_").replace("-", "_").lower())
    if key is None:
        msg = f"unknown property {prop!r}; known properties are {', '.join(PROPERTIES)}"
        raise ValueError(msg)
    return key


def check_property(pid, prop, atol=1e-5, **kwargs):
    """
    Check whether a decomposition satisfies a property.

    Parameters
    ----------
    pid : BasePID
        The decomposition.
    prop : str
        The property, by key (e.g. 'SCR') or name (e.g. 'source_chain_rule').
    atol : float
        The absolute tolerance of each comparison.
    kwargs : dict
        Options of the check: `context` (SCR), `split` (TC), `orders` (S0),
        and `other` (AD).

    Returns
    -------
    check : PropertyCheck
        The result; truthy if the property holds.
    """
    key = _resolve(prop)
    if pid._dist.is_symbolic():
        msg = "property checks need a numeric distribution"
        raise NotImplementedError(msg)
    check = PROPERTIES[key].check
    if check is None:
        return PropertyCheck(key, None, note="cannot be checked on a single distribution")
    return check(pid, atol, **kwargs)


def check_properties(pid, props=None, atol=1e-5):
    """
    Check several properties of a decomposition.

    Parameters
    ----------
    pid : BasePID
        The decomposition.
    props : iterable of str, None
        The properties to check. If None, all of them.
    atol : float
        The absolute tolerance of each comparison.

    Returns
    -------
    report : PropertyReport
        The results, next to the properties claimed for the measure.
    """
    keys = list(PROPERTIES) if props is None else [_resolve(prop) for prop in props]
    checks = {}
    for key in keys:
        try:
            checks[key] = check_property(pid, key, atol=atol)
        except Exception as error:  # a failed recomputation leaves this property undetermined
            checks[key] = PropertyCheck(key, None, note=f"error: {type(error).__name__}: {error}")
    dist_name = getattr(pid._dist, "name", "")
    title = f"Properties of {pid._name}" + (f" | {dist_name}" if dist_name else "")
    claims = claims_for(type(pid))
    two_source = isinstance(pid, (BaseBivariatePID, BaseUniquePID)) or claims.get("LP1", Claim(False)).holds is None
    if len(pid._sources) > 2 and two_source:
        title += " (claims concern two sources)"
    return PropertyReport(title, checks, claims)


# ---------------------------------------------------------------------------
# Claims from the literature
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Claim:
    """
    Whether a measure satisfies a property.

    Attributes
    ----------
    holds : bool, None
        Whether the property holds; None if it is not applicable.
    mark : str
        The footnote mark of the source table, if any, or '!' for an entry
        that corrects the source.
    caveat : str
        Qualifications of the claim, or its evidence.
    source : str
        The bibliography key of the source, or '' for a property determined
        here: a failure is then shown by the counterexample in the caveat, and
        a success by the argument there.
    """

    holds: bool | None
    mark: str = ""
    caveat: str = ""
    source: str = ""

    def __str__(self):
        return {True: "✓", False: "✗", None: "n.a."}[self.holds] + self.mark


# The columns of Table 5 of Liardi et al., in order: I_min, I_MMI, I_RDR,
# I_BROJA, I_delta, I_red, I_IG, I_RR, I_DEP, I_CT, I_RAV, I_wedge, I_alpha,
# I_prec, I_MES, I_do, I_PM, I_SX, I_CCS. Their I_delta is the Gaussian
# deficiency measure of Venkatesh & Schamberg, not dit's PID_Delta (Banerjee
# et al.), so that column is not used.
_TABLE5_MEASURES = (
    "PID_WB",
    "PID_MMI",
    "PID_RDR",
    "PID_BROJA",
    None,
    "PID_Proj",
    "PID_IG",
    "PID_RR",
    "PID_dep",
    "PID_CT",
    "PID_RAV",
    "PID_GK",
    "PID_GH",
    "PID_Prec",
    "PID_MES",
    "PID_Do",
    "PID_PM",
    "PID_SX",
    "PID_CCS",
)

# Y: satisfied, N: violated, -: not applicable. Suffixes are the footnotes.
_TABLE5_FOOTNOTES = {
    "+": ("†", "only for two sources"),
    "s": ("§", "only for full-support distributions"),
    "p": ("‡", "the pointwise version holds on the informative and misinformative lattices separately"),
    "e": ("*", "supported by simulations, not proven"),
}
_TABLE5 = {
    "SR": "Y Y Y Y Y Y Y Y Y Y Y Y Y Y Y Y Y Y Y",
    "S0": "Y Y Y Y Y Y Y Y Y Y Y Y Y Y Y Y Y Y Y",
    "M0": "Y Y Y Y Y Y Y Y Y Y Y+ Y Y Y N N Np Np N",
    "GP": "Y Y Y Y Y Y Y Y Y Y Y Y Y Y Y Y Np Np N",
    "LP0": "Y Y Y Y Y Y Y Y Y N Y N N N N N Np Np N",
    "LP1": "Y Y Y - - - - - - N N N N N - - Np Np N",
    "IID": "N N N Y Y Y N Y Y Y Y Y Y Y Y Y N N Y",
    "ID": "N N N Y Y Y N N Y Y Y N N N Y Y N N N",
    "TM": "N Y N N N N Ye N N Y N Y N N N N N N N",
    "TC": "N N N N N N N N N N N N N N N N Y Y N",
    "SE": "Y Y Y Y Y Y Y Y Y Y Y Y Y Y N N Y Y Y",
    "LB": "Y Y Y Y Y Y Y Y Y Y Y+ Y Y Y N N N N N",
    "TE": "Y Y Y+ Y Y Y Y Y Y Y Y+ N Y Y Y Y N N N",
    "M1": "Y Y Y+ Y Y Y Y Y Y Y Y+ N Y Y N N N N N",
    "S1": "N N N N N N N N N N N N N N N N N N N",
    "*": "Y Y Y Y Y Y N N N N N Y N N Y Y N N N",
    "BP": "N N Y Y Y Y N N N N N N N Y N N Np N N",
    "AD": "N N N Y N N N N N N N Y Y Y Y Y N N N",
    "CO": "Y Y Y Y Y Ys Y Y Y Y N N N N Y Y Y Y N",
    "DI": "N N N N N N Y N N N N N N N N Y N Y N",
    "EI": "Y Y Y Y Y Y Y Y Y Y Y Y Y Y Y Y Y Y Y",
}

# Entries of Table 5 that are corrected, each with its evidence; marked "!".
_TABLE5_CORRECTIONS = {
    ("PID_Prec", "*"): Claim(
        True,
        "!",
        "Table 5 marks a violation, but its Appendix E.1 and the definition (which involves only p(t) and each"
        " p(x_i | t)) give the property",
        "liardi2026landscape",
    ),
    ("PID_GK", "*"): Claim(
        False,
        "!",
        "Table 5 marks the property, but identical sources X0 = X1 share I(X0 : T), while the full-support"
        " coupling p(t) p(x0 | t) p(x1 | t), with the same source-target marginals, shares nothing",
        "liardi2026landscape",
    ),
    ("PID_RR", "LB"): Claim(
        False,
        "!",
        "Table 5 derives the property from M0 and SR (Prop. 1), which needs three collections; on 'rdn xor',"
        " I_rr = 1/2 while the common bit of the sources carries 1 bit about the target",
        "liardi2026landscape",
    ),
    ("PID_RDR", "BP"): Claim(
        False,
        "!",
        "only the 'if' direction holds: unique information also vanishes for sources that are Blackwell-ordered"
        " only event by event (T = t against T != t), as for two correlated bits and a target copying both",
        "liardi2026landscape",
    ),
    ("PID_GH", "M0"): Claim(
        False,
        "!",
        "Table 5 marks the property, but the order Q - X - Y is not transitive, so adding a source that contains"
        " another can lower the redundancy: on 'max min', I_GH({0}{1}) = 0.082, I_GH({0}{1:2}) = 0.049 and"
        " I_GH({0}{1}{1:2}) = 0",
        "liardi2026landscape",
    ),
    ("PID_GH", "M1"): Claim(
        False,
        "!",
        "Table 5 marks the property, but it implies M0 (Theo. 6), which fails",
        "liardi2026landscape",
    ),
    ("PID_IG", "GP"): Claim(
        False,
        "!",
        "Table 5 follows Theorem 4.2 of Niu & Quinn, whose proof is a sketch: on 'w', I_IG = -0.062, and on"
        " 0.99 'w' + 0.01 uniform, which has full support, I_IG = -0.011",
        "liardi2026landscape",
    ),
    ("PID_IG", "LP0"): Claim(False, "!", "the redundancy can be negative (see GP)", "liardi2026landscape"),
    ("PID_IG", "LB"): Claim(
        False,
        "!",
        "a constant is a function of both sources, and the redundancy can be negative (see GP)",
        "liardi2026landscape",
    ),
    ("PID_CCS", "SE"): Claim(
        False,
        "!",
        "Table 5 holds the property satisfied by definition, but the sign condition drops outcomes where a"
        " source and a function of it disagree in sign: for X0 = a, X1 = (a, b) and T = a and b, with a and b"
        " independent fair bits, I_ccs({0}{1}) = 0.458, but I(X0 : T) = 0.311",
        "liardi2026landscape",
    ),
}

# Caveats of Table 5 entries made precise.
_TABLE5_CAVEATS = {
    ("PID_RAV", "TE"): "only for a single source and the target (Prop. 37): on 'uniform', I_RAV({0}{1}) = 0, but"
    " adding the target as a third source gives 1",
    ("PID_RAV", "M1"): "only for two sources, from M0 and TE (Theo. 6), where TE needs a single source: adding the"
    " target to both sources fails (see TE)",
}


def _table5_claim(entry, key):
    holds = {"Y": True, "N": False, "-": None}[entry[0]]
    mark, caveat = _TABLE5_FOOTNOTES.get(entry[1:], ("", ""))
    if holds is None:
        caveat = "defined only for two sources"
    elif key in ("CO", "DI") and not holds:
        mark, caveat = "∥", "fails globally; an almost-everywhere version may hold"
    return Claim(holds, mark, caveat, "liardi2026landscape")


def _source_chain_rule_claims():
    fails = "fails for three or more sources"
    implied = fails + "; implied by its local positivity (Liardi et al., Table 5) and Thm. 4.3"
    source = "gutknecht2026partial"
    max_min = (
        "on 'max min', the counterexample of Gutknecht, I_cap({0:2}{1:2}) = 1/2, but I(X2 : T) + I_cap({0}{1} | X2) = 0"
    )
    return {
        "PID_WB": Claim(False, caveat=fails, source=source),
        "PID_MMI": Claim(False, caveat=implied, source=source),
        "PID_RDR": Claim(False, caveat=implied, source=source),
        "PID_SX": Claim(True, caveat="pointwise, Prop. 5.1", source=source),
        "PID_PM": Claim(True, caveat="pointwise, Prop. 5.1", source=source),
        "PID_GK": Claim(True, caveat="Prop. 5.2", source=source),
        "PID_Prec": Claim(False, caveat=max_min),
        "PID_Deg": Claim(False, caveat=max_min + "; as for I_prec, the same measure"),
        "PID_CCS": Claim(
            False, caveat="on 'anddup', I_cap({0:1}{0:2}) = 0.458, but I(X0 : T) + I_cap({1}{2} | X0) = 0.311"
        ),
        "PID_CI": Claim(False, caveat="on 'dbl xor', I_cap({0:1}{0:2}) = 1, but I(X0 : T) + I_cap({1}{2} | X0) = 0"),
    }


def _hifi_claims():
    """
    The properties of the information-theoretic Hi-Fi decomposition (Proposition 2 of
    :cite:`vansanten2026between`).
    """
    source = "vansanten2026between"
    two = "only for two sources"
    return {
        "SR": Claim(True, source=source),
        "S0": Claim(True, source=source),
        "M0": Claim(True, "†", two, source),
        "GP": Claim(True, source=source),
        "LP0": Claim(True, source=source),
        "LP1": Claim(
            False,
            caveat="zero overlap on every pair of groups of sources is often impossible, e.g. for a copy and a"
            " partner (Example 1)",
            source=source,
        ),
        "IID": Claim(True, source=source),
        "ID": Claim(True, source=source),
        "S1": Claim(True, "†", two + "; redundancy and synergy do not depend on which variable is the target", source),
        "*": Claim(False, source=source),
        "BP": Claim(False, source=source),
    }


def _pairwise_reference_claims():
    """
    The properties of the pairwise-reference decomposition, :class:`PID_PR`, for two
    sources (Proposition 6 of :cite:`vansanten2026between`). It is defined for two
    sources; dit's extension to more sources is experimental, so nothing is claimed
    for it.
    """
    source = "vansanten2026between"
    two = "only for two sources; the extension to more sources is experimental"
    return {
        "SR": Claim(True, "†", two, source),
        "S0": Claim(True, "†", two, source),
        "GP": Claim(True, "†", two + "; it is a nonnegative decomposition", source),
        "LP0": Claim(True, source=source),
        "LP1": Claim(
            None, caveat="defined only for two sources; the extension to more sources is experimental", source=source
        ),
        "IID": Claim(True, caveat="by the identity property", source=source),
        "ID": Claim(True, source=source),
        "S1": Claim(True, "†", two + "; redundancy and synergy do not depend on which variable is the target", source),
        "*": Claim(False, source=source),
    }


# ---------------------------------------------------------------------------
# Claims determined here
# ---------------------------------------------------------------------------

_SINGLE_SOURCE = "by construction: the redundancy of a single source is its mutual information with the target"
_LABELS = "by definition: it depends on the probabilities, not on the labels of the outcomes"
_TWO_SOURCES = "defined only for two sources"
# bits equal with probability 0.8, and a target pairing their disjunction and their parity
_OR_XOR = "for bits equal with probability 0.8 and T = (X0 or X1, X0 xor X1)"
# independent fair bits, and a target pairing their conjunction and their parity
_AND_XOR = "for independent fair bits and T = (X0 and X1, X0 xor X1)"
# independent fair bits, and a target pairing the first with their disjunction
_COPY_OR = "for independent fair bits and T = (X0, X0 or X1), the redundancy is 0, but {} about X0 or X1"
# independent fair bits X0, X1, N, and a target pairing the parity with a noisy copy of X1
_XOR_AND = (
    "for independent fair bits X0, X1, N and T = (X0 xor X1, X1 and N), the redundancy is 0, but the chain rule gives"
)


def _holds(caveat, source=""):
    return Claim(True, caveat=caveat, source=source)


def _fails(caveat, source=""):
    return Claim(False, caveat=caveat, source=source)


def _determined_claims():
    """
    Properties of measures that the literature leaves open, determined for dit's
    implementations. Each failure rests on the counterexample in its caveat, which
    `check_property` reproduces; each success on the argument there. The claims for a
    measure defined for two sources concern two sources.
    """
    single, labels, two = _holds(_SINGLE_SOURCE), _holds(_LABELS), Claim(None, caveat=_TWO_SOURCES)
    fixed = "the pairwise maximum-entropy distribution P2 is P"
    pairwise = {
        "SE": _holds(f"if X1 is a function of X0, {fixed}, so the redundancy is I(X1 : T)"),
        "TE": _holds(f"with a copy of T as a source, {fixed}, so the redundancy is I(X0 : T)"),
        "M1": _holds(f"M0 holds, and if T is a function of X1, {fixed}, so the redundancy is I(X0 : T)"),
        "EI": labels,
    }
    blackwell_sum = "on 'sum', the sources are Blackwell equivalent, yet each has {} bit of unique information"
    symmetric = _holds("the redundancy is the smaller of two terms that the sources exchange")
    swapped = _fails(
        "on 'sum', the sources are Blackwell equivalent, so the redundancy is I(X0 : T) = 1/2, but with T"
        " exchanged for X0 it is at most I(X1 : X0) = 0"
    )
    inconsistent = "the unique informations are inconsistent, and the redundancy follows that of one source"
    skar_m0 = _fails(f"on 'prob. 1', the redundancy 0.171 exceeds I(X1 : T) = 0.020; {inconsistent}")
    skar_star = _fails("on 'simple', 0.311, but 0.082 under p(t) p(x0 | t) p(x1 | t)")
    claims = {
        # C2, the co-information of the pairwise maximum-entropy distribution P2
        "PID_RA": {
            "SR": single,
            "S0": _holds("C2 is symmetric in the sources"),
            "M0": _holds(
                "C2 = I(X0 : T) + I(X1 : T) - I_P2(X0 X1 : T) is at most each I(Xi : T), as I_P2(X0 X1 : T) is at"
                f" least each I_P2(Xi : T) = I(Xi : T); and if X1 is a function of X0, {fixed}, so C2 = I(X1 : T)"
            ),
            "GP": _fails("on 'sum', I_RA = -1/2"),
            "LP0": _fails("on 'sum', the redundancy is -1/2"),
            "LP1": two,
            "IID": _holds("by the identity property"),
            "ID": _holds(f"if T = (X0, X1), {fixed}, so C2 = I(X0 : X1) - I(X0 : X1 | T) = I(X0 : X1)"),
            "TM": _fails(f"{_AND_XOR}, I_RA = -1/2, but 0 about X0 xor X1"),
            "TC": _fails(f"{_XOR_AND} 0.311"),
            "LB": _fails("a constant is a function of both sources, and on 'sum' I_RA = -1/2"),
            "S1": _fails("on 'sum', exchanging T with X0 changes the redundancy from -1/2 to 0"),
            "*": _fails("on 'sum', I_RA = -1/2, but 0.189 under p(t) p(x0 | t) p(x1 | t)"),
            "BP": _fails(blackwell_sum.format(1)),
            "AD": _holds("P2 of two independent systems is the product of theirs, so C2 adds"),
            **pairwise,
        },
        "PID_PR": {
            "M0": _holds(
                f"its atoms are nonnegative, and if X1 is a function of X0, {fixed}, so the redundancy is I(X1 : T)"
            ),
            "TM": _fails(f"{_OR_XOR}, I_PR = 0.078, but 0.249 about X0 or X1"),
            "TC": _fails(f"{_OR_XOR}, I_PR = 0.078, but the chain rule gives 0.249"),
            "LB": _fails("on 'gband', the sources share a bit that carries 1 bit about T, but I_PR = 0.811"),
            "BP": _fails(
                "on 'gband', the sources are Blackwell equivalent, yet each has 1/2 bit of unique information"
            ),
            "AD": _fails(
                "C2 adds over independent systems, but its positive part need not: 'boom' and 'sum' have I_PR 0.208"
                " and 0, the two together 0"
            ),
            **pairwise,
        },
        "PID_HiFi": {
            "TM": _fails(f"{_OR_XOR}, I_HF = 0.078, but 0.249 about X0 or X1"),
            "TC": _fails(f"{_OR_XOR}, I_HF = 0.078, but the chain rule gives 0.249"),
            "SE": _holds("if X1 is a function of X0, the co-information is I(X1 : T)"),
            "LB": _fails("on 'rdn xor', the sources share a bit that carries 1 bit about T, but I_HF = 0"),
            "TE": _holds("the co-information of X0, a copy of T, and T is I(X0 : T)"),
            "M1": _holds("M0 holds, and if T is a function of X1, the co-information is I(X0 : T)"),
            "AD": _fails(
                "co-information adds over independent systems, but its positive part need not: 'redundant' and"
                " 'synergy' have I_HF 1 and 0, the two together 0"
            ),
            "EI": labels,
        },
        "PID_IPID": {
            "SR": single,
            "S0": symmetric,
            "LP1": two,
            "TM": _fails(_COPY_OR.format("0.311")),
            "TC": _fails(f"{_XOR_AND} 0.311"),
            "S1": swapped,
            "EI": labels,
        },
        "PID_Delta": {
            "SR": single,
            "S0": symmetric,
            "LP1": two,
            "TM": _fails(_COPY_OR.format("0.311")),
            "TC": _fails(f"{_XOR_AND} 0.311"),
            "S1": swapped,
            "AD": _fails(
                "for 'prob. 1' and a copy of it with the sources exchanged, the smaller term of each is the other's"
                " larger one, so the pair has 0.058, more than twice the 0.020 of each"
            ),
            "EI": labels,
        },
        "PID_DeltaLambda": {
            "SR": single,
            "S0": symmetric,
            "LP1": two,
            "IID": _fails(
                "for independent fair bits copied into the target, mixing X0 with noise keeps the deficiency below"
                " 1 bit, so the redundancy is 0.402 rather than 0"
            ),
            "ID": _fails("it fails on independent sources (see IID)"),
            "TC": _fails("for independent fair bits copied into the target, I_δλ = 0.402, but the chain rule gives 0"),
            "EI": labels,
        },
        "PID_SKAR_nw": {
            "SR": single,
            "S0": _fails(f"{inconsistent}: on 'wb 2' it is 1/2 or 0, depending on the order of the sources"),
            "M0": skar_m0,
            "LP1": two,
            "M1": _fails("it fails M0"),
            "EI": labels,
        },
        "PID_SKAR_owa": {
            "SR": single,
            "S0": _fails(f"{inconsistent}: on 'prob. 2' it is 0.311 or 0, depending on the order of the sources"),
            "M0": skar_m0,
            "LP1": two,
            "M1": _fails("it fails M0"),
            "*": skar_star,
            "BP": _fails(blackwell_sum.format("1/2")),
            "EI": labels,
        },
        "PID_SKAR_owb": {"SR": single, "LP1": two, "EI": labels},
        "PID_SKAR_tw": {
            "SR": single,
            "LP1": two,
            "*": skar_star,
            "BP": _fails(blackwell_sum.format("1/2")),
            "EI": labels,
        },
        "PID_CI": {
            "SR": _holds("self-redundancy of the union information (Theorem 1)", "gomes2024measure"),
            "S0": _holds("symmetry of the union information (Theorem 1)", "gomes2024measure"),
            "M0": _fails(
                "monotone for two sources, but on 'anddup' I_cap({0}{1}) = 0.082 exceeds I_cap({1}{0:2}) = -0.058"
            ),
            "GP": _fails("nonnegative for two sources, but on 'xor cat' I_cap({0}{1}{2}) = -1"),
            "LP0": _holds(
                "for two sources, each a single variable, the union information lies between each I(Xi : T) and"
                " I(X0 X1 : T), and the redundancy is at least I_q(X0 : X1) >= 0 for the q of the definition"
            ),
            "LP1": _fails("on 'xor cat', the atom of {0}{1}{2} is -1"),
            "IID": _holds("by the identity property"),
            "ID": _holds(
                "if T = (X0, X1), the channels p(x_i | t) are deterministic, so q = p, the union information is"
                " H(X0 X1), and the redundancy is I(X0 : X1)"
            ),
            "TM": _fails(_COPY_OR.format("0.082")),
            "TC": _fails(f"{_AND_XOR}, I_cap = 0.189, but the chain rule gives 0.115"),
            "SE": _holds(
                "a source that is a function of another is left out of their union, so in the inclusion-exclusion"
                " sum the sub-collections with the source it determines cancel those without"
            ),
            "LB": _fails("on 'xor cat', I_cap({0}{1}{2}) = -1, less than the 0 of a constant"),
            "TE": _holds(
                "with a copy of T as a source, every union information involving it is H(T), and these cancel in"
                " the inclusion-exclusion sum"
            ),
            "M1": _fails("it fails M0"),
            "S1": _fails("on 'pnt. unq.', exchanging T with X0 changes the redundancy from 1/4 to 0"),
            "*": _fails("on 'simple', 0.311, but 0.082 under p(t) p(x0 | t) p(x1 | t)"),
            "BP": _fails(blackwell_sum.format("0.311")),
            "EI": labels,
        },
        "PID_dep_a": {
            "SR": single,
            "LP1": two,
            "TM": _fails(_COPY_OR.format("0.082")),
            "TC": _fails(f"{_AND_XOR}, the redundancy is 0.189, but the chain rule gives 0.115"),
            "S1": _fails("on 'pnt. unq.', exchanging T with X0 changes the redundancy from 1/4 to 0"),
            "*": skar_star,
            "BP": _fails(blackwell_sum.format("0.311")),
            "EI": labels,
        },
        "PID_dep_b": {
            "SR": single,
            "LP0": _fails("on 'erase', the synergy is -0.078"),
            "LP1": two,
            "TM": _fails(_COPY_OR.format("0.082")),
            "TC": _fails(f"{_AND_XOR}, the redundancy is 0.189, but the chain rule gives 0.115"),
            "S1": _fails("on 'pnt. unq.', exchanging T with X0 changes the redundancy from 1/4 to 0"),
            "BP": _fails(blackwell_sum.format("0.311")),
            "EI": labels,
        },
    }
    for name, less in (("PID_MC", "less capable than"), ("PID_LN", "noisier than")):
        claims[name] = {
            "SR": Claim(True, source="gomes2023orders"),
            "S0": Claim(True, source="gomes2023orders"),
            "M0": Claim(True, source="gomes2023orders"),
            "GP": _holds(f"a constant channel is {less} every source"),
            "LP0": _fails(
                f"on 'wb 1', the sources indicate different values of T, so a channel {less} both is constant: the"
                " redundancy is 0, and the synergy -0.252"
            ),
            "IID": _holds(
                f"if T copies independent sources, a channel {less} both conveys nothing once either source is"
                " fixed, so it is constant"
            ),
            "ID": _fails(
                "if T copies bits equal with probability 0.8, the redundancy is 0, as for independent sources, not"
                " I(X0 : X1) = 0.278"
            ),
            "SE": _holds(f"a function of a source is {less} it, and the order is transitive"),
            "LB": _holds(f"a function of every source is {less} each of them"),
            "TE": _holds(f"every channel is {less} the target itself"),
            "M1": _holds(f"M0 holds, and every channel is {less} a source that determines the target"),
            "*": _holds("by definition: it depends only on p(t) and the channels p(x_i | t)"),
            "EI": labels,
        }
    claims["PID_LN"]["ID"] = _fails(
        "for T = (X0, X1) it is the Gács-Körner common information (Theorem 3), which is 0 for bits equal with"
        " probability 0.8, not I(X0 : X1) = 0.278",
        "gomes2023orders",
    )
    return claims


def _local_positivity_claim(lp0, lp1):
    """
    LP is LP0 and LP1 together: it fails if either does, and is unknown unless both
    are claimed.
    """
    for part in (lp0, lp1):
        if part is not None and part.holds is False:
            return part
    if lp0 is None or lp1 is None:
        return None
    if lp1.holds is None:
        return Claim(lp0.holds, lp0.mark, lp1.caveat, lp0.source)
    return Claim(True, source=lp0.source)


def _same_measure_claims(claims, caveat):
    """
    The claims for another implementation of a measure.
    """
    return {
        key: Claim(claim.holds, claim.mark, "; ".join(c for c in (claim.caveat, caveat) if c), claim.source)
        for key, claim in claims.items()
    }


def _build_claims():
    claims = defaultdict(dict)
    for key, row in _TABLE5.items():
        for measure, entry in zip(_TABLE5_MEASURES, row.split(), strict=True):
            if measure is not None:
                claims[measure][key] = _table5_claim(entry, key)
    for (measure, key), claim in _TABLE5_CORRECTIONS.items():
        claims[measure][key] = claim
    for (measure, key), caveat in _TABLE5_CAVEATS.items():
        claim = claims[measure][key]
        claims[measure][key] = Claim(claim.holds, claim.mark, caveat, claim.source)
    # PID_Deg is the degradation redundancy of Gomes & Figueiredo, which is Kolchinsky's
    # Blackwell redundancy, the measure of PID_Prec.
    claims["PID_Deg"] = _same_measure_claims(claims["PID_Prec"], "as for I_prec, the same measure")
    claims["PID_HiFi"] = _hifi_claims()
    claims["PID_PR"] = _pairwise_reference_claims()
    # what the literature leaves open
    for measure, table in _determined_claims().items():
        for key, claim in table.items():
            claims[measure].setdefault(key, claim)
    for table in claims.values():
        local = _local_positivity_claim(table.get("LP0"), table.get("LP1"))
        if local is not None:
            table["LP"] = local
    for measure, claim in _source_chain_rule_claims().items():
        claims[measure]["SCR"] = claim
    return {measure: {key: table[key] for key in PROPERTIES if key in table} for measure, table in claims.items()}


_CLAIMS = _build_claims()


def claims_for(measure):
    """
    The properties claimed for a measure: those shown in the literature, and those
    determined here, which have no source and carry their evidence in the caveat.
    The claims for a measure defined for two sources concern two sources.

    Parameters
    ----------
    measure : type, BasePID, str
        A PID class, an instance of one, or the class name.

    Returns
    -------
    claims : dict
        Maps property keys to Claims; empty if the measure is not covered.
    """
    if isinstance(measure, str):
        name = measure
    elif isinstance(measure, type):
        name = measure.__name__
    else:
        name = type(measure).__name__
    return dict(_CLAIMS.get(name, {}))
