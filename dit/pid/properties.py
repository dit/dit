"""
Properties of partial information decompositions.

A property can be checked on a computed decomposition (does this PID, on this
distribution, satisfy it?) or claimed for a measure in the literature. The
checks cover every property catalogued by :cite:`liardi2026landscape` that can
be tested on a single distribution, plus the source chain rule of
:cite:`gutknecht2026partial`. The claims transcribe Table 5 of
:cite:`liardi2026landscape` and the source chain rule results of
:cite:`gutknecht2026partial`.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from itertools import chain, combinations, permutations, product

import numpy as np

from ..distribution import Distribution
from ..helpers import as_linear
from ..multivariate import coinformation
from ..params import ditParams
from ..shannon import conditional_entropy
from ..utils import build_table
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
    the properties claimed for its measure in the literature.
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


def _label(node):
    """
    Label an antichain as in `BasePID.to_string`, e.g. '{0}{1:2}'.
    """
    return "".join("{" + ":".join(map(str, element)) + "}" for element in node)


def _canonical(elements):
    """
    The lattice node of an antichain: minimal elements, ordered as in the lattice.
    """
    elements = {tuple(sorted(element)) for element in elements}
    minimal = [e for e in elements if not any(set(o) < set(e) for o in elements)]
    return tuple(sorted(minimal, key=lambda e: (len(e), e)))


def _positions(node, sources):
    """
    The antichain `node` as a set of groups of positions in `sources`.
    """
    return frozenset(frozenset(j for j, s in enumerate(sources) if set(s) <= set(e)) for e in node)


def _node(groups, sources, extra=()):
    """
    The lattice node whose elements join the sources at each group of positions, and `extra`.
    """
    return _canonical(tuple(chain(*(sources[j] for j in group), extra)) for group in groups)


def _support(dist):
    """
    The (outcome, probability) pairs of the support of `dist`.
    """
    return [(outcome, p) for outcome, p in zip(dist.outcomes, dist.pmf, strict=True) if p > 0]


def _from_table(table):
    """
    A Distribution from a mapping of outcome tuples to probabilities.
    """
    return Distribution(list(table), list(table.values()))


def _layout(groups):
    """
    Consecutive variable indices for groups of the given sizes.
    """
    starts = np.cumsum([0] + [len(group) for group in groups])
    return [tuple(range(a, b)) for a, b in zip(starts[:-1], starts[1:], strict=True)]


def _restricted(dist, groups):
    """
    The joint distribution of the variable groups, placed consecutively.
    """
    columns = list(chain(*groups))
    table = defaultdict(float)
    for outcome, p in _support(dist):
        table[tuple(outcome[c] for c in columns)] += p
    return _from_table(table), _layout(groups)


def _conditionals(dist, given, groups):
    """
    The distributions of the variable groups given each value of `given`.

    Returns
    -------
    conditionals : list of (float, Distribution)
        The probability of each value of `given`, and the conditional
        distribution, with the groups placed consecutively.
    layout : list of tuples
        The indices of each group in the conditional distributions.
    """
    columns = list(chain(*groups))
    tables = defaultdict(lambda: defaultdict(float))
    for outcome, p in _support(dist):
        tables[tuple(outcome[c] for c in given)][tuple(outcome[c] for c in columns)] += p
    conditionals = []
    for table in tables.values():
        pz = sum(table.values())
        conditionals.append((pz, _from_table({outcome: p / pz for outcome, p in table.items()})))
    return conditionals, _layout(groups)


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
    not define.
    """
    target = tuple(target)
    if not isinstance(pid, BaseIncompletePID):
        return pid._scaled(float(type(pid)._measure(dist, node, target, **pid._kwargs)))
    if len(node) == 1:
        return _mi(pid, dist, node[0], target)
    if len(node) > 2 and isinstance(pid, (BaseBivariatePID, BaseUniquePID)):
        return np.nan
    return _red(pid, _same_measure(pid, dist, node, target), node)


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
        if inferior and not vanishes:
            where = f"{_label([source])} is Blackwell inferior to {_label([other])}, so pi({_label([source])})"
            violations.append(Violation(where, unique, 0.0, "="))
        elif vanishes and not inferior:
            where = f"{_label([source])} is not Blackwell inferior to {_label([other])}, so pi({_label([source])})"
            violations.append(Violation(where, unique, 0.0, "!="))
    holds = (not violations) if tested else None
    return PropertyCheck("BP", holds, tested, violations)


# ---------------------------------------------------------------------------
# Checks that recompute the decomposition
# ---------------------------------------------------------------------------


def _context_positions(pid, context):
    """
    Positions in the sources of the context, given as sources (or single variable indices).
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


def _check_source_chain_rule(pid, atol, context=None):
    sources, target, n = pid._sources, pid._target, len(pid._sources)
    if context is None:
        contexts = [c for k in range(1, n - 1) for c in combinations(range(n), k)]
    else:
        contexts = [_context_positions(pid, context)]
    tally = _Tally(atol)
    if not contexts:
        return tally.result("SCR", "has content only for three or more sources")
    for c in contexts:
        rest = [j for j in range(n) if j not in c]
        c_rvs = tuple(chain(*(sources[j] for j in c)))
        mi_c = _mi(pid, pid._dist, c_rvs, target)
        conditional = defaultdict(float)
        conditionals, layout = _conditionals(pid._dist, c_rvs, [*(sources[j] for j in rest), target])
        *rest_layout, target_layout = layout
        for pz, dist in conditionals:
            sub = _same_measure(pid, dist, rest_layout, target_layout)
            for node in sub._lattice:
                conditional[_positions(node, rest_layout)] += pz * _red(pid, sub, node)
        for groups, value in conditional.items():
            groups = [[rest[j] for j in group] for group in groups]
            joined = _node(groups, sources, c_rvs)
            where = f"I_cap({_label(joined)}) = I(c : T) + I_cap({_label(_node(groups, sources))} | c), c = {_label([c_rvs])}"
            tally.equal(where, pid.get_red(joined), mi_c + value)
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
    for pz, dist in conditionals:
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
    if len(pid._sources) > 2 and claims.get("LP1", Claim(False)).holds is None:
        title += " (claims concern two sources)"
    return PropertyReport(title, checks, claims)


# ---------------------------------------------------------------------------
# Claims from the literature
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Claim:
    """
    Whether the literature shows that a measure satisfies a property.

    Attributes
    ----------
    holds : bool, None
        Whether the property holds; None if it is not applicable.
    mark : str
        The footnote mark of the source table, if any, or '!' for an entry
        that corrects the source.
    caveat : str
        Qualifications of the claim.
    source : str
        The bibliography key of the source.
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
    return {
        "PID_WB": Claim(False, caveat=fails, source=source),
        "PID_MMI": Claim(False, caveat=implied, source=source),
        "PID_RDR": Claim(False, caveat=implied, source=source),
        "PID_SX": Claim(True, caveat="pointwise, Prop. 5.1", source=source),
        "PID_PM": Claim(True, caveat="pointwise, Prop. 5.1", source=source),
        "PID_GK": Claim(True, caveat="Prop. 5.2", source=source),
    }


def _local_positivity_claim(lp0, lp1):
    """
    LP is LP0 and LP1 together.
    """
    if lp0.holds is False:
        return lp0
    if lp1.holds is False:
        return lp1
    if lp1.holds is None:
        return Claim(lp0.holds, lp0.mark, lp1.caveat, lp0.source)
    return Claim(True, source=lp0.source)


def _build_claims():
    claims = defaultdict(dict)
    for key, row in _TABLE5.items():
        for measure, entry in zip(_TABLE5_MEASURES, row.split(), strict=True):
            if measure is not None:
                claims[measure][key] = _table5_claim(entry, key)
    for (measure, key), claim in _TABLE5_CORRECTIONS.items():
        claims[measure][key] = claim
    for table in claims.values():
        table["LP"] = _local_positivity_claim(table["LP0"], table["LP1"])
    for measure, claim in _source_chain_rule_claims().items():
        claims[measure]["SCR"] = claim
    return {measure: {key: table[key] for key in PROPERTIES if key in table} for measure, table in claims.items()}


_CLAIMS = _build_claims()


def claims_for(measure):
    """
    The properties claimed for a measure in the literature.

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
