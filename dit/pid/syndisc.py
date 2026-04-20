"""
Synergistic Disclosure Decomposition.

Rosas, Mediano, Rassouli & Barrett (2020), "An operational information
decomposition via synergistic disclosure", J. Phys. A: Math. Theor. 53 485001.
https://doi.org/10.1088/1751-8121/abb723

The decomposition is built on the notion of alpha-synergy:

    S_alpha(X -> Y) = max_{V: I(V; X_{alpha_i})=0 for all i} I(V; Y)

where the maximisation is over all alpha-synergistic channels p_{V|X}
satisfying the Markov chain V - X - Y.

Atoms are obtained via Mobius inversion on the extended constraint lattice L*,
and a "backbone" coarse-graining provides a tractable, non-negative
decomposition that scales linearly with system size.
"""

from copy import deepcopy
from itertools import combinations

import networkx as nx
import numpy as np
from lattices import Lattice
from lattices.lattices import free_distributive_lattice

from ..algorithms import BaseAuxVarOptimizer
from ..math import prod
from ..multivariate import coinformation
from ..utils import build_table, flatten

__all__ = ("ModifiedSynDisc", "SynDisc")


# ─────────────────────────────────────────────────────────────────────────────
# Constraint lattice construction
# ─────────────────────────────────────────────────────────────────────────────


def _constraint_le(alpha, beta):
    """
    The constraint partial order: alpha <=_c beta iff for every alpha_i in
    alpha, there exists beta_j in beta such that alpha_i is a subset of beta_j.

    Parameters
    ----------
    alpha : frozenset of frozensets
    beta : frozenset of frozensets

    Returns
    -------
    le : bool
    """
    if not alpha:
        return True
    if not beta:
        return False
    return all(any(a <= b for b in beta) for a in alpha)


def _build_constraint_lattice(sources):
    """
    Build the extended constraint lattice L* for the given sources.

    The lattice is oriented so that the empty antichain (no constraints,
    S = I(X;Y)) is at the **top** and the most-constrained node ({[n]}) is
    at the **bottom**.  This orientation matches the Mobius inversion
    convention used in the paper (equations 9-11): atoms are computed by
    subtracting descendant contributions, and atoms sum to I(X;Y).

    Internally the lattice is built with <=_c and then inverted.

    Parameters
    ----------
    sources : tuple of tuples
        The source variable groups, e.g. ((0,), (1,)) for two sources.

    Returns
    -------
    lattice : Lattice
        The constraint lattice with frozenset-of-frozenset nodes,
        top = empty antichain, bottom = {[n]}.
    """
    fdl = free_distributive_lattice(sources)
    nodes = set(fdl) | {frozenset()}
    return Lattice(nodes, _constraint_le).inverse()


def _transform_constraint(lattice):
    """
    Transform a constraint lattice from frozensets of frozensets of tuples
    to tuples of tuples of integers, mirroring the PID _transform function.

    The empty antichain frozenset() maps to the empty tuple ().

    Parameters
    ----------
    lattice : Lattice
        A constraint lattice with frozenset nodes.

    Returns
    -------
    tuple_lattice : Lattice
        The lattice with tuple-of-tuples nodes.
    """
    def tuplefy(n):
        if not n:
            return ()
        return tuple(sorted(
            (tuple(sorted(sum(_, ()))) for _ in n),
            key=lambda tup: (len(tup), tup),
        ))

    def freeze(n):
        if not n:
            return frozenset()
        return frozenset(frozenset((__,) for __ in _) for _ in n)

    tuple_lattice = deepcopy(lattice)

    tuple_edges = [(tuplefy(e[0]), tuplefy(e[1])) for e in lattice._lattice.edges]
    tuple_lattice._lattice = nx.DiGraph(tuple_edges)

    tuple_lattice._relationship = lambda a, b: lattice._relationship(freeze(a), freeze(b))
    tuple_lattice.top = tuplefy(lattice.top)
    tuple_lattice.bottom = tuplefy(lattice.bottom)
    tuple_lattice._ts = [tuplefy(n) for n in lattice._ts]

    return tuple_lattice


# ─────────────────────────────────────────────────────────────────────────────
# Optimizer for computing S_alpha
# ─────────────────────────────────────────────────────────────────────────────


class SyndiscOptimizer(BaseAuxVarOptimizer):
    """
    Compute S_alpha(X -> Y) = max_{V: I(V; X_{alpha_i})=0 for all i} I(V; Y).

    The auxiliary variable V is constructed as a channel on the joint sources X,
    subject to zero mutual information constraints for each subgroup specified
    by alpha.

    Parameters
    ----------
    dist : Distribution
        The joint distribution over sources and target.
    sources : list of lists
        Each inner list gives the indices of one source variable.
    target : list
        The indices of the target variable.
    alpha : tuple of tuples
        Each inner tuple specifies a subset of *source indices* (0-based into
        the sources list) for which I(V; X_{alpha_i}) must be zero.
    crvs : list, None
        Variables to condition on.
    bound : int, None
        Cardinality bound on V.
    """

    _PENALTY_WEIGHT = 500

    def __init__(self, dist, sources, target, alpha, crvs=None, bound=None):
        self._n_sources = len(sources)
        self._alpha = alpha

        rvs = list(sources) + [list(target)]
        super().__init__(dist, rvs=rvs, crvs=crvs)

        self._source_indices = set(range(self._n_sources))
        self._target_index = {self._n_sources}

        theoretical_bound = self._compute_bound()
        bound = min(bound, theoretical_bound) if bound else theoretical_bound

        self._construct_auxvars([(self._source_indices | self._crvs, bound)])

        self._mi_target = self._mutual_information(self._target_index, self._arvs)

        self._mi_subgroups = {}
        for subgroup in self._alpha:
            key = frozenset(subgroup)
            indices = set(subgroup)
            self._mi_subgroups[key] = self._mutual_information(self._arvs, indices)

        for key, mi_fn in self._mi_subgroups.items():
            self.constraints.append({
                "type": "eq",
                "fun": lambda x, fn=mi_fn: self._squared_mi(x, fn),
            })

        self._default_hops = 5

        self._additional_options = {
            "options": {
                "maxiter": 1000,
                "ftol": 1e-6,
                "eps": 1.4901161193847656e-9,
            },
        }

    def _compute_bound(self):
        """Upper bound on |V| from the Caratheodory-Fenchel theorem."""
        return prod(self._shape[i] for i in self._source_indices) + 1

    def _squared_mi(self, x, mi_fn):
        """Equality constraint residual: I(V; X_subgroup)^2 == 0."""
        pmf = self.construct_joint(x)
        return mi_fn(pmf) ** 2

    def _objective(self):
        """
        Maximise I(V; Y) with quadratic penalty for constraint violations.
        """
        mi_target = self._mi_target
        mi_subgroups = self._mi_subgroups
        w = self._PENALTY_WEIGHT

        def objective(self, x):
            pmf = self.construct_joint(x)
            neg_mi = -mi_target(pmf)
            penalty = sum(fn(pmf) ** 2 for fn in mi_subgroups.values())
            return neg_mi + w * penalty

        return objective

    def synergistic_disclosure(self, x):
        """
        Compute I(V; Y) from an optimisation vector.

        Parameters
        ----------
        x : np.ndarray
            The optimisation vector (typically ``self._optima``).

        Returns
        -------
        s_alpha : float
        """
        pmf = self.construct_joint(x)
        return self._mi_target(pmf)


# ─────────────────────────────────────────────────────────────────────────────
# Full decomposition class
# ─────────────────────────────────────────────────────────────────────────────


def _node_to_alpha(node, sources):
    """
    Convert a constraint-lattice node (tuple of tuples of source indices)
    to the alpha parameter expected by SyndiscOptimizer.

    Each element of the node is a tuple of raw variable indices; we need to
    map them to source-list indices.

    Parameters
    ----------
    node : tuple of tuples
        A lattice node, e.g. ((0,), (1,)) or ((0, 1),).
    sources : tuple of tuples
        The source variable groups.

    Returns
    -------
    alpha : tuple of tuples
        Source-list indices for each constraint subgroup.
    """
    src_to_idx = {src: i for i, src in enumerate(sources)}

    alpha = []
    for constraint in node:
        subgroup = []
        for src in sources:
            if all(idx in constraint for idx in src):
                subgroup.append(src_to_idx[src])
        alpha.append(tuple(subgroup))
    return tuple(alpha)


class SynDisc:
    """
    Synergistic disclosure decomposition (Rosas et al. 2020).

    Decomposes I(X; Y) into atoms via Mobius inversion on the extended
    constraint lattice. Also provides the backbone decomposition, a
    non-negative coarse-graining that scales linearly with system size.

    Parameters
    ----------
    dist : Distribution
        The joint distribution over sources and target.
    sources : iter of iters, None
        The source variable groups. If None, all variables except the last
        are treated as individual sources.
    target : iter, None
        The target variable. If None, the last variable is used.
    niter : int, None
        Number of basin-hopping restarts per lattice node.
    bound : int, None
        Cardinality bound on the auxiliary variable V.
    """

    _name = "S_disc"

    def __init__(self, dist, sources=None, target=None, niter=None, bound=None):
        self._dist = dist

        if target is None:
            target = dist.rvs[-1]
        if sources is None:
            sources = [var for var in dist.rvs if var[0] not in target]

        self._sources = tuple(map(tuple, sources))
        self._target = tuple(target)
        self._niter = niter
        self._bound = bound

        self._lattice = _transform_constraint(
            _build_constraint_lattice(self._sources)
        )

        self._total = coinformation(
            self._dist, [list(flatten(self._sources)), self._target]
        )

        self._synergies = {}
        self._atoms = {}

        self._compute()

    def _compute(self):
        """Evaluate S_alpha at every lattice node, then Mobius-invert."""
        for node in self._lattice:
            self.get_synergy(node)
        self._compute_mobius_inversion()

    def _compute_mobius_inversion(self):
        """Perform Mobius inversion to obtain atoms."""
        for node in reversed(list(self._lattice)):
            self.get_atom(node)

    def _compute_s_alpha(self, node):
        """
        Compute S_alpha(X -> Y) for a given lattice node.

        The empty node () has no constraints, so S = I(X;Y).
        """
        if not node:
            return self._total

        alpha = _node_to_alpha(node, self._sources)

        if not any(alpha):
            return self._total

        all_sources_constrained = (
            len(alpha) == 1
            and set(alpha[0]) == set(range(len(self._sources)))
        )
        if all_sources_constrained:
            return 0.0

        flat_sources = [list(src) for src in self._sources]
        try:
            opt = SyndiscOptimizer(
                self._dist,
                flat_sources,
                list(self._target),
                alpha,
                bound=self._bound,
            )
            opt.optimize(niter=self._niter)
            val = opt.synergistic_disclosure(opt._optima)
            return max(val, 0.0)
        except Exception:
            return 0.0

    def get_synergy(self, node):
        """
        Get the raw S_alpha value for a lattice node.

        Parameters
        ----------
        node : tuple of tuples
            The constraint lattice node.

        Returns
        -------
        s_alpha : float
        """
        if node not in self._synergies:
            self._synergies[node] = float(self._compute_s_alpha(node))
        return self._synergies[node]

    def get_atom(self, node):
        """
        Get the Mobius-inverted atom S^alpha_partial for a lattice node.

        Parameters
        ----------
        node : tuple of tuples
            The constraint lattice node.

        Returns
        -------
        atom : float
        """
        if node not in self._atoms:
            desc_sum = sum(
                self.get_atom(n) for n in self._lattice.descendants(node)
            )
            self._atoms[node] = float(self.get_synergy(node) - desc_sum)
        return self._atoms[node]

    def __getitem__(self, node):
        """Get the atom value for a node."""
        return self.get_atom(node)

    # ─────────────────────────────────────────────────────────────────────────
    # Backbone decomposition
    # ─────────────────────────────────────────────────────────────────────────

    def _backbone_node(self, m):
        """
        Get the backbone lattice node gamma_m: the antichain of all subsets
        of sources of size m.

        Parameters
        ----------
        m : int
            The order (0 to n).

        Returns
        -------
        node : tuple of tuples
        """
        n = len(self._sources)
        if m == 0:
            return ()
        source_indices = [src for src in self._sources]
        subsets = []
        for combo in combinations(range(n), m):
            merged = tuple(sorted(set().union(*(source_indices[i] for i in combo))))
            subsets.append(merged)
        return tuple(sorted(subsets, key=lambda t: (len(t), t)))

    def get_backbone(self, m):
        """
        Get the backbone synergy term B^m(X -> Y) = S_{gamma_m}(X -> Y).

        Parameters
        ----------
        m : int
            Order from 0 to n. B^0 = I(X;Y), B^n = 0.

        Returns
        -------
        b_m : float
        """
        node = self._backbone_node(m)
        return self.get_synergy(node)

    def get_backbone_atom(self, m):
        """
        Get the backbone atom B^m_partial = B^{m-1} - B^m.

        Always non-negative by construction (Proposition 3 of the paper).

        Parameters
        ----------
        m : int
            Order from 1 to n.

        Returns
        -------
        b_m_partial : float
        """
        return self.get_backbone(m - 1) - self.get_backbone(m)

    # ─────────────────────────────────────────────────────────────────────────
    # Display
    # ─────────────────────────────────────────────────────────────────────────

    def to_string(self, digits=4):
        """
        Create a table of the synergistic disclosure decomposition.

        Parameters
        ----------
        digits : int
            Decimal precision for display.

        Returns
        -------
        table : str
        """
        s_str = "S_alpha"
        a_str = "S_d"

        dist_name = getattr(self._dist, "name", "")
        title = f"{self._name} | {dist_name}" if dist_name else self._name
        table = build_table([self._name, s_str, a_str], title=title)

        table.float_format[s_str] = f"{digits + 2}.{digits}"
        table.float_format[a_str] = f"{digits + 2}.{digits}"

        for node in self._lattice:
            if not node:
                node_label = "{}"
            else:
                node_label = "".join(
                    "{{{}}}".format(":".join(map(str, n))) for n in node
                )
            s_val = self.get_synergy(node)
            a_val = self.get_atom(node)
            if np.isclose(0, s_val, atol=10 ** -(digits - 1), rtol=10 ** -(digits - 1)):
                s_val = 0.0
            if np.isclose(0, a_val, atol=10 ** -(digits - 1), rtol=10 ** -(digits - 1)):
                a_val = 0.0
            table.add_row([node_label, s_val, a_val])

        return table.get_string()

    def backbone_to_string(self, digits=4):
        """
        Create a table of the backbone decomposition.

        Parameters
        ----------
        digits : int
            Decimal precision for display.

        Returns
        -------
        table : str
        """
        n = len(self._sources)
        b_str = "B^m"
        a_str = "B^m_d"

        dist_name = getattr(self._dist, "name", "")
        title = f"{self._name} backbone | {dist_name}" if dist_name else f"{self._name} backbone"
        table = build_table(["m", b_str, a_str], title=title)

        table.float_format[b_str] = f"{digits + 2}.{digits}"
        table.float_format[a_str] = f"{digits + 2}.{digits}"

        for m in range(n + 1):
            b_val = self.get_backbone(m)
            a_val = self.get_backbone_atom(m) if m > 0 else None
            if np.isclose(0, b_val, atol=10 ** -(digits - 1), rtol=10 ** -(digits - 1)):
                b_val = 0.0
            if a_val is not None and np.isclose(0, a_val, atol=10 ** -(digits - 1), rtol=10 ** -(digits - 1)):
                a_val = 0.0
            table.add_row([m, b_val, a_val if a_val is not None else ""])

        return table.get_string()

    def __repr__(self):
        """Return the table representation."""
        return self.to_string()

    def __str__(self):
        """Return the table representation."""
        return self.to_string()


class ModifiedSynDisc(SynDisc):
    """
    Modified synergistic disclosure decomposition.

    Gutknecht, Makkeh & Wibral (2023), "From Babel to Boole: The Logical
    Organization of Information Decompositions", arXiv:2306.00734v2.

    Identical to SynDisc except that singleton constraint nodes (|alpha| = 1)
    use conditional mutual information I(T : a^C | a) instead of the
    optimization-based synergistic disclosure.  This enforces the PID
    consistency condition: for every source collection a, the sum of atoms
    whose parthood distribution assigns 1 to a equals I(a : T).

    Parameters
    ----------
    dist : Distribution
        The joint distribution over sources and target.
    sources : iter of iters, None
        The source variable groups. If None, all variables except the last
        are treated as individual sources.
    target : iter, None
        The target variable. If None, the last variable is used.
    niter : int, None
        Number of basin-hopping restarts per lattice node.
    bound : int, None
        Cardinality bound on the auxiliary variable V.
    """

    _name = "S_msd"

    def _compute_s_alpha(self, node):
        """
        Compute the modified synergistic disclosure for a lattice node.

        For singleton constraints (|alpha| = 1), returns I(T : a^C | a)
        rather than the optimization-based S_alpha.  All other nodes
        delegate to the parent implementation.
        """
        if len(node) == 1:
            all_sources_constrained = (
                set(node[0]) == set().union(*self._sources)
            )
            if all_sources_constrained:
                return 0.0

            source_mi = coinformation(
                self._dist, [list(node[0]), list(self._target)]
            )
            return self._total - source_mi

        return super()._compute_s_alpha(node)
