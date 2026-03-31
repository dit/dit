"""
Shapley-based information decompositions.

Implements the Ay/Polani/Virgo decomposition from arXiv:1910.05979, which
decomposes mutual information I(sources; target) into non-negative
contributions for each subset of source variables using the "input lattice"
and generalized Shapley values.
"""

from collections import defaultdict

import numpy as np
from boltons.iterutils import pairwise_iter
from lattices.lattices import dependency_lattice

from ..algorithms import maxent_dist
from ..params import ditParams
from ..shannon import entropy
from ..utils import build_table
from .information_partitions import ShapleyDecomposition

__all__ = (
    "ShapleyDependencyDecomposition",
    "ShapleyShannonDecomposition",
)


def _node_to_constraint(node, source_idx_map, all_source_indices, target_indices):
    """
    Map an input-lattice node (antichain of frozensets of source labels) to a
    set of marginal constraints suitable for `maxent_dist`.

    Parameters
    ----------
    node : frozenset of frozensets
        A node from the input lattice built over source labels.
    source_idx_map : dict
        Maps each source label to its frozenset of variable indices.
    all_source_indices : frozenset
        The union of all source variable indices.
    target_indices : frozenset
        The target variable indices.

    Returns
    -------
    constraint : frozenset of frozensets
        An antichain of variable-index groups for `maxent_dist`.
    """
    parts = {all_source_indices}
    for element in node:
        group = frozenset().union(*(source_idx_map[label] for label in element))
        parts.add(group | target_indices)
    parts = {p for p in parts if not any(p < q for q in parts)}
    return frozenset(parts)


class ShapleyDependencyDecomposition:
    """
    Decompose I(sources; target) into non-negative Shapley-based contributions
    for every non-empty subset of source variables, following the method of
    Ay, Polani & Virgo (arXiv:1910.05979).

    Each contribution I_A measures how much unique predictive information about
    the target is added by the subset A of source variables, averaged over all
    orderings consistent with the input-lattice precedence constraints.
    """

    def __init__(self, dist, sources=None, target=None, maxiter=None):
        """
        Parameters
        ----------
        dist : Distribution
            The joint distribution over source and target variables.
        sources : iter of iters, optional
            Source variable groups.  Each element is an iterable of variable
            indices treated as one source.  Defaults to each variable except
            the last.
        target : iter, optional
            Target variable indices.  Defaults to the last variable.
        maxiter : int, optional
            Maximum iterations for the `maxent_dist` optimizer.
        """
        self._dist = dist

        if target is None:
            target = dist.rvs[-1]
        if sources is None:
            sources = [var for var in dist.rvs if var[0] not in target]

        self._sources = tuple(map(tuple, sources))
        self._target = tuple(target)

        self._compute(maxiter=maxiter)

    def _compute(self, maxiter=None):
        """Build the input lattice, split distributions, and Shapley values."""
        source_labels = list(range(len(self._sources)))
        source_idx_map = {
            i: frozenset(self._sources[i]) for i in source_labels
        }
        all_source_indices = frozenset().union(*source_idx_map.values())
        target_indices = frozenset(self._target)

        self._lattice = dependency_lattice(source_labels, cover=False)

        constraint_map = {}
        for node in self._lattice:
            constraint_map[node] = _node_to_constraint(
                node, source_idx_map, all_source_indices, target_indices,
            )

        self._dists = {}
        for node in reversed(list(self._lattice)):
            try:
                parent = list(self._lattice._lattice[node].keys())[0]
                x0 = self._dists[parent].pmf
            except IndexError:
                x0 = None
            self._dists[node] = maxent_dist(
                self._dist, constraint_map[node], x0=x0, sparse=False,
                maxiter=maxiter if maxiter is not None else 1000,
            )

        self._entropies = {node: entropy(d) for node, d in self._dists.items()}

        self._shapley_values()

    def _shapley_values(self):
        """Compute Shapley information contributions over maximal chains."""
        info_diffs = defaultdict(list)
        for chain in self._lattice.chains():
            for a, b in pairwise_iter(chain):
                predictor = b - a
                gain = self._entropies[a] - self._entropies[b]
                info_diffs[predictor].append(gain)
        self.contributions = {
            pred: float(np.mean(vals)) for pred, vals in info_diffs.items()
        }

    def __getitem__(self, item):
        """
        Look up the information contribution of a predictor.

        Parameters
        ----------
        item : frozenset
            The predictor subset, e.g. ``frozenset({frozenset({0})})``.

        Returns
        -------
        value : float
        """
        return self.contributions[item]

    def __repr__(self):
        """Represent using ``to_string`` when configured."""
        if ditParams["repr.print"]:
            return self.to_string()
        return super().__repr__()

    def __str__(self):
        """Pretty-print the decomposition table."""
        return self.to_string()

    @staticmethod
    def _stringify_predictor(predictor):
        """
        Convert a predictor frozenset-of-frozensets to a readable label.

        E.g. ``frozenset({frozenset({0, 1})})`` -> ``"{0,1}"``,
             ``frozenset({frozenset({0})})``    -> ``"{0}"``.
        """
        parts = sorted(
            ("{" + ",".join(map(str, sorted(fs))) + "}" for fs in predictor),
            key=lambda s: (len(s), s),
        )
        return "".join(parts)

    def to_string(self, digits=4):
        """
        Render the decomposition as a table.

        Parameters
        ----------
        digits : int
            Decimal places to display.

        Returns
        -------
        table : str
        """
        table = build_table(
            field_names=["predictor", "information"],
            title="Shapley Dependency Decomposition",
        )
        table.float_format["information"] = f" {digits + 2}.{digits}"
        items = sorted(
            self.contributions.items(),
            key=lambda kv: (len(next(iter(kv[0]))), sorted(next(iter(kv[0])))),
        )
        for predictor, value in items:
            if np.isclose(value, 0.0):
                value = 0.0
            table.add_row([self._stringify_predictor(predictor), value])
        return table.get_string()


ShapleyShannonDecomposition = ShapleyDecomposition
