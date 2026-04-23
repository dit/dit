"""
Logarithmic Decomposition of Shannon Entropy.

Implements the logarithmic decomposition (LD) of Down & Mediano, which
refines Yeung's I-measure into "logarithmic atoms" -- one for every subset
of the joint outcome space with 2 or more elements.  Each atom has an
intrinsic sign determined solely by its degree (even -> positive, odd ->
negative) and an interior-loss measure mu that sums to yield entropy,
mutual information, co-information, etc.

References
----------
.. [1] K. J. A. Down and P. A. M. Mediano, "A Logarithmic Decomposition
   and a Signed Measure Space for Entropy," arXiv:2409.03732, 2024.
.. [2] K. J. A. Down and P. A. M. Mediano, "Algebraic Representations
   of Entropy and Fixed-Parity Information Quantities,"
   arXiv:2409.04845, 2024.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..distribution import Distribution

__all__ = (
    "LogarithmicDecomposition",
    "logarithmic_decomposition",
)


def _xlogx(x):
    """Compute x * log2(x), with the convention 0 * log(0) = 0."""
    if x <= 0:
        return 0.0
    return x * np.log2(x)


def _loss(probs):
    """
    Total entropy loss L(p1, ..., pn).

    L = (sum pi) * log2(sum pi) - sum(pi * log2(pi))

    This equals H(p1,...,pn) when sum(pi) = 1, but is well-defined and
    homogeneous of degree 1 for arbitrary non-negative pi.
    """
    if len(probs) <= 1:
        return 0.0
    s = sum(probs)
    if s <= 0:
        return 0.0
    return _xlogx(s) - sum(_xlogx(p) for p in probs)


def _interior_loss(probs):
    """
    Interior loss mu(p1, ..., pn) via Mobius inversion of L.

    mu(p1,...,pn) = sum_{S subset {p1,...,pn}} (-1)^{n-|S|} L(S)

    Parameters
    ----------
    probs : tuple of float
        Probabilities associated with the outcomes in the atom.

    Returns
    -------
    float
        The interior loss (measure) of the atom.
    """
    n = len(probs)
    if n <= 1:
        return 0.0

    indices = list(range(n))
    total = 0.0
    for k in range(2, n + 1):
        for subset_idx in itertools.combinations(indices, k):
            sub_probs = tuple(probs[i] for i in subset_idx)
            sign = (-1) ** (n - k)
            total += sign * _loss(sub_probs)
    return total


class LogarithmicDecomposition:
    """
    The logarithmic decomposition of a joint distribution.

    Given a distribution over a joint outcome space Omega, this class
    computes:

    - The atom space Delta(Omega): all subsets of Omega with |S| >= 2.
    - The interior-loss measure mu(b) for each atom b.
    - The content Delta(X) for each random variable X: atoms that cross
      a partition boundary.
    - Entropy, mutual information, and co-information as sums of atom
      measures over appropriate content intersections.
    - Ideal generators and the R_n filter from the algebraic theory.

    Parameters
    ----------
    dist : Distribution
        A joint distribution. The joint outcome space is taken from the
        distribution's non-zero-probability outcomes.

    Notes
    -----
    The number of atoms is 2^|Omega| - |Omega| - 1, so this is practical
    only for small outcome spaces (up to ~15-20 outcomes).

    Examples
    --------
    >>> import dit
    >>> d = dit.example_dists.Xor()
    >>> ld = LogarithmicDecomposition(d)
    >>> abs(ld.coinformation() - (-1.0)) < 1e-10
    True
    """

    def __init__(self, dist: Distribution):
        self._dist = dist

        outcomes = dist.outcomes
        pmf = dist.pmf
        if hasattr(dist, "is_log") and dist.is_log():
            base = dist.get_base(numerical=True)
            pmf = base**pmf

        self._outcome_list = list(outcomes)
        self._prob = {o: float(p) for o, p in zip(outcomes, pmf, strict=True)}

        self._omega = frozenset(outcomes)
        self._dims = dist.dims if hasattr(dist, "dims") else tuple(range(dist.outcome_length()))
        self._outcome_length = dist.outcome_length()

        self._atoms_cache: set[frozenset] | None = None
        self._measure_cache: dict[frozenset, float] = {}

    # ------------------------------------------------------------------
    # Atom space
    # ------------------------------------------------------------------

    @property
    def omega(self) -> frozenset:
        """The joint outcome space as a frozenset of outcome tuples."""
        return self._omega

    @property
    def atoms(self) -> set[frozenset]:
        """
        All logarithmic atoms Delta(Omega): subsets of Omega with |S| >= 2.

        Returns
        -------
        set of frozenset
        """
        if self._atoms_cache is None:
            omega = list(self._omega)
            result = set()
            for k in range(2, len(omega) + 1):
                for combo in itertools.combinations(omega, k):
                    result.add(frozenset(combo))
            self._atoms_cache = result
        return self._atoms_cache

    # ------------------------------------------------------------------
    # Measure
    # ------------------------------------------------------------------

    def loss(self, subset: frozenset) -> float:
        """
        Total entropy loss L(S) for a subset of outcomes.

        Parameters
        ----------
        subset : frozenset
            A subset of outcomes from Omega.

        Returns
        -------
        float
            The loss in bits.
        """
        probs = tuple(self._prob.get(o, 0.0) for o in subset)
        return _loss(probs)

    def measure(self, atom: frozenset) -> float:
        """
        Interior loss mu(b) for a single atom.

        Parameters
        ----------
        atom : frozenset
            A subset of Omega with |atom| >= 2.

        Returns
        -------
        float
            The signed measure of the atom in bits.
        """
        if atom in self._measure_cache:
            return self._measure_cache[atom]

        probs = tuple(self._prob.get(o, 0.0) for o in atom)
        val = 0.0 if any(p <= 0 for p in probs) else _interior_loss(probs)

        self._measure_cache[atom] = val
        return val

    def measure_set(self, atom_set) -> float:
        """
        Sum of mu(b) for all atoms b in the given set.

        Parameters
        ----------
        atom_set : iterable of frozenset

        Returns
        -------
        float
        """
        return sum(self.measure(a) for a in atom_set)

    # ------------------------------------------------------------------
    # Content (partition → atom set)
    # ------------------------------------------------------------------

    def _partition_for_rv(self, rv_indices):
        """
        Compute the partition of Omega induced by one or more random variables.

        Parameters
        ----------
        rv_indices : list of int
            Indices into each outcome tuple selecting the variable(s).

        Returns
        -------
        dict
            Mapping from event-value tuple to frozenset of outcomes.
        """
        parts: dict[tuple, set] = {}
        for o in self._outcome_list:
            key = tuple(o[i] for i in rv_indices) if len(rv_indices) > 1 else (o[rv_indices[0]],)
            parts.setdefault(key, set()).add(o)
        return {k: frozenset(v) for k, v in parts.items()}

    def content(self, rvs=None) -> set[frozenset]:
        """
        Content Delta(X): atoms crossing a boundary in the partition of X.

        An atom S is in Delta(X) if S contains at least two outcomes that
        belong to different parts of X's partition.

        Parameters
        ----------
        rvs : list of int or list of list of int, optional
            Random variable indices. Each inner list is treated as a group
            (joint variable). If None, uses all variables jointly (gives
            all atoms -- equivalent to Delta(Omega) for the finest partition).

        Returns
        -------
        set of frozenset
            The content set.

        Examples
        --------
        >>> ld.content([0])      # Delta(X_0)
        >>> ld.content([0, 1])   # Delta(X_0, X_1) = Delta(X_0 join X_1)
        """
        if rvs is None:
            rvs = list(range(self._outcome_length))
        if not isinstance(rvs, (list, tuple)):
            rvs = [rvs]

        partition = self._partition_for_rv(rvs)
        parts = list(partition.values())

        if len(parts) <= 1:
            return set()

        result = set()
        for atom in self.atoms:
            part_ids = set()
            for o in atom:
                for idx, part in enumerate(parts):
                    if o in part:
                        part_ids.add(idx)
                        break
            if len(part_ids) >= 2:
                result.add(atom)
        return result

    # ------------------------------------------------------------------
    # Information quantities
    # ------------------------------------------------------------------

    def entropy(self, rvs=None) -> float:
        """
        Entropy H(X) = mu(Delta(X)).

        Parameters
        ----------
        rvs : list of int, optional
            Variable indices defining X. If None, uses all variables.

        Returns
        -------
        float
            Entropy in bits.
        """
        return self.measure_set(self.content(rvs))

    def mutual_information(self, rvs_list) -> float:
        """
        Mutual information I(X ; Y) = mu(Delta(X) ∩ Delta(Y)).

        Parameters
        ----------
        rvs_list : list of list of int
            Each inner list specifies one variable group.
            E.g. [[0], [1]] for I(X0 ; X1).

        Returns
        -------
        float
        """
        return self.coinformation(rvs_list)

    def coinformation(self, rvs_list=None) -> float:
        """
        Co-information I(X1 ; ... ; Xr) = mu(∩_i Delta(Xi)).

        Parameters
        ----------
        rvs_list : list of list of int, optional
            Each inner list specifies one variable group.
            If None, uses each single variable as a separate group.

        Returns
        -------
        float
        """
        if rvs_list is None:
            rvs_list = [[i] for i in range(self._outcome_length)]
        contents = [self.content(rv) for rv in rvs_list]
        intersection = contents[0]
        for c in contents[1:]:
            intersection = intersection & c
        return self.measure_set(intersection)

    # ------------------------------------------------------------------
    # Ideal structure (from paper 2)
    # ------------------------------------------------------------------

    @staticmethod
    def degree(atom: frozenset) -> int:
        """Degree of an atom: the number of outcomes it contains."""
        return len(atom)

    @staticmethod
    def generators(content_set: set[frozenset]) -> set[frozenset]:
        """
        Minimal elements (ideal generators) of a content set.

        In the partial order where b_S1 <= b_S2 iff S1 ⊆ S2, the generators
        are those atoms not containing any other atom in the set.

        Parameters
        ----------
        content_set : set of frozenset

        Returns
        -------
        set of frozenset
        """
        gens = set()
        sorted_atoms = sorted(content_set, key=len)
        for atom in sorted_atoms:
            if not any(g < atom for g in gens):
                gens.add(atom)
        return gens

    @staticmethod
    def r_n(content_set: set[frozenset], n: int) -> set[frozenset]:
        """
        R_n(C): atoms in C lying in the upper set of a degree-n atom in C.

        From Definition 62 of [1]:
        R_n(C) = {c in C : exists c' in C with deg(c') = n and c' ⊆ c}

        Parameters
        ----------
        content_set : set of frozenset
            A set of atoms (e.g. a co-information content).
        n : int
            The degree to filter on.

        Returns
        -------
        set of frozenset
        """
        degree_n = {a for a in content_set if len(a) == n}
        return {c for c in content_set if any(g <= c for g in degree_n)}

    # ------------------------------------------------------------------
    # Tabular summary
    # ------------------------------------------------------------------

    def atom_table(self, rvs_groups=None):
        """
        Summary table of all atoms with degree, measure, and memberships.

        Parameters
        ----------
        rvs_groups : list of list of int, optional
            Variable groups to show content membership for. Defaults to
            each single variable.

        Returns
        -------
        list of dict
            Each dict has keys 'atom', 'degree', 'measure', and one boolean
            key per variable group.
        """
        if rvs_groups is None:
            rvs_groups = [[i] for i in range(self._outcome_length)]

        contents = {tuple(rv): self.content(rv) for rv in rvs_groups}

        rows = []
        for atom in sorted(self.atoms, key=lambda a: (len(a), sorted(a))):
            row = {
                "atom": atom,
                "degree": len(atom),
                "measure": self.measure(atom),
            }
            for rv in rvs_groups:
                label = f"Delta({rv})"
                row[label] = atom in contents[tuple(rv)]
            rows.append(row)
        return rows

    def __repr__(self):
        n = len(self._omega)
        n_atoms = len(self.atoms)
        return f"LogarithmicDecomposition(|Omega|={n}, atoms={n_atoms})"


def logarithmic_decomposition(dist: Distribution) -> LogarithmicDecomposition:
    """
    Construct a LogarithmicDecomposition for the given distribution.

    Parameters
    ----------
    dist : Distribution

    Returns
    -------
    LogarithmicDecomposition
    """
    return LogarithmicDecomposition(dist)
