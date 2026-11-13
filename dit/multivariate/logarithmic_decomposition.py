"""
Logarithmic Decomposition of Shannon Entropy.

Implements the logarithmic decomposition (LD) of Down & Mediano, which
refines Yeung's I-measure into "logarithmic atoms" -- one for every subset
of the joint outcome space with 2 or more elements.  Each atom has an
intrinsic sign determined solely by its degree (even -> positive, odd ->
negative) and an interior-loss measure mu that sums to yield entropy,
mutual information, co-information, etc.

Every region of the I-diagram, conditioned or not, is a set of atoms.  The
Gacs-Korner common information is the measure of the largest *discernible*
subset of the co-information content -- the largest subset that is itself
the content of a random variable [3].  Expressions that count atoms more
than once, such as the total correlation and the O-information, are formal
sums of atoms [1].

References
----------
.. [1] K. J. A. Down and P. A. M. Mediano, "A Logarithmic Decomposition
   and a Signed Measure Space for Entropy," arXiv:2409.03732, 2024.
.. [2] K. J. A. Down and P. A. M. Mediano, "Algebraic Representations
   of Entropy and Fixed-Parity Information Quantities,"
   arXiv:2409.04845, 2024.
.. [3] K. J. A. Down and P. A. M. Mediano, "A Logarithmic Decomposition
   for Information," IEEE International Symposium on Information Theory
   (ISIT), pp. 150-155, 2023. arXiv:2305.07554.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

from ..exceptions import ditException
from ..utils import flatten

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
    Interior loss mu(p1, ..., pn) via Mobius inversion of L::

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

    - The atom space Delta(Omega): all subsets of Omega with ``|S| >= 2``.
    - The interior-loss measure mu(b) for each atom b.
    - The content Delta(X) of each random variable X, or of any partition
      of Omega: the atoms that cross a partition boundary.
    - Every region of the I-diagram, conditioned or not, as a set of atoms
      whose measure is the region's I-measure: entropy, conditional
      entropy, mutual information, co-information, and so on.
    - Maximally discernible subsets: the largest subset of a set of atoms
      that is the content of a random variable.  The Gacs-Korner common
      information is the measure of the maximally discernible subset of the
      co-information content.
    - Formal sums, which count atoms with multiplicity: the total
      correlation, dual total correlation, and O-information.
    - Ideal generators and the R_n filter from the algebraic theory.

    Parameters
    ----------
    dist : Distribution
        A joint distribution. The joint outcome space is taken from the
        distribution's non-zero-probability outcomes.

    Notes
    -----
    The number of atoms is ``2^|Omega| - |Omega| - 1``, and measuring an atom
    with ``k`` outcomes sums ``2^k`` terms, so measuring every atom takes about
    three times as long with each added outcome.  This is practical only for
    small outcome spaces (up to about 15 outcomes).

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
        self._outcome_length = dist.outcome_length()

        # Omega is the support: atoms holding a zero-probability outcome have
        # zero measure, and such outcomes would join otherwise separate blocks
        # of the partitions behind discernible subsets.
        pairs = [(o, float(p)) for o, p in dist.zipped()]
        pairs = [(o, p) for o, p in pairs if p > 0]

        self._outcome_list = [o for o, _ in pairs]
        self._prob = dict(pairs)
        # Single numerical variables have bare outcomes rather than 1-tuples.
        if self._outcome_length == 1:
            self._rows = {o: o if isinstance(o, tuple) else (o,) for o in self._outcome_list}
        else:
            self._rows = {o: o for o in self._outcome_list}

        self._omega = frozenset(self._outcome_list)
        self._dims = dist.dims if hasattr(dist, "dims") else tuple(range(dist.outcome_length()))

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
        All logarithmic atoms Delta(Omega): subsets of Omega with ``|S| >= 2``.

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
            A subset of Omega with ``|atom| >= 2``.

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
        atom_set : iterable of frozenset, or mapping of frozenset to int
            A set of atoms, or a formal sum mapping each atom to its
            multiplicity (see :meth:`formal_sum`).

        Returns
        -------
        float
        """
        if isinstance(atom_set, Mapping):
            return sum(n * self.measure(a) for a, n in atom_set.items())
        return sum(self.measure(a) for a in atom_set)

    # ------------------------------------------------------------------
    # Content (partition → atom set)
    # ------------------------------------------------------------------

    def _indices(self, rvs):
        """Flatten a variable specification into indices; None means every variable."""
        if rvs is None:
            return list(range(self._outcome_length))
        if not isinstance(rvs, (list, tuple)):
            rvs = [rvs]
        return list(flatten(rvs))

    def _labels(self, rvs):
        """Map each outcome of Omega to the values it gives the variables `rvs`."""
        indices = self._indices(rvs)
        return {o: tuple(row[i] for i in indices) for o, row in self._rows.items()}

    def _crossing(self, labels):
        """The atoms whose outcomes do not all share one label."""
        result = set()
        for atom in self.atoms:
            outcomes = iter(atom)
            first = labels[next(outcomes)]
            if any(labels[o] != first for o in outcomes):
                result.add(atom)
        return result

    def partition(self, rvs=None) -> frozenset[frozenset]:
        """
        The partition of Omega induced by one or more random variables.

        Parameters
        ----------
        rvs : list of int or list of list of int, optional
            Random variable indices, treated as one joint variable. If None,
            uses all variables jointly, which separates every outcome.

        Returns
        -------
        frozenset of frozenset
            The blocks: the outcomes that agree on every variable in `rvs`.
        """
        blocks: dict[tuple, set] = {}
        for o, label in self._labels(rvs).items():
            blocks.setdefault(label, set()).add(o)
        return frozenset(frozenset(block) for block in blocks.values())

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
        return self._crossing(self._labels(rvs))

    def partition_content(self, partition) -> set[frozenset]:
        """
        Content of an arbitrary random variable on Omega, given as a partition.

        Parameters
        ----------
        partition : iterable of iterables of outcomes
            The blocks of a partition of Omega: every outcome of Omega lies in
            exactly one block. The blocks are the level sets of the variable.

        Returns
        -------
        set of frozenset
            The atoms containing outcomes from two different blocks.

        Raises
        ------
        ditException
            Raised if the blocks overlap, miss an outcome of Omega, or contain
            something other than an outcome of Omega.
        """
        labels = {}
        for index, block in enumerate(partition):
            for outcome in block:
                if outcome not in self._omega:
                    msg = f"{outcome!r} is not an outcome of Omega (the support of the distribution)."
                    raise ditException(msg)
                if outcome in labels:
                    msg = f"{outcome!r} appears in more than one block."
                    raise ditException(msg)
                labels[outcome] = index
        if len(labels) != len(self._omega):
            missing = sorted(map(repr, self._omega - labels.keys()))
            msg = f"The blocks do not cover Omega; missing {', '.join(missing)}."
            raise ditException(msg)
        return self._crossing(labels)

    def region(self, rvs_list=None, crvs=None) -> set[frozenset]:
        """
        The atoms of the I-diagram region I[X_1 : ... : X_r | Z].

        The region is the intersection of the contents Delta(X_i), minus the
        content Delta(Z). Its measure is the region's I-measure (Down & Mediano,
        2023, Theorem 15): a single variable gives the conditional entropy
        H[X | Z], two give the conditional mutual information, and so on.
        Taking every variable, each as a single variable or in `crvs`, gives
        one atom of the I-diagram.

        Parameters
        ----------
        rvs_list : list of list of int, optional
            Each inner list specifies one variable group. If None, uses each
            single variable as a separate group.
        crvs : list of int, optional
            Random variables to condition on (removed from the region).

        Returns
        -------
        set of frozenset
        """
        if rvs_list is None:
            rvs_list = [[i] for i in range(self._outcome_length)]
        result = set(self.atoms)
        for rvs in rvs_list:
            result &= self.content(rvs)
        if crvs is not None:
            result -= self.content(crvs)
        return result

    # ------------------------------------------------------------------
    # Information quantities
    # ------------------------------------------------------------------

    def entropy(self, rvs=None, crvs=None) -> float:
        """
        Entropy H[X | Z] = mu(Delta(X) minus Delta(Z)).

        Parameters
        ----------
        rvs : list of int, optional
            Variable indices defining X. If None, uses all variables.
        crvs : list of int, optional
            Variable indices defining Z. If None, there is no conditioning.

        Returns
        -------
        float
            Entropy in bits.
        """
        return self.measure_set(self.region([rvs], crvs))

    def mutual_information(self, rvs_list, crvs=None) -> float:
        """
        Mutual information I(X ; Y | Z) = mu((Delta(X) ∩ Delta(Y)) minus Delta(Z)).

        Parameters
        ----------
        rvs_list : list of list of int
            Each inner list specifies one variable group.
            E.g. [[0], [1]] for I(X0 ; X1).
        crvs : list of int, optional
            Random variables to condition on.

        Returns
        -------
        float
        """
        return self.coinformation(rvs_list, crvs)

    def coinformation(self, rvs_list=None, crvs=None) -> float:
        """
        Co-information I(X1 ; ... ; Xr | Z) = mu((∩_i Delta(Xi)) minus Delta(Z)).

        Parameters
        ----------
        rvs_list : list of list of int, optional
            Each inner list specifies one variable group.
            If None, uses each single variable as a separate group.
        crvs : list of int, optional
            Random variables to condition on.

        Returns
        -------
        float
        """
        return self.measure_set(self.region(rvs_list, crvs))

    # ------------------------------------------------------------------
    # Discernible subsets (paper 3)
    # ------------------------------------------------------------------

    def discernible_partition(self, atom_set) -> frozenset[frozenset]:
        """
        The finest partition of Omega whose content lies inside a set of atoms.

        A random variable's content avoids an atom exactly when the variable
        is constant on that atom's outcomes. The finest partition whose
        content lies inside `atom_set` therefore joins the outcomes of every
        atom not in `atom_set`: its blocks are the connected components of the
        hypergraph on Omega with those atoms as hyperedges.

        Parameters
        ----------
        atom_set : iterable of frozenset
            Any set of atoms. It need not be closed under supersets; atoms
            that are not subsets of Omega are ignored.

        Returns
        -------
        frozenset of frozenset
            The blocks of the partition.
        """
        kept = {frozenset(atom) for atom in atom_set}
        parent = {o: o for o in self._omega}

        def find(o):
            while parent[o] != o:
                parent[o] = parent[parent[o]]
                o = parent[o]
            return o

        for atom in self.atoms:
            if atom in kept:
                continue
            outcomes = iter(atom)
            root = find(next(outcomes))
            for o in outcomes:
                other = find(o)
                if other != root:
                    parent[other] = root

        blocks: dict = {}
        for o in self._omega:
            blocks.setdefault(find(o), set()).add(o)
        return frozenset(frozenset(block) for block in blocks.values())

    def discernible(self, atom_set) -> set[frozenset]:
        """
        Dis(S): the largest subset of S that is the content of a random variable.

        A set of atoms is *discernible* when it is the content of some random
        variable on Omega (Down & Mediano, 2023, Definition 18; the 2024
        version calls such sets *representable*). Contents are closed under
        union -- Delta(X) ∪ Delta(Y) is the content of the joint variable XY
        -- so every set of atoms has a unique largest discernible subset: the
        content of :meth:`discernible_partition`.

        Parameters
        ----------
        atom_set : iterable of frozenset
            Any set of atoms.

        Returns
        -------
        set of frozenset
            The maximally discernible subset of `atom_set`.

        Examples
        --------
        The Gacs-Korner common random variable is the variable whose content
        is the maximally discernible subset of the mutual information content:

        >>> d = dit.Distribution(['00', '01', '10', '11', '22', '33'], [1/8] * 4 + [1/4] * 2)
        >>> ld = LogarithmicDecomposition(d)
        >>> round(ld.measure_set(ld.discernible(ld.region([[0], [1]]))), 10)
        1.5
        """
        return self.partition_content(self.discernible_partition(atom_set))

    def is_discernible(self, atom_set) -> bool:
        """
        Whether a set of atoms is the content of a random variable on Omega.

        Parameters
        ----------
        atom_set : iterable of frozenset
            Any set of atoms.

        Returns
        -------
        bool
            True if `atom_set` equals its maximally discernible subset.
        """
        atoms = {frozenset(atom) for atom in atom_set}
        return self.discernible(atoms) == atoms

    def gk_common_information(self, rvs_list=None, crvs=None) -> float:
        """
        Gacs-Korner common information as the measure of a discernible subset.

        The common random variable of X_1, ..., X_r is their meet, and its
        content is Dis(∩_i Delta(X_i)) (Down & Mediano, 2023, Theorem 17).
        Conditioning matches :func:`~dit.multivariate.gk_common_information`:
        the result is H[X_1 meet ... meet X_r | Z], the measure of that content
        minus Delta(Z).

        Parameters
        ----------
        rvs_list : list of list of int, optional
            Each inner list specifies one variable group. If None, uses each
            single variable as a separate group.
        crvs : list of int, optional
            Random variables to condition on.

        Returns
        -------
        float
            The common information in bits.
        """
        common = self.discernible(self.region(rvs_list))
        if crvs is not None:
            common -= self.content(crvs)
        return self.measure_set(common)

    # ------------------------------------------------------------------
    # Formal sums (paper 1)
    # ------------------------------------------------------------------

    def formal_sum(self, terms, crvs=None) -> dict[frozenset, int]:
        """
        The formal sum of atoms equal to the entropy expression sum_k c_k H[X_k | Z].

        Each atom b receives the multiplicity sum_k c_k, taken over the terms
        whose region Delta(X_k) minus Delta(Z) contains b. Measuring the formal
        sum with :meth:`measure_set` gives the value of the expression for any
        probabilities on Omega. When every multiplicity is 1, the expression is
        the measure of a set of atoms, its content (Down & Mediano, 2023).

        Parameters
        ----------
        terms : iterable of (int, list of int)
            Pairs of an integer coefficient c_k and the variable indices of
            X_k, which form one joint variable.
        crvs : list of int, optional
            Random variables Z that every entropy is conditioned on.

        Returns
        -------
        dict
            Maps each atom with nonzero multiplicity to that multiplicity.

        Examples
        --------
        The mutual information H[X] + H[Y] - H[XY] counts each atom of
        Delta(X) ∩ Delta(Y) once:

        >>> mi = ld.formal_sum([(1, [0]), (1, [1]), (-1, [0, 1])])
        >>> set(mi) == ld.region([[0], [1]]) and set(mi.values()) == {1}
        True
        """
        excluded = self.content(crvs) if crvs is not None else set()
        multiplicities: dict[frozenset, int] = {}
        for coefficient, rvs in terms:
            for atom in self.content(rvs) - excluded:
                multiplicities[atom] = multiplicities.get(atom, 0) + coefficient
        return {atom: n for atom, n in multiplicities.items() if n != 0}

    def _groups(self, rvs_list):
        """Each variable group as a list of indices; None means each single variable."""
        if rvs_list is None:
            return [[i] for i in range(self._outcome_length)]
        return [self._indices(rvs) for rvs in rvs_list]

    def _total_correlation_terms(self, rvs_list):
        """Terms of sum_i H[X_i] - H[X_1 ... X_n]."""
        groups = self._groups(rvs_list)
        joint = [i for group in groups for i in group]
        return [(1, group) for group in groups] + [(-1, joint)]

    def _dual_total_correlation_terms(self, rvs_list):
        """Terms of H[X] - sum_i H[X_i | X_others] = (1 - n) H[X] + sum_i H[X_others]."""
        groups = self._groups(rvs_list)
        joint = [i for group in groups for i in group]
        others = [[i for j, group in enumerate(groups) if j != k for i in group] for k in range(len(groups))]
        return [(1 - len(groups), joint)] + [(1, rest) for rest in others]

    def total_correlation(self, rvs_list=None, crvs=None) -> float:
        """
        Total correlation T[X_1 : ... : X_n | Z] as a formal sum.

        An atom in the contents of k of the variables X_i, and not in
        Delta(Z), is counted k - 1 times.

        Parameters
        ----------
        rvs_list : list of list of int, optional
            Each inner list specifies one variable group. If None, uses each
            single variable as a separate group.
        crvs : list of int, optional
            Random variables to condition on.

        Returns
        -------
        float
        """
        return self.measure_set(self.formal_sum(self._total_correlation_terms(rvs_list), crvs))

    def dual_total_correlation(self, rvs_list=None, crvs=None) -> float:
        """
        Dual total correlation B[X_1 : ... : X_n | Z] as a formal sum.

        An atom is counted once if it lies in the contents of at least two of
        the variables X_i and not in Delta(Z), so this formal sum is a content:
        the union of the pairwise intersections Delta(X_i) ∩ Delta(X_j), minus
        Delta(Z).

        Parameters
        ----------
        rvs_list : list of list of int, optional
            Each inner list specifies one variable group. If None, uses each
            single variable as a separate group.
        crvs : list of int, optional
            Random variables to condition on.

        Returns
        -------
        float
        """
        return self.measure_set(self.formal_sum(self._dual_total_correlation_terms(rvs_list), crvs))

    def o_information(self, rvs_list=None, crvs=None) -> float:
        """
        O-information T - B as a formal sum.

        An atom in the contents of k of the variables X_i, and not in
        Delta(Z), is counted max(k - 2, 0) times. With three variables this is
        the co-information content; with n variables a multiplicity can reach
        n - 2, so the O-information is not the measure of a set of atoms.

        Parameters
        ----------
        rvs_list : list of list of int, optional
            Each inner list specifies one variable group. If None, uses each
            single variable as a separate group.
        crvs : list of int, optional
            Random variables to condition on.

        Returns
        -------
        float
        """
        terms = self._total_correlation_terms(rvs_list)
        terms += [(-c, rvs) for c, rvs in self._dual_total_correlation_terms(rvs_list)]
        return self.measure_set(self.formal_sum(terms, crvs))

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
