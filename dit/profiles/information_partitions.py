"""
Information partitions, e.g. ways of dividing up the information in a joint
distribution.
"""

import re
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from functools import cache

import numpy as np
from boltons.iterutils import pairwise_iter
from lattices.lattices import dependency_lattice, powerset_lattice

from ..algorithms import degrees_of_freedom, maxent_dist
from ..algorithms.mprojection import (
    m_projection_eps_limit,
    mflat_subsets_from_dependency,
    symmetric_smooth,
)
from ..divergences import kullback_leibler_divergence
from ..multivariate.transmission import transmission
from ..other import extropy
from ..params import ditParams
from ..shannon import entropy
from ..utils import build_table
from .base_profile import _unit_for_base

__all__ = (
    "ShannonPartition",
    "ExtropyPartition",
    "DependencyDecomposition",
    "DualDependencyDecomposition",
)


class BaseInformationPartition(metaclass=ABCMeta):
    """
    Construct an I-Diagram-like partition from a given joint distribution.
    """

    def __init__(self, dist):
        """
        Construct a Shannon-type partition of the information contained in
        `dist`.

        Parameters
        ----------
        dist : distribution
            The distribution to partition.
        """
        self.dist = dist
        self.unit = self._compute_unit(dist)
        self._partition()

    @staticmethod
    def _compute_unit(dist):
        """
        Return the unit label used when rendering this partition.

        Subclasses whose measure uses a fixed unit (e.g. extropy) should
        override this to return a constant.
        """
        return _unit_for_base(dist)

    @staticmethod
    @abstractmethod
    def _symbol(rvs, crvs):
        """
        This method should return the information symbol for an atom.
        """
        pass

    def _stringify(self, rvs, crvs):
        """
        Construct a string representation of a measure, e.g. I[X:Y|Z]

        Parameters
        ----------
        rvs : list
            The random variable(s) for the measure.
        crvs : list
            The random variable(s) that the measure is conditioned on.
        """
        rvs = [",".join(str(_) for _ in rv) for rv in rvs]
        crvs = [str(_) for _ in crvs]
        a = ":".join(rvs)
        b = ",".join(crvs)
        symbol = self._symbol(rvs, crvs)
        sep = "|" if len(crvs) > 0 else ""
        s = f"{symbol}[{a}{sep}{b}]"
        return s

    def _partition(self):
        """
        Return all the atoms of the I-diagram for `dist`.

        Parameters
        ----------
        dist : distribution
            The distribution to compute the I-diagram of.
        """
        rvs = self.dist.get_rv_names()
        if not rvs:
            rvs = tuple(range(self.dist.outcome_length()))

        self._lattice = powerset_lattice(rvs)
        Hs = {}
        Is = {}
        atoms = {}
        new_atoms = {}

        # Entropies
        for node in self._lattice:
            Hs[node] = self._measure(self.dist, node)  # pylint: disable=no-member

        # Subset-sum type thing, basically co-information calculations.
        for node in self._lattice:
            Is[node] = sum((-1) ** (len(rv) + 1) * Hs[rv] for rv in self._lattice.descendants(node, include=True))

        # Mobius inversion of the above, resulting in the Shannon atoms.
        for node in self._lattice:
            kids = self._lattice.ascendants(node)
            atoms[node] = Is[node] - sum(atoms[child] for child in kids)

        # get the atom indices in proper format
        for atom, value in atoms.items():
            if not atom:
                continue

            a_rvs = tuple((_,) for _ in atom)
            a_crvs = tuple(sorted(set(rvs) - set(atom)))
            new_atoms[(a_rvs, a_crvs)] = value

        self.atoms = new_atoms

    def __getitem__(self, item):
        """
        Return the value of any information measure.

        Parameters
        ----------
        item : tuple
            A pair (rvs, crvs).
        """

        def is_part(atom, rvs, crvs):
            lhs = all(any(((_,) in atom[0]) for _ in rv) for rv in rvs)
            rhs = set(crvs).issubset(atom[1])
            return lhs and rhs

        return sum(value for atom, value in self.atoms.items() if is_part(atom, *item))

    def __repr__(self):
        """
        Represent using the str().
        """
        if ditParams["repr.print"]:
            return self.to_string()
        else:
            return super().__repr__()

    def __str__(self):
        """
        Use PrettyTable to create a nice table.
        """
        return self.to_string()

    def to_string(self, digits=3):
        """
        Use PrettyTable to create a nice table.
        """
        table = build_table(["measure", self.unit], title=self._name)  # pylint: disable=no-member
        ### TODO: add some logic for the format string, so things look nice
        #         with arbitrary values
        table.float_format[self.unit] = f" 5.{digits}"  # pylint: disable=no-member
        key_function = lambda row: (len(row[0][0]), row[0][0], row[0][1])
        items = self.atoms.items()
        for (rvs, crvs), value in sorted(items, key=key_function):
            # gets rid of pesky -0.0 display values
            if np.isclose(value, 0.0):
                value = 0.0
            table.add_row([self._stringify(rvs, crvs), value])
        return table.get_string()

    def get_atoms(self, string=True):
        """
        Return all the atoms for the distribution.

        Parameters
        ----------
        string : bool
            If True, return atoms as strings. Otherwise, as a pair of tuples.
        """
        f = self._stringify if string else lambda a, b: (a, b)

        return {f(rvs, crvs) for rvs, crvs in self.atoms}


class ShannonPartition(BaseInformationPartition):
    """
    Construct an I-Diagram from a given joint distribution.
    """

    _measure = staticmethod(entropy)
    _name = "Shannon Partition"
    unit = "bits"

    @staticmethod
    def _symbol(rvs, crvs):
        """
        Returns H for a conditional entropy, and I for all other atoms.
        """
        return "H" if len(rvs) == 1 else "I"


class ExtropyPartition(BaseInformationPartition):
    """
    Construct an X-Diagram from a given joint distribution. One important
    distinction regarding X-Diagrams vs I-Diagrams is that the atoms of an
    X-Diagram are strictly positive.
    """

    _measure = staticmethod(extropy)
    _name = "Extropy Partition"
    unit = "exits"

    @staticmethod
    def _compute_unit(dist):
        """
        Extropy is always reported in "exits" regardless of the distribution's base.
        """
        return "exits"

    @staticmethod
    def _symbol(rvs, crvs):
        """
        Returns X for all atoms.
        """
        return "X"


def tuplefy(dependency):
    """ """
    dependency = tuple(map(tuple, dependency))
    return tuple(tuple(sorted(_)) for _ in sorted(dependency, key=lambda d: (-len(d), d)))


class DependencyDecomposition:
    """
    Construct a decomposition of all the dependencies in a given joint
    distribution.
    """

    def __init__(self, dist, rvs=None, measures={"H": entropy}, cover=True, maxiter=None):  # noqa: B006
        """
        Construct a Krippendorff-type partition of the information contained in
        `dist`.

        Parameters
        ----------
        dist : distribution
            The distribution to partition.
        rvs : iterable
        measures : dict
        cover : bool
        maxiter : int
        """
        self.dist = dist
        self.rvs = sum(dist.rvs, []) if rvs is None else rvs
        self.measures = measures
        self.cover = cover
        self._partition(maxiter=maxiter)
        self._measure_of_interest = self.atoms

    @staticmethod
    def _stringify(dependency):
        """
        Construct a string representation of a dependency, e.g. ABC:AD:BD

        Parameters
        ----------
        dependency : tuple of tuples
        """
        s = ":".join("".join(map(str, d)) for d in tuplefy(dependency))
        return s

    def _partition(self, maxiter=None):
        """
        Computes all the dependencies of `dist`.

        Parameters
        ----------
        maxiter : int
            The number of iterations for the optimization subroutine.
        """
        names = self.dist.get_rv_names()
        rvs = [names[i] for i in self.rvs] if names else self.rvs

        self._lattice = dependency_lattice(rvs, cover=self.cover)
        dists = {}

        # Entropies
        for node in reversed(list(self._lattice)):
            try:
                parent = list(self._lattice._lattice[node].keys())[0]
                x0 = dists[parent].pmf
            except IndexError:
                x0 = None
            dists[node] = maxent_dist(self.dist, node, x0=x0, sparse=False, maxiter=maxiter)

        self.dists = dists

        atoms = defaultdict(dict)
        for name, measure in self.measures.items():
            for node in self._lattice:
                if measure is degrees_of_freedom:
                    # Model complexity is a property of the structure (node),
                    # not recoverable from the reconstructed distribution alone.
                    atoms[node][name] = degrees_of_freedom(self.dist, node)
                elif measure is transmission:
                    # The node's reconstruction is already in `dists`; reuse it
                    # rather than recomputing the maxent distribution.
                    atoms[node][name] = kullback_leibler_divergence(self.dist, dists[node])
                else:
                    atoms[node][name] = measure(dists[node])

        self.atoms = atoms

    def __repr__(self):
        """
        Represent using the str().
        """
        if ditParams["repr.print"]:
            return self.to_string()
        else:
            return super().__repr__()

    def __str__(self):
        """
        Use PrettyTable to create a nice table.
        """
        return self.to_string()

    def __getitem__(self, item):
        """
        Return the dictionary of information values associated with a node.

        Parameters
        ----------
        item : tuple
            The node of interest.

        Returns
        -------
        vars : dict
            A dictionary of {measure: value} pairs.
        """
        return self.atoms[item]

    def edges(self, constraint):
        """
        Iterate over edges which add `constraint`.

        Parameters
        ----------
        constraint : tuple
            The constraint of interest.

        Yields
        ------
        edge : tuple
            An edge that adds the constraint.
        """
        for u, v in self._lattice._lattice.edges():
            if constraint <= u - v:
                yield (u, v)

    @cache
    def delta(self, edge, measure):
        """
        Return the difference in `measure` along `edge`.

        Parameters
        ----------
        edge : tuple
            An edge in the lattice.

        measure : str
            The label for the information measure to get the difference of.

        Returns
        -------
        delta : float
            The difference in the measure.
        """
        a = self.atoms[edge[0]][measure]
        b = self.atoms[edge[1]][measure]
        return a - b

    def to_string(self, digits=3):
        """
        Use PrettyTable to create a nice table.
        """
        measures = list(self.measures.keys())
        table = build_table(
            field_names=["dependency"] + measures, title=re.sub(r"(?<!^)(?=[A-Z])", " ", self.__class__.__name__)
        )
        ### TODO: add some logic for the format string, so things look nice
        # with arbitrary values
        for m in measures:
            table.float_format[m] = f" {digits + 2}.{digits}"
        items = [(tuplefy(row[0]), row[1]) for row in self._measure_of_interest.items() if row[0]]
        items = sorted(items, key=lambda row: row[0])
        items = sorted(items, key=lambda row: sorted((len(d) for d in row[0]), reverse=True), reverse=True)
        for dependency, values in items:
            # gets rid of pesky -0.0 display values
            for m, value in values.items():
                if np.isclose(value, 0.0):
                    values[m] = 0.0
            table.add_row([self._stringify(dependency)] + [values[m] for m in measures])
        return table.get_string()

    def get_dependencies(self, string=True):
        """
        Return all the dependencies within the distribution.

        Parameters
        ----------
        string : bool
            If True, return dependencies as strings. Otherwise, as a tuple of
            tuples.
        """
        f = self._stringify if string else lambda d: d

        return set(map(f, self.atoms.keys()))


class DualDependencyDecomposition(DependencyDecomposition):
    """
    m-flat dual of :class:`DependencyDecomposition`.

    Uses the same dependency lattice, but at each node :math:`\\pi` reconstructs
    via the reverse-KL m-projection onto the additive (mixture) family

    .. math::

        \\mathcal{M}_\\pi = \\Bigl\\{ Q : Q(x) = \\sum_{S \\subseteq T,\\ T\\in\\pi}
        h_S(x_S) \\Bigr\\}

    rather than MaxEnt with marginal constraints. Sparse targets use symmetric
    smoothing :math:`P_\\varepsilon=(1-\\varepsilon)P+\\varepsilon U` and take
    :math:`\\varepsilon\\downarrow 0` along ``eps_schedule``. Default atom
    measure is :math:`D(Q_\\pi \\Vert P_\\varepsilon)` at the final ``eps``.

    See Amari (2001) and :class:`~dit.profiles.MFlatConnectedInformations` for
    the order-chain special case.
    """

    # Sentinel: measure the reverse KL from the node reconstruction to the data.
    REVERSE_KL = object()

    def __init__(
        self,
        dist,
        rvs=None,
        measures=None,
        cover=True,
        eps_schedule=(1e-4, 1e-6, 1e-8),
        eps=None,
        nrestarts=8,
        maxiter=2000,
    ):
        """
        Parameters
        ----------
        dist : Distribution
            Distribution to decompose.
        rvs : iterable or None
            Variables to include. Defaults to all.
        measures : dict or None
            ``{name: callable}`` applied to each reconstruction. The sentinel
            ``DualDependencyDecomposition.REVERSE_KL`` (the default under the
            name ``"rKL"``) records :math:`D(Q \\Vert P_\\varepsilon)`.
        cover : bool
            Passed to :func:`~lattices.lattices.dependency_lattice`.
        eps_schedule : sequence of float or None
            Decreasing smooth weights for the :math:`\\varepsilon\\downarrow 0`
            limit. Default ``(1e-4, 1e-6, 1e-8)``. Ignored if ``eps`` is set
            (single-shot smooth).
        eps : float or None
            Single symmetric smooth weight (overrides ``eps_schedule``).
        nrestarts, maxiter : int
            m-projection optimizer controls.
        """
        if measures is None:
            measures = {"rKL": self.REVERSE_KL}
        self.eps_schedule = None if eps_schedule is None else tuple(eps_schedule)
        self.eps = eps
        self._nrestarts = nrestarts
        self._mp_maxiter = maxiter
        # Replicate DependencyDecomposition.__init__ so we do not call MaxEnt.
        self.dist = dist
        self.rvs = sum(dist.rvs, []) if rvs is None else rvs
        self.measures = measures
        self.cover = cover
        self._partition()
        self._measure_of_interest = self.atoms

    def _index_map(self):
        """Map lattice element labels to dense outcome-coordinate indices."""
        names = self.dist.get_rv_names()
        if names:
            return {name: i for i, name in enumerate(names)}
        return {rv: rv for rv in self.rvs}

    def _project_node(self, subsets, warm):
        """Reverse-KL project onto the mixture family for one lattice node."""
        if self.eps is not None or self.eps_schedule is None:
            from ..algorithms.mprojection import m_projection_from_subsets

            return m_projection_from_subsets(
                self.dist,
                subsets,
                eps=self.eps if self.eps is not None else 1e-8,
                nrestarts=self._nrestarts,
                maxiter=self._mp_maxiter,
                warm_start=warm,
                criterion="reverse_kl",
            ), None

        result = m_projection_eps_limit(
            self.dist,
            subsets=subsets,
            eps_schedule=self.eps_schedule,
            nrestarts=self._nrestarts,
            maxiter=self._mp_maxiter,
            warm_start=warm,
        )
        return result["dist"], result

    def _partition(self, maxiter=None):  # noqa: ARG002
        """
        m-project onto each dependency node's mixture family.
        """
        names = self.dist.get_rv_names()
        rvs = [names[i] for i in self.rvs] if names else self.rvs

        self._lattice = dependency_lattice(rvs, cover=self.cover)
        index_map = self._index_map()
        n = self.dist.outcome_length()
        dists = {}
        meta = {}

        for node in reversed(list(self._lattice)):
            try:
                parent = next(iter(self._lattice._lattice[node].keys()))
                warm = dists[parent]
            except (IndexError, StopIteration, KeyError):
                warm = None

            subsets = mflat_subsets_from_dependency(node, index_map=index_map)
            if any(len(s) == n for s in subsets):
                # Full joint: exact recovery of the working reverse-KL target.
                if self.eps is not None or self.eps_schedule is None:
                    from ..algorithms.mprojection import _resolve_reverse_kl_target

                    target, final_eps = _resolve_reverse_kl_target(self.dist, eps=self.eps)
                    dists[node] = target
                    meta[node] = {"eps": final_eps, "target": target, "rKL": 0.0}
                else:
                    target = symmetric_smooth(self.dist, self.eps_schedule[-1])
                    dists[node] = target
                    meta[node] = {"eps": self.eps_schedule[-1], "target": target, "rKL": 0.0}
            else:
                q, info = self._project_node(subsets, warm)
                dists[node] = q
                meta[node] = info

        self.dists = dists
        self._node_meta = meta

        # Shared reverse-KL target at the final eps.
        if self.eps is not None or self.eps_schedule is None:
            from ..algorithms.mprojection import _resolve_reverse_kl_target

            target, final_eps = _resolve_reverse_kl_target(self.dist, eps=self.eps)
        else:
            final_eps = self.eps_schedule[-1]
            target = symmetric_smooth(self.dist, final_eps)
        self.target = target
        self.eps_final = final_eps

        atoms = defaultdict(dict)
        for name, measure in self.measures.items():
            for node in self._lattice:
                if measure is degrees_of_freedom:
                    atoms[node][name] = degrees_of_freedom(self.dist, node)
                elif measure is self.REVERSE_KL:
                    atoms[node][name] = kullback_leibler_divergence(dists[node], target)
                elif measure is transmission:
                    atoms[node][name] = kullback_leibler_divergence(target, dists[node])
                else:
                    atoms[node][name] = measure(dists[node])

        self.atoms = atoms


class ShapleyDecomposition(DependencyDecomposition):
    """ """

    def __init__(self, dist, rvs=None, measures=None, cover=False, maxiter=None, agg_func=np.mean):
        """ """
        if measures is None:
            measures = {"H": entropy}
        super().__init__(dist, rvs=rvs, measures=measures, cover=cover, maxiter=maxiter)
        self._func = agg_func
        self._shapley_values()
        self._measure_of_interest = self._shapleys

    def _shapley_values(self):
        """ """
        info_diffs = defaultdict(lambda: defaultdict(list))
        for chain in self._lattice.chains():
            for a, b in pairwise_iter(chain):
                dep = b - a
                for measure in self.measures:
                    info_diffs[dep][measure].append(self.delta((b, a), measure))
        self._shapleys = {atom: {measure: self._func(v) for measure, v in d.items()} for atom, d in info_diffs.items()}
