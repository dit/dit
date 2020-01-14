"""
Information partitions, e.g. ways of dividing up the information in a joint
distribution.
"""

from abc import ABCMeta, abstractmethod
from collections import defaultdict
from itertools import islice

import numpy as np
import prettytable
from lattices.lattices import dependency_lattice, powerset_lattice

from .. import ditParams
from ..algorithms import maxent_dist
from ..other import extropy
from ..shannon import entropy

__all__ = [
    'ShannonPartition',
    'ExtropyPartition',
    'DependencyDecomposition',
]


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
        self._partition()

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
        rvs = [','.join(str(_) for _ in rv) for rv in rvs]
        crvs = [str(_) for _ in crvs]
        a = ':'.join(rvs)
        b = ','.join(crvs)
        symbol = self._symbol(rvs, crvs)
        sep = '|' if len(crvs) > 0 else ''
        s = "{0}[{1}{2}{3}]".format(symbol, a, sep, b)
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
            Is[node] = sum((-1)**(len(rv)+1)*Hs[rv] for rv in self._lattice.descendants(node, include=True))

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
        if ditParams['repr.print']:
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
        table = prettytable.PrettyTable(['measure', self.unit])  # pylint: disable=no-member
        if ditParams['text.font'] == 'linechar': # pragma: no cover
            try:
                table.set_style(prettytable.UNICODE_LINES)
            except AttributeError:
                pass
        ### TODO: add some logic for the format string, so things look nice
        #         with arbitrary values
        table.float_format[self.unit] = ' 5.{0}'.format(digits)  # pylint: disable=no-member
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
        if string:
            f = self._stringify
        else:
            f = lambda a, b: (a, b)

        return {f(rvs, crvs) for rvs, crvs in self.atoms.keys()}


class ShannonPartition(BaseInformationPartition):
    """
    Construct an I-Diagram from a given joint distribution.
    """

    _measure = staticmethod(entropy)
    unit = 'bits'

    @staticmethod
    def _symbol(rvs, crvs):
        """
        Returns H for a conditional entropy, and I for all other atoms.
        """
        return 'H' if len(rvs) == 1 else 'I'


class ExtropyPartition(BaseInformationPartition):
    """
    Construct an X-Diagram from a given joint distribution. One important
    distinction regarding X-Diagrams vs I-Diagrams is that the atoms of an
    X-Diagram are strictly positive.
    """

    _measure = staticmethod(extropy)
    unit = 'exits'

    @staticmethod
    def _symbol(rvs, crvs):
        """
        Returns X for all atoms.
        """
        return 'X'


def tuplefy(dependency):
    """
    """
    dependency = tuple(map(tuple, dependency))
    return tuple([tuple(sorted(_)) for _ in sorted(dependency, key=lambda d: (-len(d), d))])


class DependencyDecomposition(object):
    """
    Construct a decomposition of all the dependencies in a given joint
    distribution.
    """

    def __init__(self, dist, rvs=None, measures={'H': entropy}, maxiter=None):
        """
        Construct a Krippendorff-type partition of the information contained in
        `dist`.

        Parameters
        ----------
        dist : distribution
            The distribution to partition.
        rvs : iterable
        measures : dict
        maxiter : int
        """
        self.dist = dist
        self.rvs = sum(dist.rvs, []) if rvs is None else rvs
        self.measures = measures
        self._partition(maxiter=maxiter)

    @staticmethod
    def _stringify(dependency):
        """
        Construct a string representation of a dependency, e.g. ABC:AD:BD

        Parameters
        ----------
        dependency : tuple of tuples
        """
        s = ':'.join(''.join(map(str, d)) for d in tuplefy(dependency))
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
        if names:
            rvs = [names[i] for i in self.rvs]
        else:
            rvs = self.rvs

        self._lattice = dependency_lattice(rvs)
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
                atoms[node][name] = measure(dists[node])

        self.atoms = atoms

    def __repr__(self):
        """
        Represent using the str().
        """
        if ditParams['repr.print']:
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
        table = prettytable.PrettyTable(['dependency'] + measures)
        if ditParams['text.font'] == 'linechar':  # pragma: no cover
            try:
                table.set_style(prettytable.UNICODE_LINES)
            except AttributeError:
                pass
        ### TODO: add some logic for the format string, so things look nice
        # with arbitrary values
        for m in measures:
            table.float_format[m] = ' {}.{}'.format(digits+2, digits)
        items = [(tuplefy(row[0]), row[1]) for row in self.atoms.items()]
        items = sorted(items, key=lambda row: row[0])
        items = sorted(items, key=lambda row: sorted([len(d) for d in row[0]], reverse=True), reverse=True)
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
        if string:
            f = self._stringify
        else:
            f = lambda d: d

        return set(map(f, self.atoms.keys()))
