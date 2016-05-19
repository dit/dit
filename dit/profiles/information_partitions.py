"""
Information partitions, e.g. ways of dividing up the information in a joint
distribution.
"""

from __future__ import absolute_import

from abc import ABCMeta, abstractmethod

from itertools import islice
from iterutils import powerset

from prettytable import PrettyTable

from networkx import DiGraph, dfs_preorder_nodes as children, topological_sort

import dit
from ..shannon import entropy
from ..other import extropy
from ..math import close

__all__ = ['ShannonPartition',
           'ExtropyPartition',
          ]


### TODO: enable caching on this?
def poset_lattice(elements):
    """
    Return the Hasse diagram of the lattice induced by `elements`.
    """
    child = lambda a, b: a.issubset(b) and (len(b) - len(a) == 1)

    lattice = DiGraph()

    for a in powerset(elements):
        for b in powerset(elements):
            if child(set(a), set(b)):
                lattice.add_edge(b, a)

    return lattice


class BaseInformationPartition(object):
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

        lattice = poset_lattice(rvs)
        rlattice = lattice.reverse()
        Hs = {}
        Is = {}
        atoms = {}
        new_atoms = {}

        # Entropies
        for node in lattice:
            Hs[node] = self._measure(self.dist, node)

        # Subset-sum type thing, basically co-information calculations.
        for node in lattice:
            Is[node] = sum((-1)**(len(rv)+1)*Hs[rv] for rv in children(lattice, node))

        # Mobius inversion of the above, resulting in the Shannon atoms.
        for node in topological_sort(lattice)[:-1]:
            kids = islice(children(rlattice, node), 1, None)
            atoms[node] = Is[node] - sum(atoms[child] for child in kids)

        # get the atom indices in proper format
        for atom, value in atoms.items():
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
        return str(self)

    def __str__(self):
        """
        Use PrettyTable to create a nice table.
        """
        return self.to_string()

    def to_string(self, digits=3):
        """
        Use PrettyTable to create a nice table.
        """
        table = PrettyTable(['measure', self.unit])
        ### TODO: add some logic for the format string, so things look nice
        # with arbitrary values
        table.float_format[self.unit] = ' 5.{0}'.format(digits)
        key_function = lambda row: (len(row[0][0]), row[0][0], row[0][1])
        items = self.atoms.items()
        for (rvs, crvs), value in sorted(items, key=key_function):
            # gets rid of pesky -0.0 display values
            if close(value, 0.0):
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

        return set([f(rvs, crvs) for rvs, crvs in self.atoms.keys()])


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
