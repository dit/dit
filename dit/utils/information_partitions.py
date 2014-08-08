"""
Information partitions, e.g. ways of dividing up the information in a joint
distribution.
"""

from __future__ import absolute_import

from itertools import islice
from iterutils import powerset

from prettytable import PrettyTable

from networkx import DiGraph, dfs_preorder_nodes as children, topological_sort

from ..shannon import entropy as H

__all__ = ['ShannonPartition',
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


class ShannonPartition(object):
    """
    Construct an I-Diagram from a given joint distribution.
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
    def _stringify(rvs, crvs):
        """
        Construct a string representation of a measure, e.g. I[X:Y|Z]

        Parameters
        ----------
        rvs : list
            The random variable(s) for the measure.
        crvs : list
            The random variable(s) that the measure is conditioned on.
        """
        rvs = [ ','.join(str(_) for _ in rv) for rv in rvs ]
        crvs = [ str(_) for _ in crvs ]
        a = ':'.join(rvs)
        b = ','.join(crvs)
        symbol = 'H' if len(rvs) == 1 else 'I'
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
        Hs = {}; Is = {}; atoms = {}; new_atoms = {}

        # Entropies
        for node in lattice:
            Hs[node] = H(self.dist, node)

        # Subset-sum type thing, basically co-information calculations.
        for node in lattice:
            Is[node] = sum( (-1)**(len(rv)+1)*Hs[rv] for rv in children(lattice, node) )

        # Mobius inversion of the above, resulting in the Shannon atoms.
        for node in topological_sort(lattice)[:-1]:
            kids = islice(children(rlattice, node), 1, None)
            atoms[node] = Is[node] - sum( atoms[child] for child in kids )

        # get the atom indices in proper format
        for atom, value in atoms.items():
            a_rvs = tuple( (_,) for _ in atom )
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
            lhs = all( any( ( (_,) in atom[0] ) for _ in rv) for rv in rvs )
            rhs = set(crvs).issubset(atom[1])
            return lhs and rhs

        return sum(value for atom, value in self.atoms.items() if is_part(atom, *item))


    def __str__(self):
        """
        Use PrettyTable to create a nice table.
        """
        table = PrettyTable(['measure', 'bits'])
        ### TODO: add some logic for the format string, so things look nice with arbitrary values
        table.float_format['bits'] = ' 5.3'
        for (rvs, crvs), value in reversed(sorted(self.atoms.items(), key=(lambda row: len(row[0][1])))):
            if abs(value) < 1e-10: # TODO: make more robust
                value = 0.0        # gets rid of pesky -0.0 display values
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

        return set([ f(rvs, crvs) for rvs, crvs in self.atoms.keys() ])
