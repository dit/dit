"""
Information partitions, e.g. ways of dividing up the information in a joint
distribution.
"""

from prettytable import PrettyTable

from dit.utils import partitions
from dit.multivariate import coinformation as I

__all__ = ['ShannonPartition',
          ]

class ShannonPartition(object):
    """
    Construct an I-Diagram from a given joint distribution.
    """

    def __init__(self, dist):
        """
        """
        self.dist = dist
        self._partition()

    @staticmethod
    def _stringify(rvs, crvs):
        """
        """
        rvs = [ ','.join(str(_) for _ in rv) for rv in rvs ]
        crvs = [ str(_) for _ in crvs ]
        a = ':'.join(rvs)
        b = ','.join(crvs)
        symbol = 'H' if len(rvs) == 1 else 'I'
        sep = '|' if len(crvs) > 0 else ''
        s = "{}[{}{}{}]".format(symbol, a, sep, b)
        return s

    ### FIXME: Computing each atom this way is terribly inefficient.
    def _partition(self):
        """
        Return all the atoms of the I-diagram for `dist`.

        Parameters
        ----------
        dist : distribution
            The distribution to compute the I-diagram of.

        Returns
        -------

        """
        rvs = self.dist.get_rv_names()
        if not rvs:
            rvs = tuple(range(self.dist.outcome_length()))
        atoms = { p for p in partitions(rvs, tuples=True) if len(p) == 2 }
        atoms = atoms.union({ (p[1], p[0]) for p in atoms })
        atoms.add((rvs, ()))

        atoms = [ (tuple( (_,) for _ in a ), b) for a, b in atoms ]

        self.atoms = { (rvs, crvs): I(self.dist, rvs, crvs) for rvs, crvs in atoms }

    def __str__(self):
        """
        Use PrettyTable to create a nice table.
        """
        table = PrettyTable(['measure', 'bits'])
        ### TODO: add some logic for the format string, so things look nice with arbitrary values
        table.float_format['bits'] = ' 5.3'
        for (rvs, crvs), value in reversed(sorted(self.atoms.iteritems(), key=(lambda row: len(row[0][1])))):
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
            f = lambda a, b: a, b

        return { f(rvs, crvs) for rvs, crvs in self.atoms.keys() }
