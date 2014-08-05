"""
Tests for dit.util.information_partitions.
"""
from nose.tools import assert_almost_equal, assert_equal

from itertools import islice
from iterutils import powerset

from dit.multivariate import coinformation as I
from dit.utils import partitions
from dit.utils.information_partitions import ShannonPartition
from dit.example_dists import n_mod_m

def all_info_measures(vars):
    """
    """
    for stuff in islice(powerset(vars), 1, None):
        others = set(vars) - set(stuff)
        for part in partitions(stuff, tuples=True):
            for cond in powerset(others):
                yield (part , cond)

def test_sp1():
    """ Test all possible info measures """
    d = n_mod_m(4, 2)
    ip = ShannonPartition(d)
    for meas in all_info_measures(range(4)):
        yield assert_almost_equal, ip[meas], I(d, meas[0], meas[1])

def test_sp2():
    """ Test all possible info measures, with rv_names """
    d = n_mod_m(4, 2)
    d.set_rv_names('xyzw')
    ip = ShannonPartition(d)
    for meas in all_info_measures('xyzw'):
        yield assert_almost_equal, ip[meas], I(d, meas[0], meas[1])

def test_sp3():
    """ Test get_atoms() """
    d = n_mod_m(3, 2)
    ip = ShannonPartition(d)
    atoms1 = {'H[0|1,2]',
              'H[1|0,2]',
              'H[2|0,1]',
              'I[0:1:2]',
              'I[0:1|2]',
              'I[0:2|1]',
              'I[1:2|0]'
             }
    atoms2 = ip.get_atoms()
    assert_equal(atoms1 - atoms2, set())
    assert_equal(atoms2 - atoms1, set())
