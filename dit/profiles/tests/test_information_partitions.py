"""
Tests for dit.util.information_partitions.
"""
import pytest

from itertools import islice

from dit.example_dists import n_mod_m
from dit.multivariate import coinformation as I, dual_total_correlation as B
from dit.profiles.information_partitions import *
from dit.utils import partitions, powerset

def all_info_measures(vars):
    """
    """
    for stuff in islice(powerset(vars), 1, None):
        others = set(vars) - set(stuff)
        for part in partitions(stuff, tuples=True):
            for cond in powerset(others):
                yield (part , cond)

@pytest.mark.parametrize('meas', all_info_measures(range(4)))
def test_sp1(meas):
    """ Test all possible info measures """
    d = n_mod_m(4, 2)
    ip = ShannonPartition(d)
    assert ip[meas] == pytest.approx(I(d, meas[0], meas[1]))

@pytest.mark.parametrize('meas', all_info_measures('xyzw'))
def test_sp2(meas):
    """ Test all possible info measures, with rv_names """
    d = n_mod_m(4, 2)
    d.set_rv_names('xyzw')
    ip = ShannonPartition(d)
    assert ip[meas] == pytest.approx(I(d, meas[0], meas[1]))

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
    assert (atoms1 - atoms2) | (atoms2 - atoms1) == set()

def test_sp4():
    """ Test printing """
    d = n_mod_m(3, 2)
    ip = ShannonPartition(d)
    string = """\
+----------+--------+
| measure  |  bits  |
+----------+--------+
| H[0|1,2] |  0.000 |
| H[1|0,2] |  0.000 |
| H[2|0,1] |  0.000 |
| I[0:1|2] |  1.000 |
| I[0:2|1] |  1.000 |
| I[1:2|0] |  1.000 |
| I[0:1:2] | -1.000 |
+----------+--------+"""
    assert str(ip) == string

def test_ep1():
    """
    Test against known values.
    """
    d = n_mod_m(3, 2)
    ep = ExtropyPartition(d)
    string = """\
+----------+--------+
| measure  | exits  |
+----------+--------+
| X[0|1,2] |  0.000 |
| X[1|0,2] |  0.000 |
| X[2|0,1] |  0.000 |
| X[0:1|2] |  0.245 |
| X[0:2|1] |  0.245 |
| X[1:2|0] |  0.245 |
| X[0:1:2] |  0.510 |
+----------+--------+"""
    assert str(ep) == string

def test_dd1():
    """
    Test against known values.
    """
    d = n_mod_m(3, 2)
    ep = DependencyDecomposition(d, measures={'B': B})
    string = """\
+------------+--------+
| dependency |   B    |
+------------+--------+
|    012     |  2.000 |
|  01:02:12  |  0.000 |
|   01:02    |  0.000 |
|   01:12    |  0.000 |
|   02:12    |  0.000 |
|    01:2    |  0.000 |
|    02:1    |  0.000 |
|    12:0    |  0.000 |
|   0:1:2    |  0.000 |
+------------+--------+"""
    assert str(ep) == string
