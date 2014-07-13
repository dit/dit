"""
Tests for dit.util.information_partitions.
"""
from nose.tools import assert_almost_equal

from itertools import islice
from iterutils import powerset

from dit.multivariate import coinformation as I
from dit.utils import partitions, ShannonPartition
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
        yield assesrt_almost_equal, ip[meas], I(d, meas[0], meas[1])

def test_sp2():
    """ Test all possible info measures, with rv_names """
    d = n_mod_m(4, 2)
    d.set_rv_names('xyzw')
    ip = ShannonPartition(d)
    for meas in all_info_measures('xyzw'):
        yield assesrt_almost_equal, ip[meas], I(d, meas[0], meas[1])
