from __future__ import division

from nose.tools import assert_almost_equal

from iterutils import powerset

from dit.algorithms import binding_information as B, tse_complexity as TSE
from dit.example_dists import n_mod_m
from dit.math.misc import combinations as nCk

# test based on Olbrich's talk
def test_tse1():
    for i, j in zip(range(3, 6), range(2, 5)):
        d = n_mod_m(i, j)
        indices = [ [k] for k in range(i) ]
        tse = TSE(d)
        x = 1/2 * sum(B(d, rv)/nCk(i, len(rv)) for rv in powerset(indices))
        assert_almost_equal(tse, x)
