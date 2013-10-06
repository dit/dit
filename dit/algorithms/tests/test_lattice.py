
from nose.tools import *

from dit.algorithms.lattice import sigma_algebra_sort

def test_sigalg_sort():
	sigalg = frozenset([
		frozenset([]),
		frozenset([1]),
		frozenset([2]),
		frozenset([1,2])
	])
	sigalg_ = [(), (1,), (2,), (1,2)]
	assert_equal( sigalg_, sigma_algebra_sort(sigalg) )
