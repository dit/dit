"""
Tests for dit.pid.lattice.
"""

from dit.pid.lattice import least_upper_bound, pid_lattice


def test_lub():
    """
    Test least upper bound
    """
    lattice = pid_lattice(((0,), (1,)))
    lub = least_upper_bound(lattice, [((0,), (1,)), ((0,),)])
    assert lub == ((0, 1),)

