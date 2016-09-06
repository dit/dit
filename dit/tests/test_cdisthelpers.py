
import pytest

import numpy as np
import dit

def test_joint_from_factors():
    d = dit.example_dists.Xor()
    for i in range(3):
        pX, pYgX = d.condition_on([i])
        pXY = dit.joint_from_factors(pX, pYgX)
        assert pXY.is_approx_equal(d)

def test_joint_from_factors_rvname():
    d = dit.example_dists.Xor()
    d.pmf = dit.math.pmfops.jittered(d.pmf, .4)
    d.set_rv_names('XYZ')
    pY, pXZgY = d.condition_on('Y')

    # Standard
    pXYZ = dit.joint_from_factors(pY, pXZgY)
    assert pXYZ.is_approx_equal(d)

    # Now we make them have incompatible masks.
    pY._new_mask()
    with pytest.raises(dit.exceptions.ditException):
        dit.joint_from_factors(pY, pXZgY, strict=True)

    # Build out of order now.
    pYXZ = dit.joint_from_factors(pY, pXZgY, strict=False)
    # This is one instance where ['YXZ'] is not treated the same as 'YXZ'
    d2 = d.coalesce(['YXZ'], extract=True)
    assert pYXZ.is_approx_equal(d2)

def test_bad_marginal():
    d = dit.example_dists.Xor()
    pY, pXZgY = d.condition_on([1])

    # Incompatible marginal
    pY = dit.Distribution(['0', '1', '2'], [.25, .5, .25])
    with pytest.raises(dit.exceptions.ditException):
        dit.joint_from_factors(pY, pXZgY, strict=False)

    # Compatible marginal that is not trim.
    pY = dit.Distribution(['0', '1', '2'], [.25, .75, 0])
    pY.make_dense()
    pYXZ = dit.joint_from_factors(pY, pXZgY, strict=False)

    # Compatible marginal that is not trim.
    # Note here that we have created a completely different distribution
    # where the outcomes for X are '0' and '2' now.
    pY = dit.Distribution(['0', '1', '2'], [.25, 0, .75])
    pY.make_dense()
    pYXZ = dit.joint_from_factors(pY, pXZgY, strict=False)
