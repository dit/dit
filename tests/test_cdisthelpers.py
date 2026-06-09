"""
Tests for dit.cdisthelpers.
"""

import pytest

import dit


def test_joint_from_factors():
    d = dit.example_dists.Xor()
    for i in range(3):
        pX, pYgX = d.condition_on([i])
        pXY = dit.joint_from_factors(pX, pYgX)
        assert pXY.is_approx_equal(d)


def test_joint_from_factors_rvname():
    d = dit.example_dists.Xor()
    d.pmf = dit.math.pmfops.jittered(d.pmf, 0.4)
    d.set_rv_names("XYZ")
    pY, pXZgY = d.condition_on(["Y"])

    pXYZ = dit.joint_from_factors(pY, pXZgY)
    assert pXYZ.is_approx_equal(d)


def test_joint_from_factors_rvname_collision():
    """When the conditional's rv name collides with the marginal's, it is
    suffixed with ``_Y`` to keep names unique."""
    mdist = dit.Distribution(["0", "1"], [0.5, 0.5])
    mdist.set_rv_names("A")
    c0 = dit.Distribution(["0", "1"], [0.8, 0.2])
    c0.set_rv_names("A")
    c1 = dit.Distribution(["0", "1"], [0.3, 0.7])
    c1.set_rv_names("A")

    d = dit.joint_from_factors(mdist, [c0, c1])
    assert d.get_rv_names() == ("A", "A_Y")


def test_bad_marginal():
    d = dit.example_dists.Xor()
    pY, pXZgY = d.condition_on([1])

    # Incompatible marginal
    pY = dit.Distribution(["0", "1", "2"], [0.25, 0.5, 0.25])
    with pytest.raises(dit.exceptions.ditException):
        dit.joint_from_factors(pY, pXZgY, strict=False)

    # Compatible marginal that is not trim.
    pY = dit.Distribution(["0", "1", "2"], [0.25, 0.75, 0])
    pY.make_dense()
    dit.joint_from_factors(pY, pXZgY, strict=False)

    # Compatible marginal that is not trim.
    # Note here that we have created a completely different distribution
    # where the outcomes for X are '0' and '2' now.
    pY = dit.Distribution(["0", "1", "2"], [0.25, 0, 0.75])
    pY.make_dense()
    dit.joint_from_factors(pY, pXZgY, strict=False)
