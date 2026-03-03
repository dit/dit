"""
Tests for dit.algorithms.channelcapacity
"""

import numpy as np
import pytest

from dit import Distribution, joint_from_factors
from dit.algorithms.channelcapacity import channel_capacity, channel_capacity_joint
from dit.cdisthelpers import cdist_array
from dit.exceptions import ditException


def BEC_joint(epsilon):
    """
    The joint distribution for the binary erase channel at channel capacity.

    Parameters
    ----------
    epsilon : float
        The noise level at which the input is erased.

    """
    pX = Distribution(["0", "1"], [1 / 2, 1 / 2])
    pYgX0 = Distribution(["0", "1", "e"], [1 - epsilon, 0, epsilon])
    pYgX1 = Distribution(["0", "1", "e"], [0, 1 - epsilon, epsilon])
    pYgX = [pYgX0, pYgX1]
    pXY = joint_from_factors(pX, pYgX, strict=False)
    return pXY


def test_channel_capacity_no_rvnames():
    epsilon = 0.3
    pXY = BEC_joint(epsilon)
    pX, pYgX = pXY.condition_on([0])
    cc, pXopt = channel_capacity(pYgX, pX)

    # Verify channel capacity.
    assert np.allclose(cc, 1 - epsilon)

    # Verify maximizing distribution.
    assert pX.is_approx_equal(pXopt)

    # Verify joint distribution at channel capacity.
    pXYopt = joint_from_factors(pXopt, pYgX)
    assert pXY.is_approx_equal(pXYopt)


def test_channel_capacity_rvnames():
    epsilon = 0.01
    pXY = BEC_joint(epsilon)
    pXY.set_rv_names("XY")
    pX, pYgX = pXY.condition_on(["X"])
    cc, pXopt = channel_capacity(pYgX, pX)

    # Verify channel capacity.
    assert np.allclose(cc, 1 - epsilon)

    # Verify maximizing distribution.
    assert pX.is_approx_equal(pXopt)

    # Verify joint distribution at channel capacity.
    pXYopt = joint_from_factors(pXopt, pYgX)
    assert pXY.is_approx_equal(pXYopt)


def test_channel_capacity_array1():
    epsilon = 0.3
    pXY = BEC_joint(epsilon)
    pX, pYgX = pXY.condition_on([0])
    cc, pXopt_pmf = channel_capacity(pYgX)

    # Verify channel capacity.
    assert np.allclose(cc, 1 - epsilon)

    # Verify maximizing distribution.
    assert np.allclose(pX.pmf, pXopt_pmf)


def test_channel_capacity_array2():
    epsilon = 0.3
    pXY = BEC_joint(epsilon)
    pX, pYgX = pXY.condition_on([0])
    pYgX = cdist_array(pYgX, base="linear", mode="dense")
    cc, pXopt_pmf = channel_capacity(pYgX, atol=1e-9, rtol=1e-9)

    # Verify channel capacity.
    assert np.allclose(cc, 1 - epsilon)

    # Verify maximizing distribution.
    assert np.allclose(pX.pmf, pXopt_pmf)


def test_bad_marginal():
    epsilon = 0.01
    pXY = BEC_joint(epsilon)
    pXY.set_rv_names("XY")
    pX, pYgX = pXY.condition_on(["X"])
    pX["0"] = 0
    # Now make its length disagree with the number of cdists.
    pX.make_sparse()
    with pytest.raises(ditException):
        channel_capacity(pYgX, pX)


def test_channel_capacity_joint1():
    """
    Test against a known value.
    """
    gm = Distribution(["00", "01", "10"], [1 / 3] * 3)
    cc = channel_capacity_joint(gm, [0], [1])
    assert cc == pytest.approx(0.3219280796196524)


def test_channel_capacity_joint2():
    """
    Test against a known value.
    """
    gm = Distribution(["00", "01", "10"], [1 / 3] * 3)
    m = Distribution(["0", "1"], [2 / 5, 3 / 5])
    _, marg = channel_capacity_joint(gm, [0], [1], marginal=True)
    assert marg.is_approx_equal(m, atol=1e-3)


# ── Distribution tests ─────────────────────────────────────────────────


def BEC_joint_xr(epsilon):
    """Build the BEC joint distribution as an Distribution."""
    pXY = BEC_joint(epsilon)
    pXY.set_rv_names("XY")
    return pXY


def test_channel_capacity_distribution():
    epsilon = 0.3
    pXY = BEC_joint_xr(epsilon)
    pX, pYgX = pXY.condition_on(["X"])
    cc, pXopt_pmf = channel_capacity(pYgX)

    assert np.allclose(cc, 1 - epsilon)
    assert np.allclose(pXopt_pmf, [0.5, 0.5])


def test_channel_capacity_distribution_with_marginal():
    epsilon = 0.3
    pXY = BEC_joint_xr(epsilon)
    pX, pYgX = pXY.condition_on(["X"])
    pX = pXY.marginal("X")
    cc, pXopt = channel_capacity(pYgX, pX)

    assert np.allclose(cc, 1 - epsilon)
    assert isinstance(pXopt, Distribution)
    assert np.allclose(pXopt.data.values, [0.5, 0.5])


def test_channel_capacity_distribution_not_conditional():
    pXY = BEC_joint_xr(0.3)
    with pytest.raises(ditException):
        channel_capacity(pXY)


def test_channel_capacity_joint_distribution():
    gm = Distribution(["00", "01", "10"], [1 / 3] * 3)
    gm.set_rv_names("XY")
    cc = channel_capacity_joint(gm, ["X"], ["Y"])
    assert cc == pytest.approx(0.3219280796196524)


def test_channel_capacity_joint_distribution_marginal():
    gm = Distribution(["00", "01", "10"], [1 / 3] * 3)
    gm.set_rv_names("XY")
    cc, marg = channel_capacity_joint(gm, ["X"], ["Y"], marginal=True)
    assert cc == pytest.approx(0.3219280796196524)
    assert isinstance(marg, Distribution)
    assert np.allclose(marg.data.values, [2 / 5, 3 / 5], atol=1e-3)
