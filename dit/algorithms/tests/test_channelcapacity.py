from __future__ import division

import pytest

import numpy as np

import dit

def BEC_joint(epsilon):
    """
    The joint distribution for the binary erase channel at channel capacity.

    Parameters
    ----------
    epsilon : float
        The noise level at which the input is erased.

    """
    pX = dit.Distribution(['0', '1'], [1/2, 1/2])
    pYgX0 = dit.Distribution(['0', '1', 'e'], [1 - epsilon, 0, epsilon])
    pYgX1 = dit.Distribution(['0', '1', 'e'], [0, 1 - epsilon, epsilon])
    pYgX = [pYgX0, pYgX1]
    pXY = dit.joint_from_factors(pX, pYgX, strict=False)
    return pXY

def test_channel_capacity_no_rvnames():
    epsilon = .3
    pXY = BEC_joint(epsilon)
    pX, pYgX = pXY.condition_on([0])
    cc, pXopt = dit.algorithms.channel_capacity(pYgX, pX)

    # Verify channel capacity.
    assert np.allclose(cc, 1 - epsilon)

    # Verify maximizing distribution.
    assert pX.is_approx_equal(pXopt)

    # Verify joint distribution at channel capacity.
    pXYopt = dit.joint_from_factors(pXopt, pYgX)
    assert pXY.is_approx_equal(pXYopt)

def test_channel_capacity_rvnames():
    epsilon = .01
    pXY = BEC_joint(epsilon)
    pXY.set_rv_names('XY')
    pX, pYgX = pXY.condition_on('X')
    cc, pXopt = dit.algorithms.channel_capacity(pYgX, pX)

    # Verify channel capacity.
    assert np.allclose(cc, 1 - epsilon)

    # Verify maximizing distribution.
    assert pX.is_approx_equal(pXopt)

    # Verify joint distribution at channel capacity.
    pXYopt = dit.joint_from_factors(pXopt, pYgX)
    assert pXY.is_approx_equal(pXYopt)

def test_channel_capacity_array():
    epsilon = .3
    pXY = BEC_joint(epsilon)
    pX, pYgX = pXY.condition_on([0])
    cc, pXopt_pmf = dit.algorithms.channel_capacity(pYgX)

    # Verify channel capacity.
    assert np.allclose(cc, 1 - epsilon)

    # Verify maximizing distribution.
    assert np.allclose(pX.pmf, pXopt_pmf)

def test_bad_marginal():
    epsilon = .01
    pXY = BEC_joint(epsilon)
    pXY.set_rv_names('XY')
    pX, pYgX = pXY.condition_on('X')
    pX['0'] = 0
    # Now make its length disagree with the number of cdists.
    pX.make_sparse()
    with pytest.raises(dit.exceptions.ditException):
        dit.algorithms.channel_capacity(pYgX, pX)
