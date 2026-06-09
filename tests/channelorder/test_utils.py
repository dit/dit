"""
Unit tests for dit.channelorder._utils.
"""

import numpy as np
import pytest

from dit import Distribution
from dit.channelorder._utils import (
    channel_matrix,
    channels_from_joint,
    compose_channels,
    reverse_channels_from_joint,
)
from dit.exceptions import ditException


def bsc(p):
    """Binary symmetric channel with crossover probability *p*."""
    return np.array([[1 - p, p], [p, 1 - p]])


class TestChannelMatrix:
    def test_2d_array_passthrough(self):
        mat = bsc(0.1)
        out = channel_matrix(mat)
        assert out.shape == (2, 2)
        assert np.allclose(out, mat)

    def test_conditional_distribution(self):
        arr = np.array([[0.8, 0.2], [0.3, 0.7]])
        cond = Distribution.from_array(arr, ["X", "Y"], [[0, 1], [0, 1]], given_vars={"X"}, free_vars={"Y"})
        assert cond.is_conditional()
        assert np.allclose(channel_matrix(cond), arr)

    def test_list_of_distributions(self):
        joint = Distribution(["00", "01", "10", "11"], [0.4, 0.1, 0.2, 0.3])
        _, cdists = joint.condition_on([0], rvs=[1])
        out = channel_matrix(cdists)
        assert out.shape == (2, 2)
        assert np.allclose(out.sum(axis=1), 1.0)

    def test_nonconditional_distribution_raises(self):
        joint = Distribution(["00", "11"], [0.5, 0.5])
        assert not joint.is_conditional()
        with pytest.raises(ditException):
            channel_matrix(joint)

    def test_unrecognized_input_raises(self):
        # 1D array is neither a 2D matrix, a Distribution, nor a list.
        with pytest.raises(ditException):
            channel_matrix(np.array([0.5, 0.5]))


def _markov_joint():
    """P(S, Z, Y) = P(S) P(Z|S) P(Y|Z), a genuine S-Z-Y Markov chain."""
    ps = np.array([0.5, 0.5])
    pz_given_s = bsc(0.2)
    py_given_z = bsc(0.3)
    joint = np.zeros((2, 2, 2))
    for s in range(2):
        for z in range(2):
            for y in range(2):
                joint[s, z, y] = ps[s] * pz_given_s[s, z] * py_given_z[z, y]
    outcomes = [(s, z, y) for s in range(2) for z in range(2) for y in range(2)]
    return Distribution(outcomes, joint.ravel(), rv_names=["S", "Z", "Y"])


class TestChannelsFromJoint:
    def test_forward_channels(self):
        d = _markov_joint()
        kappa, mu, pi_s = channels_from_joint(d, ["S"], ["Y"], ["Z"])
        assert kappa.shape == (2, 2)  # P(Y|S)
        assert mu.shape == (2, 2)  # P(Z|S)
        assert pi_s.shape == (2,)
        assert np.allclose(kappa.sum(axis=1), 1.0)
        assert np.allclose(mu.sum(axis=1), 1.0)
        assert np.allclose(pi_s, [0.5, 0.5])


class TestReverseChannelsFromJoint:
    def test_reverse_channels(self):
        d = _markov_joint()
        kappa_bar, mu_bar, pi_y, pi_z = reverse_channels_from_joint(d, ["S"], ["Y"], ["Z"])
        assert kappa_bar.shape == (2, 2)  # P(S|Y)
        assert mu_bar.shape == (2, 2)  # P(S|Z)
        assert pi_y.shape == (2,)
        assert pi_z.shape == (2,)
        assert np.allclose(kappa_bar.sum(axis=1), 1.0)
        assert np.allclose(mu_bar.sum(axis=1), 1.0)
        assert np.isclose(pi_y.sum(), 1.0)
        assert np.isclose(pi_z.sum(), 1.0)


class TestComposeChannels:
    def test_identity_preprocess_is_noop(self):
        post = bsc(0.1)
        pre = np.eye(2)
        assert np.allclose(compose_channels(post, pre), post)

    def test_bsc_composition(self):
        a, b = 0.1, 0.2
        composed = compose_channels(bsc(a), bsc(b))
        # Two cascaded BSCs form a BSC with crossover a + b - 2ab.
        assert np.allclose(composed, bsc(a + b - 2 * a * b))
