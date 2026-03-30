"""
Unit tests for dit.channelorder.deficiency.
"""

import numpy as np
import pytest

from dit.channelorder.deficiency import (
    le_cam_deficiency,
    le_cam_distance,
    output_kl_deficiency,
    weighted_input_kl_deficiency,
    weighted_le_cam_deficiency,
    weighted_output_kl_deficiency,
    weighted_output_kl_deficiency_joint,
)


# ── Helpers ────────────────────────────────────────────────────────────────


def bsc(p):
    return np.array([[1 - p, p], [p, 1 - p]])


def bec(eps):
    return np.array([[1 - eps, 0, eps], [0, 1 - eps, eps]])


def identity(n):
    return np.eye(n)


def constant(n_in, n_out):
    return np.ones((n_in, n_out)) / n_out


# ── Le Cam deficiency ──────────────────────────────────────────────────────


class TestLeCamDeficiency:
    def test_zero_when_degraded(self):
        # Identity >= BSC, so delta(identity, bsc) = 0
        d = le_cam_deficiency(identity(2), bsc(0.3))
        assert d == pytest.approx(0.0, abs=1e-7)

    def test_positive_when_not_degraded(self):
        d = le_cam_deficiency(bsc(0.3), identity(2))
        assert d > 0

    def test_nonneg(self):
        d = le_cam_deficiency(bsc(0.2), bsc(0.3))
        assert d >= -1e-10

    def test_self_zero(self):
        d = le_cam_deficiency(bsc(0.2), bsc(0.2))
        assert d == pytest.approx(0.0, abs=1e-7)

    def test_constant_channel(self):
        d = le_cam_deficiency(bsc(0.3), constant(2, 2))
        assert d == pytest.approx(0.0, abs=1e-7)


class TestLeCamDistance:
    def test_symmetric(self):
        d1 = le_cam_distance(bsc(0.1), bsc(0.3))
        d2 = le_cam_distance(bsc(0.3), bsc(0.1))
        assert d1 == pytest.approx(d2, abs=1e-7)

    def test_zero_self(self):
        assert le_cam_distance(bsc(0.2), bsc(0.2)) == pytest.approx(0.0, abs=1e-7)

    def test_positive_for_different(self):
        assert le_cam_distance(bsc(0.1), bsc(0.4)) > 0


# ── Weighted Le Cam deficiency ─────────────────────────────────────────────


class TestWeightedLeCamDeficiency:
    def test_zero_when_degraded(self):
        pi = np.array([0.5, 0.5])
        d = weighted_le_cam_deficiency(identity(2), bsc(0.3), pi)
        assert d == pytest.approx(0.0, abs=1e-7)

    def test_nonneg(self):
        pi = np.array([0.3, 0.7])
        d = weighted_le_cam_deficiency(bsc(0.2), bsc(0.4), pi)
        assert d >= -1e-10


# ── Output KL deficiency ──────────────────────────────────────────────────


class TestOutputKLDeficiency:
    def test_zero_when_degraded(self):
        d = output_kl_deficiency(identity(2), bsc(0.3))
        assert d == pytest.approx(0.0, abs=1e-5)

    def test_positive_when_not_degraded(self):
        d = output_kl_deficiency(bsc(0.4), identity(2))
        assert d > 0

    def test_nonneg(self):
        d = output_kl_deficiency(bsc(0.1), bsc(0.3))
        assert d >= -1e-10


# ── Weighted output KL deficiency ─────────────────────────────────────────


class TestWeightedOutputKLDeficiency:
    def test_zero_when_degraded(self):
        pi = np.array([0.5, 0.5])
        d = weighted_output_kl_deficiency(identity(2), bsc(0.3), pi)
        assert d == pytest.approx(0.0, abs=1e-5)

    def test_nonneg(self):
        pi = np.array([0.4, 0.6])
        d = weighted_output_kl_deficiency(bsc(0.1), bsc(0.3), pi)
        assert d >= -1e-10

    def test_pinsker_bound(self):
        """Eq. 25: weighted_le_cam <= sqrt(ln2/2 * weighted_output_kl)."""
        mu = bsc(0.2)
        kappa = bsc(0.4)
        pi = np.array([0.5, 0.5])
        tv = weighted_le_cam_deficiency(mu, kappa, pi)
        kl = weighted_output_kl_deficiency(mu, kappa, pi)
        assert tv <= np.sqrt(np.log(2) / 2 * kl) + 1e-5


# ── Weighted input KL deficiency ──────────────────────────────────────────


class TestWeightedInputKLDeficiency:
    def test_nonneg(self):
        mu_bar = np.array([[0.7, 0.3], [0.2, 0.8]])
        kappa_bar = np.array([[0.6, 0.4], [0.4, 0.6]])
        pi_y = np.array([0.5, 0.5])
        d = weighted_input_kl_deficiency(mu_bar, kappa_bar, pi_y)
        assert d >= -1e-10

    def test_zero_when_input_degraded(self):
        # kappa_bar rows in convex hull of mu_bar rows
        mu_bar = np.array([[1.0, 0.0], [0.0, 1.0]])
        lam = np.array([[0.6, 0.4], [0.3, 0.7]])
        kappa_bar = lam @ mu_bar
        pi_y = np.array([0.5, 0.5])
        d = weighted_input_kl_deficiency(mu_bar, kappa_bar, pi_y)
        assert d == pytest.approx(0.0, abs=1e-4)


# ── Joint distribution convenience ────────────────────────────────────────


class TestJointConvenience:
    def test_weighted_output_kl_joint(self):
        from dit import Distribution
        # Build a simple joint where S-Z-Y is a Markov chain
        ps = np.array([0.5, 0.5])
        pz_given_s = np.array([[0.9, 0.1], [0.1, 0.9]])
        py_given_z = np.array([[0.8, 0.2], [0.2, 0.8]])

        outcomes = []
        pmf = []
        for s in range(2):
            for z in range(2):
                for y in range(2):
                    p = ps[s] * pz_given_s[s, z] * py_given_z[z, y]
                    outcomes.append((s, z, y))
                    pmf.append(p)

        d = Distribution(outcomes, pmf, rv_names=["S", "Z", "Y"])
        val = weighted_output_kl_deficiency_joint(d, ["S"], ["Y"], ["Z"])
        assert val == pytest.approx(0.0, abs=1e-4)
