"""
Unit tests for dit.channelorder.orderings.

Uses concrete channel examples from [Banerjee 2025] and standard
information-theoretic constructions.
"""

import numpy as np

from dit.channelorder.orderings import (
    blackwell_order_joint,
    is_input_degraded,
    is_less_noisy,
    is_more_capable,
    is_output_degraded,
    is_shannon_included,
)

# ── Helper channel constructors ────────────────────────────────────────────


def bsc(p):
    """Binary symmetric channel with crossover probability *p*."""
    return np.array([[1 - p, p], [p, 1 - p]])


def bec(eps):
    """Binary erasure channel with erasure probability *eps*.  Output alphabet {0, 1, e}."""
    return np.array([[1 - eps, 0, eps], [0, 1 - eps, eps]])


def identity(n):
    """Noiseless identity channel on *n* symbols."""
    return np.eye(n)


def constant(n_in, n_out):
    """Constant channel: every input maps to a uniform output."""
    return np.ones((n_in, n_out)) / n_out


# ── Blackwell (output-degraded) order ──────────────────────────────────────


class TestOutputDegraded:
    def test_identity_dominates_bsc(self):
        assert is_output_degraded(identity(2), bsc(0.3))

    def test_identity_dominates_bec(self):
        assert is_output_degraded(identity(2), bec(0.4))

    def test_constant_dominated_by_anything(self):
        assert is_output_degraded(bsc(0.3), constant(2, 2))

    def test_constant_does_not_dominate_bsc(self):
        assert not is_output_degraded(constant(2, 2), bsc(0.3))

    def test_self_degraded(self):
        ch = bsc(0.2)
        assert is_output_degraded(ch, ch)

    def test_bsc_composition(self):
        # BSC(p1) o BSC(p2) = BSC(p1+p2-2*p1*p2)
        # So BSC(0.1) >= BSC(0.1+0.2-2*0.1*0.2) = BSC(0.26)
        assert is_output_degraded(bsc(0.1), bsc(0.26))

    def test_bsc_not_degraded_to_less_noisy(self):
        # BSC(0.3) does not output-degrade to BSC(0.1)
        assert not is_output_degraded(bsc(0.3), bsc(0.1))

    def test_bec_hierarchy(self):
        # BEC(eps1) >= BEC(eps2) iff eps1 <= eps2
        assert is_output_degraded(bec(0.2), bec(0.5))
        assert not is_output_degraded(bec(0.5), bec(0.2))

    def test_alias(self):
        from dit.channelorder.orderings import is_blackwell_sufficient

        assert is_blackwell_sufficient is is_output_degraded


# ── Input-degraded order ───────────────────────────────────────────────────


class TestInputDegraded:
    def test_trivial(self):
        # If kappa_bar has rows that are rows of mu_bar, it's input-degraded
        mu_bar = np.array([[0.8, 0.2], [0.3, 0.7], [0.5, 0.5]])
        kappa_bar = np.array([[0.5, 0.5]])  # row of mu_bar
        assert is_input_degraded(mu_bar, kappa_bar)

    def test_convex_hull(self):
        mu_bar = np.array([[1.0, 0.0], [0.0, 1.0]])
        # Any distribution on {0,1} is in conv of rows of mu_bar
        kappa_bar = np.array([[0.6, 0.4], [0.3, 0.7]])
        assert is_input_degraded(mu_bar, kappa_bar)

    def test_outside_hull(self):
        mu_bar = np.array([[0.8, 0.1, 0.1], [0.7, 0.2, 0.1]])
        kappa_bar = np.array([[0.1, 0.1, 0.8]])
        assert not is_input_degraded(mu_bar, kappa_bar)


# ── BSC/BEC broadcast channel (Example 2 from paper) ──────────────────────


class TestBscBecHierarchy:
    """
    Example 2: broadcast channel with BSC(p) and BEC(eps).
    For p=0.1:
      - eps <= 2p=0.2: Y is output-degraded from Z
      - 2p < eps <= 4p(1-p)=0.36: Z less noisy than Y, not output-degraded
      - 4p(1-p) < eps <= h(p)≈0.469: Z more capable than Y, not less noisy
      - h(p) < eps: none of the orderings hold
    """

    p = 0.1

    def test_regime_1_output_degraded(self):
        eps = 0.15  # < 2*0.1 = 0.2
        assert is_output_degraded(bec(eps), bsc(self.p))

    def test_regime_2_less_noisy_not_output_degraded(self):
        eps = 0.3  # 0.2 < 0.3 < 0.36
        assert not is_output_degraded(bec(eps), bsc(self.p))
        assert is_less_noisy(bec(eps), bsc(self.p))

    def test_regime_3_more_capable_not_less_noisy(self):
        eps = 0.4  # 0.36 < 0.4 < 0.469
        assert not is_less_noisy(bec(eps), bsc(self.p))
        assert is_more_capable(bec(eps), bsc(self.p))

    def test_regime_4_nothing(self):
        eps = 0.9  # > h(0.1) ≈ 0.469
        assert not is_more_capable(bec(eps), bsc(self.p))


# ── More capable order ─────────────────────────────────────────────────────


class TestMoreCapable:
    def test_identity_more_capable(self):
        assert is_more_capable(identity(2), bsc(0.3))

    def test_noisier_not_more_capable(self):
        assert not is_more_capable(bsc(0.4), bsc(0.1))


# ── Less noisy order ──────────────────────────────────────────────────────


class TestLessNoisy:
    def test_identity_less_noisy(self):
        assert is_less_noisy(identity(2), bsc(0.3))

    def test_output_degraded_implies_less_noisy(self):
        # BSC(0.1) >= BSC(0.26) in Blackwell, so also less noisy
        assert is_less_noisy(bsc(0.1), bsc(0.26))


# ── Shannon inclusion ─────────────────────────────────────────────────────


class TestShannonInclusion:
    def test_output_degraded_implies_shannon(self):
        # Blackwell => Shannon when same input alphabet
        assert is_shannon_included(identity(2), bsc(0.3))

    def test_self_inclusion(self):
        ch = bsc(0.2)
        assert is_shannon_included(ch, ch)


# ── Joint distribution convenience ─────────────────────────────────────────


class TestBlackwellJoint:
    def test_markov_chain(self):
        from dit import Distribution

        # S-Z-Y Markov chain: Z is more informative about S than Y
        d = Distribution(
            [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)],
            [0.25, 0.15, 0.05, 0.05, 0.05, 0.05, 0.15, 0.25],
            rv_names=["S", "Z", "Y"],
        )
        # P(Y|S) should be output-degraded from P(Z|S) if S-Z-Y
        # Not necessarily true for arbitrary joint -- but this one is
        # constructed so that it holds.
        # Actually, let's build a proper Markov chain:
        import numpy as np

        ps = np.array([0.5, 0.5])
        pz_given_s = np.array([[0.8, 0.2], [0.2, 0.8]])
        py_given_z = np.array([[0.7, 0.3], [0.3, 0.7]])

        # P(S,Z,Y) = P(S) P(Z|S) P(Y|Z)
        joint = np.zeros((2, 2, 2))
        for s in range(2):
            for z in range(2):
                for y in range(2):
                    joint[s, z, y] = ps[s] * pz_given_s[s, z] * py_given_z[z, y]

        outcomes = []
        pmf = []
        for s in range(2):
            for z in range(2):
                for y in range(2):
                    outcomes.append((s, z, y))
                    pmf.append(joint[s, z, y])

        d = Distribution(outcomes, pmf, rv_names=["S", "Z", "Y"])
        assert blackwell_order_joint(d, ["S"], ["Y"], ["Z"])
