"""
Tests for dit.multivariate.union_information.
"""

import pytest

from dit import Distribution as D
from dit.multivariate import coinformation, entropy
from dit.multivariate.union_information import (
    intersection_entropy,
    synergistic_entropy,
    union_entropy,
    unique_entropy,
)


class TestUnionEntropy:
    def test_redundant_bits(self):
        d = D(["00", "11"], [1 / 2] * 2)
        assert union_entropy(d, [[0], [1]]) == pytest.approx(1.0)

    def test_independent_bits(self):
        d = D(["00", "01", "10", "11"], [1 / 4] * 4)
        assert union_entropy(d, [[0], [1]]) == pytest.approx(1.0)

    def test_correlated_bits(self):
        d = D(["00", "01", "10", "11"], [0.4, 0.1, 0.1, 0.4])
        assert union_entropy(d, [[0], [1]]) == pytest.approx(1.0)

    def test_trivariate_independent(self):
        d = D(["000", "001", "010", "011", "100", "101", "110", "111"], [1 / 8] * 8)
        assert union_entropy(d, [[0], [1], [2]]) == pytest.approx(1.0)

    def test_geq_marginals(self):
        d = D(["00", "01", "10", "11"], [0.4, 0.1, 0.1, 0.4])
        H0 = entropy(d, [0])
        H1 = entropy(d, [1])
        Hu = union_entropy(d, [[0], [1]])
        assert Hu >= H0 - 1e-10
        assert Hu >= H1 - 1e-10


class TestIntersectionEntropy:
    def test_redundant_bits(self):
        d = D(["00", "11"], [1 / 2] * 2)
        assert intersection_entropy(d, [[0], [1]]) == pytest.approx(1.0)

    def test_independent_bits(self):
        d = D(["00", "01", "10", "11"], [1 / 4] * 4)
        assert intersection_entropy(d, [[0], [1]]) == pytest.approx(1.0)

    def test_correlated_bits(self):
        d = D(["00", "01", "10", "11"], [0.4, 0.1, 0.1, 0.4])
        assert intersection_entropy(d, [[0], [1]]) == pytest.approx(1.0)

    def test_leq_marginals(self):
        d = D(["00", "01", "10", "11"], [0.4, 0.1, 0.1, 0.4])
        H0 = entropy(d, [0])
        H1 = entropy(d, [1])
        Hi = intersection_entropy(d, [[0], [1]])
        assert Hi <= H0 + 1e-10
        assert Hi <= H1 + 1e-10


class TestSynergisticEntropy:
    def test_redundant_bits(self):
        d = D(["00", "11"], [1 / 2] * 2)
        assert synergistic_entropy(d, [[0], [1]]) == pytest.approx(0.0)

    def test_independent_bits(self):
        d = D(["00", "01", "10", "11"], [1 / 4] * 4)
        assert synergistic_entropy(d, [[0], [1]]) == pytest.approx(1.0)

    def test_nonnegative(self):
        d = D(["00", "01", "10", "11"], [0.4, 0.1, 0.1, 0.4])
        assert synergistic_entropy(d, [[0], [1]]) >= -1e-10

    def test_trivariate(self):
        d = D(["000", "001", "010", "011", "100", "101", "110", "111"], [1 / 8] * 8)
        assert synergistic_entropy(d, [[0], [1], [2]]) == pytest.approx(2.0)


class TestUniqueEntropy:
    def test_redundant_bits(self):
        d = D(["00", "11"], [1 / 2] * 2)
        assert unique_entropy(d, [[0], [1]]) == pytest.approx(0.0)
        assert unique_entropy(d, [[1], [0]]) == pytest.approx(0.0)

    def test_independent_bits(self):
        d = D(["00", "01", "10", "11"], [1 / 4] * 4)
        assert unique_entropy(d, [[0], [1]]) == pytest.approx(0.0)
        assert unique_entropy(d, [[1], [0]]) == pytest.approx(0.0)

    def test_nonnegative(self):
        d = D(["00", "01", "10", "11"], [0.4, 0.1, 0.1, 0.4])
        assert unique_entropy(d, [[0], [1]]) >= -1e-10
        assert unique_entropy(d, [[1], [0]]) >= -1e-10

    def test_requires_two_groups(self):
        d = D(["00", "01", "10", "11"], [1 / 4] * 4)
        with pytest.raises(ValueError, match="exactly 2"):
            unique_entropy(d, [[0], [1], [0, 1]])


class TestDecompositions:
    """Verify the key identities from Finn & Lizier (2020)."""

    def test_joint_entropy_decomposition_bivariate(self):
        """H(X,Y) = H(XuY) + H(X\\Y) + H(Y\\X) + H(X+Y) [Eq. 40]"""
        d = D(["00", "01", "10", "11"], [0.4, 0.1, 0.1, 0.4])
        H = entropy(d)
        Hi = intersection_entropy(d, [[0], [1]])
        Huq0 = unique_entropy(d, [[0], [1]])
        Huq1 = unique_entropy(d, [[1], [0]])
        Hs = synergistic_entropy(d, [[0], [1]])
        assert pytest.approx(Hi + Huq0 + Huq1 + Hs) == H

    def test_mutual_info_identity(self):
        """I(X;Y) = H(XuY) - H(X+Y) [Eq. 43]"""
        d = D(["00", "01", "10", "11"], [0.4, 0.1, 0.1, 0.4])
        I = coinformation(d, [[0], [1]])
        Hi = intersection_entropy(d, [[0], [1]])
        Hs = synergistic_entropy(d, [[0], [1]])
        assert pytest.approx(Hi - Hs) == I

    def test_joint_entropy_decomposition_asymmetric(self):
        """Test decomposition on an asymmetric distribution."""
        d = D(["00", "01", "10", "11", "20", "21"], [0.1, 0.2, 0.15, 0.25, 0.05, 0.25])
        H = entropy(d)
        Hi = intersection_entropy(d, [[0], [1]])
        Huq0 = unique_entropy(d, [[0], [1]])
        Huq1 = unique_entropy(d, [[1], [0]])
        Hs = synergistic_entropy(d, [[0], [1]])
        assert pytest.approx(Hi + Huq0 + Huq1 + Hs) == H

    def test_mutual_info_identity_asymmetric(self):
        d = D(["00", "01", "10", "11", "20", "21"], [0.1, 0.2, 0.15, 0.25, 0.05, 0.25])
        I = coinformation(d, [[0], [1]])
        Hi = intersection_entropy(d, [[0], [1]])
        Hs = synergistic_entropy(d, [[0], [1]])
        assert pytest.approx(Hi - Hs) == I

    def test_union_plus_synergy_equals_joint(self):
        """H(X,Y) = H(XtY) + H(X+Y) [Eq. 39-40]"""
        d = D(["00", "01", "10", "11"], [0.4, 0.1, 0.1, 0.4])
        H = entropy(d)
        Hu = union_entropy(d, [[0], [1]])
        Hs = synergistic_entropy(d, [[0], [1]])
        assert pytest.approx(Hu + Hs) == H

    def test_conditioning_not_supported(self):
        d = D(["000", "001", "010", "011", "100", "101", "110", "111"], [1 / 8] * 8)
        with pytest.raises(NotImplementedError):
            union_entropy(d, [[0], [1]], [2])
        with pytest.raises(NotImplementedError):
            intersection_entropy(d, [[0], [1]], [2])
        with pytest.raises(NotImplementedError):
            synergistic_entropy(d, [[0], [1]], [2])
        with pytest.raises(NotImplementedError):
            unique_entropy(d, [[0], [1]], [2])
