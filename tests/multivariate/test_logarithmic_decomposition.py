"""
Tests for the logarithmic decomposition module.

Validates against known examples from:
  [1] Down & Mediano, arXiv:2409.03732 (LD measure space)
  [2] Down & Mediano, arXiv:2409.04845 (algebraic / fixed-parity)
"""

import pytest
import numpy as np

import dit
from dit.multivariate.logarithmic_decomposition import (
    LogarithmicDecomposition,
    logarithmic_decomposition,
    _loss,
    _interior_loss,
)
from dit.shannon import entropy as shannon_entropy
from dit.multivariate import coinformation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _close(a, b, tol=1e-9):
    return abs(a - b) < tol


# ---------------------------------------------------------------------------
# Low-level: loss and interior loss
# ---------------------------------------------------------------------------

class TestLoss:
    """Tests for the total loss L(p1,...,pn)."""

    def test_single_prob(self):
        assert _loss((0.5,)) == 0.0

    def test_empty(self):
        assert _loss(()) == 0.0

    def test_uniform_binary(self):
        # L(0.5, 0.5) = 1*log2(1) - 2*(0.5*log2(0.5)) = 0 + 1 = 1.0
        assert _close(_loss((0.5, 0.5)), 1.0)

    def test_equals_entropy_when_sum_one(self):
        probs = (0.25, 0.25, 0.25, 0.25)
        H = -sum(p * np.log2(p) for p in probs)
        assert _close(_loss(probs), H)

    def test_non_negative(self):
        for _ in range(50):
            n = np.random.randint(2, 6)
            probs = tuple(np.random.dirichlet(np.ones(n)))
            assert _loss(probs) >= -1e-15

    def test_zero_prob(self):
        assert _close(_loss((0.5, 0.0, 0.5)), _loss((0.5, 0.5)))


class TestInteriorLoss:
    """Tests for the interior loss mu(p1,...,pn)."""

    def test_degree_2_positive(self):
        mu = _interior_loss((0.3, 0.7))
        assert mu > 0

    def test_degree_3_negative(self):
        mu = _interior_loss((0.2, 0.3, 0.5))
        assert mu < 0

    def test_degree_4_positive(self):
        mu = _interior_loss((0.1, 0.2, 0.3, 0.4))
        assert mu > 0

    def test_sign_alternates(self):
        """(-1)^n * mu >= 0 for all atoms of degree n."""
        for _ in range(30):
            n = np.random.randint(2, 7)
            probs = tuple(np.random.dirichlet(np.ones(n)))
            mu = _interior_loss(probs)
            assert (-1)**n * mu >= -1e-12

    def test_magnitude_decreases(self):
        """Corollary 19: |mu(p1,...,pn,tau)| < |mu(p1,...,pn-1)| for n>=3."""
        probs_base = (0.2, 0.3)
        mu_base = abs(_interior_loss(probs_base))
        for tau in [0.01, 0.1, 0.2, 0.5]:
            mu_ext = abs(_interior_loss(probs_base + (tau,)))
            assert mu_ext < mu_base + 1e-12

    def test_degree_2_equals_loss(self):
        probs = (0.3, 0.7)
        assert _close(_interior_loss(probs), _loss(probs))


# ---------------------------------------------------------------------------
# LogarithmicDecomposition: basic structure
# ---------------------------------------------------------------------------

class TestStructure:

    @pytest.fixture
    def xor_ld(self):
        d = dit.example_dists.Xor()
        return LogarithmicDecomposition(d)

    def test_atom_count(self, xor_ld):
        # XOR has 4 outcomes -> 2^4 - 4 - 1 = 11 atoms
        assert len(xor_ld.atoms) == 11

    def test_atom_degrees(self, xor_ld):
        degs = {xor_ld.degree(a) for a in xor_ld.atoms}
        assert degs == {2, 3, 4}

    def test_omega_size(self, xor_ld):
        assert len(xor_ld.omega) == 4

    def test_repr(self, xor_ld):
        r = repr(xor_ld)
        assert "LogarithmicDecomposition" in r
        assert "atoms=11" in r


# ---------------------------------------------------------------------------
# Entropy consistency
# ---------------------------------------------------------------------------

class TestEntropyConsistency:
    """H(X) via LD must match dit.shannon.entropy."""

    def _check_entropy(self, dist, rvs=None):
        ld = LogarithmicDecomposition(dist)
        h_ld = ld.entropy(rvs)
        if rvs is None:
            rvs_shannon = None
        else:
            rvs_shannon = rvs
        h_shannon = shannon_entropy(dist, rvs_shannon)
        assert _close(h_ld, h_shannon), f"LD={h_ld}, Shannon={h_shannon}"

    def test_xor_joint(self):
        self._check_entropy(dit.example_dists.Xor())

    def test_xor_single_var(self):
        d = dit.example_dists.Xor()
        for i in range(d.outcome_length()):
            self._check_entropy(d, [i])

    def test_binary_symmetric(self):
        d = dit.Distribution([(0, 0), (0, 1), (1, 0), (1, 1)],
                             [0.25, 0.25, 0.25, 0.25])
        self._check_entropy(d)
        self._check_entropy(d, [0])
        self._check_entropy(d, [1])

    def test_or_gate(self):
        d = dit.Distribution(
            [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)],
            [0.25, 0.25, 0.25, 0.25],
        )
        self._check_entropy(d)
        for i in range(3):
            self._check_entropy(d, [i])

    def test_dyadic(self):
        d = dit.example_dists.dyadic
        self._check_entropy(d)
        for i in range(d.outcome_length()):
            self._check_entropy(d, [i])

    def test_triadic(self):
        d = dit.example_dists.triadic
        self._check_entropy(d)
        for i in range(d.outcome_length()):
            self._check_entropy(d, [i])


# ---------------------------------------------------------------------------
# Co-information consistency
# ---------------------------------------------------------------------------

class TestCoinformationConsistency:
    """Co-information via LD must match dit.multivariate.coinformation."""

    def _check_coinfo(self, dist, rvs_list=None):
        ld = LogarithmicDecomposition(dist)
        ci_ld = ld.coinformation(rvs_list)
        ci_dit = coinformation(dist, rvs_list)
        assert _close(ci_ld, ci_dit), f"LD={ci_ld}, dit={ci_dit}"

    def test_xor_3way(self):
        d = dit.example_dists.Xor()
        self._check_coinfo(d, [[0], [1], [2]])

    def test_xor_pairwise(self):
        d = dit.example_dists.Xor()
        self._check_coinfo(d, [[0], [1]])
        self._check_coinfo(d, [[0], [2]])
        self._check_coinfo(d, [[1], [2]])

    def test_or_gate(self):
        d = dit.Distribution(
            [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)],
            [0.25, 0.25, 0.25, 0.25],
        )
        self._check_coinfo(d, [[0], [1], [2]])
        self._check_coinfo(d, [[0], [1]])

    def test_dyadic(self):
        d = dit.example_dists.dyadic
        self._check_coinfo(d, [[0], [1], [2]])
        self._check_coinfo(d, [[0], [1]])

    def test_triadic(self):
        d = dit.example_dists.triadic
        self._check_coinfo(d, [[0], [1], [2]])

    def test_independent_binary(self):
        d = dit.Distribution([(0, 0), (0, 1), (1, 0), (1, 1)],
                             [0.25, 0.25, 0.25, 0.25])
        self._check_coinfo(d, [[0], [1]])


# ---------------------------------------------------------------------------
# XOR gate specifics
# ---------------------------------------------------------------------------

class TestXorGate:

    @pytest.fixture
    def xor_ld(self):
        return LogarithmicDecomposition(dit.example_dists.Xor())

    def test_coinformation_minus_one(self, xor_ld):
        ci = xor_ld.coinformation([[0], [1], [2]])
        assert _close(ci, -1.0)

    def test_pairwise_mi_xz_zero(self, xor_ld):
        mi = xor_ld.mutual_information([[0], [2]])
        assert _close(mi, 0.0)

    def test_pairwise_mi_yz_zero(self, xor_ld):
        mi = xor_ld.mutual_information([[1], [2]])
        assert _close(mi, 0.0)


# ---------------------------------------------------------------------------
# OR gate co-information structure (Example 25 from paper 2)
# ---------------------------------------------------------------------------

class TestOrGate:

    @pytest.fixture
    def or_ld(self):
        d = dit.Distribution(
            [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)],
            [0.25, 0.25, 0.25, 0.25],
        )
        return LogarithmicDecomposition(d)

    def test_coinformation_negative(self, or_ld):
        ci = or_ld.coinformation([[0], [1], [2]])
        assert ci < 0


# ---------------------------------------------------------------------------
# Sign property (Theorem 18 of paper 1)
# ---------------------------------------------------------------------------

class TestSignProperty:

    def test_all_atoms_correct_sign(self):
        d = dit.Distribution(
            [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)],
            [0.25, 0.25, 0.25, 0.25],
        )
        ld = LogarithmicDecomposition(d)
        for atom in ld.atoms:
            mu = ld.measure(atom)
            n = ld.degree(atom)
            if abs(mu) > 1e-15:
                assert (-1)**n * mu > -1e-15, (
                    f"atom {atom} (deg {n}): mu={mu}, sign wrong"
                )

    def test_sign_xor(self):
        d = dit.example_dists.Xor()
        ld = LogarithmicDecomposition(d)
        for atom in ld.atoms:
            mu = ld.measure(atom)
            n = ld.degree(atom)
            if abs(mu) > 1e-15:
                assert (-1)**n * mu > -1e-15


# ---------------------------------------------------------------------------
# Dyadic vs Triadic distinction (Theorem 63 of paper 1)
# ---------------------------------------------------------------------------

class TestDyadicTriadic:

    def test_r2_distinguishes(self):
        """
        mu(R2(Delta_X ∩ Delta_Y ∩ Delta_Z)) = 0 for dyadic, = 1 for triadic.
        """
        for label, dist, expected in [
            ("dyadic", dit.example_dists.dyadic, 0.0),
            ("triadic", dit.example_dists.triadic, 1.0),
        ]:
            ld = LogarithmicDecomposition(dist)
            co_content = ld.content([0]) & ld.content([1]) & ld.content([2])
            r2 = ld.r_n(co_content, 2)
            val = ld.measure_set(r2)
            assert _close(val, expected), (
                f"{label}: R2 measure = {val}, expected {expected}"
            )


# ---------------------------------------------------------------------------
# Ideal generators
# ---------------------------------------------------------------------------

class TestGenerators:

    def test_generators_simple(self):
        a = frozenset({1, 2})
        b = frozenset({1, 2, 3})
        c = frozenset({3, 4})
        content = {a, b, c}
        gens = LogarithmicDecomposition.generators(content)
        assert gens == {a, c}

    def test_entropy_content_generated_by_2atoms(self):
        """Theorem 18 of paper 2: Delta(X) is generated by degree-2 atoms."""
        d = dit.Distribution(
            [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)],
            [0.25, 0.25, 0.25, 0.25],
        )
        ld = LogarithmicDecomposition(d)
        for rv_idx in range(d.outcome_length()):
            c = ld.content([rv_idx])
            if not c:
                continue
            gens = ld.generators(c)
            for g in gens:
                assert ld.degree(g) == 2, f"Generator {g} has degree {ld.degree(g)}"

    def test_mi_generated_by_2atoms(self):
        """Theorem 22 of paper 2: I(X;Y) content is a degree-2 ideal."""
        d = dit.Distribution(
            [(0, 0), (0, 1), (1, 0), (1, 1)],
            [0.1, 0.2, 0.3, 0.4],
        )
        ld = LogarithmicDecomposition(d)
        mi_content = ld.content([0]) & ld.content([1])
        if mi_content:
            gens = ld.generators(mi_content)
            for g in gens:
                assert ld.degree(g) == 2


# ---------------------------------------------------------------------------
# R_n filter
# ---------------------------------------------------------------------------

class TestRn:

    def test_r_n_basic(self):
        a2 = frozenset({1, 2})
        a3 = frozenset({1, 2, 3})
        a3b = frozenset({3, 4, 5})
        content = {a2, a3, a3b}
        r2 = LogarithmicDecomposition.r_n(content, 2)
        assert a2 in r2
        assert a3 in r2  # contains {1,2} which is degree 2
        assert a3b not in r2

    def test_r_n_empty(self):
        a3 = frozenset({1, 2, 3})
        content = {a3}
        r2 = LogarithmicDecomposition.r_n(content, 2)
        assert len(r2) == 0


# ---------------------------------------------------------------------------
# Atom table
# ---------------------------------------------------------------------------

class TestAtomTable:

    def test_table_rows(self):
        d = dit.Distribution([(0, 0), (0, 1), (1, 0), (1, 1)],
                             [0.25, 0.25, 0.25, 0.25])
        ld = LogarithmicDecomposition(d)
        table = ld.atom_table()
        # 4 outcomes -> 11 atoms
        assert len(table) == 11
        for row in table:
            assert 'atom' in row
            assert 'degree' in row
            assert 'measure' in row


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

class TestConvenience:

    def test_logarithmic_decomposition_func(self):
        d = dit.example_dists.Xor()
        ld = logarithmic_decomposition(d)
        assert isinstance(ld, LogarithmicDecomposition)
        assert _close(ld.coinformation([[0], [1], [2]]), -1.0)
