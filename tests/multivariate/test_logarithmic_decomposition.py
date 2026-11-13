"""
Tests for the logarithmic decomposition module.

Validates against known examples from:
  [1] Down & Mediano, arXiv:2409.03732 (LD measure space)
  [2] Down & Mediano, arXiv:2409.04845 (algebraic / fixed-parity)
  [3] Down & Mediano, arXiv:2305.07554 (ISIT 2023: regions, discernible sets, Gacs-Korner)
"""

import importlib
import itertools

import numpy as np
import pytest

import dit
from dit.algorithms import insert_meet
from dit.exceptions import ditException
from dit.multivariate import (
    coinformation,
    dual_total_correlation,
    gk_common_information,
    o_information,
    total_correlation,
)
from dit.multivariate.logarithmic_decomposition import (
    LogarithmicDecomposition,
    _interior_loss,
    _loss,
    logarithmic_decomposition,
)
from dit.profiles import ShannonPartition
from dit.shannon import conditional_entropy
from dit.shannon import entropy as shannon_entropy

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
            assert (-1) ** n * mu >= -1e-12

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
        rvs_shannon = None if rvs is None else rvs
        h_shannon = shannon_entropy(dist, rvs_shannon)
        assert _close(h_ld, h_shannon), f"LD={h_ld}, Shannon={h_shannon}"

    def test_xor_joint(self):
        self._check_entropy(dit.example_dists.Xor())

    def test_xor_single_var(self):
        d = dit.example_dists.Xor()
        for i in range(d.outcome_length()):
            self._check_entropy(d, [i])

    def test_binary_symmetric(self):
        d = dit.Distribution([(0, 0), (0, 1), (1, 0), (1, 1)], [0.25, 0.25, 0.25, 0.25])
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
        d = dit.Distribution([(0, 0), (0, 1), (1, 0), (1, 1)], [0.25, 0.25, 0.25, 0.25])
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
                assert (-1) ** n * mu > -1e-15, f"atom {atom} (deg {n}): mu={mu}, sign wrong"

    def test_sign_xor(self):
        d = dit.example_dists.Xor()
        ld = LogarithmicDecomposition(d)
        for atom in ld.atoms:
            mu = ld.measure(atom)
            n = ld.degree(atom)
            if abs(mu) > 1e-15:
                assert (-1) ** n * mu > -1e-15


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
            assert _close(val, expected), f"{label}: R2 measure = {val}, expected {expected}"


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
        d = dit.Distribution([(0, 0), (0, 1), (1, 0), (1, 1)], [0.25, 0.25, 0.25, 0.25])
        ld = LogarithmicDecomposition(d)
        table = ld.atom_table()
        # 4 outcomes -> 11 atoms
        assert len(table) == 11
        for row in table:
            assert "atom" in row
            assert "degree" in row
            assert "measure" in row


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


class TestConvenience:
    def test_logarithmic_decomposition_func(self):
        d = dit.example_dists.Xor()
        ld = logarithmic_decomposition(d)
        assert isinstance(ld, LogarithmicDecomposition)
        assert _close(ld.coinformation([[0], [1], [2]]), -1.0)


# ---------------------------------------------------------------------------
# Degenerate inputs, log-probability pmfs, argument normalisation, caching
# ---------------------------------------------------------------------------


def _skewed_pair():
    """Two dependent bits with distinct outcome probabilities."""
    return dit.Distribution([(0, 0), (0, 1), (1, 0), (1, 1)], [0.1, 0.2, 0.3, 0.4])


class TestDegenerateLoss:
    def test_zero_mass_loss(self):
        """A multiset of outcomes carrying no probability mass has no loss."""
        assert _loss((0.0, 0.0)) == 0.0
        assert _loss((0.0, 0.0, 0.0)) == 0.0

    def test_interior_loss_below_degree_two(self):
        """Fewer than two outcomes carry no interior loss."""
        assert _interior_loss(()) == 0.0
        assert _interior_loss((0.4,)) == 0.0

    def test_homogeneous_of_degree_one(self):
        """Scaling every probability by c scales L and mu by c."""
        probs = (0.1, 0.2, 0.3)
        for c in (0.5, 2.0, 7.0):
            scaled = tuple(c * p for p in probs)
            assert _loss(scaled) == pytest.approx(c * _loss(probs))
            assert _interior_loss(scaled) == pytest.approx(c * _interior_loss(probs))


class TestLossMethod:
    @pytest.fixture
    def ld(self):
        return LogarithmicDecomposition(_skewed_pair())

    def test_loss_of_omega_is_joint_entropy(self, ld):
        """L(Omega) is the joint entropy of a normalised distribution."""
        assert ld.loss(ld.omega) == pytest.approx(shannon_entropy(_skewed_pair()))

    def test_loss_of_pair(self, ld):
        """L({a, b}) = (pa + pb) * h(pa / (pa + pb)), with h the binary entropy."""
        q = 0.1 / (0.1 + 0.4)
        h = -(q * np.log2(q) + (1 - q) * np.log2(1 - q))
        assert ld.loss(frozenset({(0, 0), (1, 1)})) == pytest.approx(0.5 * h)

    def test_loss_is_sum_of_contained_atom_measures(self, ld):
        """L(S) is the total measure of the atoms contained in S (inverse Mobius relation)."""
        for subset in ld.atoms:
            contained = [atom for atom in ld.atoms if atom <= subset]
            assert ld.loss(subset) == pytest.approx(ld.measure_set(contained), abs=1e-12)

    def test_outcomes_outside_omega_carry_nothing(self, ld):
        """Unknown outcomes add no loss, and atoms containing them have zero measure."""
        a, b, unknown = (0, 0), (1, 1), (2, 2)
        assert ld.loss(frozenset({a})) == 0.0
        assert ld.loss(frozenset({unknown, (3, 3)})) == 0.0
        assert ld.loss(frozenset({a, b, unknown})) == pytest.approx(ld.loss(frozenset({a, b})))
        assert ld.measure(frozenset({a, unknown})) == 0.0


class TestLogPmf:
    @pytest.mark.parametrize("base", [2, "e"])
    def test_log_pmf_matches_linear(self, base):
        """A log-probability pmf decomposes exactly like the linear one, in bits."""
        d = _skewed_pair()
        d_log = d.copy(base=base)
        assert d_log.is_log()
        ld, ld_log = LogarithmicDecomposition(d), LogarithmicDecomposition(d_log)
        assert ld_log.omega == ld.omega
        for atom in ld.atoms:
            assert ld_log.measure(atom) == pytest.approx(ld.measure(atom), abs=1e-12)
        assert ld_log.entropy() == pytest.approx(shannon_entropy(d))
        assert ld_log.entropy([0]) == pytest.approx(shannon_entropy(d, [0]))
        assert ld_log.coinformation() == pytest.approx(coinformation(d), abs=1e-12)


class TestMeasureCache:
    def test_measure_is_memoised(self, monkeypatch):
        """Repeated measure() calls reuse the cached value instead of recomputing it."""
        module = importlib.import_module(LogarithmicDecomposition.__module__)
        calls = []
        real_interior_loss = module._interior_loss

        def counting_interior_loss(probs):
            calls.append(probs)
            return real_interior_loss(probs)

        monkeypatch.setattr(module, "_interior_loss", counting_interior_loss)
        ld = LogarithmicDecomposition(dit.example_dists.Xor())
        atom = frozenset(ld.omega)
        first = ld.measure(atom)
        assert ld.measure(atom) == first
        assert len(calls) == 1

        # every atom is computed once, however many quantities use it
        h = ld.entropy()
        assert ld.entropy() == h
        assert ld.coinformation() == pytest.approx(-1.0)
        assert len(calls) == len(ld.atoms)


class TestContentArguments:
    @pytest.fixture
    def ld(self):
        return LogarithmicDecomposition(dit.example_dists.Xor())

    def test_single_index(self, ld):
        """A bare variable index is treated as a one-variable list."""
        for i in range(3):
            assert ld.content(i) == ld.content([i])
            assert ld.entropy(i) == pytest.approx(1.0)

    def test_grouped_indices_form_joint_variable(self, ld):
        """Nested groups are flattened into one joint variable, whose content is the union."""
        joint = ld.content([[0], [1]])
        assert joint == ld.content([0, 1])
        assert joint == ld.content([0]) | ld.content([1])
        assert ld.entropy([[0], [1]]) == pytest.approx(2.0)

    def test_default_content_is_every_atom(self, ld):
        """With distinct joint outcomes, the content of all variables is every atom."""
        assert ld.content() == ld.atoms


class TestConstantVariable:
    @pytest.fixture
    def dist(self):
        return dit.Distribution([(0, 0), (0, 1), (0, 2)], [0.2, 0.3, 0.5])

    def test_constant_variable_has_empty_content(self, dist):
        """A variable taking a single value induces no boundary: empty content, zero information."""
        ld = LogarithmicDecomposition(dist)
        assert ld.content([0]) == set()
        assert ld.content(0) == set()
        assert ld.generators(ld.content([0])) == set()
        assert ld.entropy([0]) == 0.0
        assert ld.mutual_information([[0], [1]]) == 0.0
        assert ld.entropy([1]) == pytest.approx(shannon_entropy(dist, [1]))


class TestZeroProbabilityOutcomes:
    def test_dense_matches_sparse(self):
        """Zero-probability outcomes contribute nothing to any information quantity."""
        sparse = dit.Distribution([(0, 0), (1, 1)], [0.5, 0.5])
        dense = sparse.copy()
        dense.make_dense()
        ld_sparse, ld_dense = LogarithmicDecomposition(sparse), LogarithmicDecomposition(dense)
        assert ld_dense.measure(frozenset({(0, 0), (0, 1)})) == 0.0
        assert ld_dense.entropy() == pytest.approx(ld_sparse.entropy())
        assert ld_dense.entropy([0]) == pytest.approx(1.0)
        assert ld_dense.coinformation() == pytest.approx(ld_sparse.coinformation())
        assert ld_dense.coinformation() == pytest.approx(1.0)


class TestDefaultGroups:
    def test_coinformation_defaults_to_single_variables(self):
        """With no groups, co-information is taken over every single variable."""
        xor = dit.example_dists.Xor()
        ld = LogarithmicDecomposition(xor)
        assert ld.coinformation() == pytest.approx(-1.0)
        assert ld.coinformation() == pytest.approx(ld.coinformation([[0], [1], [2]]))

        pair = _skewed_pair()
        assert LogarithmicDecomposition(pair).coinformation() == pytest.approx(coinformation(pair), abs=1e-12)

    def test_atom_table_with_explicit_groups(self):
        """Membership columns follow the requested groups and their measures add up to entropies."""
        d = _skewed_pair()
        ld = LogarithmicDecomposition(d)
        table = ld.atom_table([[0], [0, 1]])
        assert len(table) == len(ld.atoms) == 11
        assert [row["degree"] for row in table] == sorted(row["degree"] for row in table)
        content_0 = ld.content([0])
        for row in table:
            assert set(row) == {"atom", "degree", "measure", "Delta([0])", "Delta([0, 1])"}
            assert row["degree"] == len(row["atom"])
            assert row["measure"] == ld.measure(row["atom"])
            assert row["Delta([0])"] == (row["atom"] in content_0)
        h_0 = sum(row["measure"] for row in table if row["Delta([0])"])
        h_01 = sum(row["measure"] for row in table if row["Delta([0, 1])"])
        assert h_0 == pytest.approx(shannon_entropy(d, [0]))
        assert h_01 == pytest.approx(shannon_entropy(d))


# ---------------------------------------------------------------------------
# Helpers for the I-diagram, discernibility and formal-sum tests [3]
# ---------------------------------------------------------------------------

# Bell numbers: the number of partitions of an n-element set.
BELL = {1: 1, 2: 2, 3: 5, 4: 15, 5: 52, 6: 203, 7: 877}


def _scalar_dist(n, rng=None):
    """One numerical variable on the outcomes 0, ..., n - 1, which dit stores bare."""
    pmf = np.full(n, 1 / n) if rng is None else rng.dirichlet(np.ones(n))
    return dit.Distribution(list(range(n)), pmf)


def _random_sparse(rng, cards, size):
    """A distribution on `size` random outcomes of the product of the alphabets `cards`."""
    outcomes = list(itertools.product(*(range(c) for c in cards)))
    keep = sorted(rng.choice(len(outcomes), size=size, replace=False))
    return dit.Distribution([outcomes[i] for i in keep], rng.dirichlet(np.ones(size)))


def _random_blocky(rng, cards, size):
    """
    Like _random_sparse, but every outcome lies in one of a few random blocks:
    each variable's values are split among the blocks, so the variables often
    share a nontrivial common part.
    """
    k = int(rng.integers(2, min(cards) + 1))
    labels = [rng.permutation(np.arange(c) % k) for c in cards]
    outcomes = [
        o
        for o in itertools.product(*(range(c) for c in cards))
        if len({int(labels[i][v]) for i, v in enumerate(o)}) == 1
    ]
    size = min(size, len(outcomes))
    keep = sorted(rng.choice(len(outcomes), size=size, replace=False))
    return dit.Distribution([outcomes[i] for i in keep], rng.dirichlet(np.ones(size)))


def _with_random_bit(rng, d):
    """Append a variable that is a random binary function of the outcome."""
    bits = {o: int(rng.integers(2)) for o in d.outcomes}
    return dit.insert_rvf(d, lambda o: (bits[o],))


def _sorted_atoms(ld):
    """The atoms in a fixed order, so that seeded random choices are reproducible."""
    return sorted(ld.atoms, key=lambda atom: (len(atom), sorted(atom)))


def _set_partitions(elements):
    """Every partition of `elements`, each a frozenset of frozenset blocks."""
    elements = list(elements)
    if not elements:
        yield frozenset()
        return
    first, rest = elements[0], elements[1:]
    for smaller in _set_partitions(rest):
        blocks = list(smaller)
        yield frozenset([*blocks, frozenset([first])])
        for i, block in enumerate(blocks):
            yield frozenset([*blocks[:i], block | {first}, *blocks[i + 1 :]])


def _crossing_atoms(atoms, partition):
    """The atoms meeting two blocks of `partition`, computed without the class."""
    block_of = {o: i for i, block in enumerate(partition) for o in block}
    return frozenset(atom for atom in atoms if len({block_of[o] for o in atom}) > 1)


def _all_contents(ld):
    """Map every partition of Omega to its content."""
    return {p: _crossing_atoms(ld.atoms, p) for p in _set_partitions(sorted(ld.omega))}


def _finest_inside(contents, atom_set):
    """
    By exhaustive search, the partition with the largest content inside `atom_set`.

    Also checks that this content contains every other content inside `atom_set`,
    so the largest discernible subset exists and is unique.
    """
    inside = [p for p, content in contents.items() if content <= atom_set]
    finest = max(inside, key=lambda p: len(contents[p]))
    assert all(contents[p] <= contents[finest] for p in inside)
    return finest


def _meet(partitions, omega):
    """The finest common coarsening: join any two outcomes sharing a block of any partition."""
    parent = {o: o for o in omega}

    def find(o):
        while parent[o] != o:
            o = parent[o]
        return o

    for partition in partitions:
        for block in partition:
            first, *rest = block
            for o in rest:
                parent[find(o)] = find(first)
    blocks = {}
    for o in omega:
        blocks.setdefault(find(o), set()).add(o)
    return frozenset(frozenset(block) for block in blocks.values())


def _common_refinement(p, q):
    return frozenset(a & b for a in p for b in q if a & b)


def _random_atom_sets(ld, contents, rng, count):
    """
    Atom sets of many shapes: independent random atoms, up-sets of random edges,
    contents with atoms removed or added, Boolean combinations of contents,
    degree bands, and down-sets. Only some are closed under supersets, and about
    half have a nonempty discernible part.
    """
    atoms = _sorted_atoms(ld)
    edges = [atom for atom in atoms if len(atom) == 2]
    content_list = list(contents.values())
    everything = frozenset(atoms)

    def random_content():
        return content_list[rng.integers(len(content_list))]

    def edge_upset(excluded):
        """The atoms containing an edge that survives exclusion with probability `excluded`."""
        chosen = [e for e in edges if rng.random() >= excluded]
        return {atom for atom in atoms if any(e <= atom for e in chosen)}

    def random_atoms(density):
        return {atom for atom in atoms if rng.random() < density}

    def without_some(subset):
        subset = list(subset)
        if not subset:
            return set()
        drop = rng.choice(len(subset), size=min(len(subset), int(rng.integers(1, 3))), replace=False)
        return set(subset) - {subset[i] for i in drop}

    sets = []
    for _ in range(count):
        kind = rng.integers(10)
        if kind == 0:
            subset = random_atoms(rng.choice([0.5, 0.9, 0.99]))
        elif kind == 1:
            subset = edge_upset(rng.choice([0.1, 0.25, 0.5]))
        elif kind == 2:
            subset = without_some(random_content())
        elif kind == 3:
            subset = set(random_content()) | random_atoms(0.3)
        elif kind == 4:
            c1, c2 = random_content(), random_content()
            subset = set([c1 & c2, c1 | c2, c1 - c2, c1 ^ c2, everything - c1][rng.integers(5)])
        elif kind == 5:
            subset = set(random_content() & random_content() & random_content())
        elif kind == 6:
            degrees = {k for k in range(2, len(ld.omega) + 1) if rng.random() < 0.5}
            subset = {atom for atom in atoms if len(atom) in degrees}
        elif kind == 7:
            subset = set(everything) - edge_upset(0.5)
        elif kind == 8:
            subset = without_some(random_content() | random_content())
        else:
            subset = set(random_content()) & edge_upset(0.15)
        sets.append(subset)
    return sets


# ---------------------------------------------------------------------------
# Omega is the support; single numerical variables
# ---------------------------------------------------------------------------


class TestSupport:
    def test_dense_distribution_uses_its_support(self):
        """make_dense adds zero-probability outcomes; they are not outcomes of Omega."""
        sparse = dit.Distribution([(0, 0), (1, 1)], [0.5, 0.5])
        dense = sparse.copy()
        dense.make_dense()
        assert len(dense.outcomes) == 4
        ld_sparse, ld_dense = LogarithmicDecomposition(sparse), LogarithmicDecomposition(dense)
        assert ld_dense.omega == ld_sparse.omega == frozenset({(0, 0), (1, 1)})
        assert len(ld_dense.atoms) == 1
        assert ld_dense.gk_common_information() == pytest.approx(1.0)

    def test_log_pmf_uses_its_support(self):
        dense = dit.Distribution([(0, 0), (1, 1)], [0.25, 0.75])
        dense.make_dense()
        ld = LogarithmicDecomposition(dense.copy(base=2))
        assert ld.omega == frozenset({(0, 0), (1, 1)})
        assert ld.entropy() == pytest.approx(shannon_entropy(dense))


class TestScalarOutcomes:
    @pytest.mark.parametrize("outcomes", [[(0,), (1,), (2,)], [0, 1, 2], [0.5, 1.5, 2.5]])
    def test_single_numerical_variable(self, outcomes):
        """dit stores these outcomes bare (0, not (0,)); each is still one variable's value."""
        d = dit.Distribution(outcomes, [0.2, 0.3, 0.5])
        assert d.outcome_length() == 1
        ld = LogarithmicDecomposition(d)
        h = shannon_entropy(d)
        assert ld.entropy() == pytest.approx(h)
        assert ld.entropy([0]) == pytest.approx(h)
        assert ld.content(0) == ld.atoms
        assert ld.partition([0]) == frozenset(frozenset([o]) for o in ld.omega)
        assert ld.coinformation() == pytest.approx(h)
        assert ld.gk_common_information() == pytest.approx(h)

    def test_point_mass(self):
        """A single outcome has no atoms: every quantity vanishes."""
        ld = LogarithmicDecomposition(dit.Distribution([0], [1.0]))
        assert ld.atoms == set()
        assert ld.discernible(set()) == set()
        assert ld.discernible_partition(set()) == frozenset([ld.omega])
        assert ld.is_discernible(set())
        assert ld.gk_common_information() == 0


# ---------------------------------------------------------------------------
# Partitions and their contents
# ---------------------------------------------------------------------------


class TestPartitionContent:
    def test_matches_variable_contents(self):
        ld = LogarithmicDecomposition(dit.example_dists.dyadic)
        for rvs in ([0], [1], [2], [0, 1], [[0], [2]], [0, 1, 2], []):
            assert ld.partition_content(ld.partition(rvs)) == ld.content(rvs)
        assert ld.partition([]) == frozenset([ld.omega])
        assert ld.content([]) == set()
        assert ld.partition() == frozenset(frozenset([o]) for o in ld.omega)

    def test_matches_independent_computation(self):
        """partition_content agrees with a direct computation for every partition of 5 outcomes."""
        ld = LogarithmicDecomposition(_scalar_dist(5))
        for partition, content in _all_contents(ld).items():
            assert ld.partition_content(partition) == content
            assert ld.partition_content([list(block) for block in partition]) == content

    def test_invalid_partitions(self):
        ld = LogarithmicDecomposition(_skewed_pair())
        a, b, c, d = sorted(ld.omega)
        with pytest.raises(ditException, match="not an outcome"):
            ld.partition_content([[a, b], [c, d, (5, 5)]])
        with pytest.raises(ditException, match="more than one block"):
            ld.partition_content([[a, b], [b, c, d]])
        with pytest.raises(ditException, match="do not cover"):
            ld.partition_content([[a, b], [c]])


# ---------------------------------------------------------------------------
# I-diagram regions (Theorem 15 of [3])
# ---------------------------------------------------------------------------


def _region_dists():
    rng = np.random.default_rng(2)
    return {
        "xor": dit.example_dists.Xor(),
        "dyadic": dit.example_dists.dyadic,
        "triadic": dit.example_dists.triadic,
        "random3": _random_sparse(rng, (2, 3, 2), 8),
        "random4": _random_sparse(rng, (2, 2, 2, 2), 9),
    }


class TestRegions:
    @pytest.mark.parametrize("name", ["xor", "dyadic", "triadic", "random3", "random4"])
    def test_every_idiagram_atom(self, name):
        """Each I-diagram atom is the measure of its region, and the regions partition Delta(Omega)."""
        dist = _region_dists()[name]
        ld = LogarithmicDecomposition(dist)
        seen = set()
        for (rvs, crvs), value in ShannonPartition(dist).atoms.items():
            groups = [list(group) for group in rvs]
            region = ld.region(groups, list(crvs))
            assert ld.measure_set(region) == pytest.approx(value, abs=1e-10)
            assert ld.coinformation(groups, list(crvs)) == pytest.approx(value, abs=1e-10)
            assert not region & seen
            seen |= region
        assert seen == ld.atoms

    def test_conditional_quantities_match_dit(self):
        rng = np.random.default_rng(23)
        for _ in range(10):
            d = _random_sparse(rng, (2, 2, 3), int(rng.integers(4, 10)))
            ld = LogarithmicDecomposition(d)
            assert ld.entropy([0], [1]) == pytest.approx(conditional_entropy(d, [0], [1]), abs=1e-10)
            assert ld.entropy([0, 1], [2]) == pytest.approx(conditional_entropy(d, [0, 1], [2]), abs=1e-10)
            assert ld.entropy(None, [2]) == pytest.approx(conditional_entropy(d, [0, 1, 2], [2]), abs=1e-10)
            for rvs, crvs in [([[0], [1]], [2]), ([[0], [2]], [1]), ([[0], [1, 2]], [])]:
                expected = coinformation(d, rvs, crvs)
                assert ld.mutual_information(rvs, crvs) == pytest.approx(expected, abs=1e-10)
            assert ld.coinformation([[0], [1], [2]], []) == pytest.approx(coinformation(d), abs=1e-10)

    def test_abstract_identities(self):
        """I(X;Y) = mu(X ∩ Y), H(X,Y) = mu(X ∪ Y), and H(X|Y) = mu(X minus Y)."""
        d = _skewed_pair()
        ld = LogarithmicDecomposition(d)
        cx, cy = ld.content([0]), ld.content([1])
        assert ld.region([[0], [1]]) == cx & cy
        assert ld.content([0, 1]) == cx | cy
        assert ld.region([[0]], [1]) == cx - cy
        assert ld.measure_set(cx - cy) == pytest.approx(conditional_entropy(d, [0], [1]), abs=1e-12)
        assert ld.measure_set(cx | cy) == pytest.approx(shannon_entropy(d), abs=1e-12)


# ---------------------------------------------------------------------------
# Examples and analytic results from [3]
# ---------------------------------------------------------------------------


class TestPaperExamples:
    def test_figure_3_memberships(self):
        """
        Example 14 and Fig. 3 of [3]: outcomes 1-4 on a 2x2 grid with p = (0.1, 0.2, 0.3, 0.4),
        X the columns {1, 3}, {2, 4} and Y the rows {1, 2}, {3, 4}.
        """
        ld = LogarithmicDecomposition(_skewed_pair())
        label = {(0, 0): 1, (0, 1): 2, (1, 0): 3, (1, 1): 4}

        def names(atoms):
            return {"".join(str(i) for i in sorted(label[o] for o in atom)) for atom in atoms}

        x, y = ld.content([1]), ld.content([0])
        assert names(x - y) == {"12", "34"}
        assert names(y - x) == {"13", "24"}
        assert names(x & y) == {"14", "23", "123", "124", "134", "234", "1234"}

    def test_figure_4_gacs_korner(self):
        """
        Fig. 4 of [3]: X = {1}{3}{2,4} and Y = {1}{2}{3,4}. The common variable separates
        outcome 1 from the rest, and I - K is the measure of the atoms 23 and 234.
        """
        one, two, three, four = (0, 0), (1, 1), (2, 2), (1, 2)
        d = dit.Distribution([one, two, three, four], [0.1, 0.2, 0.3, 0.4])
        ld = LogarithmicDecomposition(d)
        mi = ld.region([[0], [1]])
        b = frozenset

        def edges(atoms):
            return {atom for atom in atoms if len(atom) == 2}

        assert edges(mi) == {b({one, two}), b({one, three}), b({one, four}), b({two, three})}
        assert edges(ld.discernible(mi)) == {b({one, two}), b({one, three}), b({one, four})}
        assert ld.discernible_partition(mi) == b({b({one}), b({two, three, four})})
        h = -(0.1 * np.log2(0.1) + 0.9 * np.log2(0.9))
        assert ld.gk_common_information() == pytest.approx(h)
        assert ld.gk_common_information() == pytest.approx(gk_common_information(d))
        gap = ld.measure(b({two, three})) + ld.measure(b({two, three, four}))
        assert ld.mutual_information([[0], [1]]) - ld.gk_common_information() == pytest.approx(gap)

    def test_lemma_10_limit_at_infinity(self):
        """mu(p, x) tends to -mu(p) as x grows, so |mu(p, x)| increases towards |mu(p)|."""
        rng = np.random.default_rng(37)
        for size in (2, 3, 4):
            for _ in range(5):
                probs = tuple(rng.uniform(0.05, 1.0, size=size))
                target = -_interior_loss(probs)
                values = [_interior_loss((*probs, x)) for x in (1e1, 1e2, 1e3, 1e4)]
                gaps = [abs(value - target) for value in values]
                # the gap shrinks until it reaches floating-point noise (about 1e-10)
                assert all(later < earlier or later < 1e-10 for earlier, later in itertools.pairwise(gaps))
                assert gaps[3] < 1e-3 * abs(target)
                assert all(np.sign(value) == np.sign(target) for value in values)

    def test_corollary_12_magnitude_strictly_decreases(self):
        """|mu(p_1, ..., p_n, t)| < |mu(p_1, ..., p_n)| for t >= 0, n >= 2, all p_i > 0, even off [0, 1]."""
        rng = np.random.default_rng(41)
        for _ in range(200):
            probs = tuple(rng.uniform(0.01, 1.0, size=int(rng.integers(2, 6))))
            tau = rng.uniform(0, 1) if rng.random() < 0.5 else rng.uniform(1, 10)
            assert abs(_interior_loss((*probs, tau))) < abs(_interior_loss(probs))
        assert _interior_loss((0.2, 0.3, 0.0)) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Discernible subsets (Definition 18 of [3]) in many contexts
# ---------------------------------------------------------------------------


class TestDiscernibleBruteForce:
    """Dis(S) against an exhaustive search over every partition of Omega."""

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_every_atom_set(self, n):
        """For |Omega| <= 4, every one of the 2^(2^n - n - 1) atom sets."""
        ld = LogarithmicDecomposition(_scalar_dist(n))
        contents = _all_contents(ld)
        assert len(contents) == BELL[n]
        atoms = _sorted_atoms(ld)
        n_discernible = 0
        for mask in range(2 ** len(atoms)):
            subset = {atom for i, atom in enumerate(atoms) if mask >> i & 1}
            finest = _finest_inside(contents, subset)
            assert ld.discernible_partition(subset) == finest
            assert ld.discernible(subset) == contents[finest]
            n_discernible += ld.is_discernible(subset)
        # one discernible set per partition: a content determines its partition
        assert n_discernible == BELL[n]

    @pytest.mark.parametrize("n", [5, 6])
    def test_random_atom_sets(self, n):
        rng = np.random.default_rng(n)
        ld = LogarithmicDecomposition(_scalar_dist(n, rng))
        contents = _all_contents(ld)
        assert len(contents) == BELL[n]
        discernible_sets = set(contents.values())
        for subset in _random_atom_sets(ld, contents, rng, count=150):
            finest = _finest_inside(contents, subset)
            assert ld.discernible_partition(subset) == finest
            assert ld.discernible(subset) == contents[finest]
            assert ld.is_discernible(subset) == (frozenset(subset) in discernible_sets)

    def test_atom_sets_built_from_variables(self):
        """Regions, R_n filters, generators and the dual total correlation content of three variables."""
        rng = np.random.default_rng(3)
        for i in range(6):
            d = _random_blocky(rng, (3, 3, 3), 7) if i % 2 else _random_sparse(rng, (2, 3, 2), 7)
            ld = LogarithmicDecomposition(d)
            contents = _all_contents(ld)
            cx, cy, cz = (ld.content([i]) for i in range(3))
            candidates = [
                cx & cy,
                cx & cy & cz,
                (cx & cy) - cz,
                cx - cy,
                cx ^ cy,
                cx | cy,
                ld.content([0, 1]) & cz,
                ld.r_n(cx & cy & cz, 2),
                ld.r_n(cx & cy, 3),
                (cx & cy) | (cx & cz) | (cy & cz),
                ld.generators(cx & cy),
                ld.atoms - (cx & cy & cz),
            ]
            for subset in candidates:
                finest = _finest_inside(contents, set(subset))
                assert ld.discernible_partition(subset) == finest
                assert ld.discernible(subset) == contents[finest]


class TestDiscernibleLaws:
    """Dis is an interior operator, and its fixed points are exactly the contents."""

    @pytest.fixture
    def setup(self):
        rng = np.random.default_rng(11)
        ld = LogarithmicDecomposition(_scalar_dist(5, rng))
        return ld, _random_atom_sets(ld, _all_contents(ld), rng, count=120), rng

    def test_deflationary_and_idempotent(self, setup):
        ld, sets, _ = setup
        for subset in sets:
            dis = ld.discernible(subset)
            assert dis <= subset
            assert ld.discernible(dis) == dis
            assert ld.is_discernible(dis)

    def test_monotone(self, setup):
        ld, sets, rng = setup
        atoms = _sorted_atoms(ld)
        for subset in sets:
            bigger = subset | {atom for atom in atoms if rng.random() < 0.3}
            assert ld.discernible(subset) <= ld.discernible(bigger)

    def test_intersections_and_unions(self, setup):
        ld, sets, _ = setup
        for s, t in zip(sets[::2], sets[1::2], strict=True):
            ds, dt = ld.discernible(s), ld.discernible(t)
            assert ld.discernible(s & t) == ld.discernible(ds & dt)
            assert ds | dt <= ld.discernible(s | t)
            assert ld.is_discernible(ds | dt)

    def test_contents_are_the_fixed_points(self):
        """Every partition of up to 6 outcomes is recovered from its content."""
        for n in range(1, 7):
            ld = LogarithmicDecomposition(_scalar_dist(n))
            for partition, content in _all_contents(ld).items():
                assert ld.discernible_partition(content) == partition
                assert ld.discernible(content) == content
                assert ld.is_discernible(content)

    def test_variable_contents_are_discernible(self):
        """Contents of variables, joint variables, and functions of variables."""
        d = dit.insert_rvf(_random_sparse(np.random.default_rng(4), (3, 3), 7), lambda o: ((o[0] + o[1]) % 2,))
        ld = LogarithmicDecomposition(d)
        for rvs in ([0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2], []):
            assert ld.discernible_partition(ld.content(rvs)) == ld.partition(rvs)
            assert ld.is_discernible(ld.content(rvs))
        # the parity is a function of X_0 X_1, so its content is inside theirs
        assert ld.discernible(ld.content([0, 1]) & ld.content([2])) == ld.content([2])


class TestDiscernibleStructuredSets:
    """Closed forms for families of atom sets, most of them not intersections of contents."""

    @pytest.fixture(params=[3, 4, 5])
    def ld(self, request):
        return LogarithmicDecomposition(_scalar_dist(request.param))

    def test_empty_and_full(self, ld):
        assert ld.discernible(set()) == set()
        assert ld.discernible_partition(set()) == frozenset([ld.omega])
        assert ld.discernible(ld.atoms) == ld.atoms
        assert ld.discernible_partition(ld.atoms) == frozenset(frozenset([o]) for o in ld.omega)

    def test_single_atoms_and_fixed_pairs(self, ld):
        """With 3 or more outcomes, no single atom, nor the atoms holding a fixed pair, has a discernible part."""
        for atom in ld.atoms:
            assert ld.discernible({atom}) == set()
        for a, b in itertools.combinations(sorted(ld.omega), 2):
            assert ld.discernible({atom for atom in ld.atoms if {a, b} <= atom}) == set()

    def test_degree_bands(self, ld):
        n = len(ld.omega)
        for k in range(3, n + 1):
            assert ld.discernible({atom for atom in ld.atoms if len(atom) >= k}) == set()
        for k in range(2, n):
            assert ld.discernible({atom for atom in ld.atoms if len(atom) <= k}) == set()

    def test_stars_are_discernible(self, ld):
        """The atoms meeting a set A are the content of the variable naming each outcome of A."""
        omega = sorted(ld.omega)
        for r in range(1, len(omega)):
            for a in itertools.combinations(omega, r):
                star = {atom for atom in ld.atoms if atom & set(a)}
                expected = frozenset([*(frozenset([o]) for o in a), ld.omega - set(a)])
                assert ld.discernible_partition(star) == expected
                assert ld.is_discernible(star)

    def test_higher_atoms_matter(self):
        """An atom outside the set joins its outcomes even when its edges are inside."""
        ld = LogarithmicDecomposition(_scalar_dist(3))
        b = frozenset
        assert ld.is_discernible({b({0, 1}), b({0, 2}), b({0, 1, 2})})
        assert ld.discernible({b({0, 1}), b({0, 2})}) == set()
        assert ld.discernible({b({0, 1}), b({0, 2}), b({1, 2})}) == set()


class TestDiscernibleContentCombinations:
    """Dis of Boolean combinations of contents, over every pair of partitions of 4 outcomes."""

    @pytest.fixture
    def setup(self):
        ld = LogarithmicDecomposition(_scalar_dist(4))
        return ld, _all_contents(ld)

    def test_intersections_give_the_meet(self, setup):
        ld, contents = setup
        for p, q in itertools.product(contents, repeat=2):
            meet = _meet([p, q], ld.omega)
            assert ld.discernible_partition(contents[p] & contents[q]) == meet
            assert ld.discernible(contents[p] & contents[q]) == contents[meet]
        for p, q, r in itertools.product(contents, repeat=3):
            assert ld.discernible_partition(contents[p] & contents[q] & contents[r]) == _meet([p, q, r], ld.omega)

    def test_unions_give_the_common_refinement(self, setup):
        ld, contents = setup
        for p, q in itertools.product(contents, repeat=2):
            union = contents[p] | contents[q]
            assert ld.discernible_partition(union) == _common_refinement(p, q)
            assert ld.is_discernible(union)

    def test_differences_have_no_discernible_part(self, setup):
        """No variable's content fits in Delta(X) minus Delta(Y) unless Y is constant."""
        ld, contents = setup
        trivial = frozenset([ld.omega])
        for p, q in itertools.product(contents, repeat=2):
            expected = contents[p] if q == trivial else set()
            assert ld.discernible(contents[p] - contents[q]) == expected

    def test_complements_have_no_discernible_part(self, setup):
        ld, contents = setup
        trivial = frozenset([ld.omega])
        for p, content in contents.items():
            expected = ld.atoms if p == trivial else set()
            assert ld.discernible(ld.atoms - content) == expected


class TestDiscernibleInvariance:
    def test_relabeling_outcomes_commutes_with_dis(self):
        rng = np.random.default_rng(5)
        n = 5
        pmf = rng.dirichlet(np.ones(n))
        perm = rng.permutation(n)
        names = {i: (f"w{perm[i]}",) for i in range(n)}
        ld = LogarithmicDecomposition(dit.Distribution(list(range(n)), pmf))
        relabeled = LogarithmicDecomposition(dit.Distribution([names[i] for i in range(n)], pmf))

        def image(atom_set):
            return {frozenset(names[o] for o in atom) for atom in atom_set}

        for subset in _random_atom_sets(ld, _all_contents(ld), rng, count=80):
            partition = ld.discernible_partition(subset)
            assert relabeled.discernible(image(subset)) == image(ld.discernible(subset))
            assert relabeled.discernible_partition(image(subset)) == frozenset(
                frozenset(names[o] for o in block) for block in partition
            )

    def test_independent_of_probabilities(self):
        """Discernibility is set-theoretic; the pmf only sets the measure, H(partition)."""
        rng = np.random.default_rng(6)
        dists = [_scalar_dist(5, rng) for _ in range(3)]
        lds = [LogarithmicDecomposition(d) for d in dists]
        for subset in _random_atom_sets(lds[0], _all_contents(lds[0]), rng, count=60):
            partitions = {ld.discernible_partition(subset) for ld in lds}
            assert len(partitions) == 1
            (partition,) = partitions
            for d, ld in zip(dists, lds, strict=True):
                probs = dict(d.zipped())
                weights = [sum(probs[o] for o in block) for block in partition]
                entropy = -sum(w * np.log2(w) for w in weights)
                assert ld.measure_set(ld.discernible(subset)) == pytest.approx(entropy, abs=1e-12)

    def test_measure_is_dit_entropy_of_the_variable(self):
        """Inserting the discernible partition as a new variable, dit gives the same entropy."""
        rng = np.random.default_rng(8)
        d = _random_sparse(rng, (3, 3), 7)
        ld = LogarithmicDecomposition(d)
        for subset in _random_atom_sets(ld, _all_contents(ld), rng, count=40):
            blocks = sorted(ld.discernible_partition(subset), key=sorted)
            label = {o: i for i, block in enumerate(blocks) for o in block}
            d2 = dit.insert_rvf(d, lambda o, label=label: (label[o],))
            assert ld.measure_set(ld.discernible(subset)) == pytest.approx(shannon_entropy(d2, [2]), abs=1e-10)

    def test_atoms_outside_omega_are_ignored(self):
        ld = LogarithmicDecomposition(_skewed_pair())
        junk = {frozenset({(0, 0), (5, 5)}), frozenset({(0, 0)})}
        for subset in (set(), ld.content([0]), ld.region([[0], [1]]), ld.atoms):
            assert ld.discernible(subset | junk) == ld.discernible(subset)
            assert ld.discernible_partition(subset | junk) == ld.discernible_partition(subset)
            assert not ld.is_discernible(subset | junk)


# ---------------------------------------------------------------------------
# Gacs-Korner common information (Theorem 17 of [3])
# ---------------------------------------------------------------------------


class TestGacsKorner:
    def test_matches_dit_on_random_supports(self):
        rng = np.random.default_rng(13)
        for i in range(40):
            cards = tuple(int(c) for c in rng.integers(2, 5, size=2))
            size = int(rng.integers(2, min(9, cards[0] * cards[1]) + 1))
            d = _random_blocky(rng, cards, size) if i % 2 else _random_sparse(rng, cards, size)
            ld = LogarithmicDecomposition(d)
            assert ld.gk_common_information() == pytest.approx(gk_common_information(d), abs=1e-10)

    def test_groups_and_conditioning_match_dit(self):
        rng = np.random.default_rng(17)
        cases = [
            ([[0], [1], [2]], None),
            ([[0], [1]], [2]),
            ([[0], [2]], [1]),
            ([[0, 1], [2]], None),
            ([[0], [1, 2]], [2]),
        ]
        for i in range(18):
            size = int(rng.integers(3, 10))
            if i % 3 == 0:
                d = _random_sparse(rng, (2, 3, 2), size)
            elif i % 3 == 1:
                d = _random_blocky(rng, (3, 3, 3), size)
            else:  # X_2 is a random function of X_0 X_1, so conditioning on it leaves a common part
                d = _with_random_bit(rng, _random_blocky(rng, (3, 3), size))
            ld = LogarithmicDecomposition(d)
            for rvs, crvs in cases:
                expected = gk_common_information(d, rvs, crvs)
                assert ld.gk_common_information(rvs, crvs) == pytest.approx(expected, abs=1e-10)

    def test_common_variable_is_the_meet(self):
        """The discernible partition of the co-information content is dit's meet of the variables."""
        rng = np.random.default_rng(19)
        for i in range(16):
            size = int(rng.integers(3, 10))
            d = _random_blocky(rng, (3, 3, 3), size) if i % 2 else _random_sparse(rng, (3, 3, 2), size)
            ld = LogarithmicDecomposition(d)
            for rvs in ([[0], [1]], [[0], [1], [2]], [[0, 1], [1, 2]]):
                meet = insert_meet(d, -1, rvs, support_only=True)
                blocks = {}
                for outcome in meet.outcomes:
                    blocks.setdefault(outcome[-1], set()).add(outcome[:-1])
                expected = frozenset(frozenset(block) for block in blocks.values())
                assert ld.discernible_partition(ld.region(rvs)) == expected

    def test_gap_to_mutual_information(self):
        """K <= I, and I - K is the measure of the atoms that Dis removes."""
        rng = np.random.default_rng(29)
        for i in range(20):
            size = int(rng.integers(3, 9))
            d = _random_blocky(rng, (3, 3), size) if i % 2 else _random_sparse(rng, (3, 3), size)
            ld = LogarithmicDecomposition(d)
            mi = ld.region([[0], [1]])
            gap = ld.measure_set(mi - ld.discernible(mi))
            assert gap >= -1e-12
            assert ld.mutual_information([[0], [1]]) - ld.gk_common_information() == pytest.approx(gap, abs=1e-12)

    def test_more_variables_less_common_information(self):
        rng = np.random.default_rng(31)
        for i in range(10):
            size = int(rng.integers(3, 9))
            d = _random_blocky(rng, (3, 3, 3), size) if i % 2 else _random_sparse(rng, (2, 3, 2), size)
            ld = LogarithmicDecomposition(d)
            three = ld.discernible(ld.region([[0], [1], [2]]))
            assert three <= ld.discernible(ld.region([[0], [1]]))
            assert ld.measure_set(three) <= ld.gk_common_information([[0], [1]]) + 1e-12


# ---------------------------------------------------------------------------
# Formal sums: atoms counted with multiplicity
# ---------------------------------------------------------------------------


class TestFormalSums:
    @pytest.fixture
    def four(self):
        d = _random_sparse(np.random.default_rng(43), (2, 2, 2, 2), 10)
        return d, LogarithmicDecomposition(d)

    @staticmethod
    def _crossed(ld, n):
        """For each atom, the number of single variables whose content contains it."""
        contents = [ld.content([i]) for i in range(n)]
        return {atom: sum(atom in content for content in contents) for atom in ld.atoms}

    def test_multiplicities(self, four):
        """TC counts an atom k - 1 times, DTC once if k >= 2, O-information max(k - 2, 0) times."""
        _, ld = four
        n = 4
        k = self._crossed(ld, n)
        assert max(k.values()) == n
        singles = [[i] for i in range(n)]
        others = [[j for j in range(n) if j != i] for i in range(n)]
        everything = list(range(n))
        tc = ld.formal_sum([(1, s) for s in singles] + [(-1, everything)])
        dtc = ld.formal_sum([(1 - n, everything)] + [(1, rest) for rest in others])
        oi = ld.formal_sum([(1, s) for s in singles] + [(n - 2, everything)] + [(-1, rest) for rest in others])
        assert tc == {atom: k[atom] - 1 for atom in ld.atoms if k[atom] > 1}
        assert dtc == {atom: 1 for atom in ld.atoms if k[atom] >= 2}
        assert oi == {atom: k[atom] - 2 for atom in ld.atoms if k[atom] > 2}
        assert max(oi.values()) == n - 2

    def test_down2024_proposition_40(self, four):
        """DTC is the union of pairwise intersections; TC is the sum of contents minus their union."""
        _, ld = four
        contents = [ld.content([i]) for i in range(4)]
        pairwise = set().union(*(a & b for a, b in itertools.combinations(contents, 2)))
        dtc = ld.formal_sum([(-3, [0, 1, 2, 3])] + [(1, [j for j in range(4) if j != i]) for i in range(4)])
        assert dtc == dict.fromkeys(pairwise, 1)
        tc = ld.formal_sum([(1, [i]) for i in range(4)] + [(-1, [0, 1, 2, 3])])
        union = set().union(*contents)
        assert tc == {
            atom: sum(atom in c for c in contents) - 1 for atom in union if sum(atom in c for c in contents) > 1
        }

    def test_values_match_dit(self):
        rng = np.random.default_rng(47)
        cases = [(None, None), ([[0], [1], [2]], [3]), ([[0, 1], [2], [3]], None), ([[0], [1]], [2, 3])]
        for _ in range(6):
            d = _random_sparse(rng, (2, 2, 2, 2), int(rng.integers(5, 10)))
            ld = LogarithmicDecomposition(d)
            for rvs, crvs in cases:
                assert ld.total_correlation(rvs, crvs) == pytest.approx(total_correlation(d, rvs, crvs), abs=1e-10)
                assert ld.dual_total_correlation(rvs, crvs) == pytest.approx(
                    dual_total_correlation(d, rvs, crvs), abs=1e-10
                )
                assert ld.o_information(rvs, crvs) == pytest.approx(o_information(d, rvs, crvs), abs=1e-10)

    def test_o_information_of_three_variables_is_the_coinformation_content(self):
        ld = LogarithmicDecomposition(dit.example_dists.Xor())
        oi = ld.formal_sum([(1, [0]), (1, [1]), (1, [2]), (1, [0, 1, 2]), (-1, [1, 2]), (-1, [0, 2]), (-1, [0, 1])])
        assert oi == dict.fromkeys(ld.region(), 1)
        assert ld.o_information() == pytest.approx(-1.0)
        assert ld.total_correlation() == pytest.approx(1.0)
        assert ld.dual_total_correlation() == pytest.approx(2.0)

    def test_entropies_mutual_information_and_conditioning(self):
        ld = LogarithmicDecomposition(_skewed_pair())
        assert ld.formal_sum([(1, [0])]) == dict.fromkeys(ld.content([0]), 1)
        assert ld.formal_sum([(1, [0]), (1, [1]), (-1, [0, 1])]) == dict.fromkeys(ld.region([[0], [1]]), 1)
        assert ld.formal_sum([(1, [0])], crvs=[1]) == dict.fromkeys(ld.region([[0]], [1]), 1)
        assert ld.formal_sum([(2, [0]), (-2, [0])]) == {}
        atom = next(iter(ld.atoms))
        assert ld.measure_set({atom: 3}) == pytest.approx(3 * ld.measure(atom))

    def test_down2024_example_39(self):
        """I(X;Y) - H(X|Y) + H(X,Y) is ab + cd + 2(ad + bc + abc + abd + acd + bcd + abcd)."""
        ld = LogarithmicDecomposition(_skewed_pair())
        a, b, c, d = (0, 0), (0, 1), (1, 0), (1, 1)  # X = {ab}{cd} is X_0, Y = {ac}{bd} is X_1
        expression = ld.formal_sum([(1, [0]), (2, [1]), (-1, [0, 1])])  # = H(X) + 2 H(Y) - H(X, Y)
        f = frozenset
        expected = {f({a, b}): 1, f({c, d}): 1, f({a, d}): 2, f({b, c}): 2, f({a, b, c, d}): 2}
        expected |= {f(triple): 2 for triple in itertools.combinations((a, b, c, d), 3)}
        assert expression == expected
