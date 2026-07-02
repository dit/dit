"""
Tests for symbolic (sympy-backed) distributions and measures.
"""

import pytest

from dit.divergences import cross_entropy, kullback_leibler_divergence
from dit.multivariate import (
    coinformation,
    dual_total_correlation,
    entropy,
    gk_common_information,
    interaction_information,
    o_information,
    total_correlation,
)
from dit.pid import PID_MMI, PID_WB
from dit.shannon import conditional_entropy, mutual_information
from dit.symbolic import simplify, symbolic_distribution, symbols

sympy = pytest.importorskip("sympy")


def binary_entropy(p):
    """H(p) in bits, symbolically."""
    return (-p * sympy.log(p) - (1 - p) * sympy.log(1 - p)) / sympy.log(2)


def equal(expr1, expr2):
    """True if two symbolic expressions are equal.

    First tries exact simplification; if that is inconclusive (log-of-negative
    branch issues can leave spurious imaginary terms), falls back to numeric
    equality at several interior points of the unit interval for each free
    symbol.
    """
    diff = expr1 - expr2
    if simplify(diff) == 0:
        return True
    syms = sorted(diff.free_symbols, key=str)
    grid = [sympy.Rational(k, 10) for k in range(1, 10)]
    import itertools

    for combo in itertools.product(grid, repeat=len(syms)):
        subs = dict(zip(syms, combo, strict=True))
        val = complex(diff.subs(subs))
        if abs(val) > 1e-9:
            return False
    return True


# ── Construction ──────────────────────────────────────────────────────────


def test_is_symbolic():
    """A symbolic distribution reports is_symbolic() True."""
    p = symbols("p")
    d = symbolic_distribution(["0", "1"], [p, 1 - p])
    assert d.is_symbolic()


def test_numeric_not_symbolic():
    """A numeric distribution reports is_symbolic() False."""
    from dit import Distribution

    d = Distribution(["0", "1"], [0.5, 0.5])
    assert not d.is_symbolic()


def test_pmf_preserves_symbols():
    """The pmf property returns symbolic values without float coercion."""
    p = symbols("p")
    d = symbolic_distribution(["0", "1"], [p, 1 - p])
    assert list(d.pmf) == [p, 1 - p]


def test_getitem_preserves_symbols():
    """Indexing returns the symbolic probability."""
    p = symbols("p")
    d = symbolic_distribution(["0", "1"], [p, 1 - p])
    assert d["0"] == p


def test_validate_free_symbols_ok():
    """Validation passes for pmfs with free symbols (undecidable normalisation)."""
    p = symbols("p")
    d = symbolic_distribution(["0", "1"], [p, 1 - p])
    assert d.validate()


# ── Entropy ───────────────────────────────────────────────────────────────


def test_binary_entropy():
    """H of a symbolic coin equals the binary entropy function."""
    p = symbols("p")
    d = symbolic_distribution(["0", "1"], [p, 1 - p])
    assert equal(entropy(d), binary_entropy(p))


def test_entropy_evaluates():
    """Symbolic entropy evaluates to the numeric value at a point."""
    p = symbols("p")
    d = symbolic_distribution(["0", "1"], [p, 1 - p])
    assert entropy(d).subs(p, sympy.Rational(1, 2)) == 1


def test_uniform_entropy():
    """H of a uniform symbolic distribution over 4 outcomes is 2 bits."""
    a = symbols("a")
    d = symbolic_distribution(["00", "01", "10", "11"], [a, a, a, a])
    # substitute the normalised value
    H = entropy(d).subs(a, sympy.Rational(1, 4))
    assert simplify(H) == 2


# ── Multivariate (giant bit) ──────────────────────────────────────────────


def test_giant_bit_measures():
    """For a 2-bit giant bit, H(XY) = I(X:Y) = T = coI = H(p)."""
    p = symbols("p")
    d = symbolic_distribution(["00", "11"], [p, 1 - p])
    Hp = binary_entropy(p)
    assert equal(entropy(d), Hp)
    assert equal(mutual_information(d, [0], [1]), Hp)
    assert equal(total_correlation(d), Hp)
    assert equal(coinformation(d), Hp)


def test_giant_bit_3var():
    """For a 3-bit giant bit, DTC = O = coI = H(p)."""
    p = symbols("p")
    d = symbolic_distribution(["000", "111"], [p, 1 - p])
    Hp = binary_entropy(p)
    assert equal(dual_total_correlation(d), Hp)
    assert equal(o_information(d), Hp)
    assert equal(coinformation(d), Hp)


def test_interaction_information_symbolic():
    """Interaction information returns a symbolic expression for a giant bit."""
    p = symbols("p")
    d = symbolic_distribution(["000", "111"], [p, 1 - p])
    II = interaction_information(d, [[0], [1], [2]])
    assert II.free_symbols == {p}


def test_conditional_entropy_zero():
    """H(X|X) is exactly zero even symbolically."""
    p = symbols("p")
    d = symbolic_distribution(["00", "11"], [p, 1 - p])
    assert conditional_entropy(d, [0], [0]) == 0


# ── Divergences ───────────────────────────────────────────────────────────


def test_kl_self_is_zero():
    """KL(p || p) = 0."""
    p, q = symbols("p q")
    d1 = symbolic_distribution(["0", "1"], [p, 1 - p])
    d2 = symbolic_distribution(["0", "1"], [q, 1 - q])
    dkl = kullback_leibler_divergence(d1, d2)
    assert simplify(dkl.subs(q, p)) == 0


def test_cross_entropy_symbolic():
    """Cross entropy is symbolic and reduces to entropy when p == q."""
    p, q = symbols("p q")
    d1 = symbolic_distribution(["0", "1"], [p, 1 - p])
    d2 = symbolic_distribution(["0", "1"], [q, 1 - q])
    xh = cross_entropy(d1, d2)
    assert equal(xh.subs(q, p), binary_entropy(p))


# ── Common information ────────────────────────────────────────────────────


def test_gk_common_information_giant_bit():
    """Gács-Körner common information of a giant bit is H(p)."""
    p = symbols("p")
    d = symbolic_distribution(["00", "11"], [p, 1 - p])
    assert equal(gk_common_information(d), binary_entropy(p))


# ── PID (closed-form) ─────────────────────────────────────────────────────


def test_pid_wb_giant_bit():
    """I_min: a giant bit is purely redundant, unique/synergy are zero."""
    p = symbols("p")
    d = symbolic_distribution(["000", "111"], [p, 1 - p])
    pid = PID_WB(d)
    red = pid.get_pi(((0,), (1,)))
    unique0 = pid.get_pi(((0,),))
    synergy = pid.get_pi(((0, 1),))
    assert unique0 == 0
    assert synergy == 0
    assert equal(red, binary_entropy(p))


def test_pid_mmi_giant_bit():
    """I_mmi: a giant bit is purely redundant."""
    p = symbols("p")
    d = symbolic_distribution(["000", "111"], [p, 1 - p])
    pid = PID_MMI(d)
    assert equal(pid.get_pi(((0,), (1,))), binary_entropy(p))
    assert pid.get_pi(((0, 1),)) == 0
