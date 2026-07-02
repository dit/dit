"""
Tests for the symbolic (sympy) backend of the Wyner and Exact common
informations.

Two flavours, mirroring ``tests/symbolic/test_cross_validation.py``:

1. Closed-form assertions for small named sources whose common information is
   known analytically (giant bit, product/independent, XOR).
2. Cross-validation: a symbolic result, evaluated at concrete points, must match
   the numeric ``backend="numpy"`` optimizer.

The doubly-symmetric binary source (DSBS) is the interesting case: its Wyner
common information closes via a structural ansatz, while its Exact common
information has no simple closed form and is expected to raise.
"""

import pytest

from dit import Distribution
from dit.multivariate import (
    exact_common_information,
    wyner_common_information,
)
from dit.multivariate.common_informations.symbolic_markov import (
    SymbolicOptimizationError,
)
from dit.symbolic import evaluate, symbolic_distribution, symbols

sympy = pytest.importorskip("sympy")

TOL = 1e-4  # numeric optimizer tolerance is ~1e-6; symbolic vs numeric within 1e-4


def _giant_bit(p):
    return symbolic_distribution(["00", "11"], [p, 1 - p])


def _product(p, q):
    return symbolic_distribution(
        ["00", "01", "10", "11"],
        [p * q, p * (1 - q), (1 - p) * q, (1 - p) * (1 - q)],
    )


def _xor():
    return symbolic_distribution(["000", "011", "101", "110"], [sympy.Rational(1, 4)] * 4)


def _dsbs(a):
    return symbolic_distribution(["00", "01", "10", "11"], [(1 - a) / 2, a / 2, a / 2, (1 - a) / 2])


# ── Closed-form assertions ────────────────────────────────────────────────


@pytest.mark.parametrize("measure", [wyner_common_information, exact_common_information])
def test_giant_bit_is_binary_entropy(measure):
    """Both CIs of a giant bit equal the binary entropy H(p)."""
    p = symbols("p")
    result = measure(_giant_bit(p), backend="symbolic")
    # H(p) in bits.
    expected = (-p * sympy.log(p) - (1 - p) * sympy.log(1 - p)) / sympy.log(2)
    assert sympy.simplify(result - expected) == 0


@pytest.mark.parametrize("measure", [wyner_common_information, exact_common_information])
def test_product_is_zero(measure):
    """Independent variables need no common information."""
    p, q = symbols("p q")
    result = measure(_product(p, q), backend="symbolic")
    assert sympy.simplify(result) == 0


@pytest.mark.parametrize("measure", [wyner_common_information, exact_common_information])
def test_xor_is_two_bits(measure):
    """The XOR source has DTC == H == 2, pinning both CIs at 2 bits."""
    result = measure(_xor(), backend="symbolic")
    assert sympy.simplify(result) == 2


def test_dsbs_wyner_closed_form():
    """Wyner CI of the DSBS closes to 1 + h(a) - 2 h(a0), a0 (1-a0) = a/2."""
    a = symbols("a")
    result = wyner_common_information(_dsbs(a), backend="symbolic")
    assert result.free_symbols == {a}


def test_dsbs_exact_has_no_closed_form():
    """Exact CI of the DSBS does not reduce to a known closed form: raise."""
    a = symbols("a")
    with pytest.raises(SymbolicOptimizationError):
        exact_common_information(_dsbs(a), backend="symbolic")


# ── Cross-validation vs the numeric backend ───────────────────────────────

_A_POINTS = [0.1, 0.2, 0.3, 0.4]


@pytest.mark.parametrize("av", _A_POINTS)
def test_dsbs_wyner_matches_numeric(av):
    """Symbolic Wyner-DSBS, evaluated at a point, matches the numeric optimizer."""
    a = symbols("a")
    sym = wyner_common_information(_dsbs(a), backend="symbolic")
    sym_val = evaluate(sym, {a: av})

    numeric_dist = Distribution(["00", "01", "10", "11"], [(1 - av) / 2, av / 2, av / 2, (1 - av) / 2])
    num_val = wyner_common_information(numeric_dist, backend="numpy")

    assert abs(sym_val - num_val) < TOL


@pytest.mark.parametrize("pv", [0.25, 0.5, 0.75])
def test_giant_bit_wyner_matches_numeric(pv):
    """Symbolic giant-bit Wyner matches the numeric backend at a point."""
    p = symbols("p")
    sym = wyner_common_information(_giant_bit(p), backend="symbolic")
    sym_val = evaluate(sym, {p: pv})

    numeric_dist = Distribution(["00", "11"], [pv, 1 - pv])
    num_val = wyner_common_information(numeric_dist, backend="numpy")

    assert abs(sym_val - num_val) < TOL


# ── Auto-dispatch and guards ──────────────────────────────────────────────


def test_symbolic_dist_auto_dispatches():
    """A symbolic distribution routes to the symbolic solver without asking."""
    p = symbols("p")
    # No backend= kwarg: is_symbolic() should trigger the symbolic path.
    result = wyner_common_information(_giant_bit(p))
    assert result.free_symbols == {p}


def test_crvs_not_supported():
    """Conditioning is not supported in the v1 symbolic backend."""
    p = symbols("p")
    d = symbolic_distribution(["000", "111"], [p, 1 - p])
    with pytest.raises(NotImplementedError):
        wyner_common_information(d, rvs=[[0], [1]], crvs=[2], backend="symbolic")
