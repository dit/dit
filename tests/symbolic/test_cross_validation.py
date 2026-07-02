"""
Cross-validation tests: symbolic measures, after substituting concrete values,
must match the same measure computed on a purely numeric distribution built
with those values from the start.

The pattern for every test:

1. Build a symbolic distribution whose probabilities are expressions in one or
   more symbols (e.g. ``p``, ``q``).
2. Compute a measure symbolically -> a sympy expression.
3. Substitute concrete rational values for the symbols and coerce to float.
4. Build the *numeric* distribution with those same values plugged in from the
   start, compute the same measure.
5. Assert the two agree.
"""

import pytest

import dit
from dit.divergences import (
    cross_entropy,
    kullback_leibler_divergence,
)
from dit.multivariate import (
    caekl_mutual_information,
    coinformation,
    dual_total_correlation,
    entropy,
    gk_common_information,
    interaction_information,
    o_information,
    total_correlation,
    tse_complexity,
)
from dit.other import extropy
from dit.pid import PID_MMI, PID_WB
from dit.shannon import conditional_entropy, mutual_information
from dit.symbolic import evaluate, symbolic_distribution, symbols

sympy = pytest.importorskip("sympy")

# A few concrete (p, q) points at which to compare symbolic-vs-numeric.
# Rationals keep substitution exact; the final comparison is done in float.
POINTS = [
    (sympy.Rational(1, 2), sympy.Rational(1, 2)),
    (sympy.Rational(1, 3), sympy.Rational(1, 4)),
    (sympy.Rational(7, 10), sympy.Rational(1, 5)),
    (sympy.Rational(9, 10), sympy.Rational(2, 3)),
]

TOL = 1e-9


def _subs_map(point):
    """Map the actual (positive) symbols ``p``/``q`` to the point's values.

    ``symbols`` returns cached symbols carrying ``positive=True``; keying by the
    symbol objects (not the bare names) is required for ``.subs`` to match.
    """
    p_sym, q_sym = symbols("p"), symbols("q")
    pv, qv = point
    return {p_sym: pv, q_sym: qv}


def _subs_float(expr, point):
    """Substitute ``point`` into a (possibly non-sympy) measure result -> float.

    Uses :func:`dit.symbolic.evaluate`, which handles ``Min``/``Max`` in the
    result robustly (a plain ``.subs`` can raise on unsimplified constants).
    """
    if hasattr(expr, "free_symbols"):
        return evaluate(expr, _subs_map(point))
    return float(expr)


def _point_values(point):
    """Return (p_value, q_value) as Python floats for the given point."""
    pv, qv = point
    return float(pv), float(qv)


# ── distribution builders (symbolic + matching numeric) ────────────────────


def _coin(symbolic):
    """A single biased coin: [p, 1-p]."""
    if symbolic:
        p = symbols("p")
        return symbolic_distribution(["0", "1"], [p, 1 - p])

    def build(pv, qv):
        return dit.Distribution(["0", "1"], [pv, 1 - pv])

    return build


def _independent(symbolic):
    """Two independent bits X~[p,1-p], Y~[q,1-q]."""
    if symbolic:
        p, q = symbols("p q")
        return symbolic_distribution(
            ["00", "01", "10", "11"],
            [p * q, p * (1 - q), (1 - p) * q, (1 - p) * (1 - q)],
        )

    def build(pv, qv):
        return dit.Distribution(
            ["00", "01", "10", "11"],
            [pv * qv, pv * (1 - qv), (1 - pv) * qv, (1 - pv) * (1 - qv)],
        )

    return build


def _correlated(symbolic):
    """Two-parameter joint on two bits: [p*q, p*(1-q), (1-p)*q, (1-p)*(1-q)]
    is independent; use a genuinely correlated one instead."""
    if symbolic:
        p, q = symbols("p q")
        # p is P(X=Y region split), q mixes the diagonal vs antidiagonal
        return symbolic_distribution(
            ["00", "01", "10", "11"],
            [p * q, (1 - p) * q, (1 - p) * (1 - q), p * (1 - q)],
        )

    def build(pv, qv):
        return dit.Distribution(
            ["00", "01", "10", "11"],
            [pv * qv, (1 - pv) * qv, (1 - pv) * (1 - qv), pv * (1 - qv)],
        )

    return build


def _giant_bit3(symbolic):
    """Three-variable giant bit: [p, 1-p] on 000/111."""
    if symbolic:
        p = symbols("p")
        return symbolic_distribution(["000", "111"], [p, 1 - p])

    def build(pv, qv):
        return dit.Distribution(["000", "111"], [pv, 1 - pv])

    return build


def _three_param(symbolic):
    """A generic 3-variable distribution over 4 support points."""
    if symbolic:
        p, q = symbols("p q")
        return symbolic_distribution(
            ["000", "011", "101", "110"],
            [p * q, p * (1 - q), (1 - p) * q, (1 - p) * (1 - q)],
        )

    def build(pv, qv):
        return dit.Distribution(
            ["000", "011", "101", "110"],
            [pv * qv, pv * (1 - qv), (1 - pv) * qv, (1 - pv) * (1 - qv)],
        )

    return build


# ── helper to run a measure symbolically and numerically and compare ───────


def _check(measure, builder, point, **kwargs):
    """Assert symbolic(measure).subs(point) == numeric(measure) at point."""
    sym_dist = builder(symbolic=True)
    sym_val = _subs_float(measure(sym_dist, **kwargs), point)

    num_build = builder(symbolic=False)
    pv, qv = _point_values(point)
    num_dist = num_build(pv, qv)
    num_val = float(measure(num_dist, **kwargs))

    assert sym_val == pytest.approx(num_val, abs=TOL), f"symbolic={sym_val} numeric={num_val} at {point}"


# ── entropy ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("point", POINTS)
def test_entropy_coin(point):
    _check(entropy, _coin, point)


@pytest.mark.parametrize("point", POINTS)
def test_entropy_independent(point):
    _check(entropy, _independent, point)


@pytest.mark.parametrize("point", POINTS)
def test_entropy_correlated(point):
    _check(entropy, _correlated, point)


@pytest.mark.parametrize("point", POINTS)
def test_extropy_coin(point):
    _check(extropy, _coin, point)


# ── mutual information / conditional entropy ───────────────────────────────


@pytest.mark.parametrize("point", POINTS)
def test_mutual_information_independent(point):
    _check(lambda d: mutual_information(d, [0], [1]), _independent, point)


@pytest.mark.parametrize("point", POINTS)
def test_mutual_information_correlated(point):
    _check(lambda d: mutual_information(d, [0], [1]), _correlated, point)


@pytest.mark.parametrize("point", POINTS)
def test_conditional_entropy_correlated(point):
    _check(lambda d: conditional_entropy(d, [0], [1]), _correlated, point)


# ── multivariate ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("point", POINTS)
def test_coinformation_correlated(point):
    _check(coinformation, _correlated, point)


@pytest.mark.parametrize("point", POINTS)
def test_total_correlation_correlated(point):
    _check(total_correlation, _correlated, point)


@pytest.mark.parametrize("point", POINTS)
def test_total_correlation_three_param(point):
    _check(total_correlation, _three_param, point)


@pytest.mark.parametrize("point", POINTS)
def test_dual_total_correlation_three_param(point):
    _check(dual_total_correlation, _three_param, point)


@pytest.mark.parametrize("point", POINTS)
def test_coinformation_three_param(point):
    _check(coinformation, _three_param, point)


@pytest.mark.parametrize("point", POINTS)
def test_interaction_information_three_param(point):
    _check(interaction_information, _three_param, point)


@pytest.mark.parametrize("point", POINTS)
def test_o_information_three_param(point):
    _check(o_information, _three_param, point)


@pytest.mark.parametrize("point", POINTS)
def test_tse_complexity_three_param(point):
    _check(tse_complexity, _three_param, point)


@pytest.mark.parametrize("point", POINTS)
def test_caekl_three_param(point):
    _check(caekl_mutual_information, _three_param, point)


# ── conditional multivariate ───────────────────────────────────────────────


@pytest.mark.parametrize("point", POINTS)
def test_conditional_coinformation(point):
    _check(lambda d: coinformation(d, [[0], [1]], [2]), _three_param, point)


@pytest.mark.parametrize("point", POINTS)
def test_conditional_total_correlation(point):
    _check(lambda d: total_correlation(d, [[0], [1]], [2]), _three_param, point)


# ── common information ─────────────────────────────────────────────────────


@pytest.mark.parametrize("point", POINTS)
def test_gk_common_information_giant_bit(point):
    _check(gk_common_information, _giant_bit3, point)


# ── divergences ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("point", POINTS)
def test_kl_divergence(point):
    """KL(P_p || P_q) symbolic vs numeric, with distinct p and q."""
    p, q = symbols("p q")
    d1 = symbolic_distribution(["0", "1"], [p, 1 - p])
    d2 = symbolic_distribution(["0", "1"], [q, 1 - q])
    sym_val = _subs_float(kullback_leibler_divergence(d1, d2), point)

    pv, qv = _point_values(point)
    n1 = dit.Distribution(["0", "1"], [pv, 1 - pv])
    n2 = dit.Distribution(["0", "1"], [qv, 1 - qv])
    num_val = float(kullback_leibler_divergence(n1, n2))

    assert sym_val == pytest.approx(num_val, abs=TOL)


@pytest.mark.parametrize("point", POINTS)
def test_cross_entropy(point):
    p, q = symbols("p q")
    d1 = symbolic_distribution(["0", "1"], [p, 1 - p])
    d2 = symbolic_distribution(["0", "1"], [q, 1 - q])
    sym_val = _subs_float(cross_entropy(d1, d2), point)

    pv, qv = _point_values(point)
    n1 = dit.Distribution(["0", "1"], [pv, 1 - pv])
    n2 = dit.Distribution(["0", "1"], [qv, 1 - qv])
    num_val = float(cross_entropy(n1, n2))

    assert sym_val == pytest.approx(num_val, abs=TOL)


# ── PID ────────────────────────────────────────────────────────────────────

PID_NODES = [((0, 1),), ((0,),), ((1,),), ((0,), (1,))]


@pytest.mark.parametrize("point", POINTS)
@pytest.mark.parametrize("measure", [PID_WB, PID_MMI])
def test_pid_giant_bit(point, measure):
    """Each PID atom: symbolic-substituted matches numeric, for a giant bit."""
    p = symbols("p")
    sym = measure(symbolic_distribution(["000", "111"], [p, 1 - p]))

    pv, _ = _point_values(point)
    num = measure(dit.Distribution(["000", "111"], [pv, 1 - pv]))

    for node in PID_NODES:
        sym_val = _subs_float(sym.get_pi(node), point)
        num_val = float(num.get_pi(node))
        assert sym_val == pytest.approx(num_val, abs=TOL), f"node {node} at {point}"


@pytest.mark.parametrize("point", POINTS)
@pytest.mark.parametrize("measure", [PID_WB, PID_MMI])
def test_pid_three_param(point, measure):
    """PID atoms match numeric on a genuinely structured 3-variable dist."""
    p, q = symbols("p q")
    outcomes = ["000", "011", "101", "110"]
    sym_pmf = [p * q, p * (1 - q), (1 - p) * q, (1 - p) * (1 - q)]
    sym = measure(symbolic_distribution(outcomes, sym_pmf))

    pv, qv = _point_values(point)
    num_pmf = [pv * qv, pv * (1 - qv), (1 - pv) * qv, (1 - pv) * (1 - qv)]
    num = measure(dit.Distribution(outcomes, num_pmf))

    for node in PID_NODES:
        sym_val = _subs_float(sym.get_pi(node), point)
        num_val = float(num.get_pi(node))
        assert sym_val == pytest.approx(num_val, abs=TOL), f"node {node} at {point}"
