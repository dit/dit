"""
Targeted coverage tests for dit.distribution.Distribution.

These exercise constructor variants, display helpers, indexing edge cases,
scalar-distribution arithmetic/comparison operators, and conditioning paths
that the main test_distribution.py suite leaves uncovered.
"""

import numpy as np
import pytest

from dit import Distribution, pruned_samplespace
from dit.distribution import _is_zero, _symbolic_nonzero_mask
from dit.exceptions import InvalidOutcome
from dit.samplespace import CartesianProduct
from dit.symbolic import symbolic_distribution, symbols


def die(n=6):
    """A fair n-sided die over the numeric outcomes 1..n (scalar dist)."""
    return Distribution(list(range(1, n + 1)), [1 / n] * n)


def joint():
    """A 2-variable joint distribution with named RVs."""
    return Distribution(["00", "01", "10", "11"], [0.4, 0.1, 0.1, 0.4], rv_names=["X", "Y"])


# ── Constructor variants ──────────────────────────────────────────────────


def test_base_none_autodetect_from_outcomes():
    """base=None on a valid pmf auto-detects 'linear'."""
    d = Distribution(["00", "11"], [0.5, 0.5], base=None)
    assert d.get_base() == "linear"


def test_base_none_from_dataarray():
    """base=None when building from a DataArray falls back to 'linear'."""
    d = Distribution(["00", "11"], [0.5, 0.5])
    d2 = Distribution(d.DataArray, base=None)
    assert d2.get_base() == "linear"


def test_variable_length_string_outcomes():
    """Variable-length string outcomes are treated as opaque scalar outcomes."""
    d = Distribution(["red", "blue"], [0.5, 0.5])
    assert d.outcome_length() == 1
    assert set(d.outcomes) == {("red",), ("blue",)}


def test_sample_space_cartesian_product():
    """An explicit CartesianProduct sample space is honored."""
    ss = CartesianProduct([["0", "1"], ["0", "1"]])
    d = Distribution(["00", "11"], [0.5, 0.5], sample_space=ss)
    assert ("0", "1") in list(d.sample_space())


def test_sample_space_iterable():
    """An explicit iterable sample space is honored."""
    d = Distribution(["00", "11"], [0.5, 0.5], sample_space=["00", "01", "10", "11"])
    assert len(list(d.sample_space())) == 4


# ── pmf setter / dense path ────────────────────────────────────────────────


def test_pmf_setter_dense_length():
    """Setting pmf with full sample-space length densifies then re-sparsifies."""
    d = Distribution(["00", "11"], [0.5, 0.5])
    assert d.is_sparse()
    d.pmf = [0.25, 0.25, 0.25, 0.25]
    assert d["01"] == pytest.approx(0.25)
    assert d.is_sparse()


def test_pmf_setter_bad_length():
    d = Distribution(["00", "11"], [0.5, 0.5])
    with pytest.raises(ValueError, match="doesn't match outcomes"):
        d.pmf = [0.5, 0.3, 0.2]


# ── rv name handling ───────────────────────────────────────────────────────


def test_set_rv_names_swap():
    """Swapping names uses the two-pass temporary-rename path."""
    d = joint()
    d.set_rv_names(["Y", "X"])
    assert set(d.get_rv_names()) == {"X", "Y"}


def test_set_rv_names_noop():
    """Renaming to the identical names is a no-op."""
    d = joint()
    d.set_rv_names(["X", "Y"])
    assert d.get_rv_names() == ("X", "Y")


# ── container protocol ─────────────────────────────────────────────────────


def test_len_dense_vs_sparse():
    d = Distribution(["00", "11"], [0.5, 0.5])
    assert len(d) == 2
    d.make_dense()
    assert len(d) == 4


def test_iter_and_reversed():
    d = die(3)
    assert list(iter(d)) == [1, 2, 3]
    assert list(reversed(d)) == [3, 2, 1]


def test_is_joint_conditional_is_false():
    c = joint().condition_on("Y")
    assert c.is_conditional()
    assert not c.is_joint()


# ── has_outcome / atoms / rand ─────────────────────────────────────────────


def test_has_outcome_string_multidim():
    d = joint()
    assert d.has_outcome("00")
    assert d.has_outcome("00", null=False)


def test_has_outcome_null_false_zero():
    d = Distribution(["00", "11"], [0.5, 0.5])
    d.make_dense()
    assert d.has_outcome(("0", "1"), null=True)
    assert not d.has_outcome(("0", "1"), null=False)


def test_atoms():
    d = Distribution(["00", "11"], [0.5, 0.5])
    d.make_dense()
    assert set(d.atoms()) == {("0", "0"), ("0", "1"), ("1", "0"), ("1", "1")}
    assert set(d.atoms(patoms=True)) == {("0", "0"), ("1", "1")}


def test_rand_reproducible():
    d = die()
    sample = d.rand()
    assert sample in {1, 2, 3, 4, 5, 6}


def test_sample_space_override_roundtrip():
    d = Distribution(["00", "11"], [0.5, 0.5])
    override = CartesianProduct([["0", "1"], ["0", "1"]])
    d._sample_space = override
    assert d._sample_space is override


# ── display ────────────────────────────────────────────────────────────────


def test_to_string_scalar():
    out = die(3).to_string()
    assert "p(x)" in out


def test_to_string_str_outcomes():
    out = joint().to_string(str_outcomes=True)
    assert "00" in out


def test_to_string_conditional():
    out = joint().condition_on("Y").to_string()
    assert "|" in out


def test_to_string_exact_and_digits():
    out = die(4).to_string(exact=True)
    assert "1/4" in out
    out2 = joint().to_string(digits=2)
    assert "0.4" in out2


def test_to_html():
    html = joint()._to_html()
    assert "<table" in html


def test_plain_to_string_and_repr():
    d = joint()
    assert "Notation" in d._to_string()
    assert "Distribution" in repr(d)


# ── indexing edge cases ────────────────────────────────────────────────────


def test_getitem_string_scalar_dist():
    d = Distribution(["red", "blue"], [0.5, 0.5])
    assert d["red"] == pytest.approx(0.5)


def test_getitem_bad_scalar_key():
    d = Distribution(["red", "blue"], [0.5, 0.5])
    with pytest.raises(InvalidOutcome):
        d["green"]


def test_getitem_wrong_length_string():
    d = joint()
    with pytest.raises(InvalidOutcome):
        d["000"]


def test_delitem():
    d = joint()
    del d["00"]
    assert d["00"] == pytest.approx(0.0)


def test_setitem_string_and_scalar():
    d = joint()
    d["00"] = 0.5
    assert d["00"] == pytest.approx(0.5)
    s = die(3)
    s[1] = 0.5
    assert s[1] == pytest.approx(0.5)


# ── scalar arithmetic / comparison operators ───────────────────────────────


def test_scalar_binary_with_distribution():
    d = die(3)
    assert (d % d).is_joint() is False
    assert set((d // d).outcomes) <= {0, 1, 2, 3}
    assert set((d**2).outcomes) == {1, 4, 9}
    assert set((d**d).outcomes)  # nonempty


def test_scalar_right_operators():
    d = die(3)
    assert set((10 - d).outcomes) == {7, 8, 9}
    assert set((10 % d).outcomes)
    assert set((10 // d).outcomes)
    assert set((2**d).outcomes) == {2, 4, 8}


def test_scalar_comparisons():
    d = die(6)
    assert set((d <= 3).outcomes) == {0, 1}
    assert set((d < 3).outcomes) == {0, 1}
    assert set((d >= 3).outcomes) == {0, 1}
    assert set((d > 3).outcomes) == {0, 1}
    assert set((d <= d).outcomes) == {0, 1}
    assert set((d < d).outcomes) == {0, 1}
    assert set((d >= d).outcomes) == {0, 1}
    assert set((d > d).outcomes) == {0, 1}


def test_scalar_unary():
    d = die(3)
    assert set((-d).outcomes) == {-1, -2, -3}
    assert set(abs(-d).outcomes) == {1, 2, 3}


@pytest.mark.parametrize(
    "op",
    [
        lambda d, o: d % o,
        lambda d, o: d // o,
        lambda d, o: d**o,
        lambda d, o: d <= o,
        lambda d, o: d < o,
        lambda d, o: d >= o,
        lambda d, o: d > o,
    ],
)
def test_scalar_ops_notimplemented(op):
    d = die(3)
    with pytest.raises(TypeError):
        op(d, object())


# ── classmethods ───────────────────────────────────────────────────────────


def test_from_ndarray():
    arr = np.array([[0.25, 0.25], [0.25, 0.25]])
    d = Distribution.from_ndarray(arr)
    assert d.outcome_length() == 2
    assert sum(d.pmf) == pytest.approx(1.0)


def test_from_rv_discrete():
    stats = pytest.importorskip("scipy.stats")
    rv = stats.rv_discrete(values=([0, 1, 2], [0.2, 0.3, 0.5]))
    d = Distribution.from_rv_discrete(rv)
    assert d[(2,)] == pytest.approx(0.5)


# ── approximate equality ───────────────────────────────────────────────────


def test_is_approx_equal_named_dim_mismatch():
    a = Distribution(["00", "11"], [0.5, 0.5], rv_names=["X", "Y"])
    b = Distribution(["00", "11"], [0.5, 0.5], rv_names=["X", "Z"])
    assert not a.is_approx_equal(b)


def test_is_approx_equal_alphabet_size_mismatch():
    a = Distribution(["00", "11"], [0.5, 0.5], rv_names=["X", "Y"])
    b = Distribution(["00", "01", "12"], [0.4, 0.3, 0.3], rv_names=["X", "Y"])
    assert not a.is_approx_equal(b)


def test_is_approx_equal_unnamed():
    a = Distribution(["00", "11"], [0.5, 0.5])
    b = Distribution(["00", "11"], [0.5, 0.5])
    c = Distribution(["00", "11"], [0.4, 0.6])
    assert a.is_approx_equal(b)
    assert not a.is_approx_equal(c)


def test_is_approx_equal_alphabet_mismatch_unnamed():
    a = Distribution(["00", "11"], [0.5, 0.5])
    b = Distribution(["aa", "bb"], [0.5, 0.5])
    assert not a.is_approx_equal(b)


# ── normalize ──────────────────────────────────────────────────────────────


def test_normalize_log_unconditional():
    d = Distribution(["00", "11"], [0.4, 0.4], base=2, validate=False)
    d.normalize()
    lin = d.copy(base="linear")
    assert sum(lin.pmf) == pytest.approx(1.0)


def test_normalize_conditional():
    c = joint().condition_on("Y")
    c.normalize()
    assert c.is_conditional()


# ── misc helpers ───────────────────────────────────────────────────────────


def test_is_homogeneous():
    assert joint().is_homogeneous()
    d = Distribution(["00", "12"], [0.5, 0.5], rv_names=["X", "Y"])
    assert not d.is_homogeneous()


def test_has_outcome_invalid_returns_false():
    d = joint()
    assert not d.has_outcome(("9", "9"))


def test_event_space():
    d = Distribution(["0", "1"], [0.5, 0.5])
    events = list(d.event_space())
    assert len(events) == 4  # powerset of 2 outcomes


def test_product_property():
    import itertools

    assert joint()._product is itertools.product


def test_set_base_log_to_log():
    d = joint().copy(base=2)
    d.set_base("e")
    assert d.is_log()
    assert d.get_base() == "e"


def test_str_dunder():
    assert "p(x)" in str(die(3))


def test_resolve_rv_names_out_of_range():
    from dit.exceptions import ditException

    with pytest.raises(ditException, match="out of range"):
        joint().marginal([5])


def test_condition_on_dit_compat_list():
    marg, cdists = joint().condition_on(["Y"])
    assert len(cdists) == 2
    assert all(cd.is_joint() is False for cd in cdists)


def test_condition_on_dit_compat_rvs():
    marg, cdists = joint().condition_on(["Y"], rvs=["X"])
    assert len(cdists) == 2


def test_condition_on_dit_compat_string_positional_with_rvs():
    marg, cdists = joint().condition_on("Y", rvs=["X"])
    assert len(cdists) == 2


def test_joint_subtraction():
    d = joint()
    diff = d - d
    assert float(diff.DataArray.sum()) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "op",
    [
        lambda d: d - object(),
        lambda d: 10 % d,
        lambda d: 10 // d,
        lambda d: 2**d,
    ],
)
def test_joint_op_notimplemented(op):
    with pytest.raises(TypeError):
        op(joint())


def test_getitem_numeric_bad_key():
    with pytest.raises(InvalidOutcome):
        die(3)[99]


def test_setitem_string_scalar_dist():
    d = Distribution(["red", "blue"], [0.5, 0.5])
    d["red"] = 0.25
    assert d["red"] == pytest.approx(0.25)


def test_unhashable():
    with pytest.raises(TypeError):
        hash(joint())


def test_sample_space_override_non_cartesian():
    d = Distribution(["00", "11"], [0.5, 0.5])
    d._sample_space = [("0", "0"), ("1", "1")]
    assert list(d.sample_space()) == [("0", "0"), ("1", "1")]


def test_event_probability_scalar_outcomes():
    d = die(6)
    assert d.event_probability([1, 2, 3]) == pytest.approx(0.5)


# ── Edge cases: undecidable zeros, symbolic paths, empty/trivial dists ─────


@pytest.fixture
def sympy():
    """The sympy module; symbolic tests are skipped when it is not installed."""
    return pytest.importorskip("sympy")


class _Undecidable:
    """A value whose comparison with zero has no truth value, like a sympy relational."""

    def __eq__(self, other):
        return self

    def __bool__(self):
        raise TypeError("truth value is undecidable")


def test_undecidable_values_are_in_the_support():
    """A value whose equality with zero cannot be decided is not a structural zero."""
    assert _is_zero(0)
    assert _is_zero(0.0)
    assert not _is_zero(1e-300)
    assert not _is_zero(_Undecidable())
    values = np.array([0, _Undecidable(), 0.0, 0.5], dtype=object)
    assert _symbolic_nonzero_mask(values).tolist() == [False, True, False, True]


def test_symbolic_base_none_is_linear(sympy):
    """base=None skips float-based auto-detection for a symbolic pmf and uses 'linear'."""
    p = symbols("p")
    d = Distribution(["0", "1"], [p, 1 - p], base=None)
    assert d.is_symbolic()
    assert d.get_base() == "linear"
    assert list(d.pmf) == [p, 1 - p]


def test_symbolic_dense_pmf_keeps_structural_zeros(sympy):
    """A dense symbolic pmf reports every outcome, with exact zeros off the support."""
    p = symbols("p")
    d = symbolic_distribution(["00", "11"], [p, 1 - p])
    assert list(d.pmf) == [p, 1 - p]
    d.make_dense()
    assert d.outcomes == (("0", "0"), ("0", "1"), ("1", "0"), ("1", "1"))
    assert list(d.pmf) == [p, 0, 0, 1 - p]
    assert len(d) == 4


def test_symbolic_lookup_of_opaque_string_outcome(sympy):
    """Looking up a multi-character scalar outcome keeps the exact symbolic probability."""
    p = symbols("p")
    d = symbolic_distribution(["red", "blue"], [p, 1 - p])
    assert d.outcome_length() == 1
    assert d["red"] == p
    assert d["blue"] == 1 - p
    assert isinstance(d["red"], sympy.Expr)


def test_symbolic_condition_on(sympy):
    """Conditioning a symbolic joint divides exactly by the symbolic marginal."""
    a, b, c = symbols("a b c")
    d = symbolic_distribution(["00", "01", "10", "11"], [a, b, c, 1 - a - b - c], rv_names=["X", "Y"])
    cond = d.condition_on("X")
    assert cond.is_symbolic()
    assert cond.free_vars == {"Y"}
    assert cond.given_vars == {"X"}
    assert sympy.simplify(cond[("0", "0")] - a / (a + b)) == 0
    assert sympy.simplify(cond[("1", "1")] - (1 - a - b - c) / (1 - a - b)) == 0
    for x in "01":
        assert sympy.simplify(cond[(x, "0")] + cond[(x, "1")]) == 1
    assert cond.validate()


def test_symbolic_validate_rejects_bad_total(sympy):
    """A symbolic pmf without free symbols must sum exactly to one."""
    third = sympy.Rational(1, 3)
    with pytest.raises(ValueError, match="sums to 2/3"):
        symbolic_distribution(["0", "1"], [third, third])


def test_validate_negative_probabilities():
    """Negative probabilities are rejected even when the pmf sums to one."""
    with pytest.raises(ValueError, match="negative probabilities"):
        Distribution(["0", "1"], [1.5, -0.5])
    d = Distribution(["0", "1"], [1.5, -0.5], validate=False)
    with pytest.raises(ValueError, match=r"negative probabilities: \[-0\.5\]"):
        d.validate()
    # Round-off below the tolerance is accepted.
    assert Distribution(["0", "1"], [1 + 1e-12, -1e-12]).validate()


def test_outcomes_index():
    """_outcomes_index maps each reported outcome to its position in `outcomes`."""
    d = Distribution(["00", "11"], [0.5, 0.5])
    assert d._outcomes_index == {("0", "0"): 0, ("1", "1"): 1}
    d.make_dense()
    assert d._outcomes_index == {("0", "0"): 0, ("0", "1"): 1, ("1", "0"): 2, ("1", "1"): 3}


def test_marginal_onto_no_variables():
    """Marginalizing out every variable leaves the trivial distribution p() with mass one."""
    t = joint().marginal([])
    assert t.outcome_length() == 0
    assert t.alphabet == ()
    assert t.is_homogeneous()
    assert t.outcomes == ((),)
    assert t.pmf == pytest.approx([1.0])
    assert t.to_string(exact=False).splitlines()[-2:] == ["x    p(x)", "()   1.0"]


def test_condition_on_every_variable_gives_trivial_slices():
    """Conditioning on all kept variables (dit-compat form) gives one trivial slice per outcome."""
    d = joint()
    marg, slices = d.condition_on(crvs=["X", "Y"])
    assert marg.is_approx_equal(d)
    assert len(slices) == 4
    for s in slices:
        assert s.outcome_length() == 0
        assert float(s.data) == pytest.approx(1.0)
    marg, slices = d.condition_on(["X"], rvs=[])
    assert marg.pmf == pytest.approx([0.5, 0.5])
    assert [float(s.data) for s in slices] == pytest.approx([1.0, 1.0])


def test_zipped_over_pruned_sample_space():
    """On a restricted sample space, zipped() skips its null outcomes and 'atoms' lists exactly it."""
    d = pruned_samplespace(Distribution(["00", "11"], [0.5, 0.5]), sample_space=[("0", "1")])
    assert list(d.zipped()) == [(("0", "0"), 0.5), (("1", "1"), 0.5)]
    # ('1', '0') is in the Cartesian product of the alphabets but was pruned.
    assert list(d.zipped("atoms")) == [(("0", "0"), 0.5), (("0", "1"), 0.0), (("1", "1"), 0.5)]


def _table_body(html):
    """The rows of the probability table in an HTML rendering of a distribution."""
    return html.split("<tbody>")[1].split("</tbody>")[0]


def test_display_lists_only_positive_outcomes():
    """The plain-text and HTML tables of a joint distribution omit null outcomes."""
    d = Distribution(["00", "11"], [0.25, 0.75], rv_names=["X", "Y"])
    table = d._to_string(digits=2).split("\n\n")[1].splitlines()
    assert table == ["X   Y      p", "0   0   0.25", "1   1   0.75"]
    body = _table_body(d._to_html(digits=2))
    assert body.count("<tr>") == 2
    assert "0.25" in body
    assert "0.75" in body


def test_display_of_empty_distribution():
    """A distribution without mass renders an explicit empty marker in text and HTML."""
    d = Distribution(["00", "11"], [0.0, 0.0], rv_names=["X", "Y"], validate=False)
    text = d._to_string()
    assert "Notation:  p(X,Y)" in text
    assert text.endswith("(empty distribution)")
    body = _table_body(d._repr_html_())
    assert body.count("<tr>") == 1
    assert 'colspan="3"' in body
    assert "(empty)" in body


def test_coalesce_without_mass_raises():
    """Coalescing needs at least one outcome with positive probability."""
    d = Distribution(["00", "11"], [0.0, 0.0], validate=False)
    with pytest.raises(ValueError, match="no non-zero outcomes to coalesce"):
        d.coalesce([[0], [1]])


def test_radd_nonzero_scalar_to_joint_is_unsupported():
    """Only ``0 + p`` (as used by sum()) is defined for named distributions."""
    d = joint()
    assert d.__radd__(2) is NotImplemented
    assert d.__radd__("x") is NotImplemented
    with pytest.raises(TypeError, match="unsupported operand"):
        2 + d


def test_conditional_entropy_ignores_empty_slices():
    """Conditional entropy averages over slices with mass; with no mass at all it is zero."""
    alphabets = [[0, 1], [0, 1]]
    half = Distribution.from_array(np.array([[0.5, 0.5], [0.0, 0.0]]), ["X", "Y"], alphabets, given_vars={"X"})
    assert half.entropy() == pytest.approx(1.0)
    empty = Distribution.from_array(np.zeros((2, 2)), ["X", "Y"], alphabets, given_vars={"X"})
    assert empty.entropy() == 0.0


def test_normalize_without_mass_leaves_zeros():
    """Normalizing a distribution with zero total mass leaves it unchanged instead of making NaNs."""
    d = Distribution(["00", "11"], [0.0, 0.0], validate=False)
    d.normalize()
    np.testing.assert_array_equal(d.to_numpy(), np.zeros((2, 2)))


def test_matmul_pairs_colliding_string_symbols_as_tuples():
    """Pairing falls back to tuples when concatenated strings would collide and merge mass."""
    # '0' + '11' and '01' + '1' would both be '011'.
    a = Distribution([("0",), ("01",)], [0.5, 0.5], rv_names=["X"])
    b = Distribution([("1",), ("11",)], [0.5, 0.5], rv_names=["X"])
    prod = a @ b
    assert prod.get_rv_names() == ("X",)
    assert prod.to_dict() == pytest.approx(
        {(("0", "1"),): 0.25, (("0", "11"),): 0.25, (("01", "1"),): 0.25, (("01", "11"),): 0.25}
    )


def test_outcome_transforms_coerce_booleans_to_int():
    """Boolean results of outcome transforms become the integer outcomes 0 and 1."""
    d = die(6)
    even = d._apply_outcome_fn(lambda o: o % 2 == 0)
    assert even.to_dict() == pytest.approx({0: 0.5, 1: 0.5})
    big = d._apply_outcome_fn(lambda o: np.bool_(o > 4))
    assert big.to_dict() == pytest.approx({0: 2 / 3, 1: 1 / 3})
    same = d._combine_independent(d, lambda a, b: a == b)
    assert same.to_dict() == pytest.approx({0: 5 / 6, 1: 1 / 6})
    # bool == int, so check the types to see the coercion.
    assert all(type(o) is int for o in even.outcomes + big.outcomes + same.outcomes)


def test_native_alphabet_converts_numpy_scalars():
    """_native_alphabet turns numpy scalars into the equivalent Python values for display."""
    clean = Distribution._native_alphabet(((np.int64(1), np.int64(2)), (np.str_("a"), "b")))
    assert clean == ((1, 2), ("a", "b"))
    assert [type(v) for alpha in clean for v in alpha] == [int, int, str, str]
