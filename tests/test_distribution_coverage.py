"""
Targeted coverage tests for dit.distribution.Distribution.

These exercise constructor variants, display helpers, indexing edge cases,
scalar-distribution arithmetic/comparison operators, and conditioning paths
that the main test_distribution.py suite leaves uncovered.
"""

import numpy as np
import pytest

from dit import Distribution
from dit.exceptions import InvalidOutcome
from dit.samplespace import CartesianProduct


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
    d = Distribution(["00", "11"], [0.4, 0.4], base=2)
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


def test_hash():
    assert isinstance(hash(joint()), int)


def test_sample_space_override_non_cartesian():
    d = Distribution(["00", "11"], [0.5, 0.5])
    d._sample_space = [("0", "0"), ("1", "1")]
    assert list(d.sample_space()) == [("0", "0"), ("1", "1")]


def test_event_probability_scalar_outcomes():
    d = die(6)
    assert d.event_probability([1, 2, 3]) == pytest.approx(0.5)
