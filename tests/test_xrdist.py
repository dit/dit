"""
Tests for dit.xrdist.XRDistribution.
"""

import numpy as np
import pytest

from dit.xrdist import XRDistribution

xr = pytest.importorskip("xarray")

# ─── Helpers ─────────────────────────────────────────────────────────────


def _make_pxy():
    """Joint distribution p(X,Y) with X,Y in {0,1}."""
    arr = np.array([[0.1, 0.2], [0.3, 0.4]])
    return XRDistribution.from_array(
        arr,
        dim_names=["X", "Y"],
        alphabets=[[0, 1], [0, 1]],
    )


def _make_uniform_xyz():
    """Uniform joint distribution p(X,Y,Z), each binary."""
    arr = np.ones((2, 2, 2)) / 8
    return XRDistribution.from_array(
        arr,
        dim_names=["X", "Y", "Z"],
        alphabets=[[0, 1], [0, 1], [0, 1]],
    )


def _make_xor():
    """XOR distribution: Z = X xor Y, uniform X,Y."""
    arr = np.zeros((2, 2, 2))
    for x in range(2):
        for y in range(2):
            arr[x, y, x ^ y] = 0.25
    return XRDistribution.from_array(
        arr,
        dim_names=["X", "Y", "Z"],
        alphabets=[[0, 1], [0, 1], [0, 1]],
    )


# ─── Construction ────────────────────────────────────────────────────────


class TestInit:
    def test_basic(self):
        p = _make_pxy()
        assert p.free_vars == frozenset({"X", "Y"})
        assert p.given_vars == frozenset()
        assert p.is_joint()
        assert not p.is_conditional()
        assert p.shape == (2, 2)

    def test_outcomes_pmf_missing(self):
        with pytest.raises(ValueError, match="pmf is required"):
            XRDistribution(["00", "01", "10", "11"])

    def test_free_given_mismatch(self):
        arr = xr.DataArray(np.array([[0.5, 0.5]]), dims=["X", "Y"], coords={"X": [0], "Y": [0, 1]})
        with pytest.raises(ValueError, match="disjoint"):
            XRDistribution(arr, free_vars={"X", "Y"}, given_vars={"Y"})

    def test_free_given_incomplete(self):
        arr = xr.DataArray(np.array([[0.5, 0.5]]), dims=["X", "Y"], coords={"X": [0], "Y": [0, 1]})
        with pytest.raises(ValueError, match="cover all"):
            XRDistribution(arr, free_vars={"X"}, given_vars=set())

    def test_free_only(self):
        arr = xr.DataArray(np.array([[0.25, 0.25], [0.25, 0.25]]), dims=["X", "Y"], coords={"X": [0, 1], "Y": [0, 1]})
        p = XRDistribution(arr, free_vars={"X"})
        assert p.free_vars == frozenset({"X"})
        assert p.given_vars == frozenset({"Y"})

    def test_given_only(self):
        arr = xr.DataArray(np.array([[0.5, 0.5], [0.5, 0.5]]), dims=["X", "Y"], coords={"X": [0, 1], "Y": [0, 1]})
        p = XRDistribution(arr, given_vars={"Y"})
        assert p.free_vars == frozenset({"X"})
        assert p.given_vars == frozenset({"Y"})


class TestOutcomesPmfConstruction:
    def test_basic_outcomes_pmf(self):
        """Construct from outcomes + pmf, like dit.Distribution."""
        xrd = XRDistribution(
            ["00", "01", "10", "11"],
            [0.25, 0.25, 0.25, 0.25],
            rv_names=["X", "Y"],
        )
        assert xrd.free_vars == frozenset({"X", "Y"})
        assert xrd.given_vars == frozenset()
        assert xrd.shape == (2, 2)
        np.testing.assert_allclose(float(xrd.data.sel(X="0", Y="0")), 0.25)
        np.testing.assert_allclose(float(xrd.data.sel(X="1", Y="1")), 0.25)
        assert xrd.validate()

    def test_default_rv_names(self):
        """Without rv_names, default to X0, X1, ..."""
        xrd = XRDistribution(["00", "01", "10", "11"], [0.25, 0.25, 0.25, 0.25])
        assert xrd.dims == ("X0", "X1")
        assert xrd.free_vars == frozenset({"X0", "X1"})

    def test_sparse_outcomes(self):
        """Not all outcomes in the full product space need to be given."""
        xrd = XRDistribution(["00", "11"], [0.5, 0.5], rv_names=["X", "Y"])
        assert xrd.shape == (2, 2)
        np.testing.assert_allclose(float(xrd.data.sel(X="0", Y="0")), 0.5)
        np.testing.assert_allclose(float(xrd.data.sel(X="0", Y="1")), 0.0)
        np.testing.assert_allclose(float(xrd.data.sel(X="1", Y="1")), 0.5)

    def test_tuple_outcomes(self):
        """Outcomes can be tuples."""
        xrd = XRDistribution(
            [(0, 0), (0, 1), (1, 0), (1, 1)],
            [0.1, 0.2, 0.3, 0.4],
            rv_names=["A", "B"],
        )
        assert xrd.shape == (2, 2)
        np.testing.assert_allclose(float(xrd.data.sel(A=0, B=0)), 0.1)
        np.testing.assert_allclose(float(xrd.data.sel(A=1, B=1)), 0.4)

    def test_three_variables(self):
        """Outcomes with three characters give three dimensions."""
        outcomes = ["000", "001", "010", "011", "100", "101", "110", "111"]
        pmf = [1 / 8] * 8
        xrd = XRDistribution(outcomes, pmf, rv_names=["X", "Y", "Z"])
        assert xrd.shape == (2, 2, 2)
        np.testing.assert_allclose(float(xrd.data.sum()), 1.0)

    def test_wrong_rv_names_count(self):
        with pytest.raises(ValueError, match="Expected 2"):
            XRDistribution(["00", "01"], [0.5, 0.5], rv_names=["X"])

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            XRDistribution(["00", "01", "10"], [0.5, 0.5])

    def test_empty_outcomes(self):
        with pytest.raises(ValueError, match="non-empty"):
            XRDistribution([], [])

    def test_with_given_vars(self):
        """Can set given_vars when constructing from outcomes."""
        xrd = XRDistribution(
            ["00", "01", "10", "11"],
            [0.5, 0.5, 0.5, 0.5],
            rv_names=["X", "Y"],
            free_vars={"Y"},
            given_vars={"X"},
        )
        assert xrd.free_vars == frozenset({"Y"})
        assert xrd.given_vars == frozenset({"X"})

    def test_equivalence_with_from_distribution(self):
        """Outcomes+pmf construction should match from_distribution."""
        import dit

        d = dit.Distribution([(0, 0), (0, 1), (1, 0), (1, 1)], [0.1, 0.2, 0.3, 0.4])
        xrd_old = XRDistribution.from_distribution(d, ["A", "B"])
        xrd_new = XRDistribution(
            [(0, 0), (0, 1), (1, 0), (1, 1)],
            [0.1, 0.2, 0.3, 0.4],
            rv_names=["A", "B"],
        )
        np.testing.assert_allclose(xrd_new.data.values, xrd_old.data.values, atol=1e-12)

    def test_with_custom_base(self):
        """Can specify a log base."""
        xrd = XRDistribution(["00", "11"], [0.5, 0.5], rv_names=["X", "Y"], base=2)
        assert xrd.is_log()
        assert xrd.get_base() == 2


class TestDictConstruction:
    def test_basic_dict(self):
        """Construct from a dict mapping outcomes to probabilities."""
        xrd = XRDistribution({"00": 0.5, "11": 0.5}, rv_names=["X", "Y"])
        assert xrd.free_vars == frozenset({"X", "Y"})
        assert xrd.shape == (2, 2)
        np.testing.assert_allclose(float(xrd.data.sel(X="0", Y="0")), 0.5)
        np.testing.assert_allclose(float(xrd.data.sel(X="0", Y="1")), 0.0)
        np.testing.assert_allclose(float(xrd.data.sel(X="1", Y="1")), 0.5)

    def test_dict_default_names(self):
        """Dict construction defaults to X0, X1, ..."""
        xrd = XRDistribution({"00": 0.5, "11": 0.5})
        assert xrd.dims == ("X0", "X1")

    def test_dict_three_outcomes(self):
        xrd = XRDistribution(
            {"00": 0.25, "01": 0.25, "10": 0.25, "11": 0.25},
            rv_names=["A", "B"],
        )
        assert xrd.validate()
        np.testing.assert_allclose(float(xrd.data.sum()), 1.0)

    def test_dict_tuple_keys(self):
        """Dict with tuple keys."""
        xrd = XRDistribution(
            {(0, 0): 0.5, (1, 1): 0.5},
            rv_names=["X", "Y"],
        )
        assert xrd.shape == (2, 2)
        np.testing.assert_allclose(float(xrd.data.sel(X=0, Y=0)), 0.5)

    def test_dict_empty(self):
        with pytest.raises(ValueError, match="non-empty"):
            XRDistribution({})


class TestFromDistribution:
    def test_basic(self):
        import dit

        d = dit.Distribution([(0, 0), (0, 1), (1, 0), (1, 1)], [0.1, 0.2, 0.3, 0.4])
        xrd = XRDistribution.from_distribution(d, ["A", "B"])
        assert xrd.free_vars == frozenset({"A", "B"})
        assert xrd.shape == (2, 2)
        np.testing.assert_allclose(float(xrd.data.sel(A=0, B=0)), 0.1)
        np.testing.assert_allclose(float(xrd.data.sel(A=1, B=1)), 0.4)

    def test_auto_names(self):
        import dit

        d = dit.Distribution([(0, 0), (1, 1)], [0.5, 0.5])
        xrd = XRDistribution.from_distribution(d)
        assert xrd.dims == ("X0", "X1")

    def test_wrong_name_count(self):
        import dit

        d = dit.Distribution([(0, 0)], [1.0])
        with pytest.raises(ValueError, match="Expected 2"):
            XRDistribution.from_distribution(d, ["A"])


class TestFromArray:
    def test_basic(self):
        arr = np.array([0.5, 0.5])
        xrd = XRDistribution.from_array(arr, ["X"], [[0, 1]])
        assert xrd.free_vars == frozenset({"X"})
        assert xrd.shape == (2,)

    def test_with_given(self):
        arr = np.array([[0.5, 0.5], [0.3, 0.7]])
        xrd = XRDistribution.from_array(
            arr,
            ["X", "Y"],
            [[0, 1], [0, 1]],
            free_vars={"Y"},
            given_vars={"X"},
        )
        assert xrd.free_vars == frozenset({"Y"})
        assert xrd.given_vars == frozenset({"X"})


class TestFromFactors:
    def test_chain_rule(self):
        p_xy = _make_pxy()
        p_x = p_xy.marginal("X")
        p_y_given_x = p_xy.condition_on("X")
        joint = XRDistribution.from_factors(p_x, p_y_given_x)
        assert joint.free_vars == frozenset({"X", "Y"})
        assert joint.given_vars == frozenset()
        np.testing.assert_allclose(joint.data.values, p_xy.data.values, atol=1e-12)


# ─── Properties ──────────────────────────────────────────────────────────


class TestProperties:
    def test_dims(self):
        p = _make_pxy()
        assert p.dims == ("X", "Y")

    def test_all_vars(self):
        p = _make_pxy()
        assert p.all_vars == frozenset({"X", "Y"})

    def test_alphabet(self):
        p = _make_pxy()
        assert p.alphabet == ((0, 1), (0, 1))

    def test_outcome_length(self):
        p = _make_pxy()
        assert p.outcome_length() == 2

    def test_get_rv_names(self):
        p = _make_pxy()
        assert p.get_rv_names() == ("X", "Y")

    def test_set_rv_names(self):
        p = _make_pxy()
        p.set_rv_names(["A", "B"])
        assert p.dims == ("A", "B")
        assert p.free_vars == frozenset({"A", "B"})

    def test_set_rv_names_wrong_count(self):
        p = _make_pxy()
        with pytest.raises(ValueError):
            p.set_rv_names(["A"])

    def test_make_dense_sparse_noop(self):
        p = _make_pxy()
        assert p.make_dense() == 0
        assert p.make_sparse() == 0


# ─── Validation ──────────────────────────────────────────────────────────


class TestValidation:
    def test_valid_joint(self):
        p = _make_pxy()
        assert p.validate()

    def test_invalid_joint(self):
        arr = np.array([[0.1, 0.2], [0.3, 0.5]])
        p = XRDistribution.from_array(arr, ["X", "Y"], [[0, 1], [0, 1]])
        with pytest.raises(ValueError, match="sums to"):
            p.validate()

    def test_valid_conditional(self):
        p = _make_pxy()
        c = p.condition_on("X")
        assert c.validate()

    def test_invalid_conditional(self):
        arr = np.array([[0.3, 0.5], [0.3, 0.7]])
        p = XRDistribution.from_array(
            arr,
            ["X", "Y"],
            [[0, 1], [0, 1]],
            free_vars={"Y"},
            given_vars={"X"},
        )
        with pytest.raises(ValueError, match="normalise"):
            p.validate()


# ─── Marginal / Marginalize ─────────────────────────────────────────────


class TestMarginal:
    def test_marginal_keeps_var(self):
        p = _make_pxy()
        p_x = p.marginal("X")
        assert p_x.free_vars == frozenset({"X"})
        np.testing.assert_allclose(p_x.data.sel(X=0), 0.3)
        np.testing.assert_allclose(p_x.data.sel(X=1), 0.7)

    def test_marginal_sum_1(self):
        p = _make_pxy()
        p_x = p.marginal("X")
        assert p_x.validate()

    def test_marginal_invalid_var(self):
        p = _make_pxy()
        with pytest.raises(ValueError, match="not free"):
            p.marginal("Z")

    def test_marginal_keep_all(self):
        p = _make_pxy()
        p2 = p.marginal("X", "Y")
        np.testing.assert_allclose(p2.data.values, p.data.values)

    def test_marginalize(self):
        p = _make_pxy()
        p_x = p.marginalize("Y")
        assert p_x.free_vars == frozenset({"X"})
        np.testing.assert_allclose(p_x.data.sel(X=0), 0.3)

    def test_marginalize_invalid_var(self):
        p = _make_pxy()
        with pytest.raises(ValueError, match="not free"):
            p.marginalize("Z")


# ─── Condition On ────────────────────────────────────────────────────────


class TestConditionOn:
    def test_basic(self):
        p = _make_pxy()
        c = p.condition_on("X")
        assert c.free_vars == frozenset({"Y"})
        assert c.given_vars == frozenset({"X"})
        assert c.validate()

    def test_conditional_sums_to_1(self):
        p = _make_pxy()
        c = p.condition_on("X")
        # p(Y|X=0): [0.1/0.3, 0.2/0.3]
        np.testing.assert_allclose(float(c.data.sel(X=0).sum()), 1.0, atol=1e-12)
        np.testing.assert_allclose(float(c.data.sel(X=1).sum()), 1.0, atol=1e-12)

    def test_invalid_var(self):
        p = _make_pxy()
        with pytest.raises(ValueError):
            p.condition_on("Z")

    def test_condition_all_free(self):
        p = _make_pxy()
        with pytest.raises(ValueError, match="all free"):
            p.condition_on("X", "Y")

    def test_multiple_cond_vars(self):
        p = _make_xor()
        c = p.condition_on("X", "Y")
        assert c.free_vars == frozenset({"Z"})
        assert c.given_vars == frozenset({"X", "Y"})
        assert c.validate()


# ─── Multiplication ─────────────────────────────────────────────────────


class TestMultiplication:
    def test_chain_rule(self):
        """p(X) * p(Y|X) should reconstruct p(X,Y)."""
        p_xy = _make_pxy()
        p_x = p_xy.marginal("X")
        p_y_given_x = p_xy.condition_on("X")
        result = p_x * p_y_given_x
        assert result.free_vars == frozenset({"X", "Y"})
        assert result.given_vars == frozenset()
        np.testing.assert_allclose(result.data.values, p_xy.data.values, atol=1e-12)

    def test_three_way_chain(self):
        """p(X) * p(Y|X) * p(Z|X,Y) should reconstruct p(X,Y,Z)."""
        p_xyz = _make_xor()
        p_xy = p_xyz.marginal("X", "Y")
        p_x = p_xyz.marginal("X")
        p_y_given_x = p_xy.condition_on("X")
        p_z_given_xy = p_xyz.condition_on("X", "Y")

        result = p_x * p_y_given_x * p_z_given_xy
        assert result.free_vars == frozenset({"X", "Y", "Z"})
        assert result.given_vars == frozenset()
        np.testing.assert_allclose(result.data.values, p_xyz.data.values, atol=1e-12)

    def test_partial_application(self):
        """p(Z|X,Y) * p(X) should give p(X,Z|Y)."""
        p_xyz = _make_xor()
        p_x = p_xyz.marginal("X")
        p_z_given_xy = p_xyz.condition_on("X", "Y")
        result = p_x * p_z_given_xy
        assert result.free_vars == frozenset({"X", "Z"})
        assert result.given_vars == frozenset({"Y"})

    def test_scalar_mul(self):
        p = _make_pxy()
        p2 = p * 2
        np.testing.assert_allclose(p2.data.values, p.data.values * 2)
        assert p2.free_vars == p.free_vars

    def test_rmul_scalar(self):
        p = _make_pxy()
        p2 = 3 * p
        np.testing.assert_allclose(p2.data.values, p.data.values * 3)

    def test_free_overlap_error(self):
        p = _make_pxy()
        with pytest.raises(ValueError, match="free variables"):
            p * p

    def test_unsatisfied_given_passthrough(self):
        """Unsatisfied given vars pass through to the result."""
        arr_cond = np.array([[0.5, 0.5], [0.5, 0.5]])
        c = XRDistribution.from_array(
            arr_cond,
            ["Z", "W"],
            [[0, 1], [0, 1]],
            free_vars={"Z"},
            given_vars={"W"},
        )
        p = _make_pxy()
        result = p * c
        assert result.free_vars == frozenset({"X", "Y", "Z"})
        assert result.given_vars == frozenset({"W"})

    def test_not_implemented_type(self):
        p = _make_pxy()
        assert p.__mul__("hello") is NotImplemented
        assert p.__rmul__("hello") is NotImplemented


# ─── Division ────────────────────────────────────────────────────────────


class TestDivision:
    def test_conditioning_by_division(self):
        """p(X,Y) / p(X) should give p(Y|X)."""
        p_xy = _make_pxy()
        p_x = p_xy.marginal("X")
        result = p_xy / p_x
        assert result.free_vars == frozenset({"Y"})
        assert result.given_vars == frozenset({"X"})
        # Should equal condition_on result
        expected = p_xy.condition_on("X")
        np.testing.assert_allclose(result.data.values, expected.data.values, atol=1e-12)

    def test_scalar_div(self):
        p = _make_pxy()
        p2 = p / 2
        np.testing.assert_allclose(p2.data.values, p.data.values / 2)
        assert p2.free_vars == p.free_vars

    def test_invalid_division(self):
        p_xy = _make_pxy()
        p = _make_xor()
        with pytest.raises(ValueError, match="not in numerator"):
            p_xy / p

    def test_zero_denominator(self):
        arr = np.array([[0.5, 0.0], [0.0, 0.5]])
        p_xy = XRDistribution.from_array(arr, ["X", "Y"], [[0, 1], [0, 1]])
        p_x = p_xy.marginal("X")
        result = p_xy / p_x
        # Where p(X) = 0, result should be 0
        assert float(result.data.sel(X=1, Y=0)) == 0.0

    def test_not_implemented_type(self):
        p = _make_pxy()
        assert p.__truediv__("hello") is NotImplemented

    def test_roundtrip_mul_div(self):
        """p(X,Y) / p(X) * p(X) should give back p(X,Y)."""
        p_xy = _make_pxy()
        p_x = p_xy.marginal("X")
        p_y_given_x = p_xy / p_x
        reconstructed = p_x * p_y_given_x
        np.testing.assert_allclose(reconstructed.data.values, p_xy.data.values, atol=1e-12)


# ─── Entropy and Mutual Information ─────────────────────────────────────


class TestInformationTheory:
    def test_entropy_uniform(self):
        arr = np.array([0.5, 0.5])
        p = XRDistribution.from_array(arr, ["X"], [[0, 1]])
        np.testing.assert_allclose(p.entropy(base=2), 1.0, atol=1e-12)

    def test_entropy_deterministic(self):
        arr = np.array([1.0, 0.0])
        p = XRDistribution.from_array(arr, ["X"], [[0, 1]])
        np.testing.assert_allclose(p.entropy(base=2), 0.0, atol=1e-12)

    def test_entropy_joint(self):
        p = _make_pxy()
        h = p.entropy(base=2)
        # Manually: -sum(p log2 p)
        vals = p.data.values.ravel()
        vals = vals[vals > 0]
        expected = float(-np.sum(vals * np.log2(vals)))
        np.testing.assert_allclose(h, expected, atol=1e-12)

    def test_conditional_entropy(self):
        """Conditional entropy is the average per-slice entropy."""
        p = _make_pxy()
        p_y_given_x = p.condition_on("X")
        h_y_given_x = p_y_given_x.entropy(base=2)
        # Average of H(Y|X=0) and H(Y|X=1)
        # p(Y|X=0) = [1/3, 2/3], p(Y|X=1) = [3/7, 4/7]
        h_0 = -((1 / 3) * np.log2(1 / 3) + (2 / 3) * np.log2(2 / 3))
        h_1 = -((3 / 7) * np.log2(3 / 7) + (4 / 7) * np.log2(4 / 7))
        expected = (h_0 + h_1) / 2
        np.testing.assert_allclose(h_y_given_x, expected, atol=1e-10)

    def test_conditional_entropy_from_joint(self):
        """H(X,Y) - H(X) gives the true conditional entropy."""
        p = _make_pxy()
        h_xy = p.entropy(base=2)
        h_x = p.marginal("X").entropy(base=2)
        true_h_y_given_x = h_xy - h_x
        assert true_h_y_given_x > 0

    def test_mutual_information(self):
        p = _make_pxy()
        mi = p.mutual_information("X", "Y", base=2)
        h_x = p.marginal("X").entropy(base=2)
        h_y = p.marginal("Y").entropy(base=2)
        h_xy = p.entropy(base=2)
        np.testing.assert_allclose(mi, h_x + h_y - h_xy, atol=1e-12)

    def test_mutual_information_independent(self):
        # Independent X,Y
        arr = np.array([[0.25, 0.25], [0.25, 0.25]])
        p = XRDistribution.from_array(arr, ["X", "Y"], [[0, 1], [0, 1]])
        mi = p.mutual_information("X", "Y", base=2)
        np.testing.assert_allclose(mi, 0.0, atol=1e-12)

    def test_mutual_information_conditional_error(self):
        p = _make_pxy().condition_on("X")
        with pytest.raises(ValueError, match="joint"):
            p.mutual_information("X", "Y")

    def test_mi_set_variables(self):
        p_xyz = _make_xor()
        mi = p_xyz.mutual_information({"X"}, {"Y"}, base=2)
        np.testing.assert_allclose(mi, 0.0, atol=1e-12)


# ─── Selection / Indexing ────────────────────────────────────────────────


class TestSelection:
    def test_sel(self):
        p = _make_pxy()
        result = p.sel(X=0)
        assert isinstance(result, XRDistribution)
        assert result.free_vars == frozenset({"Y"})

    def test_sel_scalar(self):
        p = _make_pxy()
        val = p.sel(X=0, Y=0)
        assert isinstance(val, float)
        np.testing.assert_allclose(val, 0.1)

    def test_getitem_dict(self):
        p = _make_pxy()
        result = p[{"X": 0}]
        assert isinstance(result, XRDistribution)

    def test_getitem_tuple(self):
        p = _make_pxy()
        val = p[(0, 1)]
        np.testing.assert_allclose(val, 0.2)

    def test_getitem_invalid(self):
        p = _make_pxy()
        with pytest.raises(KeyError):
            p[42]


# ─── Copy and Conversion ────────────────────────────────────────────────


class TestCopyConversion:
    def test_copy_independent(self):
        p = _make_pxy()
        p2 = p.copy()
        p2.data.values[0, 0] = 999
        assert float(p.data.sel(X=0, Y=0)) == 0.1

    def test_copy_with_base(self):
        p = _make_pxy()
        p2 = p.copy(base=2)
        assert p2.is_log()
        np.testing.assert_allclose(p2.ops.exp(p2.data.values), p.data.values, atol=1e-12)

    def test_roundtrip_to_distribution(self):
        import dit

        d = dit.Distribution([(0, 0), (0, 1), (1, 0), (1, 1)], [0.1, 0.2, 0.3, 0.4])
        xrd = XRDistribution.from_distribution(d, ["A", "B"])
        d2 = xrd.to_distribution()
        for o in d.outcomes:
            np.testing.assert_allclose(d[o], d2[o], atol=1e-12)

    def test_to_distribution_conditional_error(self):
        p = _make_pxy().condition_on("X")
        with pytest.raises(ValueError, match="joint"):
            p.to_distribution()

    def test_to_numpy(self):
        p = _make_pxy()
        arr = p.to_numpy()
        assert isinstance(arr, np.ndarray)
        np.testing.assert_allclose(arr, p.data.values)

    def test_DataArray_property(self):
        p = _make_pxy()
        da = p.DataArray
        assert isinstance(da, xr.DataArray)
        assert da is p.data

    def test_outcomes_pmf(self):
        p = _make_pxy()
        assert len(p.outcomes) == 4
        assert len(p.pmf) == 4
        np.testing.assert_allclose(sum(p.pmf), 1.0)


# ─── Log Base Support ───────────────────────────────────────────────────


class TestLogBase:
    def test_default_linear(self):
        p = _make_pxy()
        assert p.get_base() == "linear"
        assert not p.is_log()

    def test_set_base_log2(self):
        p = _make_pxy()
        p.set_base(2)
        assert p.is_log()
        assert p.get_base() == 2
        np.testing.assert_allclose(
            2**p.data.values,
            np.array([[0.1, 0.2], [0.3, 0.4]]),
            atol=1e-12,
        )

    def test_set_base_back_to_linear(self):
        p = _make_pxy()
        original = p.data.values.copy()
        p.set_base(2)
        p.set_base("linear")
        assert not p.is_log()
        np.testing.assert_allclose(p.data.values, original, atol=1e-12)

    def test_log_to_log(self):
        p = _make_pxy()
        original = p.data.values.copy()
        p.set_base(2)
        p.set_base("e")
        p.set_base("linear")
        np.testing.assert_allclose(p.data.values, original, atol=1e-12)

    def test_set_base_noop(self):
        p = _make_pxy()
        p.set_base("linear")
        np.testing.assert_allclose(p.data.values, np.array([[0.1, 0.2], [0.3, 0.4]]))

    def test_validate_still_works_in_log(self):
        p = _make_pxy()
        p.set_base(2)
        assert p.validate()

    def test_entropy_same_across_bases(self):
        p = _make_pxy()
        h_lin = p.entropy(base=2)
        p.set_base(2)
        h_log = p.entropy(base=2)
        np.testing.assert_allclose(h_lin, h_log, atol=1e-10)

    def test_marginal_in_log_space(self):
        p = _make_pxy()
        p.set_base(2)
        p_x = p.marginal("X")
        assert p_x.is_log()
        # Convert back to linear and check
        expected = np.array([0.3, 0.7])
        np.testing.assert_allclose(p_x.ops.exp(p_x.data.values), expected, atol=1e-12)

    def test_condition_on_in_log_space(self):
        p = _make_pxy()
        p.set_base(2)
        c = p.condition_on("X")
        assert c.is_log()
        assert c.validate()

    def test_multiply_in_log_space(self):
        p_xy = _make_pxy()
        p_xy.set_base(2)
        p_x = p_xy.marginal("X")
        p_y_given_x = p_xy.condition_on("X")
        result = p_x * p_y_given_x
        assert result.is_log()
        # Should reconstruct p(X,Y) in log space
        np.testing.assert_allclose(
            result.ops.exp(result.data.values),
            np.array([[0.1, 0.2], [0.3, 0.4]]),
            atol=1e-10,
        )

    def test_outcomes_pmf_in_log(self):
        p = _make_pxy()
        outs_lin = p.outcomes
        pmf_lin = p.pmf
        p.set_base(2)
        outs_log = p.outcomes
        pmf_log = p.pmf
        assert outs_lin == outs_log
        np.testing.assert_allclose(pmf_lin, pmf_log, atol=1e-12)


# ─── Zipped ─────────────────────────────────────────────────────────────


class TestZipped:
    def test_pmf_mode(self):
        p = _make_pxy()
        items = list(p.zipped())
        assert len(items) == 4
        assert all(prob > 0 for _, prob in items)

    def test_atoms_mode(self):
        arr = np.array([0.5, 0.0, 0.5])
        p = XRDistribution.from_array(arr, ["X"], [[0, 1, 2]])
        pmf_items = list(p.zipped(mode="pmf"))
        atoms_items = list(p.zipped(mode="atoms"))
        assert len(pmf_items) == 2
        assert len(atoms_items) == 3


# ─── Repr / Notation ────────────────────────────────────────────────────


class TestRepr:
    def test_joint_notation(self):
        p = _make_pxy()
        assert "p(X,Y)" in repr(p)

    def test_conditional_notation(self):
        p = _make_pxy().condition_on("X")
        assert "p(Y|X)" in repr(p)


# ─── Equality ────────────────────────────────────────────────────────────


class TestEquality:
    def test_equal(self):
        p1 = _make_pxy()
        p2 = _make_pxy()
        assert p1 == p2

    def test_not_equal_data(self):
        p1 = _make_pxy()
        p2 = _make_pxy()
        p2.data.values[0, 0] = 0.99
        assert p1 != p2

    def test_not_equal_vars(self):
        p = _make_pxy()
        c = p.condition_on("X")
        assert p != c

    def test_approx_equal(self):
        p1 = _make_pxy()
        p2 = _make_pxy()
        p2.data.values[0, 0] += 1e-11
        assert p1.is_approx_equal(p2, atol=1e-9)

    def test_approx_not_equal(self):
        p1 = _make_pxy()
        assert not p1.is_approx_equal("not a dist")

    def test_eq_not_implemented(self):
        p = _make_pxy()
        assert p.__eq__(42) is NotImplemented


# ─── Normalize ───────────────────────────────────────────────────────────


class TestNormalize:
    def test_normalize_joint(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        p = XRDistribution.from_array(arr, ["X", "Y"], [[0, 1], [0, 1]])
        p.normalize()
        np.testing.assert_allclose(float(p.data.sum()), 1.0)

    def test_normalize_conditional(self):
        arr = np.array([[1.0, 3.0], [2.0, 2.0]])
        p = XRDistribution.from_array(
            arr,
            ["X", "Y"],
            [[0, 1], [0, 1]],
            free_vars={"Y"},
            given_vars={"X"},
        )
        p.normalize()
        assert p.validate()


# ─── Edge Cases ──────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_single_variable(self):
        arr = np.array([0.3, 0.7])
        p = XRDistribution.from_array(arr, ["X"], [[0, 1]])
        assert p.validate()
        assert p.entropy(base=2) > 0

    def test_zero_probabilities(self):
        arr = np.array([1.0, 0.0])
        p = XRDistribution.from_array(arr, ["X"], [[0, 1]])
        assert p.validate()
        assert len(p.outcomes) == 1

    def test_division_zero_denom(self):
        arr = np.array([[0.5, 0.0], [0.0, 0.5]])
        p_xy = XRDistribution.from_array(arr, ["X", "Y"], [[0, 1], [0, 1]])
        p_y = p_xy.marginal("Y")
        result = p_xy / p_y
        # p(Y=1) = 0.5, p(X=0,Y=1) = 0 => p(X=0|Y=1) = 0
        assert float(result.data.sel(X=0, Y=1)) == 0.0

    def test_sample_space(self):
        p = _make_pxy()
        ss = list(p.sample_space())
        assert len(ss) == 4
        assert (0, 0) in ss
        assert (1, 1) in ss

    def test_event_probability(self):
        p = _make_pxy()
        prob = p.event_probability([(0, 0), (1, 1)])
        np.testing.assert_allclose(prob, 0.5)

    def test_is_numerical(self):
        p = _make_pxy()
        assert p.is_numerical()
