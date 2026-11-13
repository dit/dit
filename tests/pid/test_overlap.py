"""Overlap and sharp bounds for consistent two-source PIDs."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from dit import Distribution, ditParams
from dit.multivariate import coinformation
from dit.pid import (
    PED_CS,
    PID_CCS,
    PID_MMI,
    PID_PM,
    PID_PR,
    PID_RA,
    PID_WB,
    PID_dep,
    PID_dep_b,
    PIDOverlap,
    pid_overlap,
)
from dit.pid.distributions import bivariates, trivariates
from dit.pid.helpers import compare_measures


def test_mmi_endpoint():
    """MMI has maximal overlap, with the bounds of Proposition 1."""
    result = pid_overlap(PID_MMI(bivariates["and"]))
    assert isinstance(result, PIDOverlap)
    assert result.coinformation == pytest.approx(-0.188721875540867)
    assert result.delta == pytest.approx(0.311278124459133)
    assert result.delta_max == pytest.approx(result.delta)
    assert result.position == pytest.approx(1.0)
    assert result.redundancy_bounds == pytest.approx((0.0, 0.311278124459133))
    assert result.synergy_bounds == pytest.approx((0.188721875540867, 0.5))
    for interval in result.unique_bounds:
        assert interval == pytest.approx((0.0, 0.311278124459133))
    with pytest.raises(FrozenInstanceError):
        result.position = 0.0


def test_pairwise_reference_endpoint():
    """The pairwise-reference PID is at Hi-Fi on AND, but not on RdnXor."""
    assert pid_overlap(PID_PR(bivariates["and"])).position == pytest.approx(0.0, abs=1e-8)
    result = pid_overlap(PID_PR(bivariates["rdn xor"]))
    assert result.coinformation == pytest.approx(0.0)
    assert result.delta == pytest.approx(1.0)
    assert result.position == pytest.approx(1.0)


@pytest.mark.parametrize("name", ["synergy", "redundant", "unique 1", "uniform", "null"])
def test_collapsed_segment(name):
    """A unique nonnegative decomposition has an undefined normalized position."""
    result = pid_overlap(PID_MMI(bivariates[name]))
    assert result.delta_max == pytest.approx(0.0, abs=1e-14)
    assert np.isnan(result.position)
    assert result.redundancy_bounds[0] == pytest.approx(result.redundancy_bounds[1])
    assert result.synergy_bounds[0] == pytest.approx(result.synergy_bounds[1])


def test_signed_positions_not_clipped():
    """Negative redundancy and negative unique atoms leave opposite ends of the segment."""
    below = pid_overlap(PID_RA(bivariates["and"]))
    above = pid_overlap(PID_PM(bivariates["and"]))
    assert below.delta < 0.0
    assert below.position < 0.0
    assert above.delta > above.delta_max
    assert above.position > 1.0


@pytest.mark.parametrize("measure", [PID_MMI, PID_WB, PID_dep, PID_PR, PID_PM, PID_RA])
@pytest.mark.parametrize("name", ["and", "sum", "cat", "rdn xor", "pnt. unq."])
def test_overlap_reconstructs_atoms(measure, name):
    """Every consistent signed or nonnegative PID has the same one-parameter form."""
    pid = measure(bivariates[name])
    result = pid_overlap(pid)
    c = result.coinformation
    red = pid.get_pi(pid._lattice.bottom)
    syn = pid.get_pi(pid._lattice.top)
    assert result.delta == pytest.approx(min(red, syn), abs=1e-7)
    assert red == pytest.approx(max(0.0, c) + result.delta, abs=1e-6)
    assert syn == pytest.approx(max(0.0, -c) + result.delta, abs=1e-6)
    for source, bounds in zip(pid._sources, result.unique_bounds, strict=True):
        mi = coinformation(pid._dist, [source, pid._target])
        assert pid.get_pi((source,)) == pytest.approx(mi - max(0.0, c) - result.delta, abs=1e-6)
        assert bounds[0] == pytest.approx(bounds[1] - result.delta_max)


def test_source_order_names_and_composite_groups():
    """Unique bounds follow the supplied sources, including named composite sources."""
    d = Distribution(["0000", "0010", "1101", "1111"], [0.25] * 4)
    pid = PID_MMI(d, [[0, 1], [2]], [3])
    result = pid_overlap(pid)
    assert result.unique_bounds[0] == pytest.approx((1.0, 1.0))
    assert result.unique_bounds[1] == pytest.approx((0.0, 0.0))
    d.set_rv_names(["A", "B", "C", "Y"])
    reverse = pid_overlap(PID_MMI(d, [["C"], ["A", "B"]], ["Y"]))
    assert reverse.unique_bounds == result.unique_bounds[::-1]
    assert reverse.coinformation == pytest.approx(result.coinformation)


@pytest.mark.parametrize("base", [np.e, 10, 3])
@pytest.mark.parametrize("units", [False, True])
@pytest.mark.parametrize("measure", [PID_MMI, PID_dep])
def test_units(base, units, measure, monkeypatch):
    """Information values use the PID's log base; normalized position is dimensionless."""
    if units:
        pytest.importorskip("pint")
    reference = pid_overlap(measure(bivariates["and"]))
    monkeypatch.setitem(ditParams, "units", units)
    result = pid_overlap(measure(bivariates["and"].copy(base=base)))
    scale = 1 / np.log2(base)
    assert result.coinformation == pytest.approx(reference.coinformation * scale)
    assert result.delta == pytest.approx(reference.delta * scale)
    assert result.delta_max == pytest.approx(reference.delta_max * scale)
    assert result.position == pytest.approx(reference.position)
    assert result.redundancy_bounds == pytest.approx(np.array(reference.redundancy_bounds) * scale)
    assert result.synergy_bounds == pytest.approx(np.array(reference.synergy_bounds) * scale)
    assert isinstance(result.delta, float)
    for actual, expected in zip(result.unique_bounds, reference.unique_bounds, strict=True):
        assert actual == pytest.approx(np.array(expected) * scale)


def test_lazy_pid():
    """The helper computes a lazy decomposition before reading its atoms."""
    pid = PID_dep(bivariates["and"], compute=False)
    assert not pid._computed
    result = pid_overlap(pid)
    assert pid._computed
    assert result.position == pytest.approx(0.262476432247, abs=1e-5)


@pytest.mark.parametrize("atol", [-1.0, np.inf, np.nan])
def test_bad_tolerance(atol):
    """The collapse tolerance must be finite and nonnegative."""
    with pytest.raises(ValueError, match="atol"):
        pid_overlap(PID_MMI(bivariates["and"]), atol=atol)


def test_explicit_collapse_tolerance():
    """A caller may leave the position undefined for a narrow segment, without changing delta."""
    pid = PID_MMI(bivariates["and"])
    reference = pid_overlap(pid)
    result = pid_overlap(pid, atol=0.4)
    assert np.isnan(result.position)
    assert result.delta == reference.delta
    assert result.delta_max == reference.delta_max


@pytest.mark.parametrize("value", [None, bivariates["and"], object()])
def test_requires_pid(value):
    with pytest.raises(TypeError, match="PID"):
        pid_overlap(value)


def test_reject_ped():
    with pytest.raises(TypeError, match="target"):
        pid_overlap(PED_CS(bivariates["and"].marginal([0, 1])))


@pytest.mark.parametrize("sources", [[[0]], [[0], [1], [2]]])
def test_requires_two_sources(sources):
    with pytest.raises(ValueError, match="two sources"):
        pid_overlap(PID_MMI(trivariates["anddup"], sources, [3]))


def test_inconsistent_pid():
    """A quadruple that fails the PID equations has no single overlap parameter."""
    with pytest.raises(ValueError, match="consistent"):
        pid_overlap(PID_dep_b(bivariates["simple"]))


def test_incomplete_pid():
    pid = PID_MMI(bivariates["and"])
    pid._pis[pid._lattice.bottom] = np.nan
    with pytest.raises(ValueError, match="finite"):
        pid_overlap(pid)


def test_symbolic_not_supported():
    sympy = pytest.importorskip("sympy")
    p = sympy.Symbol("p", positive=True)
    d = Distribution(["000", "111"], [p, 1 - p])
    with pytest.raises(NotImplementedError, match="symbolic"):
        pid_overlap(PID_MMI(d, compute=False))


def test_log_base_below_one():
    """The nonnegative segment assumes information units with a positive bit conversion."""
    with pytest.raises(ValueError, match="base"):
        pid_overlap(PID_MMI(bivariates["and"].copy(base=0.5)))


def _table_rows(output):
    """The rows of a printed comparison table, keyed by their first column."""
    rows = {}
    for line in output.splitlines():
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) > 1 and cells[0]:
            rows[cells[0]] = cells[1:]
    return rows


def test_compare_measures_overlap(capsys):
    """The comparison table can append each measure's overlap and position."""
    compare_measures(bivariates["and"], pids=[PID_WB, PID_MMI, PID_dep, PID_CCS, PID_RA], overlap=True)
    rows = _table_rows(capsys.readouterr().out)
    lambdas = [float(value) for value in rows["lambda"]]
    assert lambdas == pytest.approx([1.0, 1.0, 0.262476, 1 / 3, -0.606275], abs=1e-5)
    assert [float(value) for value in rows["delta"]] == pytest.approx(
        [0.311278, 0.311278, 0.081704, 0.103759, -0.188722], abs=1e-5
    )


def test_compare_measures_overlap_nan(capsys):
    """Collapsed segments and inconsistent decompositions have no position."""
    compare_measures(bivariates["simple"], pids=[PID_MMI, PID_dep_b], overlap=True)
    rows = _table_rows(capsys.readouterr().out)
    assert rows["lambda"] == ["nan", "nan"]
    assert rows["delta"][1] == "nan"


def test_compare_measures_overlap_two_sources():
    with pytest.raises(ValueError, match="two sources"):
        compare_measures(trivariates["anddup"], pids=[PID_MMI], overlap=True)
