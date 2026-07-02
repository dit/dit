"""
Tests for dit.visualization.upset.InformationUpsetPlot.

These tests exercise the data model of the UpSet plot (atoms, membership, and
sizes). The ``draw`` method is matplotlib-only and excluded from coverage; a
minimal smoke test using the Agg backend guards against import/layout breakage.
"""

import pytest

from dit import Distribution
from dit.profiles import ShannonPartition
from dit.profiles.information_partitions import ExtropyPartition
from dit.shannon import entropy
from dit.visualization import InformationUpsetPlot

# canonical examples from Allen2014 (as used in the profiles docs)
xor = Distribution(["000", "011", "101", "110"], [1 / 4] * 4)
giant_bit = Distribution(["000", "111"], [1 / 2] * 2)
independent = Distribution(["00", "01", "10", "11"], [1 / 4] * 4)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_atom_count(n):
    """There are 2**n - 1 atoms for n variables."""
    outcomes = [format(i, f"0{n}b") for i in range(2**n)]
    d = Distribution(outcomes, [1 / 2**n] * 2**n)
    u = InformationUpsetPlot(d)
    assert len(u.atoms) == 2**n - 1


def test_values_match_shannon_partition():
    """Atom values agree with the underlying ShannonPartition."""
    u = InformationUpsetPlot(xor)
    sp = ShannonPartition(xor)
    got = {(a["members"], a["conditions"]): a["value"] for a in u.atoms}
    for (rvs, crvs), value in sp.atoms.items():
        key = (frozenset(rv[0] for rv in rvs), frozenset(crvs))
        assert got[key] == pytest.approx(value)


def test_negative_atom_preserved():
    """The XOR co-information atom is -1 bit and survives into the plot."""
    u = InformationUpsetPlot(xor)
    top = {frozenset({0, 1, 2}): None}
    coinfo = next(a for a in u.atoms if a["members"] == frozenset({0, 1, 2}))
    assert coinfo["conditions"] == frozenset()
    assert coinfo["value"] == pytest.approx(-1.0)
    assert top  # sanity


def test_membership_partitions_all_variables():
    """Every atom's members and conditions partition the full variable set."""
    u = InformationUpsetPlot(xor)
    allvars = set(u.variables)
    for atom in u.atoms:
        assert atom["members"] | atom["conditions"] == allvars
        assert atom["members"] & atom["conditions"] == set()
        assert atom["degree"] == len(atom["members"])


def test_sizes_are_marginal_entropies():
    """The side "set sizes" are the per-variable marginal entropies."""
    u = InformationUpsetPlot(xor)
    for i, var in enumerate(u.variables):
        assert u.sizes[var] == pytest.approx(entropy(xor, [i]))


def test_rv_names_used_for_variables():
    """Named random variables label the rows/sizes."""
    d = Distribution(["00", "01", "10", "11"], [1 / 4] * 4)
    d.set_rv_names("XY")
    u = InformationUpsetPlot(d)
    assert u.variables == ["X", "Y"]
    assert set(u.sizes) == {"X", "Y"}
    assert u.sizes["X"] == pytest.approx(1.0)


def test_min_degree_filters_atoms():
    """min_degree drops low-order atoms."""
    u = InformationUpsetPlot(xor)
    only_pairs_up = u._sorted_atoms(min_degree=2)
    assert all(a["degree"] >= 2 for a in only_pairs_up)
    assert len(only_pairs_up) == 4  # three pairwise + one triple


def test_sort_by_value_descending():
    """sort_by='value' orders atoms by descending signed value."""
    u = InformationUpsetPlot(xor)
    vals = [a["value"] for a in u._sorted_atoms(sort_by="value")]
    assert vals == sorted(vals, reverse=True)
    assert vals[-1] == pytest.approx(-1.0)  # co-information last


def test_sort_by_magnitude():
    """sort_by='magnitude' orders atoms by descending absolute value."""
    u = InformationUpsetPlot(xor)
    mags = [abs(a["value"]) for a in u._sorted_atoms(sort_by="magnitude")]
    assert mags == sorted(mags, reverse=True)


def test_sort_by_degree():
    """sort_by='degree' orders atoms by ascending degree."""
    u = InformationUpsetPlot(giant_bit)
    degs = [a["degree"] for a in u._sorted_atoms(sort_by="degree")]
    assert degs == sorted(degs)


def test_sort_by_invalid():
    """An unknown sort_by raises ValueError."""
    u = InformationUpsetPlot(xor)
    with pytest.raises(ValueError):
        u._sorted_atoms(sort_by="nope")


def test_nonlinear_base_unit():
    """A log-base distribution reports its unit (e.g. nats)."""
    d = Distribution(["00", "11"], [1 / 2] * 2)
    d.set_base("e")
    u = InformationUpsetPlot(d)
    assert u.unit == "nats"


def test_extropy_partition_backend():
    """The plot accepts an alternate partition class."""
    u = InformationUpsetPlot(independent, partition=ExtropyPartition)
    assert u.unit == "exits"
    assert len(u.atoms) == 3  # 2**2 - 1


def test_to_string_delegates_to_partition():
    """to_string/str render the partition table."""
    u = InformationUpsetPlot(giant_bit)
    assert u.to_string() == ShannonPartition(giant_bit).to_string()
    assert str(u) == u.to_string()


def test_draw_smoke():
    """draw executes end-to-end on the Agg backend and returns the panels."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # a 5-variable distribution to exercise the many-variable regime
    outcomes = [format(i, "05b") for i in range(32)]
    d = Distribution(outcomes, [1 / 32] * 32)
    u = InformationUpsetPlot(d)
    axes = u.draw()
    assert set(axes) == {"atoms", "matrix", "sizes"}
    plt.close("all")


def test_draw_on_existing_axis():
    """draw can host its panels on a provided axis's location."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    u = InformationUpsetPlot(xor)
    axes = u.draw(ax=ax, min_degree=2, show_values=False)
    assert set(axes) == {"atoms", "matrix", "sizes"}
    plt.close("all")
