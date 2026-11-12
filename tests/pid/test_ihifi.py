"""
Tests for the information-theoretic Hi-Fi decomposition and the Hi-Fi diagnostics.
"""

from itertools import product

import numpy as np
import pytest

from dit import Distribution, ditParams
from dit.multivariate import coinformation
from dit.pid import (
    PID_BROJA,
    PID_MMI,
    PID_WB,
    HiFi,
    PID_dep,
    PID_dep_c,
    PID_HiFi,
    TripletReport,
    hifi,
    pid_overlap,
    triplet_report,
)
from dit.pid.distributions import bivariates, trivariates
from dit.pid.measures.ihifi import _pair_nodes


def _uniform(outcomes):
    return Distribution([tuple(outcome) for outcome in outcomes], [1 / len(outcomes)] * len(outcomes))


# a copy and a partner (arXiv:2611.08131, Example 1): X0 = (a, b), X1 = a, X2 = c, Y = (a, b xor c)
COPY_PARTNER = _uniform([(2 * a + b, a, c, 2 * a + (b ^ c)) for a, b, c in product((0, 1), repeat=3)])
# the parity of three bits
PARITY = _uniform([(a, b, c, a ^ b ^ c) for a, b, c in product((0, 1), repeat=3)])
# Y = X0, X1 = X0 xor N, X2 = N
NOISY_COPY = _uniform([(a, a ^ n, n, a) for a, n in product((0, 1), repeat=2)])


@pytest.mark.parametrize("name", sorted(bivariates))
def test_hifi_is_zero_overlap_end(name):
    """For two sources Hi-Fi is the zero-overlap end of every bound of Proposition 1."""
    pid = PID_HiFi(bivariates[name])
    result = pid_overlap(pid)
    assert pid.consistent
    assert pid.nonnegative
    assert result.delta == pytest.approx(0.0, abs=1e-9)
    assert pid.get_pi(pid._lattice.bottom) == pytest.approx(result.redundancy_bounds[0], abs=1e-9)
    assert pid.get_pi(pid._lattice.top) == pytest.approx(result.synergy_bounds[0], abs=1e-9)
    for source, (_, upper) in zip(pid._sources, result.unique_bounds, strict=True):
        assert pid.get_pi((source,)) == pytest.approx(upper, abs=1e-9)


@pytest.mark.parametrize("name", ["and", "sum", "rdn xor", "pnt. unq.", "cat", "reduced or"])
def test_nonnegative_measures_bounded_by_hifi(name):
    """Every nonnegative decomposition has at least Hi-Fi's redundancy and synergy, and at
    most its unique informations (Corollary 1)."""
    d = bivariates[name]
    hf = PID_HiFi(d)
    for measure in (PID_WB, PID_MMI, PID_BROJA, PID_dep):
        pid = measure(d)
        assert pid.get_pi(pid._lattice.bottom) >= hf.get_pi(hf._lattice.bottom) - 1e-5
        assert pid.get_pi(pid._lattice.top) >= hf.get_pi(hf._lattice.top) - 1e-5
        for source in pid._sources:
            assert pid.get_pi((source,)) <= hf.get_pi((source,)) + 1e-5


def test_target_symmetry():
    """Hi-Fi's redundancy and synergy do not depend on which variable is the target."""
    rng = np.random.default_rng(5)
    d = Distribution.from_ndarray(rng.dirichlet(np.ones(12)).reshape((2, 3, 2)))
    pids = [PID_HiFi(d, [[a], [b]], [t]) for a, b, t in ((0, 1, 2), (0, 2, 1), (1, 2, 0))]
    reds = [pid.get_pi(pid._lattice.bottom) for pid in pids]
    syns = [pid.get_pi(pid._lattice.top) for pid in pids]
    assert reds == pytest.approx([reds[0]] * 3)
    assert syns == pytest.approx([syns[0]] * 3)


@pytest.mark.parametrize("n, count", [(2, 1), (3, 6), (4, 25), (5, 90)])
def test_pair_nodes(n, count):
    """The pairs of disjoint groups of sources, as counted in Table IV."""
    nodes = _pair_nodes(tuple((i,) for i in range(n)))
    assert len(nodes) == count
    assert all(not set(first) & set(second) for first, second in nodes)


def test_more_sources_forced_overlap():
    """Zero overlap on every pair is impossible for a copy and a partner, so the three-source
    extension has a negative atom (Example 1)."""
    pid = PID_HiFi(COPY_PARTNER)
    assert pid.consistent
    assert not pid.nonnegative
    assert pid.get_pi(((0,), (1, 2))) == pytest.approx(-1.0, abs=1e-9)


def test_more_sources_undetermined():
    """On Rauh et al.'s distribution the pairs leave some atoms undetermined."""
    pid = PID_HiFi(trivariates["xor cat"])
    assert pid.consistent
    assert not pid.complete


def test_overlapping_sources():
    """Two sources that share a variable still form the pair of the decomposition."""
    d = Distribution(
        [("0", "0", "0", "0"), ("1", "0", "0", "1"), ("1", "0", "1", "0"), ("1", "1", "1", "1")], [0.25] * 4
    )
    pid = PID_HiFi(d, [[0, 2], [1, 2]], [3])
    assert pid.complete
    assert pid.get_red(((0, 2), (1, 2))) == pytest.approx(max(0.0, coinformation(d, [[0, 2], [1, 2], [3]])))


@pytest.mark.parametrize("name", ["and", "sum", "rdn xor", "pnt. unq."])
def test_hifi_two_sources_matches_pid(name):
    """For two sources hifi gives the atoms of PID_HiFi."""
    d = bivariates[name]
    pid = PID_HiFi(d)
    result = hifi(d)
    assert isinstance(result, HiFi)
    for source in pid._sources:
        assert result[source].unique == pytest.approx(pid.get_pi((source,)), abs=1e-9)
        assert result[source].redundancy == pytest.approx(pid.get_pi(pid._lattice.bottom), abs=1e-9)
        assert result[source].synergy == pytest.approx(pid.get_pi(pid._lattice.top), abs=1e-9)


def test_hifi_parity_needs_exhaustive_search():
    """No single partner reveals the parity bit, which a greedy search misses (Remark 2)."""
    result = hifi(PARITY)
    driver = result[(0,)]
    assert driver.synergy == pytest.approx(1.0)
    assert driver.synergy_partners == ((1,), (2,))
    assert driver.redundancy_partners is None
    assert result.coinformations[((0,), ((1,),))] == pytest.approx(0.0)


def test_hifi_noisy_copy():
    """The copy is redundant with the pair of the noisy copy and the noise (Remark 2)."""
    driver = hifi(NOISY_COPY)[(0,)]
    assert (driver.unique, driver.redundancy, driver.synergy) == pytest.approx((0.0, 1.0, 0.0))
    assert driver.redundancy_partners == ((1,), (2,))


def test_hifi_copy_partner():
    """Redundancy and synergy of one source, with different partners (Example 1)."""
    driver = hifi(COPY_PARTNER)[(0,)]
    assert (driver.unique, driver.redundancy, driver.synergy) == pytest.approx((0.0, 1.0, 1.0))
    assert driver.redundancy_partners == ((1,),)
    assert driver.synergy_partners == ((2,),)


@pytest.mark.parametrize("seed", range(4))
def test_hifi_bounds_unique_atoms(seed):
    """No nonnegative decomposition has more unique information than U_i (Theorem 1(ii)).
    I_min and MMI are nonnegative for any number of sources."""
    rng = np.random.default_rng(seed)
    d = Distribution.from_ndarray(rng.dirichlet(np.ones(16)).reshape((2,) * 4))
    result = hifi(d)
    for measure in (PID_WB, PID_MMI):
        pid = measure(d)
        for source in pid._sources:
            assert pid.get_pi((source,)) <= result[source].unique + 1e-9


def test_hifi_coinformation_table():
    """Each source is paired with every nonempty group of the others."""
    result = hifi(PARITY)
    assert len(result.coinformations) == 3 * 3


@pytest.mark.parametrize("units", [False, True])
def test_hifi_units(units, monkeypatch):
    """Values are in the units of the distribution."""
    if units:
        pytest.importorskip("pint")
    reference = hifi(bivariates["and"])[(0,)]
    monkeypatch.setitem(ditParams, "units", units)
    actual = hifi(bivariates["and"].copy(base=10))[(0,)]
    assert actual.unique == pytest.approx(reference.unique / np.log2(10))
    assert actual.synergy == pytest.approx(reference.synergy / np.log2(10))
    assert isinstance(actual.unique, float)


def test_hifi_symbolic():
    sympy = pytest.importorskip("sympy")
    p = sympy.Symbol("p", positive=True)
    with pytest.raises(NotImplementedError, match="symbolic"):
        hifi(Distribution(["000", "111"], [p, 1 - p]))


def test_triplet_report_rdn_xor():
    """A vanishing co-information can hide a bit of pairwise redundancy and one of triadic synergy."""
    report = triplet_report(bivariates["rdn xor"])
    assert isinstance(report, TripletReport)
    assert report.coinformation == pytest.approx(0.0, abs=1e-9)
    assert report.pairwise == pytest.approx(1.0, abs=1e-6)
    assert report.connected == pytest.approx(1.0, abs=1e-6)
    assert report.delta_max == pytest.approx(1.0)
    for decomposition, (red, unique, syn) in (
        (report.hifi, (0.0, 1.0, 0.0)),
        (report.pairwise_reference, (1.0, 0.0, 1.0)),
    ):
        assert decomposition[0] == pytest.approx(red, abs=1e-6)
        assert decomposition[1] == pytest.approx((unique, unique), abs=1e-6)
        assert decomposition[2] == pytest.approx(syn, abs=1e-6)
    assert "connected I3" in str(report)


@pytest.mark.parametrize("name", ["and", "sum", "cat", "pnt. unq.", "imp. rdn", "boom"])
def test_triplet_report_matches_measures(name):
    """The report's decompositions are those of PID_HiFi, PID_dep_c and PID_dep."""
    d = bivariates[name]
    report = triplet_report(d)
    for decomposition, measure in (
        (report.hifi, PID_HiFi),
        (report.pairwise_reference, PID_dep_c),
        (report.dependency, PID_dep),
    ):
        pid = measure(d)
        red, uniques, syn = decomposition
        assert red == pytest.approx(pid.get_pi(pid._lattice.bottom), abs=1e-6)
        assert syn == pytest.approx(pid.get_pi(pid._lattice.top), abs=1e-6)
        assert uniques == pytest.approx(tuple(pid.get_pi((source,)) for source in pid._sources), abs=1e-6)
    i_0, i_1 = report.mutual_informations
    assert report.joint_information == pytest.approx(coinformation(d, [[0, 1], [2]]))
    assert report.conditional_informations == pytest.approx(
        (report.joint_information - i_1, report.joint_information - i_0)
    )


def test_triplet_report_units():
    reference = triplet_report(bivariates["and"])
    actual = triplet_report(bivariates["and"].copy(base=np.e))
    assert actual.delta_max == pytest.approx(reference.delta_max * np.log(2))
    assert actual.connected == pytest.approx(reference.connected * np.log(2), abs=1e-9)


def test_triplet_report_two_sources():
    with pytest.raises(ValueError, match="two sources"):
        triplet_report(trivariates["anddup"])
