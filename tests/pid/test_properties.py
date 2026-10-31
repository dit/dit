"""
Tests for dit.pid.properties.
"""

import numpy as np
import pytest

from dit import Distribution
from dit.pid import (
    PID_BROJA,
    PID_CCS,
    PID_CI,
    PID_CT,
    PID_GK,
    PID_IG,
    PID_LN,
    PID_MMI,
    PID_PM,
    PID_RAV,
    PID_RDR,
    PID_RR,
    PID_SX,
    PID_WB,
    PID_Deg,
    PID_Delta,
    PID_Do,
    PID_Prec,
    PID_Proj,
)
from dit.pid.distributions import bivariates, trivariates
from dit.pid.helpers import property_table
from dit.pid.measures import __all_pids
from dit.pid.properties import _CLAIMS, PROPERTIES, PropertyReport, claims_for

# correlated sources, and a target copying both
COPY = Distribution(
    [("0", "0", "0", "0"), ("0", "1", "0", "1"), ("1", "0", "1", "0"), ("1", "1", "1", "1")], [0.4, 0.1, 0.1, 0.4]
)
# independent sources, and the target (X0 and X1, X0 xor X1)
AND_XOR = Distribution([(a, b, str(int(a) & int(b)), str(int(a) ^ int(b))) for a in "01" for b in "01"], [1 / 4] * 4)
# identical sources, and a noisy copy of them
SAME = Distribution([("0", "0", "0"), ("0", "0", "1"), ("1", "1", "0"), ("1", "1", "1")], [0.4, 0.1, 0.1, 0.4])
# S2 decides which of S0, S1 copies the target (gutknecht2026partial)
MAX_MIN = trivariates["max min"]


def test_check_truthiness():
    """
    A check is truthy exactly when the property holds.
    """
    check = PID_WB(bivariates["and"]).check("SR")
    assert check
    assert check.holds is True
    assert check.tested == 3
    assert not check.violations


@pytest.mark.parametrize("name", ["SCR", "scr", "source_chain_rule", "source chain rule", "Source-Chain-Rule"])
def test_aliases(name):
    """
    Properties can be given by key or by name.
    """
    assert PID_SX(MAX_MIN).check(name).key == "SCR"


def test_unknown_property():
    """
    Unknown properties are rejected.
    """
    with pytest.raises(ValueError, match="unknown property"):
        PID_WB(bivariates["and"]).check("XYZ")


@pytest.mark.parametrize("key", ["CO", "DI"])
def test_uncheckable(key):
    """
    Continuity and differentiability cannot be checked on one distribution.
    """
    check = PID_WB(bivariates["and"]).check(key)
    assert check.holds is None
    assert not check


def test_check_str():
    """
    The string form of a check lists its violations.
    """
    text = str(PID_WB(MAX_MIN).check("SCR"))
    assert "fails" in text
    assert "{0:2}{1:2}" in text


def test_report():
    """
    A report checks every property, next to the claims for the measure.
    """
    report = PID_WB(bivariates["and"]).check_properties()
    assert isinstance(report, PropertyReport)
    assert len(report) == len(PROPERTIES)
    assert report["self_redundancy"]
    assert report["IID"].holds is False
    text = str(report)
    assert "Properties of I_min" in text
    assert "SCR" in text


def test_report_subset():
    """
    A report can be restricted to some properties.
    """
    report = PID_SX(MAX_MIN).check_properties(props=["SCR", "LP"])
    assert [check.key for check in report] == ["SCR", "LP"]
    assert report["SCR"]
    assert not report["LP"]


def test_report_two_source_claims():
    """
    The claims for a two-source measure do not cover a decomposition of three sources.
    """
    report = PID_RR(MAX_MIN).check_properties(props=["SR"])
    assert "claims concern two sources" in report.title


def test_scr_counterexample():
    """
    I_min violates the source chain rule on the counterexample of Gutknecht (2026).
    """
    check = PID_WB(MAX_MIN).check("SCR")
    assert not check
    assert check.max_violation == pytest.approx(0.5)
    assert check.violations[0].lhs == pytest.approx(0.5)
    assert check.violations[0].rhs == pytest.approx(0.0)


@pytest.mark.parametrize("pid", [PID_SX, PID_PM, PID_GK])
def test_scr_holds(pid):
    """
    I_sx, I_pm, and I_wedge satisfy the source chain rule.
    """
    assert pid(MAX_MIN).check("SCR", atol=1e-9)
    rng = np.random.default_rng(0)
    outcomes = [f"{a}{b}{c}{t}" for a in "01" for b in "01" for c in "01" for t in "01"]
    for _ in range(5):
        k = rng.integers(5, 17)
        d = Distribution(list(rng.choice(outcomes, size=k, replace=False)), rng.dirichlet(np.ones(k)))
        assert pid(d).check("SCR", atol=1e-9)


@pytest.mark.parametrize("pid", [PID_Prec, PID_Deg])
def test_scr_prec(pid):
    """
    I_prec fails the source chain rule on the counterexample, where it also has a negative atom.
    """
    pid = pid(MAX_MIN)
    assert pid.check("SCR").max_violation == pytest.approx(0.5)
    assert not pid.check("LP1")


def test_rdr_degenerate_conditionals():
    """
    I_rdr handles the constant sources and targets of conditional decompositions.
    """
    assert PID_RDR(AND_XOR, [[0], [1]], [2, 3]).check("TC")
    assert not PID_RDR(MAX_MIN).check("SCR")


def test_ln_degenerate_conditionals():
    """
    I_ln handles the constant targets of conditional decompositions.
    """
    assert PID_LN(AND_XOR, [[0], [1]], [2, 3], rng=0).check("TC")


def test_scr_context():
    """
    A single context can be checked.
    """
    pid = PID_WB(MAX_MIN)
    check = pid.check("SCR", context=[2])
    assert not check
    assert check.tested == 4
    with pytest.raises(ValueError, match="not one of the sources"):
        pid.check("SCR", context=[3])


def test_scr_two_sources():
    """
    The source chain rule has no content for two sources.
    """
    check = PID_WB(bivariates["and"]).check("SCR")
    assert check.holds is True
    assert check.tested == 0


@pytest.mark.parametrize("pid", [PID_SX, PID_PM])
def test_tc_holds(pid):
    """
    The pointwise measures satisfy the target chain rule.
    """
    assert pid(AND_XOR, [[0], [1]], [2, 3]).check("TC", atol=1e-9)
    assert pid(COPY, [[0], [1]], [2, 3]).check("TC", atol=1e-9)


def test_tc_fails():
    """
    I_mmi violates the target chain rule.
    """
    assert not PID_MMI(COPY, [[0], [1]], [2, 3]).check("TC")


def test_tc_split():
    """
    The target can be split in other ways.
    """
    pid = PID_SX(AND_XOR, [[0], [1]], [2, 3])
    assert pid.check("TC", split=([3], [2]), atol=1e-9)
    with pytest.raises(ValueError, match="does not partition"):
        pid.check("TC", split=([2], [0]))


def test_target_univariate():
    """
    The target chain rule and target monotonicity need a target of several variables.
    """
    pid = PID_WB(bivariates["and"])
    assert pid.check("TC").holds is None
    assert pid.check("TM").holds is None


def test_tm():
    """
    I_mmi satisfies target monotonicity; I_ccs does not.
    """
    assert PID_MMI(AND_XOR, [[0], [1]], [2, 3]).check("TM")
    assert not PID_CCS(AND_XOR, [[0], [1]], [2, 3]).check("TM")


def test_te_and_m1():
    """
    I_min satisfies target equality and strong monotonicity; I_wedge does not.
    """
    assert PID_WB(bivariates["and"]).check("TE")
    assert PID_WB(bivariates["and"]).check("M1")
    assert not PID_GK(bivariates["and"]).check("TE")
    assert not PID_GK(bivariates["and"]).check("M1")


@pytest.mark.parametrize("pid", [PID_WB, PID_SX])
def test_symmetry_and_relabeling(pid):
    """
    Measures are invariant under reordering sources and relabeling outcomes.
    """
    decomposition = pid(MAX_MIN)
    assert decomposition.check("S0", orders="all")
    assert decomposition.check("EI")


def test_s0_orders():
    """
    Orders must be permutations of the sources.
    """
    with pytest.raises(ValueError, match="not an ordering"):
        PID_WB(MAX_MIN).check("S0", orders=[(0, 0, 1)])


def test_s1():
    """
    I_min violates strong symmetry.
    """
    assert not PID_WB(bivariates["and"]).check("S1")


def test_assumption_star():
    """
    I_min depends only on the source-target marginals; I_ccs does not.
    """
    assert PID_WB(bivariates["and"]).check("*")
    assert not PID_CCS(bivariates["and"]).check("star")


def test_blackwell():
    """
    I_prec has the Blackwell property; I_min does not.
    """
    assert PID_Prec(bivariates["and"]).check("BP")
    assert not PID_WB(COPY, [[0], [1]], [2, 3]).check("BP")
    assert PID_WB(MAX_MIN).check("BP").holds is None


def test_additivity():
    """
    I_wedge is additive; I_min is not.
    """
    assert PID_GK(bivariates["rdn xor"]).check("AD")
    assert PID_GK(bivariates["and"]).check("AD", other=bivariates["rdn xor"])
    assert not PID_WB(MAX_MIN).check("AD")


def test_positivity_and_monotonicity():
    """
    I_sx has negative atoms and is not monotone on the counterexample; I_min is both.
    """
    pid = PID_SX(MAX_MIN)
    assert not pid.check("GP")
    assert not pid.check("LP")
    assert not pid.check("LP1")
    assert not pid.check("M0")
    assert pid.check("LP0").holds is None
    good = PID_WB(MAX_MIN)
    assert good.check("GP")
    assert good.check("LP")
    assert good.check("M0")
    assert PID_WB(bivariates["and"]).check("LP1").holds is None


def test_subset_equality():
    """
    I_do changes redundancy when a source duplicates another.
    """
    assert PID_WB(SAME).check("SE")
    check = PID_Do(SAME).check("SE")
    assert not check
    assert check.tested == 2


def test_lower_bound():
    """
    I_rr is 0.5 on 'rdn xor', but the common bit of the sources carries a bit about the target.
    """
    assert PID_WB(MAX_MIN).check("LB")
    check = PID_RR(bivariates["rdn xor"]).check("LB")
    assert not check
    assert check.max_violation == pytest.approx(0.5)


def test_identity():
    """
    I_min violates both identity properties; I_wedge only the strong one.
    """
    and_ = bivariates["and"]
    assert not PID_WB(and_).check("IID")
    assert not PID_WB(and_).check("ID")
    assert PID_GK(and_).check("IID")
    assert not PID_GK(and_).check("ID")
    assert PID_Do(and_).check("ID")


def test_units():
    """
    Checks compare values in the units of the distribution.
    """
    d = MAX_MIN.copy(base=np.e)
    assert PID_WB(d).check("SR")
    assert PID_WB(d).check("SCR").max_violation == pytest.approx(0.5 * np.log(2))


def test_claims():
    """
    Claims transcribe Table 5 of Liardi et al. and the source chain rule results of Gutknecht.
    """
    assert PID_WB.claims()["LP1"].holds is True
    assert PID_WB.claims()["SCR"].holds is False
    assert PID_SX.claims()["SCR"].holds is True
    assert PID_BROJA.claims()["LP1"].holds is None
    assert str(PID_RAV.claims()["M0"]) == "✓†"
    assert str(PID_IG.claims()["TM"]) == "✓*"
    assert str(PID_Proj.claims()["CO"]) == "✓§"
    assert str(PID_WB.claims()["DI"]) == "✗∥"
    assert str(PID_SX.claims()["GP"]) == "✗‡"
    assert claims_for("PID_WB") == PID_WB.claims() == claims_for(PID_WB(bivariates["and"]))


def test_claims_corrections():
    """
    Entries of Table 5 contradicted by its appendix or by a counterexample are corrected.
    """
    assert str(PID_Prec.claims()["*"]) == "✓!"
    assert str(PID_GK.claims()["*"]) == "✗!"
    assert str(PID_RR.claims()["LB"]) == "✗!"
    assert str(PID_RDR.claims()["BP"]) == "✗!"
    assert not PID_GK(SAME).check("*")
    assert not PID_RR(bivariates["rdn xor"]).check("LB")
    check = PID_RDR(COPY, [[0], [1]], [2, 3]).check("BP")
    assert not check
    assert "must not vanish" in str(check)


def test_claims_degradation():
    """
    PID_Deg is the measure of PID_Prec, so it takes the claims for I_prec.
    """
    deg, prec = PID_Deg.claims(), PID_Prec.claims()
    assert list(deg) == list(prec)
    assert str(deg["*"]) == "✓!"
    assert deg["LP1"].holds is False
    for key in prec:
        assert deg[key].holds == prec[key].holds
        assert deg[key].caveat.endswith("as for I_prec, the same measure")


def test_claims_local_positivity():
    """
    LP is claimed when both LP0 and LP1 are.
    """
    assert PID_WB.claims()["LP"].holds is True
    assert PID_BROJA.claims()["LP"].holds is True
    assert PID_CT.claims()["LP"].holds is False


def test_claims_uncovered():
    """
    Measures the literature does not cover have no claims.
    """
    assert PID_Delta.claims() == {}
    assert PID_CI.claims() == {}


def test_claims_cover_known_measures_and_properties():
    """
    Claims name existing measures and properties.
    """
    assert set(_CLAIMS) <= {pid.__name__ for pid in __all_pids}
    for claims in _CLAIMS.values():
        assert set(claims) <= set(PROPERTIES)


@pytest.mark.parametrize("pid", [PID_WB, PID_MMI, PID_GK, PID_SX, PID_PM, PID_CCS, PID_Deg, PID_Prec, PID_RDR])
def test_claims_agree_with_checks(pid):
    """
    No claim made without caveat is contradicted on small examples.
    """
    claims = pid.claims()
    for d in (bivariates["and"], bivariates["rdn xor"], SAME):
        for check in pid(d).check_properties():
            claim = claims.get(check.key)
            if claim is not None and claim.holds and not claim.mark:
                assert check.holds is not False, str(check)


def test_property_table(capsys):
    """
    The claims of several measures print as a table.
    """
    property_table(pids=[PID_WB, PID_SX], props=["LP", "SCR"])
    out = capsys.readouterr().out
    assert "I_min" in out
    assert "SCR" in out
    property_table()
    assert "I_∧" in capsys.readouterr().out
