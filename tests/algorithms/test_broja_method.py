"""
Tests for dit.algorithms.broja_method.
"""

import sys

import pytest

from dit.algorithms import broja_method
from dit.algorithms.broja_method import broja_solve_bivariate, ecos_available, select_broja_method
from dit.exceptions import OptimizationException
from dit.pid.distributions import bivariates

SOURCES = ((0,), (1,))
TARGET = (2,)

# BROJA unique informations (bits) of source 0 and source 1
KNOWN_UNIQUES = {
    "synergy": (0.0, 0.0),  # XOR
    "redundant": (0.0, 0.0),  # Y = X0 = X1
    "unique 1": (0.0, 1.0),  # Y = X1
    "cat": (1.0, 1.0),  # Y = (X0, X1)
    "diff": (0.18872187554086706, 0.18872187554086706),
}


@pytest.fixture
def without_ecos(monkeypatch):
    """
    Make ``import ecos`` fail, as on platforms without the optional package.
    """
    monkeypatch.setitem(sys.modules, "ecos", None)


def test_ecos_unavailable(without_ecos):
    """
    Without ECOS, 'auto' picks admUI, the cone program cannot be requested, and the
    fallback chain skips it.
    """
    assert not ecos_available()
    assert select_broja_method([2, 2, 2]) == "admui"
    assert select_broja_method([2, 2, 2], "scipy") == "scipy"
    assert broja_method._methods_fallback_chain("admui") == ["admui", "scipy"]
    assert broja_method._methods_fallback_chain("scipy") == ["scipy", "admui"]
    with pytest.raises(ImportError, match="method='cone' requires the optional 'ecos' package"):
        select_broja_method([2, 2, 2], "cone")


def test_select_unknown_method():
    with pytest.raises(ValueError, match=r"Unknown BROJA method 'newton'; expected one of \['admui', 'auto'"):
        select_broja_method([2, 2, 2], "newton")


@pytest.mark.parametrize("name", sorted(KNOWN_UNIQUES))
@pytest.mark.parametrize("method", ["admui", "cone", "auto"])
def test_known_uniques(method, name):
    """
    The solvers recover the BROJA unique informations of the canonical gates, keyed by source.
    """
    if method == "cone":
        pytest.importorskip("ecos")
    uniques, meta = broja_solve_bivariate(bivariates[name], SOURCES, TARGET, method=method)
    expected_method = method if method != "auto" else ("cone" if ecos_available() else "admui")
    assert meta["method"] == expected_method
    assert meta["converged"]
    assert set(uniques) == set(SOURCES)
    assert uniques[(0,)] == pytest.approx(KNOWN_UNIQUES[name][0], abs=1e-5)
    assert uniques[(1,)] == pytest.approx(KNOWN_UNIQUES[name][1], abs=1e-5)


def test_auto_without_ecos_uses_admui(without_ecos):
    uniques, meta = broja_solve_bivariate(bivariates["cat"], SOURCES, TARGET)
    assert meta["method"] == "admui"
    assert meta["converged"]
    assert uniques == pytest.approx({(0,): 1.0, (1,): 1.0}, abs=1e-6)


def test_unconverged_cone_falls_back_to_admui(monkeypatch):
    """
    A cone solution failing its feasibility check is discarded in favour of admUI's. The
    ECOS options go to the cone solver only.
    """
    calls = []

    def unconverged_cone(d, sources, target, **ecos_kwargs):
        calls.append(ecos_kwargs)
        return {sources[0]: 9.0, sources[1]: 9.0}, {"converged": False, "method": "cone"}

    monkeypatch.setattr(broja_method, "ecos_available", lambda: True)
    monkeypatch.setattr(broja_method, "_solve_cone", unconverged_cone)
    uniques, meta = broja_solve_bivariate(bivariates["diff"], SOURCES, TARGET, method="cone", feastol=1e-9)
    assert calls == [{"feastol": 1e-9}]
    assert meta["method"] == "admui"
    assert uniques == pytest.approx(dict(zip(SOURCES, KNOWN_UNIQUES["diff"], strict=True)), abs=1e-5)


def test_cone_iteration_limit_falls_back():
    """
    One interior-point iteration of ECOS cannot pass the feasibility check, so admUI answers.
    """
    pytest.importorskip("ecos")
    uniques, meta = broja_solve_bivariate(bivariates["diff"], SOURCES, TARGET, method="cone", max_iters=1)
    assert meta["method"] == "admui"
    assert uniques == pytest.approx(dict(zip(SOURCES, KNOWN_UNIQUES["diff"], strict=True)), abs=1e-5)


def test_failing_solver_falls_back(monkeypatch, without_ecos):
    """
    A solver that raises is skipped for the next most reliable one: admUI, then SLSQP.
    """

    def broken(*args, **kwargs):
        raise RuntimeError("admui broke")

    monkeypatch.setattr(broja_method, "_solve_admui", broken)
    uniques, meta = broja_solve_bivariate(bivariates["cat"], SOURCES, TARGET, method="admui", rng=0)
    assert meta["method"] == "scipy"
    assert uniques == pytest.approx({(0,): 1.0, (1,): 1.0}, abs=1e-6)


def test_all_solvers_fail(monkeypatch):
    """
    When every solver raises, the error reports each failure in the order they were tried.
    """

    def failing(name):
        def solve(*args, **kwargs):
            raise RuntimeError(f"{name} broke")

        return solve

    monkeypatch.setattr(broja_method, "ecos_available", lambda: True)
    for name in ("cone", "admui", "scipy"):
        monkeypatch.setattr(broja_method, f"_solve_{name}", failing(name))
    with pytest.raises(
        OptimizationException,
        match="All BROJA solvers failed: cone: cone broke; admui: admui broke; scipy: scipy broke",
    ):
        broja_solve_bivariate(bivariates["and"], SOURCES, TARGET)
