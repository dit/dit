"""
Tests for dit.pid.conditional: the conditional and contextual decompositions of
Gutknecht (2026), arXiv:2610.33743.
"""

import numpy as np
import pytest

from dit import Distribution
from dit.exceptions import ditException
from dit.multivariate import coinformation
from dit.pid import (
    PED_SX,
    PID_GH,
    PID_GK,
    PID_MMI,
    PID_PM,
    PID_SX,
    PID_WB,
    ConditionalPID,
    ContextualPID,
)
from dit.pid.distributions import bivariates, trivariates

MAX_MIN = trivariates["max min"]  # max(C, T), min(C, T), C, T
RED = ((0,), (1,))
U0, U1, SYN = ((0,),), ((1,),), ((0, 1),)


def _random(seed, shape):
    rng = np.random.default_rng(seed)
    pmf = rng.dirichlet(np.ones(int(np.prod(shape))))
    return Distribution(["".join(map(str, o)) for o in np.ndindex(*shape)], pmf)


def test_imin_max_min():
    """
    Section 7 of the paper: for I_min, the contextual redundancy of the maximum and the
    minimum given C is 1/2 bit, their conditional redundancy 0.
    """
    contextual = ContextualPID(PID_WB(MAX_MIN), context=[2])
    conditional = ConditionalPID(PID_WB, MAX_MIN, [[0], [1]], [3], crvs=[2])
    assert contextual.get_red(RED) == pytest.approx(0.5)
    assert conditional.get_red(RED) == pytest.approx(0.0)
    assert [contextual[node] for node in (RED, U0, U1, SYN)] == pytest.approx([0.5, 0, 0, 0.5])
    assert [conditional[node] for node in (RED, U0, U1, SYN)] == pytest.approx([0, 0.5, 0.5, 0])
    assert contextual != conditional


@pytest.mark.parametrize("measure", [PID_SX, PID_PM, PID_GK])
def test_scr_measures_agree(measure):
    """
    Measures satisfying the source chain rule give the same contextual and conditional
    decompositions (Theorem 3.9).
    """
    contextual = ContextualPID(measure(MAX_MIN), context=[2])
    conditional = ConditionalPID(measure, MAX_MIN, [[0], [1]], [3], crvs=[2])
    assert contextual == conditional
    assert [conditional[node] for node in (RED, U0, U1, SYN)] == pytest.approx([0, 0.5, 0.5, 0])


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("measure", [PID_SX, PID_WB, PID_MMI])
def test_agreement_is_the_source_chain_rule(measure, seed):
    """
    For every context, the decompositions agree exactly when the source chain rule holds.
    """
    d = _random(seed, (2, 2, 2, 2))
    pid = measure(d)
    for context in ([0], [1], [2]):
        contextual = ContextualPID(pid, context=context)
        conditional = ConditionalPID(measure, d, contextual._sources, [3], crvs=context)
        assert (contextual == conditional) == bool(pid.check("SCR", context=context))
    if measure is PID_SX:
        assert pid.check("SCR")


def test_contextual_atoms():
    """
    Table 1 of the paper: the atoms of three sources grouped by their access given the third.
    """
    contextual = ContextualPID(PID_SX(MAX_MIN), context=[2])
    expected = {
        RED: {((0,), (1,)), ((0,), (1, 2)), ((1,), (0, 2)), ((0, 2), (1, 2)), ((0, 1), (0, 2), (1, 2))},
        U0: {((0,),), ((0, 2),), ((0, 1), (0, 2))},
        U1: {((1,),), ((1, 2),), ((0, 1), (1, 2))},
        SYN: {((0, 1),), ((0, 1, 2),)},
    }
    for node, atoms in expected.items():
        assert set(contextual.atoms(node)) == atoms
    grouped = set().union(*expected.values())
    provided = {node for node in contextual._pid._lattice if node not in grouped}
    assert provided == {((2,),), ((0,), (2,)), ((1,), (2,)), ((0,), (1,), (2,)), ((2,), (0, 1))}


@pytest.mark.parametrize("measure", [PID_SX, PID_WB])
def test_conditional_mutual_informations(measure):
    """
    Both decompositions decompose I(rest : T | c): Propositions 3.2 and 3.5.
    """
    d = _random(7, (2, 2, 3, 2))
    contextual = ContextualPID(measure(d), context=[2])
    conditional = ConditionalPID(measure, d, [[0], [1]], [3], crvs=[2])
    for decomposition in (contextual, conditional):
        assert decomposition._total == pytest.approx(coinformation(d, [[0, 1], [3]], [2]))
        for i in (0, 1):
            assert decomposition.get_red(((i,),)) == pytest.approx(coinformation(d, [[i], [3]], [2]))
        assert sum(decomposition[node] for node in decomposition._lattice) == pytest.approx(decomposition._total)
        assert decomposition.consistent
        assert decomposition.complete


def test_external_condition():
    """
    Conditioning on a variable outside the decomposition averages the decompositions of
    the conditional distributions: AND given Z = 0, XOR given Z = 1.
    """
    joint = {}
    for z, name in enumerate(["and", "synergy"]):
        for outcome, p in zip(bivariates[name].outcomes, bivariates[name].pmf, strict=True):
            joint[(*outcome, str(z))] = p / 2
    d = Distribution(["".join(o) for o in joint], list(joint.values()))
    conditional = ConditionalPID(PID_WB, d, [[0], [1]], [2], crvs=[3])
    pid_and, pid_xor = PID_WB(bivariates["and"]), PID_WB(bivariates["synergy"])
    for node in conditional._lattice:
        assert conditional[node] == pytest.approx((pid_and[node] + pid_xor[node]) / 2)
        assert conditional.get_red(node) == pytest.approx((pid_and.get_red(node) + pid_xor.get_red(node)) / 2)
    assert sorted(z for z, _, _ in conditional.pids) == [("0",), ("1",)]
    assert sum(p for _, p, _ in conditional.pids) == pytest.approx(1.0)


def test_defaults_and_names():
    """
    The target defaults to the last variable and the sources to the others not
    conditioned on; variables may be named.
    """
    by_index = ConditionalPID(PID_SX, MAX_MIN, crvs=[2])
    assert by_index._sources == ((0,), (1,))
    named = MAX_MIN.copy()
    named.set_rv_names("XYCT")
    by_name = ConditionalPID(PID_SX, named, [["X"], ["Y"]], ["T"], crvs=["C"])
    for node, other in zip(sorted(by_index._lattice), sorted(by_name._lattice), strict=True):
        assert by_index[node] == pytest.approx(by_name[other])
    assert by_name.get_pi([["X"]]) == pytest.approx(0.5)


def test_log_base():
    """
    Values are in the units of the distribution.
    """
    d = MAX_MIN.copy(base=np.e)
    conditional = ConditionalPID(PID_WB, d, [[0], [1]], [3], crvs=[2])
    contextual = ContextualPID(PID_WB(d), context=[2])
    assert conditional.get_pi(U0) == pytest.approx(0.5 * np.log(2))
    assert contextual.get_red(RED) == pytest.approx(0.5 * np.log(2))


def test_options_and_pids():
    """
    Options are passed to the measure under every conditional distribution.
    """
    conditional = ConditionalPID(PID_SX, MAX_MIN, [[0], [1]], [3], crvs=[2], pointwise=True)
    assert all(pid._pointwise for _, _, pid in conditional.pids)
    conditional = ConditionalPID(PID_GH, MAX_MIN, [[0], [1]], [3], crvs=[2], method="exact")
    assert all(pid._kwargs == {"method": "exact"} for _, _, pid in conditional.pids)


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"crvs": None}, ditException),
        ({"crvs": []}, ditException),
        ({"crvs": [0]}, ditException),
        ({"crvs": [3]}, ditException),
        ({"crvs": [2], "sources": [[0], [0]]}, ditException),
        ({"crvs": [2], "bogus": 1}, TypeError),
    ],
)
def test_conditional_validation(kwargs, error):
    """
    The variables conditioned on must be given, and be neither sources nor the target.
    """
    kwargs = {"sources": [[0], [1]], "target": [3], **kwargs}
    with pytest.raises(error):
        ConditionalPID(PID_SX, MAX_MIN, **kwargs)


@pytest.mark.parametrize("measure", [PED_SX, PID_SX(MAX_MIN), "PID_SX"])
def test_conditional_needs_a_pid_class(measure):
    """
    The measure is a partial information decomposition class.
    """
    with pytest.raises(TypeError):
        ConditionalPID(measure, MAX_MIN, crvs=[2])


@pytest.mark.parametrize("context", [[], [0, 1, 2], [3], [[0, 1]]])
def test_contextual_validation(context):
    """
    The context is made of some, but not all, of the sources.
    """
    with pytest.raises(ValueError, match="sources"):
        ContextualPID(PID_SX(MAX_MIN), context=context)


def test_nodes_outside_the_lattice():
    """
    Nodes of the lattice of all the sources are not nodes of the remaining sources.
    """
    contextual = ContextualPID(PID_SX(MAX_MIN), context=[2])
    conditional = ConditionalPID(PID_SX, MAX_MIN, [[0], [1]], [3], crvs=[2])
    for decomposition in (contextual, conditional):
        with pytest.raises(ditException, match="not in the lattice"):
            decomposition.get_red(((2,),))
        with pytest.raises(ditException, match="not in the lattice"):
            decomposition.get_pi(((0,), (2,)))


def test_strings():
    """
    The tables name the measure and the context.
    """
    contextual = str(ContextualPID(PID_WB(MAX_MIN), context=[2]))
    conditional = ConditionalPID(PID_WB, MAX_MIN, [[0], [1]], [3], crvs=[2]).to_string(digits=2)
    assert "I_min\\{2}" in contextual
    assert "{0}{1}" in contextual
    assert "I_min|{2}" in conditional
    assert "0.50" in conditional


def test_symbolic_distributions():
    """
    Symbolic distributions are not supported.
    """
    pytest.importorskip("sympy")
    from dit.symbolic import symbolic_distribution, symbols

    p = symbols("p")
    d = symbolic_distribution(["000", "011", "101", "110"], [p / 2, p / 2, (1 - p) / 2, (1 - p) / 2])
    with pytest.raises(NotImplementedError):
        ConditionalPID(PID_SX, d, [[0]], [2], crvs=[1])
