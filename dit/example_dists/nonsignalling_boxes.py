"""
Canonical non-signalling boxes.
"""

from collections import defaultdict
from itertools import product

import numpy as np

from dit.distribution import Distribution
from dit.exceptions import ditException
from dit.helpers import parse_rvs

__all__ = (
    "chsh_value",
    "correlated_pr_box",
    "is_nonsignalling",
    "pr_box",
    "pr_box_d",
    "tripartite_pr_box",
)


def pr_box(eta=1, name=True):
    """
    The Popescu-Rohrlich box, or PR box, is the canonical non-signalling,
    non-local probability distribution used in the study of superquantum
    correlations :cite:`popescu1994quantum`. It has two space-like separated
    inputs, X and Y, and two associated outputs, A and B.

    `eta` is the noise level of this correlation. For 0 <= eta <= 1/2 the box
    can be realized classically. For 1/2 < eta <= 1/sqrt(2) the box can be
    realized quantum-mechanically.

    Parameters
    ----------
    eta : float, 0 <= eta <= 1
        The noise level of the box. Defaults to 1.

    name : bool
        Whether to set rv names or not. Defaults to True.

    Returns
    -------
    pr : Distribution
        The PR box distribution.
    """
    return pr_box_d(2, eta=eta, name=name)


def pr_box_d(d, eta=1, name=True):
    """
    The d-output generalization of the PR box :cite:`barrett2005nonlocal`.
    Inputs X and Y are binary and outputs A and B take values in
    {0, ..., d-1}; the ideal box is uniform on outcomes with
    (b - a) = x * y (mod d), and every non-local vertex of the two-input
    non-signalling polytope is equivalent to one of these under reversible
    local operations. Inputs are uniformly distributed.

    Parameters
    ----------
    d : int >= 2
        The size of each output alphabet.

    eta : float, 0 <= eta <= 1
        Mixing weight of the ideal box against uniform noise. Defaults to 1.

    name : bool
        Whether to set rv names (XYAB) or not. Defaults to True.

    Returns
    -------
    box : Distribution
        The d-output PR box distribution.
    """
    outcomes = list(product(range(2), range(2), range(d), range(d)))
    pmf = [(eta * ((b - a) % d == x * y) / d + (1 - eta) / d**2) / 4 for x, y, a, b in outcomes]
    box = Distribution(outcomes, pmf)

    if name:
        box.set_rv_names("XYAB")

    return box


def tripartite_pr_box(eta=1, name=True):
    """
    The tripartite extension of the PR box, uniform on outcomes with
    a xor b xor c = x * y * z :cite:`barrett2005popescu`. It has three binary
    inputs X, Y, Z (uniformly distributed) and three binary outputs A, B, C.

    Parameters
    ----------
    eta : float, 0 <= eta <= 1
        Mixing weight of the ideal box against uniform noise. Defaults to 1.

    name : bool
        Whether to set rv names (XYZABC) or not. Defaults to True.

    Returns
    -------
    box : Distribution
        The tripartite PR box distribution.
    """
    outcomes = list(product([0, 1], repeat=6))
    pmf = [(eta * (a ^ b ^ c == x * y * z) / 4 + (1 - eta) / 8) / 8 for x, y, z, a, b, c in outcomes]
    box = Distribution(outcomes, pmf)

    if name:
        box.set_rv_names("XYZABC")

    return box


def correlated_pr_box(eps, name=True):
    """
    The correlated non-local box eps * PR + (1 - eps) * C, where C is the
    fully correlated box uniform on a xor b = 0 :cite:`brunner2009nonlocality`.
    Its CHSH value is 2 * (1 + eps); it is the family on which the known
    non-locality distillation protocols operate. Inputs are uniformly
    distributed.

    Parameters
    ----------
    eps : float, 0 <= eps <= 1
        Weight of the PR box.

    name : bool
        Whether to set rv names (XYAB) or not. Defaults to True.

    Returns
    -------
    box : Distribution
        The correlated non-local box distribution.
    """
    outcomes = list(product([0, 1], repeat=4))
    pmf = [(eps * (a ^ b == x * y) + (1 - eps) * (a ^ b == 0)) / 8 for x, y, a, b in outcomes]
    box = Distribution(outcomes, pmf)

    if name:
        box.set_rv_names("XYAB")

    return box


def _joint_table(dist, rvs):
    """
    Marginal pmf over `rvs` as a dict from value-index tuples to
    probabilities, along with each variable's alphabet.
    """
    _, indices = parse_rvs(dist, list(rvs), sort=False)
    alphabets = [sorted(dist.alphabet[i]) for i in indices]
    lookups = [{v: k for k, v in enumerate(alphabet)} for alphabet in alphabets]
    d = dist.copy(base="linear")
    table = defaultdict(float)
    for outcome, p in zip(d.outcomes, d.pmf, strict=True):
        table[tuple(lookup[outcome[i]] for lookup, i in zip(lookups, indices, strict=True))] += p
    return table, alphabets


def chsh_value(dist, inputs=(0, 1), outputs=(2, 3)):
    """
    The Clauser-Horne-Shimony-Holt value of a bipartite binary box
    :cite:`clauser1969proposed`. With correlators
    E(x, y) = sum_{a, b} (-1)^(a xor b) p(a, b | x, y), it is

        max | E(x, y) + E(x, y') + E(x', y) - E(x', y') |

    over the choice of which input pair is negated. A box is local iff its
    CHSH value is at most 2; quantum boxes satisfy CHSH <= 2 sqrt(2).

    Parameters
    ----------
    dist : Distribution
        A joint distribution over the inputs and outputs.

    inputs : sequence of two rvs
        The input variables (X, Y), by index or name. Defaults to (0, 1).

    outputs : sequence of two rvs
        The output variables (A, B), by index or name. Defaults to (2, 3).

    Returns
    -------
    chsh : float
        The CHSH value, in [0, 4].

    Raises
    ------
    ditException
        If any variable is not binary or any input pair has zero probability.
    """
    if len(inputs) != 2 or len(outputs) != 2:
        msg = "`chsh_value` requires exactly two inputs and two outputs."
        raise ditException(msg)

    table, alphabets = _joint_table(dist, list(inputs) + list(outputs))
    if any(len(alphabet) != 2 for alphabet in alphabets):
        msg = "`chsh_value` requires binary inputs and outputs."
        raise ditException(msg)

    correlators = np.zeros((2, 2))
    for x, y in product(range(2), repeat=2):
        p_xy = sum(table[x, y, a, b] for a, b in product(range(2), repeat=2))
        if p_xy <= 0:
            msg = f"Input pair {(x, y)} has zero probability."
            raise ditException(msg)
        correlators[x, y] = sum((-1) ** (a ^ b) * table[x, y, a, b] for a, b in product(range(2), repeat=2)) / p_xy

    total = correlators.sum()
    return float(max(abs(total - 2 * correlators[x, y]) for x, y in product(range(2), repeat=2)))


def is_nonsignalling(dist, inputs, outputs, atol=1e-9):
    """
    Test whether a box is non-signalling :cite:`barrett2005nonlocal`: for
    every party i, the joint distribution of the other parties' outputs,
    conditioned on all inputs, does not depend on the input x_i. Party i
    holds ``inputs[i]`` and ``outputs[i]``.

    Parameters
    ----------
    dist : Distribution
        A joint distribution over the inputs and outputs.

    inputs : sequence of rvs
        The input variable of each party, by index or name.

    outputs : sequence of rvs
        The output variable of each party, by index or name.

    atol : float
        Absolute tolerance for comparing conditional probabilities.

    Returns
    -------
    nonsignalling : bool
        Whether the box is non-signalling.

    Raises
    ------
    ditException
        If `inputs` and `outputs` differ in length, or any input
        configuration has zero probability.
    """
    n = len(inputs)
    if n != len(outputs):
        msg = "`inputs` and `outputs` must have the same length."
        raise ditException(msg)

    table, alphabets = _joint_table(dist, list(inputs) + list(outputs))
    in_sizes = [len(alphabet) for alphabet in alphabets[:n]]
    out_sizes = [len(alphabet) for alphabet in alphabets[n:]]

    shape = in_sizes + out_sizes
    joint = np.zeros(shape)
    for key, p in table.items():
        joint[key] += p

    p_inputs = joint.sum(axis=tuple(range(n, 2 * n)))
    if np.any(p_inputs <= 0):
        msg = "Every input configuration must have positive probability."
        raise ditException(msg)
    conditional = joint / p_inputs.reshape(in_sizes + [1] * n)

    for i in range(n):
        others = conditional.sum(axis=n + i)
        if not np.allclose(others, others.take([0], axis=i), atol=atol):
            return False

    return True
