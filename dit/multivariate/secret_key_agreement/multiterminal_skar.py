"""
Multiterminal secret key capacity with helper terminals, and the companion
minimum rate of communication for omniscience, from Csiszár and Narayan
:cite:`csiszar2004secrecy`.
"""

from itertools import combinations

import numpy as np
from scipy.optimize import linprog

from ...exceptions import ditException
from ...helpers import flatten, normalize_rvs
from ...utils import unitful
from ..entropy import entropy

__all__ = (
    "omniscience_rate",
    "secret_key_capacity",
)


def _parse_key_terminals(rvs, key_terminals):
    """Validate ``key_terminals`` as a set of indices into ``rvs``."""
    n = len(rvs)
    if key_terminals is None:
        return frozenset(range(n))
    key = frozenset(key_terminals)
    if not key <= frozenset(range(n)):
        msg = f"key_terminals {sorted(key)} must index into the {n} terminals of rvs."
        raise ditException(msg)
    if len(key) < 2:
        msg = "At least two key terminals are required."
        raise ditException(msg)
    return key


def _omniscience_lp(dist, rvs, key_terminals, crvs):
    """Return ``(H(X_M | crvs), R_CO)`` for the given terminals and key set."""
    if dist.is_symbolic():
        msg = "Multiterminal secret key capacity is only implemented for numeric distributions."
        raise ditException(msg)

    rvs, crvs = normalize_rvs(dist, rvs, crvs)
    rvs = [list(flatten(rv)) for rv in rvs]
    key = _parse_key_terminals(rvs, key_terminals)
    n = len(rvs)

    h_total = entropy(dist, rvs=list(flatten(rvs)), crvs=crvs)

    A_ub, b_ub = [], []
    for size in range(1, n):
        for B in combinations(range(n), size):
            if key <= frozenset(B):
                continue
            B_rvs = [i for j in B for i in rvs[j]]
            Bc_rvs = [i for j in range(n) if j not in B for i in rvs[j]]
            row = np.zeros(n)
            row[list(B)] = -1.0
            A_ub.append(row)
            b_ub.append(-entropy(dist, rvs=B_rvs, crvs=Bc_rvs + crvs))

    res = linprog(np.ones(n), A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=(0, None), method="highs")
    if not res.success:  # pragma: no cover
        msg = f"Omniscience linear program failed: {res.message}"
        raise ditException(msg)

    return h_total, res.fun


@unitful
def omniscience_rate(dist, rvs=None, key_terminals=None, crvs=None):
    """
    Compute the minimum rate of public communication for omniscience
    :cite:`csiszar2004secrecy`.

    This is the smallest total rate :math:`R_{CO}^{\\mathcal{A}}` of public
    discussion after which every terminal in the key set :math:`\\mathcal{A}`
    can recover the observations of all terminals :math:`\\mathcal{M}`:

    .. math::
        R_{CO}^{\\mathcal{A}} = \\min \\sum_{j \\in \\mathcal{M}} R_j
        \\quad \\text{s.t.} \\quad
        \\sum_{j \\in \\mathcal{B}} R_j \\geq \\H{X_\\mathcal{B} \\mid X_{\\mathcal{B}^c}}
        \\quad \\forall\\, \\emptyset \\neq \\mathcal{B} \\subsetneq \\mathcal{M},\\
        \\mathcal{A} \\not\\subseteq \\mathcal{B}.

    Parameters
    ----------
    dist : Distribution
        The distribution of interest.
    rvs : list, None
        A list of lists. Each inner list specifies the indexes of the random
        variables observed by one terminal. If None, each random variable is
        its own terminal.
    key_terminals : iterable of int, None
        Indices into ``rvs`` of the terminals that must become omniscient.
        If None, all terminals must.
    crvs : list, None
        Random variables known to everyone, including the eavesdropper (e.g.
        those of compromised terminals); all entropies are conditioned on them.

    Returns
    -------
    R_CO : float
        The minimum communication rate for omniscience.

    Raises
    ------
    ditException
        Raised if fewer than two key terminals are given, if
        ``key_terminals`` does not index into ``rvs``, or if ``dist`` is
        symbolic.
    """
    return _omniscience_lp(dist, rvs, key_terminals, crvs)[1]


@unitful
def secret_key_capacity(dist, rvs=None, key_terminals=None, crvs=None):
    """
    Compute the multiterminal secret key capacity of Csiszár and Narayan
    :cite:`csiszar2004secrecy`.

    The terminals in ``key_terminals`` (the set :math:`\\mathcal{A}`) agree on
    a secret key using unlimited public discussion; the remaining terminals
    act as helpers and need not be kept ignorant of the key. The capacity is

    .. math::
        C_{SK}^{\\mathcal{A}} = \\H{X_\\mathcal{M}} - R_{CO}^{\\mathcal{A}},

    where :math:`R_{CO}^{\\mathcal{A}}` is the :func:`omniscience_rate`. When
    ``crvs`` is given, every entropy is conditioned on it, which yields the
    private key capacity with those variables revealed to the eavesdropper.
    When every terminal is a key terminal this equals the
    :func:`~dit.multivariate.caekl_mutual_information`.

    Parameters
    ----------
    dist : Distribution
        The distribution of interest.
    rvs : list, None
        A list of lists. Each inner list specifies the indexes of the random
        variables observed by one terminal. If None, each random variable is
        its own terminal.
    key_terminals : iterable of int, None
        Indices into ``rvs`` of the terminals that share the key. If None,
        all terminals share it.
    crvs : list, None
        Random variables known to the eavesdropper (e.g. those of compromised
        terminals, which still cooperate in the protocol).

    Returns
    -------
    C_SK : float
        The secret key capacity.

    Raises
    ------
    ditException
        Raised if fewer than two key terminals are given, if
        ``key_terminals`` does not index into ``rvs``, or if ``dist`` is
        symbolic.

    Examples
    --------
    A helper who knows the xor of Alice's and Bob's independent bits lets
    them agree on one secret bit:

    >>> d = dit.Distribution(['000', '011', '101', '110'], [1 / 4] * 4)
    >>> secret_key_capacity(d, [[0], [1], [2]], key_terminals=[0, 1])
    1.0
    """
    h_total, r_co = _omniscience_lp(dist, rvs, key_terminals, crvs)
    return h_total - r_co
