"""
Distributions useful for illustrating the behavior of the various intrinsic
measures.
"""

import numpy as np

from ..distribution import Distribution

__all__ = (
    "intrinsic_1",
    "intrinsic_2",
    "intrinsic_3",
    "bound_information",
    "deterministic_erasure",
    "deterministic_erasure_source",
    "xor_key",
    "chitambar_bob_speaks",
    "chitambar_two_way",
    "james_problem",
    "reversely_degraded",
    "gisin_wolf",
)


# from the intrinsic information paper
intrinsic_1 = Distribution(["000", "011", "101", "110", "222", "333"], [1 / 8] * 4 + [1 / 4] * 2)
intrinsic_1.secret_rate = 0.0


# from the reduced intrinsic information paper
intrinsic_2 = Distribution(["000", "011", "101", "110", "220", "331"], [1 / 8] * 4 + [1 / 4] * 2)
intrinsic_2.secret_rate = 1.0


# from the minimal intrinsic information paper, with alpha_1 = 1/3 and alpha_2 = 1/2
intrinsic_3 = Distribution(
    ["000", "001", "012", "013", "102", "103", "110", "111", "220", "221", "332", "333"],
    [1 / 24, 1 / 12, 1 / 24, 1 / 12, 1 / 24, 1 / 12, 1 / 24, 1 / 12, 1 / 8, 1 / 8, 1 / 8, 1 / 8],
)
intrinsic_3.secret_rate = 0.97927916037609197


# from the bipartite bound information paper; Eve's symbol 2 is the erasure
bound_information = Distribution(
    ["000", "002", "010", "011", "012", "100", "101", "102", "111", "112"],
    [5 / 36, 5 / 36, 2 / 36, 2 / 36, 4 / 36, 2 / 36, 2 / 36, 4 / 36, 5 / 36, 5 / 36],
)
bound_information.secret_rate = 0.0


def deterministic_erasure(eps):
    """
    A deterministic-erasure source :cite:`abin2026source`: Alice's X is uniform
    on {0, 1, 2, 3}, Eve sees Z = X // 2, and Bob sees X through an erasure
    channel, Y = X with probability 1 - eps and the erasure symbol 4 otherwise.

    Since Z is a function of X, Y - X - Z is a Markov chain, so the lower bound
    I[X:Y] - I[Y:Z] meets the upper bound I[X:Y|Z] = (1 - eps) H[X|Z]
    :cite:`maurer1993secret`, and the secret key agreement rate is 1 - eps.

    Parameters
    ----------
    eps : float
        The erasure probability, in [0, 1].

    Returns
    -------
    dist : Distribution
        The source, with its ``secret_rate`` attribute set.
    """
    outcomes = []
    pmf = []
    for x in range(4):
        z = x // 2
        outcomes += [f"{x}{x}{z}", f"{x}4{z}"]
        pmf += [(1 - eps) / 4, eps / 4]
    dist = Distribution(outcomes, pmf)
    dist.secret_rate = 1 - eps
    return dist


deterministic_erasure_source = deterministic_erasure(0.3)


def xor_key(pmf):
    """
    The XOR key source :cite:`abin2026source`: X and Y are binary with joint
    distribution ``pmf`` and Eve sees Z = X xor Y.

    Its secret key agreement rate is unknown. Every known upper bound equals
    I[X:Y], and Abin & Gohari conjecture the rate is smaller for some ``pmf``,
    so no ``secret_rate`` attribute is set.

    Parameters
    ----------
    pmf : iterable of float, len(pmf) == 4
        The probabilities of (X, Y) = 00, 01, 10, 11.

    Returns
    -------
    dist : Distribution
        The source.
    """
    return Distribution(["000", "011", "101", "110"], list(pmf))


# Figure 4 of the Chitambar, Fortescue & Hsieh paper: the rate I[X:Y|Z] = 1/3 is
# achieved when Bob announces whether y is in {0, 1}, but not by Alice alone
chitambar_bob_speaks = Distribution(
    ["000", "110", "021", "121", "022"],
    [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 3],
)
chitambar_bob_speaks.secret_rate = 1 / 3


# Figures 4 and 5 of the Chitambar, Fortescue & Hsieh paper: the rate
# I[X:Y|Z] = 1/5 needs both parties to announce whether their symbol is in {0, 1}
chitambar_two_way = Distribution(
    ["000", "110", "021", "121", "022", "203", "213", "204"],
    [1 / 10, 1 / 10, 1 / 10, 1 / 10, 1 / 5, 1 / 10, 1 / 10, 1 / 5],
)
chitambar_two_way.secret_rate = 1 / 5


# the "Problem" distribution of the unique information and secret key agreement
# paper, reordered as (X1, Y, X0) so that X0 is the eavesdropper; the one-way
# rate from Alice meets the intrinsic mutual information at 1/2
james_problem = Distribution(["000", "110", "200", "011"], [1 / 4] * 4)
james_problem.secret_rate = 1 / 2


def _h(p):
    return -sum(q * np.log2(q) for q in (p, 1 - p) if q > 0)


def _bsc(p, q):
    return p * (1 - q) + q * (1 - p)


def reversely_degraded(a, b, c, e):
    """
    A reversely degraded source :cite:`ahlswede1993common`: two independent
    components, the first degraded toward Bob and the second toward Alice.

    In the first, X1 is uniform, Y1 is X1 through a binary symmetric channel
    with crossover ``a``, and Z1 is Y1 through one with crossover ``b``, so
    X1 - Y1 - Z1. In the second, Y2 is uniform, X2 is Y2 through crossover
    ``c``, and Z2 is X2 through crossover ``e``, so Y2 - X2 - Z2. The secret
    key agreement rate is I[X:Y|Z] = I[X1:Y1|Z1] + I[X2:Y2|Z2], which needs
    two-way communication.

    Parameters
    ----------
    a, b, c, e : float
        The crossover probabilities, in [0, 1/2].

    Returns
    -------
    dist : Distribution
        The source, with its ``secret_rate`` attribute set.
    """
    outcomes = []
    pmf = []
    for x1, y1, z1, y2, x2, z2 in np.ndindex(2, 2, 2, 2, 2, 2):
        p1 = (a if x1 != y1 else 1 - a) * (b if y1 != z1 else 1 - b) / 2
        p2 = (c if y2 != x2 else 1 - c) * (e if x2 != z2 else 1 - e) / 2
        outcomes.append((f"{x1}{x2}", f"{y1}{y2}", f"{z1}{z2}"))
        pmf.append(p1 * p2)
    dist = Distribution(outcomes, pmf)
    dist.secret_rate = _h(_bsc(a, b)) - _h(a) + _h(_bsc(c, e)) - _h(c)
    return dist


def gisin_wolf(alpha):
    """
    The standard-basis measurement of a family of qutrit states
    :cite:`gisin2000linking`. In units of 1/21, each diagonal pair (i, i) has
    weight 2 and Eve sees the erasure symbol 6; each pair (i, i + 1 mod 3) has
    weight 5 - alpha and each (i, i - 1 mod 3) weight alpha, and on these Eve
    learns the pair, labeled 0 through 5.

    For 2 <= alpha <= 3 a degradation of Z makes X and Y independent, so the
    secret key agreement rate is 0 :cite:`gisin2000linking`. Outside that
    interval the rate is positive :cite:`pauwels2026bipartite` but unknown, and
    no ``secret_rate`` attribute is set.

    Parameters
    ----------
    alpha : float
        The family parameter, in [0, 5].

    Returns
    -------
    dist : Distribution
        The source.
    """
    labels = {(0, 1): 0, (1, 2): 1, (2, 0): 2, (0, 2): 3, (1, 0): 4, (2, 1): 5}
    outcomes = []
    pmf = []
    for x, y in np.ndindex(3, 3):
        if x == y:
            weight, z = 2, 6
        else:
            weight = 5 - alpha if y == (x + 1) % 3 else alpha
            z = labels[(x, y)]
        if weight > 0:
            outcomes.append(f"{x}{y}{z}")
            pmf.append(weight / 21)
    dist = Distribution(outcomes, pmf)
    if 2 <= alpha <= 3:
        dist.secret_rate = 0.0
    return dist
