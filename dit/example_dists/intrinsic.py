"""
Distributions useful for illustrating the behavior of the various intrinsic
measures.
"""

from ..distribution import Distribution

__all__ = (
    "intrinsic_1",
    "intrinsic_2",
    "intrinsic_3",
    "bound_information",
    "deterministic_erasure",
    "deterministic_erasure_source",
    "xor_key",
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
