"""
Giant bit type distributions.
"""

from itertools import product

from .. import Distribution


def giant_bit(n=3, k=2):
    """
    Return a 'giant bit' distribution of size `n` and alphabet size `k`.

    Parameters
    ----------
    n : int
        The number of identical bits. Defaults to 3.
    k : int
        The number of states for each bit. Defaults to 2.

    Returns
    -------
    gb : Distribution
        The giant bit distribution.
    """
    alpha = list(map(str, range(k)))

    outcomes = [a * n for a in alpha]
    pmf = [1 / k] * k

    return Distribution(outcomes, pmf)


def jeff(n):
    """
    The JEFF distribution, where the conditional probability :math:`p(y|X)` is
    linear in X.

    Parameters
    ----------
    n : int
        The number of inputs

    Returns
    -------
    dist : Distribution
        The JEFF distribution with `n` inputs.
    """
    xs = list(product((0, 1), repeat=n))
    prob = lambda x, y: ((1 - y) + ((-1)**(1 + y)) * sum(x) / n) / 2**n
    dist = {''.join(map(str, x + (y,))): prob(x, y) for x, y in product(xs, (0, 1))}
    return Distribution(dist)
