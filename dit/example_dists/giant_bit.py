"""
Giant bit type distributions.
"""
from __future__ import division

from itertools import product

from .. import Distribution
from ..distconst import uniform

def giant_bit(n, k, fuzzy=False):
    """
    Return a 'giant bit' distribution of size `n` and alphabet size `k`.

    Parameters
    ----------
    n : int
        The number of identical bits.
    k : int
        The number of states for each bit.
    fuzzy : bool
        If true, add some noise to the giant bit.

    Returns
    -------
    gb : Distribution
        The giant bit distribution.
    """
    if fuzzy:
        alpha = list(map(str, range(k)))
        N = k**n - k
        pr1 = 0.99/k
        pr2 = 0.01/N
        outcomes = [''.join(o) for o in product(alpha, repeat=n)]
        pmf = [(pr1 if all(_ == o[0] for _ in o) else pr2) for o in outcomes]
        return Distribution(outcomes, pmf)
    else:
        return uniform([str(i)*n for i in range(k)])


def jeff(n):
    """
    The JEFF distribution, where the conditional probability p(y|X) is linear in X.

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
    prob = lambda x, y: ((1-y) + ((-1)**(1+y))*sum(x)/n) / 2**n
    dist = {''.join(map(str, x + (y,))): prob(x, y) for x, y in product(xs, (0, 1))}
    return Distribution(dist)
