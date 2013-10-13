"""
Some standard discrete distribution.
"""

from .. import ScalarDistribution
from ..math.misc import combinations, is_integer

def bernoulli(p):
    """
    """
    return binomial(1, p)

def binomial(n, p):
    """
    """
    if not is_integer(n) and n >=0:
        raise ValueError("{} is not a positive integer.".format(n))
    if not 0 <= p <= 1:
        raise ValueError("{} is not a valid probability.".format(p))
    pp = lambda n, k: combinations(n, k) * p**k * (1-p)**(n-k)
    outcomes = list(range(n+1))
    probs = [ pp(n, k) for k in outcomes ]
    return ScalarDistribution(outcomes, probs)