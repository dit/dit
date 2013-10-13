"""
Some standard discrete distribution.
"""

from __future__ import division

from .. import ScalarDistribution
from ..math.misc import combinations as C, is_integer, is_number

def bernoulli(p):
    """
    The Bernoulli distribution:

      x  P(x)
      0  1-p
      1  p

    Parameters
    ----------
    p : float
        A float between 0 and 1, the probability of a 1.

    Returns
    -------
    d : ScalarDistribution
        The Bernoulli distribution with probability `p`.

    Raises
    ------
    ValueError
        Raised if not 0 <= `p` <= 1.
    """
    return binomial(1, p)

def binomial(n, p):
    """
    The binomial distribution:

    f(k;n,p) = P(X = k) = nCk p^k (1-p)^(n-k)

    Parameters
    ----------
    n : int
        A positive integer, the number of trials.
    p : float
        A float between 0 and 1, the probabilty of success of a trial.

    Returns
    -------
    d : ScalarDistribution
        The binomial distribution of `n` trials with probabily `p` of success.

    Raises
    ------
    ValueError
        Raised if `n` is not a positive integer.
        Raised if not 0 <= `p` <= 1.
    """
    if not is_integer(n) or n < 0:
        raise ValueError("{0} is not a positive integer.".format(n))
    if not is_number(p) or not 0 <= p <= 1:
       raise ValueError("{0} is not a valid probability.".format(p))
    pp = lambda n, k: C(n, k) * p**k * (1-p)**(n-k)
    outcomes = list(range(n+1))
    pmf = [ pp(n, k) for k in outcomes ]
    return ScalarDistribution(outcomes, pmf)

def hypergeometric(N, K, n):
    """
    The binomial distribution:

    f(k;N,K,n) = P(X = k) = KCk * (N-K)C(n-k) / NCn

    Parameters
    ----------
    N : int
        A positive integer, the size of the population.
    K : int
        The number of successes in the population.
    n : int
        The number of draws to make (without replacement) from the population.

    Returns
    -------
    d : ScalarDistribution
        The hypergeometric distribution of a population of size `N` with `K` 
        successes in the population, and `n` draws are made, without 
        replacement, from that population. P(k) is the probability of k 
        successes among the `n` draws.

    Raises
    ------
    ValueError
        Raised if `N`, `K`, or `n` are not positive integers.
    """
    if not is_integer(N) or N < 0:
        raise ValueError("{0} is not a positive integer.".format(N))
    if not is_integer(K) or K < 0:
        raise ValueError("{0} is not a positive integer.".format(K))
    if not is_integer(n) or n < 0:
        raise ValueError("{0} is not a positive integer.".format(n))
    outcomes = list(range(max(0, n+K-N), min(K, n)+1))
    pmf = [ C(K, k)*C(N-K,n-k)/C(N, n) for k in outcomes ]
    return ScalarDistribution(outcomes, pmf)

def uniform(n):
    """
    The discrete uniform distribution:
    P(x in 0..n-1) = 1/n

    Parameters
    ----------
    n : int
        The range of the uniform distribution.

    Returns
    -------
    d : ScalarDistribution
        The uniform distribution of range `n`.

    Raises
    ------
    ValueError
        Raised if `n` is not a positive integer.
    """
    if not is_integer(n) or n < 0:
        raise ValueError("{0} is not a positive integer.".format(n))
    outcomes = list(range(n))
    pmf = [1/n]*n
    return ScalarDistribution(outcomes, pmf)