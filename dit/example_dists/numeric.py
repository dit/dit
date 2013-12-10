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

    describes the number of successes in n i.i.d. draws each with probability
    of success p.

    Parameters
    ----------
    n : int
        A positive integer, the number of trials.
    p : float
        A float between 0 and 1, the probabilty of success of a trial.

    Returns
    -------
    d : ScalarDistribution
        The binomial distribution describes the number of successes in `n`
        trials (identically and independently distributed draws) each with
        probability of success `p`.

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
    pmf = [pp(n, k) for k in outcomes]
    return ScalarDistribution(outcomes, pmf)

def hypergeometric(N, K, n):
    """
    The hypergeometric distribution:

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
    pmf = [C(K, k)*C(N-K, n-k)/C(N, n) for k in outcomes]
    return ScalarDistribution(outcomes, pmf)

def uniform(a, b=None):
    """
    The discrete uniform distribution:
    P(x in a..b-1) = 1/(b-a)

    Parameters
    ----------
    a : int
        The lower bound of the uniform distribution. If `b` is None, then `a` is
        taken to be the length of the uniform with a lower bound of 0.
    b : int, None
        The upper bound of the uniform distribution. If None, a uniform
        distribution over 0 .. `a` is returned.

    Returns
    -------
    d : ScalarDistribution
        The uniform distribution from `a` to `b`-1.

    Raises
    ------
    ValueError
        Raised if `b` is not an integer or None, or if `a` is not an integer and
        positive if `b` is None or larger than `b` if be is not None.
    """
    if not (b is None or is_integer(b)):
        msg = "{0} is not an integer or None."
        raise ValueError(msg.format(b))
    if b is None:
        if not is_integer(a) or a <= 0:
            msg = "{0} is not a positive integer."
            raise ValueError(msg.format(a))
        a, b = 0, a
    else:
        if not is_integer(a) or a >= b:
            msg = "{0} is not an integer larger than {1}."
            raise ValueError(msg.format(a, b))
    outcomes = list(range(a, b))
    pmf = [1/(b-a)]*(b-a)
    return ScalarDistribution(outcomes, pmf)
