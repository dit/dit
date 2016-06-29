from __future__ import division

import itertools
import dit

def iid_sum(n, k=2):
    """
    Returns a distribution relating n iid uniform draws to their sum.

    """
    alphabet = range(k)
    outcomes = list(itertools.product(alphabet, repeat=n))
    pmf = [1] * len(outcomes)
    d = dit.Distribution(outcomes, pmf, base='linear', validate=False)
    d.normalize()
    d = dit.insert_rvf(d, lambda x: (sum(x),))
    return d

def summed_dice(a=1, b=1):
    """
    Two die X and Y are summed to form Z in the following way:

        Z = X + b * Y

    X and Y are distributed as:

        P(X=i, Y=j) = a / 36 + (1 - a) * delta_{ij} / 6

    Parameters
    ----------
    a : float
        Specifies how independent X and Y are. a=0 means X and Y are perfectly
        correlated. a=1 means X and Y are independent.
    b : integer
        An integer specifying how to add X and Y.

    Returns
    -------
    d : distribution
        The joint distribution for P(X, Y, Z).

    References
    ----------
    Malte Harder, Christoph Salge, Daniel Polani
        A Bivariate Measure of Redundant Information

    """
    outcomes = list(itertools.product(range(1, 7), repeat=2))

    def pmf_func(i, j):
        return a / 36 + (1 - a) * int(i == j) / 6

    pmf = [pmf_func(i, j) for i, j in outcomes]
    d = dit.Distribution(outcomes, pmf)

    b = int(b)
    d = dit.insert_rvf(d, lambda x: (x[0] + b * x[1],))
    return d

def Wolfs_dice():
    """
    An emperical distribution resulting from rolling two dice---one white and
    one red---20,000 times. For an analysis by Jaynes, see
    http://bayes.wustl.edu/etj/articles/entropy.concentration.pdf
    """
    outcomes = list(itertools.product(range(1, 7), repeat=2))
    pmf = [547, 587, 500, 462, 621, 690,
           609, 655, 497, 535, 651, 684,
           514, 540, 468, 438, 587, 629,
           462, 507, 414, 413, 509, 611,
           551, 562, 499, 506, 658, 672,
           563, 598, 519, 487, 609, 646,
          ]
    pmf = [ p/20000 for p in pmf]
    d = dit.Distribution(outcomes, pmf)
    d.set_rv_names(('R', 'W'))
    return d
