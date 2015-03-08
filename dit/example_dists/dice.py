from __future__ import division

import itertools
import dit

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
    outcomes = list(itertools.product(range(1,7), repeat=2))

    def pmf_func(i, j):
        return a / 36 + (1 - a) * int(i == j) / 6

    pmf = [pmf_func(i, j) for i, j in outcomes]
    d = dit.Distribution(outcomes, pmf)

    b = int(b)
    d = dit.insert_rvf(d, lambda x: (x[0] + b * x[1],) )
    return d
