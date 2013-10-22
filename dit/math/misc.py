"""
miscellaneous math.
"""

from numbers import Integral, Number

def is_number(x):
    """
    Tests if `x` is a number.

    Parameters
    ----------
    x : object
        The object to test the numericalness of.

    Returns
    -------
    b : bool
        True if `x` is a number, False otherwise.
    """
    return isinstance(x, Number)

def is_integer(x):
    """
    Tests if `x` is a integer.

    Parameters
    ----------
    x : object
        The object to test the integerness of.

    Returns
    -------
    b : bool
        True if `x` is an integer, False otherwise.
    """
    return isinstance(x, Integral)

def factorial(n):
    """
    Computes n!

    Parameters
    ----------
    n : int
        A positive integer

    Returns
    -------
    f : int
        n!

    Raises
    ------
    TypeError
        If `n` is not a number.
    ValueError
        If `n` is not non-negative integer.
    """
    if not is_number(n):
        raise TypeError("{0} is not a number.".format(n))
    if not is_integer(n) or n < 0:
        raise ValueError("{0} is not a positive integer.".format(n))
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)

def combinations(n, k):
    """
    Returns the binomial coefficient of `n` choose `k`.

    Parameters
    ----------
    n : int
        The size of the set to draw from.
    k : int
        The number of elements to draw.

    Returns
    -------
    nck : int
        `n` choose `k`

    Raises
    ------
    TypeError
        If `n` and `k` are not numbers.
    ValueError
        If `n` and `k` are not positive integers, and `n` >= `k`.
    """
    nf = factorial(n)
    kf = factorial(k)
    if k > n:
        raise ValueError("{0} is larger than {1}.".format(k, n))
    nmkf = factorial(n-k)
    return nf/(kf*nmkf)
