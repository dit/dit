"""
Construct generalizations of the xor process to have an alphabet of size m, and
the i'th symbol be the sum mod m of the others.
"""

from __future__ import division

from itertools import product

from six.moves import map, range # pylint: disable=redefined-builtin

from ..npdist import Distribution
from ..math.misc import is_integer

def n_mod_m(n, m):
    """
    Constructs a generalized form of the XOR distribution, having an arbitrary
    alphabet and arbitrary word size.

    Parameters
    ----------
    n : int > 0
        The length of the words.
    m : int > 0
        The size of the alphabet.

    Returns
    -------
    d : Distribution
        The generalized XOR distribution of size `n` over alphabet `m`.

    Raises
    ------
    ValueError
        Raised if n or m are not positive integers.
    """
    if not (is_integer(n) and n > 0):
        raise ValueError("{0} is not a positive integer.".format(n))
    if not (is_integer(m) and m > 0):
        raise ValueError("{0} is not a positive integer.".format(m))
    size = m**(n-1)
    alpha = range(m)
    subwords = product(alpha, repeat=n-1)
    outcomes = [''.join(map(str, w)) + str(sum(w)%m) for w in subwords]
    pmf = [1/size]*size
    d = Distribution(outcomes, pmf)
    return d
