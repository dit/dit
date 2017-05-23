"""
Giant bit type distributions.
"""

from ..distconst import uniform

def giant_bit(n, k):
    """
    Return a 'giant bit' distribution of size `n` and alphabet size `k`.

    Parameters
    ----------
    n : int
        The number of identical bits.
    k : int
        The number of states for each bit.

    Returns
    -------
    gb : Distribution
        The giant bit distribution.
    """
    return uniform([str(i)*n for i in range(k)])