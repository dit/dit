# -*- coding: utf-8 -*-
"""
Combinatorial functions.

"""

def unitsum_tuples(n, k, mn, mx):
    """Generates unitsum k-tuples with elements from mn to mx.

    This function is more general than slots(n,k,normalized=True), as it can
    return unitsum vectors with elements outside of (0,1).

    In order to generate unitsum samples, the following must be satisfied:
        1 = mx + (k-1) * mn

    Parameters
    ----------
    n : int
        The number of increments to include between mn and mx. n >= 1. The
        meaning of n is similar to the n in slots(n,k) and represents the
        number of ``items'' to place in each slot.
    k : int
        The length of the tuples (equivalently, the number of slots).
    mn : float
        The minimum value in the unitsum samples.
    mx : float
        The maximum value in the unitsum samples.

    Examples
    --------
    >>> s = unitsum_tuples(3, 2, .2, .8)
    >>> s.next()
    (0.20000000000000001, 0.80000000000000004)
    >>> s.next()
    (0.40000000000000008, 0.60000000000000009)
    >>> s.next()
    (0.60000000000000009, 0.40000000000000002)
    >>> s.next()
    (0.80000000000000004, 0.19999999999999996)

    """
    # In order to add up to 1 properly...we must have:
    #   sum((mx, mn/(k-1), ... , mn/(k-1))) == 1
    s = mx + (k-1) * mn
    tol = 1e-9
    if not (abs(s - 1) <= tol):
        msg = "Specified min and max will not create unitsum tuples."
        e = Exception(msg)
        raise(e)


    # Now we convert from "number of increments/items" to "number of points"
    # The number of points behaviors similar to numpy.linspace(mn,mx,n)
    n += 1

    if mn < 0:
        shift = float(abs(mn))
    else:
        shift = -float(mn)
    seq, i = [mx + shift] * k + [0], k
    while i:
        t = tuple((seq[i] - seq[i + 1] - shift) for i in range(k))
        # This should be a unitsum tuple.
        s = float(sum(t))
        assert( s > .001 )
        yield tuple(t)

        for idx, val in enumerate(seq):
            if abs(val) < 1e-9:
                i = idx - 1
                break
        seq[i:k] = [seq[i] - (mx - mn) / float(n - 1)] * (k - i)


# Thanks to Arnaud Delobelle
def slots(n, k, normalized=False):
    """Generates distributions of n identical items into k distinct slots.

    A generator over distributions of n indistinguishable items into k
    distinguishable slots, where each slot can hold up to n items.
    Selection of items is done without replacement, and the order within the
    slots cannot matter since the items are indistinguishable.

    The number of distributions is (n + k - 1)! / n! / (k-1)!

    Parameters
    ----------
    n : int
        The number of indistinguishable items.
    k : int
        The number of distinguishable slots.
    normalized : bool
        If True, then we divide each term in the tuple by the number of items.
        The default value is False.

    Yields
    ------
    t : tuple
        A tuple of length k where each element is an integer representing
        the number of indistinguishable items within the slot.

    Examples
    --------
    >>> list(slots(3,2))
    [(0, 3), (1, 2), (2, 1), (3, 0)]

    """
    seq, i = [n] * k + [0], k
    if normalized:
        nf = float(n)
        while i:
            yield tuple((seq[i] - seq[i + 1]) / nf for i in range(k))
            i = seq.index(0) - 1
            seq[i:k] = [seq[i] - 1] * (k-i)
    else:
        while i:
            yield tuple((seq[i] - seq[i + 1]) for i in range(k))
            i = seq.index(0) - 1
            seq[i:k] = [seq[i] - 1] * (k - i)


