"""
Shared helpers for the coding module.
"""

from ..exceptions import ditException

# Digit alphabet for radix-ary codewords; supports radices up to 36.
DIGITS = "0123456789abcdefghijklmnopqrstuvwxyz"


def check_radix(radix):
    """
    Validate a radix and return it.

    Parameters
    ----------
    radix : int
        The candidate code-alphabet size.

    Returns
    -------
    radix : int
    """
    if not isinstance(radix, int) or radix < 2 or radix > len(DIGITS):
        raise ditException(f"radix must be an integer in [2, {len(DIGITS)}], got {radix!r}.")
    return radix


def linear_outcomes_probs(dist):
    """
    Return ``(outcomes, probs)`` for `dist` in linear probability space.

    Zero-probability outcomes are dropped.

    Parameters
    ----------
    dist : Distribution

    Returns
    -------
    outcomes : list
    probs : list of float
    """
    d = dist.copy(base="linear") if dist.is_log() else dist
    pairs = [(o, float(p)) for o, p in zip(d.outcomes, d.pmf, strict=True) if p > 0]
    outcomes = [o for o, _ in pairs]
    probs = [p for _, p in pairs]
    return outcomes, probs


def radix_expansion(frac, length, radix):
    """
    The first `length` digits of the base-`radix` expansion of ``frac``.

    Parameters
    ----------
    frac : float
        A number in ``[0, 1)``.
    length : int
        The number of digits to emit.
    radix : int
        The base of the expansion.

    Returns
    -------
    codeword : str
    """
    digits = []
    for _ in range(length):
        frac *= radix
        d = int(frac)
        if d >= radix:  # guard against floating-point overshoot
            d = radix - 1
        digits.append(DIGITS[d])
        frac -= d
    return "".join(digits)
