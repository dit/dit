"""
Universal codes for the positive integers.

These prefix-free codes encode the positive integers without reference to a
source distribution, yet remain within a constant factor of optimal across a wide
range of distributions. Provided are the unary code, the Elias gamma / delta /
omega codes (Elias, 1975), and the Fibonacci code.

Each function maps a positive integer to its codeword string. :func:`universal_code`
wraps a chosen family into a :class:`SymbolCode` over an integer-valued source.
"""

from ..exceptions import ditException
from ._util import linear_outcomes_probs
from .symbol_code import SymbolCode

__all__ = (
    "elias_delta",
    "elias_gamma",
    "elias_omega",
    "fibonacci",
    "unary",
    "universal_code",
)


def _check_positive(n):
    if not isinstance(n, int) or n < 1:
        raise ditException(f"Universal codes encode positive integers; got {n!r}.")


def unary(n):
    """
    The unary codeword for ``n >= 1``: ``n - 1`` ones followed by a zero.
    """
    _check_positive(n)
    return "1" * (n - 1) + "0"


def elias_gamma(n):
    """
    The Elias gamma codeword for ``n >= 1``.

    The codeword is ``floor(log2(n))`` zeros followed by the binary
    representation of ``n``.
    """
    _check_positive(n)
    binary = format(n, "b")
    return "0" * (len(binary) - 1) + binary


def elias_delta(n):
    """
    The Elias delta codeword for ``n >= 1``.

    The codeword is the gamma code of the bit-length of ``n`` followed by the
    bits of ``n`` below its leading one.
    """
    _check_positive(n)
    binary = format(n, "b")
    return elias_gamma(len(binary)) + binary[1:]


def elias_omega(n):
    """
    The Elias omega (recursive) codeword for ``n >= 1``.
    """
    _check_positive(n)
    code = "0"
    k = n
    while k > 1:
        binary = format(k, "b")
        code = binary + code
        k = len(binary) - 1
    return code


def fibonacci(n):
    """
    The Fibonacci codeword for ``n >= 1`` (Zeckendorf representation + a ``1``).
    """
    _check_positive(n)
    fibs = [1, 2]
    while fibs[-1] <= n:
        fibs.append(fibs[-1] + fibs[-2])
    fibs = fibs[:-1]
    used = []
    remainder = n
    for f in reversed(fibs):
        if f <= remainder:
            used.append(True)
            remainder -= f
        else:
            used.append(False)
    used.reverse()
    return "".join("1" if u else "0" for u in used) + "1"


_FAMILIES = {
    "unary": unary,
    "gamma": elias_gamma,
    "delta": elias_delta,
    "omega": elias_omega,
    "fibonacci": fibonacci,
}


def universal_code(dist, kind="gamma"):
    """
    Build a :class:`SymbolCode` over integer outcomes using a universal code.

    Parameters
    ----------
    dist : Distribution
        The source distribution; its outcomes must be positive integers.
    kind : str
        One of ``'unary'``, ``'gamma'``, ``'delta'``, ``'omega'``, or
        ``'fibonacci'``.

    Returns
    -------
    code : SymbolCode
    """
    try:
        encoder = _FAMILIES[kind]
    except KeyError:
        raise ditException(f"Unknown universal code {kind!r}; choose from {sorted(_FAMILIES)}.") from None
    outcomes, _ = linear_outcomes_probs(dist)
    codebook = {o: encoder(o) for o in outcomes}
    return SymbolCode(codebook, dist=dist, radix=2)
