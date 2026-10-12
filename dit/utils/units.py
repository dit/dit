"""
Utilities for adding units to information values.

Information measures report values in the logarithmic base of the
distribution they are computed from: bits for linear and base-2
distributions, nats for base-e, and so on. The :func:`unitful` decorator
enforces this uniformly: it hands the wrapped measure linear-probability
distributions (so the measure itself only ever computes in bits), then
rescales the result to the base of the first distribution argument and,
when ``ditParams['units']`` is set, attaches the matching unit.
"""

from functools import wraps
from numbers import Number

import numpy as np

from ..params import ditParams

__all__ = ("unitful",)


try:
    import pint

    ureg = pint.UnitRegistry()
    ureg.define(f"nat = {np.log2(np.e)} * bit")
    ureg.define(f"dit = {np.log2(10)} * bit")
    pint.set_application_registry(ureg)
except ImportError:  # pragma: no cover
    pint = None
    ureg = None


def _is_dist(obj):
    return hasattr(obj, "is_log") and hasattr(obj, "get_base")


_NO_DIST = object()


def _linearize(obj):
    """
    Return `obj` with any log-base distributions replaced by linear copies,
    along with the base of the first distribution in `obj` (None for linear,
    `_NO_DIST` if `obj` holds no distribution).
    """
    if _is_dist(obj):
        if obj.is_log():
            return obj.copy(base="linear"), obj.get_base(numerical=True)
        return obj, None
    if isinstance(obj, (list, tuple)) and obj and all(_is_dist(o) for o in obj):
        pairs = [_linearize(o) for o in obj]
        return type(obj)(p[0] for p in pairs), pairs[0][1]
    return obj, _NO_DIST


def _unit(base):
    if base is None or base == 2:
        return ureg.bit
    if base == np.e:
        return ureg.nat
    if base == 10:
        return ureg.dit
    return np.log2(base) * ureg.bit


def _rescale(value, base):
    """Convert a value computed in bits into units of `base`."""
    if pint is not None and isinstance(value, pint.Quantity):
        return value.to(_unit(base))
    if not isinstance(value, (Number, np.ndarray)) or isinstance(value, bool):
        return value
    if base is not None:
        value = value / np.log2(base)
    if ditParams["units"] and ureg is not None:
        value = value * _unit(base)
    return value


def unitful(f):
    """
    Report the wrapped measure's value in the base of its distribution.

    Parameters
    ----------
    f : func
        The function to wrap. It must compute its result in bits from
        linear-probability distributions.

    Returns
    -------
    wrapper : func
        A function whose distribution arguments are linearized and whose
        (first) return value is expressed in the base of the first
        distribution argument, optionally with units attached.
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        base = _NO_DIST
        new_args = []
        for arg in args:
            arg, b = _linearize(arg)
            base = b if base is _NO_DIST else base
            new_args.append(arg)
        new_kwargs = {}
        for key, arg in kwargs.items():
            arg, b = _linearize(arg)
            base = b if base is _NO_DIST else base
            new_kwargs[key] = arg
        if base is _NO_DIST:
            base = None

        value = f(*new_args, **new_kwargs)

        if isinstance(value, tuple) and value:
            return (_rescale(value[0], base),) + value[1:]
        return _rescale(value, base)

    return wrapper
