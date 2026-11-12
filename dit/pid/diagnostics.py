"""Overlap coordinates and sharp bounds for two-source PIDs."""

from dataclasses import dataclass

import numpy as np

from ..multivariate import coinformation
from .pid import BasePID

__all__ = ("PIDOverlap", "pid_overlap")


@dataclass(frozen=True)
class PIDOverlap:
    """
    Overlap and nonnegative-PID bounds returned by :func:`pid_overlap`.

    The segment and its overlap coordinate are Proposition 1 of
    :cite:`vansanten2026between`, following :cite:`goodwell2017temporal`.
    Information values are plain numbers in the same logarithmic units as
    the PID, even when ``ditParams['units']`` is enabled.

    Attributes
    ----------
    coinformation : float
        The fixed net balance, redundancy minus synergy.
    delta : float
        The overlap, the smaller of the PID's redundancy and synergy. Signed
        PIDs can have negative overlap or exceed ``delta_max``; it is not clipped.
    delta_max : float
        The length of the segment of nonnegative PIDs.
    position : float
        ``delta / delta_max``, or ``nan`` when the segment is collapsed within
        the requested tolerance. Dimensionless and not restricted to [0, 1].
    redundancy_bounds : tuple of float
        The sharp (lower, upper) bounds on redundancy among nonnegative PIDs.
    synergy_bounds : tuple of float
        The corresponding bounds on synergy.
    unique_bounds : tuple of tuples of float
        The corresponding bounds on each unique atom, in the supplied source order.
    """

    coinformation: float
    delta: float
    delta_max: float
    position: float
    redundancy_bounds: tuple[float, float]
    synergy_bounds: tuple[float, float]
    unique_bounds: tuple[tuple[float, float], tuple[float, float]]


def pid_overlap(pid, atol=1e-9):
    r"""
    Report the overlap of a consistent, numeric two-source PID and its sharp bounds.

    Write :math:`I_i=I(X_i;Y)` and
    :math:`C=I_0+I_1-I(X_0,X_1;Y)`. Every two-source PID has

    .. math::

        R=[C]_+ + \delta, \qquad S=[-C]_+ + \delta, \qquad
        U_i=I_i-[C]_+-\delta.

    All four atoms are nonnegative exactly when
    :math:`0\leq\delta\leq\delta_{\max}=\min(I_0,I_1)-[C]_+`.
    This helper returns the overlap :math:`\delta=\min(R,S)`, its normalized
    position, and the bounds attained at the Hi-Fi and MMI endpoints
    :cite:`vansanten2026between`, Proposition 1 and Corollary 1. The endpoint
    bounds were given by :cite:`goodwell2017temporal`.

    Parameters
    ----------
    pid : BasePID
        A two-source decomposition with a target, including signed measures.
        A lazy decomposition is computed first. Conditional/contextual PIDs and
        entropy decompositions are not supported.
    atol : float
        Nonnegative finite absolute tolerance, in the PID's information units.
        The normalized position is undefined (``nan``) if ``delta_max <= atol``;
        neither the overlap nor the bounds are clipped to this tolerance.

    Returns
    -------
    result : PIDOverlap
        Numeric information values in the PID's logarithmic units and a
        dimensionless normalized position. No maximum-entropy fit is required
        beyond any computation of the supplied PID itself.

    Raises
    ------
    TypeError
        If `pid` is not a PID with a target.
    ValueError
        If there are not exactly two sources, the PID is inconsistent or has
        nonfinite atoms, the log base is below one, or `atol` is invalid.
    NotImplementedError
        If the distribution is symbolic.

    Notes
    -----
    The bounds describe nonnegative PIDs, not requirements on signed measures.
    A signed measure's overlap and position are returned unchanged even outside
    the segment. Neither endpoint is a universally preferred decomposition.

    Examples
    --------
    >>> from dit.pid import PID_MMI, bivariates, pid_overlap
    >>> result = pid_overlap(PID_MMI(bivariates['and']))
    >>> round(result.position, 6)
    1.0
    >>> round(result.synergy_bounds[1], 6)
    0.5
    """
    if not isinstance(pid, BasePID) or pid._target is None:
        raise TypeError("pid_overlap requires a PID with a target.")
    if len(pid._sources) != 2:
        raise ValueError("pid_overlap requires exactly two sources.")
    if pid._dist.is_symbolic():
        raise NotImplementedError("Overlap diagnostics do not support symbolic distributions.")
    if pid._unit_scale <= 0:
        raise ValueError("Overlap bounds require a logarithmic base greater than one (or linear probabilities).")
    if not np.isfinite(atol) or atol < 0:
        raise ValueError("atol must be finite and nonnegative.")

    pid._ensure_computed()
    red = float(pid.get_pi(pid._lattice.bottom))
    syn = float(pid.get_pi(pid._lattice.top))
    uniques = [float(pid.get_pi((source,))) for source in pid._sources]
    if not np.isfinite([red, syn, *uniques]).all():
        raise ValueError("Overlap diagnostics require finite values for all four PID atoms.")
    if not pid.consistent:
        raise ValueError("Overlap diagnostics require a consistent PID.")

    mis = [float(pid._scaled(float(coinformation(pid._dist, [source, pid._target])))) for source in pid._sources]
    c = sum(mis) - float(pid._total)
    r_min = max(0.0, c)
    s_min = max(0.0, -c)
    delta_max = max(0.0, min(mis) - r_min)
    delta = min(red, syn)
    unique_bounds = [(max(0.0, mi - r_min - delta_max), max(0.0, mi - r_min)) for mi in mis]

    return PIDOverlap(
        coinformation=c,
        delta=delta,
        delta_max=delta_max,
        position=delta / delta_max if delta_max > atol else float("nan"),
        redundancy_bounds=(r_min, r_min + delta_max),
        synergy_bounds=(s_min, s_min + delta_max),
        unique_bounds=(unique_bounds[0], unique_bounds[1]),
    )
