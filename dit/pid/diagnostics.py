"""Overlap coordinates, Hi-Fi bounds, and sharp bounds for PIDs."""

from dataclasses import dataclass
from itertools import chain, combinations

import numpy as np

from ..algorithms.pid_references import reference_coinformations
from ..helpers import as_linear
from ..multivariate import coinformation
from .pid import BasePID, _unit_scale, _validate_sources

__all__ = ("HiFi", "HiFiDriver", "PIDOverlap", "TripletReport", "hifi", "pid_overlap", "triplet_report")


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


def _sources_and_target(dist, sources, target):
    """
    The sources and target of a decomposition, with the defaults of :class:`BasePID`.
    """
    if target is None:
        target = dist.rvs[-1]
    if sources is None:
        sources = [var for var in dist.rvs if var[0] not in target]
    sources = tuple(map(tuple, sources))
    target = tuple(target)
    _validate_sources(dist, sources, target)
    return sources, target


def _linear_and_scale(dist):
    """
    The distribution with linear probabilities, and the factor converting bits to its units.
    """
    if dist.is_symbolic():
        raise NotImplementedError("Hi-Fi diagnostics do not support symbolic distributions.")
    scale = _unit_scale(dist)
    if scale <= 0:
        raise ValueError("Information bounds require a logarithmic base greater than one (or linear probabilities).")
    return as_linear(dist), scale


@dataclass(frozen=True)
class HiFiDriver:
    """
    Hi-Fi's decomposition of the dependence of the target on one source.

    Attributes
    ----------
    unique : float
        :math:`U_i = \\min_Z I(X_i : Y | X_Z)`, over every group :math:`Z` of the other
        sources, including none. No nonnegative decomposition gives the source more unique
        information.
    redundancy : float
        :math:`R_i = \\max_Z [I(X_i : X_Z : Y)]_+`, over nonempty :math:`Z`. Every
        nonnegative decomposition has at least this much redundancy between the source and
        the group attaining it.
    synergy : float
        :math:`S_i = \\max_Z [-I(X_i : X_Z : Y)]_+`, over nonempty :math:`Z`. Every
        nonnegative decomposition has at least this much synergy between the source and the
        group attaining it.
    redundancy_partners : tuple of tuples, None
        The group of other sources attaining ``redundancy``, or None if it is zero.
    synergy_partners : tuple of tuples, None
        The group of other sources attaining ``synergy``, or None if it is zero.
    """

    unique: float
    redundancy: float
    synergy: float
    redundancy_partners: tuple | None
    synergy_partners: tuple | None


@dataclass(frozen=True)
class HiFi:
    """
    Hi-Fi's decomposition of each source, returned by :func:`hifi`.

    Attributes
    ----------
    drivers : dict
        Maps each source to its :class:`HiFiDriver`.
    coinformations : dict
        Maps each pair of a source and a nonempty group of the other sources to the
        co-information :math:`I(X_i : X_Z : Y)` of the source, the merged group and the
        target.
    """

    drivers: dict
    coinformations: dict

    def __getitem__(self, source):
        return self.drivers[tuple(source)]


def hifi(dist, sources=None, target=None, atol=1e-9):
    r"""
    The information-theoretic Hi-Fi decomposition of each source, by exhaustive search.

    For a driver :math:`X_i` and a target :math:`Y`, Hi-Fi conditions on every group
    :math:`Z` of the other sources :cite:`ontiveroortega2025assessing,lazic2026information`.
    With :math:`I(X_i : Y | X_Z) = I(X_i : Y) - I(X_i : X_Z : Y)`, its atoms are

    .. math::

        U_i = I(X_i : Y) - R_i, \qquad
        R_i = \max_Z [I(X_i : X_Z : Y)]_+, \qquad
        S_i = \max_Z [-I(X_i : X_Z : Y)]_+.

    Each is the zero-overlap decomposition of the pair formed by :math:`X_i` and the merged
    group :math:`X_Z`. They bound every nonnegative partial information decomposition
    :cite:`vansanten2026between`, Theorem 1:

    * its unique information of :math:`X_i` is at most :math:`U_i`, and one nonnegative
      decomposition attains these bounds for every source at once;
    * its redundancy between :math:`X_i` and the group attaining :math:`R_i` is at least
      :math:`R_i`, and likewise for the synergy and :math:`S_i`.

    Parameters
    ----------
    dist : Distribution
        The distribution.
    sources : iterable of iterables, None
        The sources. If None, every variable but the target.
    target : iterable, None
        The target. If None, the last variable.
    atol : float
        Redundancy or synergy at most this large is reported as having no partners.

    Returns
    -------
    result : HiFi
        The atoms of each source, in the units of `dist`.

    Raises
    ------
    NotImplementedError
        If `dist` is symbolic.

    Notes
    -----
    The two maxima are generally attained with different partners, so :math:`R_i` and
    :math:`S_i` are not parts of one decomposition and should not be added across sources.
    The search is exhaustive, over :math:`2^{n-1} - 1` groups for each source: the greedy
    forward search used with sampled data can miss a whole bit, e.g. :math:`S_0 = 1` for the
    parity of three bits, which no single partner reveals.

    Examples
    --------
    >>> from dit import Distribution
    >>> from dit.pid import hifi
    >>> parity = Distribution(['0000', '0011', '0101', '0110', '1001', '1010', '1100', '1111'], [1 / 8] * 8)
    >>> driver = hifi(parity)[(0,)]
    >>> driver.synergy, driver.synergy_partners
    (1.0, ((1,), (2,)))
    """
    sources, target = _sources_and_target(dist, sources, target)
    linear, scale = _linear_and_scale(dist)
    resolved = list(sources)

    def variables(group):
        return sorted(set(chain.from_iterable(resolved[i] for i in group)))

    informations = {}

    def information(group):
        if group not in informations:
            informations[group] = float(coinformation(linear, [variables(group), list(target)]))
        return informations[group]

    drivers, coinformations = {}, {}
    for i, source in enumerate(sources):
        others = [j for j in range(len(sources)) if j != i]
        best_red, best_syn = (0.0, None), (0.0, None)
        for size in range(1, len(others) + 1):
            for group in combinations(others, size):
                c = information((i,)) + information(group) - information(tuple(sorted((i, *group))))
                partners = tuple(sources[j] for j in group)
                coinformations[(source, partners)] = c * scale
                if c > best_red[0]:
                    best_red = (c, partners)
                if -c > best_syn[0]:
                    best_syn = (-c, partners)
        red, syn = best_red[0], best_syn[0]
        drivers[source] = HiFiDriver(
            unique=(information((i,)) - red) * scale,
            redundancy=red * scale,
            synergy=syn * scale,
            redundancy_partners=best_red[1] if red > atol else None,
            synergy_partners=best_syn[1] if syn > atol else None,
        )
    return HiFi(drivers=drivers, coinformations=coinformations)


@dataclass(frozen=True)
class TripletReport:
    """
    The quantities :cite:`vansanten2026between` recommends reporting for two sources and a
    target, returned by :func:`triplet_report`.

    Each decomposition is a tuple ``(redundancy, (unique_0, unique_1), synergy)``.

    Attributes
    ----------
    mutual_informations : tuple of float
        :math:`I(X_0 : Y)` and :math:`I(X_1 : Y)`.
    conditional_informations : tuple of float
        :math:`I(X_0 : Y | X_1)` and :math:`I(X_1 : Y | X_0)`.
    joint_information : float
        :math:`I(X_0, X_1 : Y)`.
    coinformation : float
        :math:`C`, redundancy minus synergy in every decomposition.
    pairwise : float
        :math:`C^{(2)}`, the co-information of the maximum-entropy distribution with the
        pairwise marginals.
    connected : float
        :math:`I^{(3)} = C^{(2)} - C`, the third-order connected information.
    independent : float
        :math:`I_{Q_{\\mathrm{CI}}}(X_0 : X_1)`, under
        :math:`Q_{\\mathrm{CI}} = p(y) p(x_0|y) p(x_1|y)`.
    delta_max : float
        The length of the segment of nonnegative decompositions.
    hifi : tuple
        The zero-overlap decomposition, :class:`PID_HiFi`.
    pairwise_reference : tuple
        The pairwise-reference decomposition, :class:`PID_dep_c`.
    dependency : tuple
        The decomposition of :class:`PID_dep`.
    """

    mutual_informations: tuple[float, float]
    conditional_informations: tuple[float, float]
    joint_information: float
    coinformation: float
    pairwise: float
    connected: float
    independent: float
    delta_max: float
    hifi: tuple
    pairwise_reference: tuple
    dependency: tuple

    def to_string(self, digits=4):
        """
        A table of the quantities.

        Parameters
        ----------
        digits : int
            The number of decimal places.

        Returns
        -------
        table : str
        """

        def number(value):
            return f"{value:.{digits}f}"

        def atoms(decomposition):
            red, (u_0, u_1), syn = decomposition
            return f"R {number(red)}, U {number(u_0)} / {number(u_1)}, S {number(syn)}"

        rows = [
            ("I(X_0 : Y), I(X_1 : Y)", " / ".join(map(number, self.mutual_informations))),
            ("I(X_0 : Y | X_1), I(X_1 : Y | X_0)", " / ".join(map(number, self.conditional_informations))),
            ("I(X_0, X_1 : Y)", number(self.joint_information)),
            ("co-information C", number(self.coinformation)),
            ("pairwise C2", number(self.pairwise)),
            ("connected I3", number(self.connected)),
            ("I_QCI(X_0 : X_1)", number(self.independent)),
            ("delta_max", number(self.delta_max)),
            ("Hi-Fi", atoms(self.hifi)),
            ("pairwise reference", atoms(self.pairwise_reference)),
            ("I_dep", atoms(self.dependency)),
        ]
        width = max(len(label) for label, _ in rows)
        return "\n".join(f"{label.ljust(width)}  {value}" for label, value in rows)

    def __str__(self):
        return self.to_string()


def triplet_report(dist, sources=None, target=None, maxiter=None):
    r"""
    Report the quantities that fix the nonnegative decompositions of two sources.

    :cite:`vansanten2026between` recommend reporting, for two sources and a target, the
    co-information :math:`C`, its split :math:`C = C^{(2)} - I^{(3)}` into the
    co-information of the pairwise maximum-entropy model and the third-order connected
    information :cite:`Amari2001,Schneidman2003`, and the length :math:`\delta_{\max}` of the
    segment of nonnegative decompositions. These give the Hi-Fi and pairwise-reference
    decompositions and the range of every nonnegative one. :math:`I^{(3)} > 0` can be tested
    on samples with :math:`G^2 = 2 N \ln 2 \, \hat I^{(3)}`.

    Parameters
    ----------
    dist : Distribution
        The distribution.
    sources : iterable of iterables, None
        The two sources; each may be a group of variables. If None, every variable but the
        target, which must be two.
    target : iterable, None
        The target. If None, the last variable.
    maxiter : int, None
        The iteration limit of the maximum entropy fit.

    Returns
    -------
    report : TripletReport
        The quantities, in the units of `dist`.

    Raises
    ------
    ValueError
        If there are not two sources.
    NotImplementedError
        If `dist` is symbolic.
    """
    sources, target = _sources_and_target(dist, sources, target)
    if len(sources) != 2:
        raise ValueError("A triplet report needs exactly two sources.")
    linear, scale = _linear_and_scale(dist)
    references = reference_coinformations(linear, sources, target, maxiter=maxiter)
    i_0, i_1 = references.mutual_informations
    c, c2 = references.coinformation, references.pairwise

    def decomposition(red):
        return (red * scale, ((i_0 - red) * scale, (i_1 - red) * scale), (red - c) * scale)

    return TripletReport(
        mutual_informations=(i_0 * scale, i_1 * scale),
        conditional_informations=((i_0 - c) * scale, (i_1 - c) * scale),
        joint_information=(i_0 + i_1 - c) * scale,
        coinformation=c * scale,
        pairwise=c2 * scale,
        connected=references.connected * scale,
        independent=references.independent * scale,
        delta_max=max(0.0, min(i_0, i_1) - max(0.0, c)) * scale,
        hifi=decomposition(max(0.0, c)),
        pairwise_reference=decomposition(max(0.0, c2)),
        dependency=decomposition(max(c2, references.independent)),
    )
