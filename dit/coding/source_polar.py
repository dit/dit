"""
Source polarization for binary sources (Arikan, 2010).

Where a :class:`~dit.coding.PolarCode` polarizes the synthesized *channels* seen
by successive-cancellation decoding, source polarization applies the same Arikan
transform on the *source* side. For ``N = 2^m`` i.i.d. copies of a binary source
``X`` (optionally with side information ``Y``), the transform ``U^N = X^N G_N``
produces synthesized coordinates whose conditional entropies

.. math::

    H(U_i \\mid U^{i-1}, Y^N)

polarize toward ``0`` (almost deterministic given the past) or ``1`` (almost
uniform given the past) as ``N`` grows. The *high-entropy set* -- the indices with
conditional entropy near one -- is exactly what a lossless polar source code must
store; the remaining low-entropy indices are recovered by sequential decisions.
These finite-block utilities compute the exact profile (no Monte Carlo, no
density evolution) and are therefore limited to small block lengths.

References
----------
Arikan, "Source polarization," ISIT 2010.
"""

import itertools

import numpy as np

from ..exceptions import ditException
from ..shannon import conditional_entropy, entropy
from .base import SourceCoding

__all__ = (
    "PolarSourceCode",
    "polar_source",
    "source_bhattacharyya",
    "source_high_entropy_set",
    "source_polarization_profile",
)


def _joint_table(dist, rv, crvs):
    """
    The joint table ``P[x, *y]`` of a binary variable ``rv`` and side info ``crvs``.

    Returns ``P`` with leading axis the (binary) alphabet of ``rv`` and one
    trailing axis per conditioning variable, plus the alphabet of ``rv``.
    """
    from ..distribution import Distribution

    if not isinstance(dist, Distribution):
        raise ditException("A dit.Distribution is required.")
    crvs = list(crvs) if crvs is not None else []
    groups = [[rv], *[[c] for c in crvs]]
    d = dist.copy().coalesce(groups)
    d.make_dense()
    shape = [len(a) for a in d.alphabet]
    P = np.asarray(d.pmf, dtype=float).reshape(shape)
    x_alphabet = list(d.alphabet[0])
    if len(x_alphabet) != 2:
        raise ditException(f"Source polarization requires a binary variable, got alphabet {x_alphabet}.")
    return P, x_alphabet


def source_bhattacharyya(dist, rv=0, crvs=None):
    """
    The source Bhattacharyya parameter of a binary variable.

    For a binary ``X`` with side information ``Y``,

    .. math::

        Z(X \\mid Y) = 2 \\sum_y \\sqrt{p(0, y)\\,p(1, y)},

    which reduces to ``2 sqrt(p(0) p(1))`` when there is no side information. It
    satisfies ``0 <= Z <= 1``, with ``Z`` near ``1`` when ``X`` is nearly uniform
    given ``Y`` and near ``0`` when ``X`` is nearly determined by ``Y``.

    Parameters
    ----------
    dist : Distribution
        The source distribution.
    rv : int
        The index of the binary source variable. Default is 0.
    crvs : list, None
        Indices of the side-information variables. If None, ``Z(X)`` is computed.

    Returns
    -------
    Z : float
    """
    P, _ = _joint_table(dist, rv, crvs)
    p0 = P[0].ravel()
    p1 = P[1].ravel()
    return float(2 * np.sum(np.sqrt(p0 * p1)))


def _polarized_joint(dist, block_length, rv, crvs):
    """
    Build the joint distribution of ``U^N`` (and side info ``Y^N``) for ``N`` copies.

    ``U^N = X^N G_N`` is formed by enumerating the ``2^N`` (times side-info)
    outcome combinations of ``N`` i.i.d. copies of ``(X, Y)`` and applying the
    Arikan transform to the ``X`` bits. Returns a dense :class:`Distribution` whose
    first ``N`` variables are ``U_0, ..., U_{N-1}`` and whose remaining variables
    are the side information ``Y`` of each copy (in order).
    """
    from ..distribution import Distribution
    from ._util import polar_transform

    N = block_length
    crvs = list(crvs) if crvs is not None else []
    groups = [[rv], *[[c] for c in crvs]]
    pair = dist.copy().coalesce(groups)
    pair.make_dense()
    # The (sorted) binary alphabet of X maps its first symbol to bit 0.
    x_zero = pair.alphabet[0][0]
    # Outcomes of a single (X, Y1, ..., Yk) copy with linear probabilities.
    copies = [(o, float(p)) for o, p in zip(pair.outcomes, pair.pmf, strict=True) if p > 0]

    table = {}
    for combo in itertools.product(copies, repeat=N):
        xs = [0 if o[0][0] == x_zero else 1 for o in combo]
        us = polar_transform(xs)
        ys = tuple(sym for o in combo for sym in o[0][1:])
        prob = 1.0
        for o in combo:
            prob *= o[1]
        outcome = tuple(str(b) for b in us) + ys
        table[outcome] = table.get(outcome, 0.0) + prob

    return Distribution(table)


def source_polarization_profile(dist, block_length, rv=0, crvs=None, metrics=("entropy", "bhattacharyya")):
    """
    The exact source-polarization profile of ``N = block_length`` i.i.d. copies.

    Applies the Arikan transform ``U^N = X^N G_N`` and reports, for each
    synthesized coordinate ``i``, the requested metrics conditioned on the past
    ``U^{i-1}`` and the side information ``Y^N``:

    - ``"entropy"`` -- the conditional entropy ``H(U_i | U^{i-1}, Y^N)``,
    - ``"bhattacharyya"`` -- the conditional source Bhattacharyya parameter
      ``Z(U_i | U^{i-1}, Y^N)``.

    The conditional entropies sum to ``N * H(X | Y)`` (entropy conservation), and
    as ``N`` grows they polarize toward ``0`` or ``1``.

    A third, optional Goela-style diagnostic reuses
    :func:`dit.divergences.maximum_correlation`:

    - ``"max_correlation_with_past"`` -- the maximal correlation
      ``rho_m(U_i ; U^{i-1})`` between each coordinate and its past (``0.0`` at
      ``i = 0``, and ignoring side information). Goela et al. (2014) show these
      also polarize.

    Parameters
    ----------
    dist : Distribution
        The source distribution for a single copy ``(X)`` or ``(X, Y, ...)``.
    block_length : int
        The number of i.i.d. copies ``N``; must be a power of two.
    rv : int
        The index of the binary source variable. Default is 0.
    crvs : list, None
        Indices of the side-information variables. If None, no side information.
    metrics : tuple of str
        Which metrics to compute per coordinate.

    Returns
    -------
    profile : list of dict
        One dict per coordinate ``i`` (in ``[0, N)``), keyed ``"index"`` plus the
        requested metric names.
    """
    N = block_length
    if N < 1 or (N & (N - 1)) != 0:
        raise ditException("The source polarization block length must be a power of two.")
    unknown = set(metrics) - {"entropy", "bhattacharyya", "max_correlation_with_past"}
    if unknown:
        raise ditException(f"Unknown source-polarization metric(s): {sorted(unknown)}.")

    crvs = list(crvs) if crvs is not None else []
    n_side = len(crvs)
    joint = _polarized_joint(dist, N, rv, crvs)
    side_indices = list(range(N, N + n_side * N))

    if "max_correlation_with_past" in metrics:
        from ..divergences import maximum_correlation

        u_marginal = joint.marginal(list(range(N)))

    profile = []
    for i in range(N):
        context = list(range(i)) + side_indices
        row = {"index": i}
        if "entropy" in metrics:
            row["entropy"] = float(conditional_entropy(joint, [i], context)) if context else float(entropy(joint, [i]))
        if "bhattacharyya" in metrics:
            row["bhattacharyya"] = source_bhattacharyya(joint, rv=i, crvs=context or None)
        if "max_correlation_with_past" in metrics:
            row["max_correlation_with_past"] = (
                float(maximum_correlation(u_marginal, [[i], list(range(i))])) if i else 0.0
            )
        profile.append(row)
    return profile


def source_high_entropy_set(dist, block_length, rate=None, size=None, rv=0, crvs=None, rank_by="entropy", tol=1e-9):
    """
    The high-entropy indices selected by a polar source code.

    Ranks the ``N`` synthesized coordinates by their conditional metric (largest
    conditional entropy, or largest source Bhattacharyya, given the past and any
    side information) and returns the top indices -- those a lossless polar source
    code must transmit. At most one of ``rate`` or ``size`` may be given:

    - ``size`` -- keep exactly this many coordinates,
    - ``rate`` -- keep ``round(rate * N)`` coordinates,
    - neither -- keep every coordinate whose conditional entropy exceeds ``tol``,
      i.e. the *lossless* set (the dropped coordinates are deterministic given the
      past and side information, so the code reconstructs the block exactly).

    Parameters
    ----------
    dist : Distribution
        The source distribution.
    block_length : int
        The number of i.i.d. copies ``N``; must be a power of two.
    rate : float, None
        The target fraction of coordinates to keep, in ``[0, 1]``. The set size is
        ``round(rate * N)``.
    size : int, None
        The exact number of coordinates to keep.
    rv : int
        The index of the binary source variable. Default is 0.
    crvs : list, None
        Indices of the side-information variables.
    rank_by : str
        ``"entropy"`` (default) or ``"bhattacharyya"``.
    tol : float
        The conditional-entropy threshold for the lossless default; coordinates at
        or below ``tol`` are treated as deterministic and dropped.

    Returns
    -------
    indices : list of int
        The selected coordinate indices, sorted ascending.
    """
    if rank_by not in ("entropy", "bhattacharyya"):
        raise ditException(f"rank_by must be 'entropy' or 'bhattacharyya', got {rank_by!r}.")
    if rate is not None and size is not None:
        raise ditException("Give at most one of `rate` or `size`.")

    N = block_length
    # The lossless default needs conditional entropies regardless of `rank_by`.
    metrics = (rank_by,) if (rate is not None or size is not None) else ("entropy", rank_by)
    profile = source_polarization_profile(dist, N, rv=rv, crvs=crvs, metrics=metrics)

    if size is None and rate is None:
        return sorted(i for i in range(N) if profile[i]["entropy"] > tol)

    if size is None:
        if not 0 <= rate <= 1:
            raise ditException("rate must be in [0, 1].")
        size = int(round(rate * N))
    if not 0 <= size <= N:
        raise ditException(f"The high-entropy set size must be in [0, {N}], got {size}.")

    order = sorted(range(N), key=lambda i: (-profile[i][rank_by], i))
    return sorted(order[:size])


class PolarSourceCode(SourceCoding):
    """
    An exact finite-block polar source code for a small binary source.

    The code applies the Arikan transform ``U^N = X^N G_N`` to a block of ``N``
    source bits and stores only the coordinates in the *high-entropy set* (those
    that stay nearly uniform given the past and any side information). Decoding
    fills in the remaining low-entropy coordinates by sequential maximum a
    posteriori (MAP) decisions against the exact ``p(U^N, Y^N)`` table and then
    inverts the transform.

    Because the code enumerates the joint distribution exactly (no density
    evolution or list decoding), it is intended for small blocks: the number of
    enumerated states is ``|copy support|^N``, guarded by ``max_states``.

    Parameters
    ----------
    dist : Distribution
        A single-copy source ``(X)`` or source-with-side-information ``(X, Y, ...)``
        distribution. ``X`` (indexed by ``rv``) must be binary.
    block_length : int
        The block length ``N``; must be a power of two.
    rv : int
        The index of the binary source variable. Default is 0.
    crvs : list, None
        Indices of the side-information variables available at the decoder. If
        None, the code is a plain (side-information-free) polar source code.
    rank_by : str
        How to rank coordinates for the high-entropy set: ``"entropy"`` (default)
        or ``"bhattacharyya"``.
    rate : float, None
        A fixed target rate in ``[0, 1]`` (keep ``round(rate * N)`` coordinates).
        Mutually exclusive with ``size``.
    size : int, None
        The exact high-entropy set size. By default the set is chosen *losslessly*
        -- every coordinate whose conditional entropy exceeds ``tol`` is kept, so
        the code reconstructs each block exactly. Fixing ``size`` (or ``rate``)
        below the lossless size yields a fixed-rate code that may not be exact.
    tol : float
        The conditional-entropy threshold used by the lossless default.
    max_states : int
        A guard on the number of enumerated joint outcomes.
    """

    def __init__(
        self, dist, block_length, rv=0, crvs=None, rank_by="entropy", rate=None, size=None, tol=1e-9, max_states=1 << 16
    ):
        super().__init__(dist=dist, radix=2)
        N = block_length
        if N < 1 or (N & (N - 1)) != 0:
            raise ditException("The polar source block length must be a power of two.")

        crvs = list(crvs) if crvs is not None else []
        self.block_length = N
        self.rv = rv
        self.crvs = crvs
        self.n_side = len(crvs)

        _, x_alphabet = _joint_table(dist, rv, crvs)
        self._x_alphabet = x_alphabet

        # Guard against exponential blowup before building the joint table.
        from ._util import polar_transform

        pair = dist.copy().coalesce([[rv], *[[c] for c in crvs]])
        pair.make_dense()
        support = sum(1 for p in pair.pmf if float(p) > 0)
        if support**N > max_states:
            raise ditException(
                f"The polar source code would enumerate {support}**{N} states, exceeding "
                f"max_states={max_states}. Reduce block_length or raise max_states."
            )
        self._polar_transform = polar_transform

        self.high_entropy_set = source_high_entropy_set(
            dist, N, rate=rate, size=size, rv=rv, crvs=crvs, rank_by=rank_by, tol=tol
        )
        self.low_entropy_set = [i for i in range(N) if i not in set(self.high_entropy_set)]

        # Exact joint p(U^N, Y^N), used for sequential MAP decoding.
        joint = _polarized_joint(dist, N, rv, crvs)
        joint.make_dense()
        self._u_probs = self._index_joint(joint)

    def _index_joint(self, joint):
        """A list of ``(u_tuple, y_tuple, prob)`` rows for MAP decoding."""
        N = self.block_length
        rows = []
        for outcome, p in zip(joint.outcomes, joint.pmf, strict=True):
            p = float(p)
            if p <= 0:
                continue
            u = tuple(int(b) for b in outcome[:N])
            y = tuple(outcome[N:])
            rows.append((u, y, p))
        return rows

    @property
    def message_length(self):
        """The number of stored (high-entropy) coordinates per block."""
        return len(self.high_entropy_set)

    def _x_bits(self, x_block):
        """Map a block of source outcomes to ``0/1`` bits."""
        zero = self._x_alphabet[0]
        bits = []
        for value in x_block:
            if value in (0, 1):
                bits.append(int(value))
            else:
                bits.append(0 if value == zero else 1)
        return bits

    def encode(self, x_block):
        """
        Encode a block of ``N`` source bits into the high-entropy coordinates.

        Parameters
        ----------
        x_block : sequence
            A length-``N`` block of source outcomes (``0/1`` or the source's own
            binary alphabet symbols).

        Returns
        -------
        encoded : list of int
            The transform bits at the high-entropy indices.
        """
        if len(x_block) != self.block_length:
            raise ditException(f"Expected a block of length {self.block_length}, got {len(x_block)}.")
        u = self._polar_transform(self._x_bits(x_block))
        return [u[i] for i in self.high_entropy_set]

    def decode(self, encoded, side_information=None):
        """
        Decode a block from its high-entropy coordinates (and any side info).

        The high-entropy coordinates are filled from ``encoded``; the low-entropy
        coordinates are recovered one at a time by an exact MAP decision against
        ``p(U_i | U^{i-1}, Y^N)``. The inverse Arikan transform then returns the
        source bits.

        Parameters
        ----------
        encoded : sequence of int
            The stored high-entropy bits, in ascending index order.
        side_information : sequence, None
            The decoder side information ``Y^N``: one symbol per copy when the code
            was built with ``crvs`` (flattened copy-major when multiple side
            variables). Required iff the code has side information.

        Returns
        -------
        x_block : list of int
            The recovered length-``N`` block of source bits.
        """
        N = self.block_length
        if len(encoded) != self.message_length:
            raise ditException(f"Expected {self.message_length} encoded bits, got {len(encoded)}.")

        expected_side = self.n_side * N
        if expected_side == 0:
            if side_information:
                raise ditException("This polar source code has no side information.")
            y = ()
        else:
            if side_information is None or len(side_information) != expected_side:
                raise ditException(
                    f"Expected {expected_side} side-information symbols, got "
                    f"{0 if side_information is None else len(side_information)}."
                )
            y = tuple(side_information)

        stored = dict(zip(self.high_entropy_set, encoded, strict=True))
        high = set(self.high_entropy_set)

        u = [None] * N
        for i in range(N):
            if i in high:
                u[i] = int(stored[i])
                continue
            weights = [0.0, 0.0]
            prefix = tuple(u[:i])
            for u_row, y_row, p in self._u_probs:
                if y and y_row != y:
                    continue
                if u_row[:i] != prefix:
                    continue
                weights[u_row[i]] += p
            u[i] = 0 if weights[0] >= weights[1] else 1

        return self._polar_transform(u)

    def rate(self):
        """The code rate, ``|high_entropy_set| / block_length``."""
        return self.message_length / self.block_length


def polar_source(
    dist, block_length, rv=0, crvs=None, rank_by="entropy", rate=None, size=None, tol=1e-9, max_states=1 << 16
):
    """
    Build an exact finite-block polar source code.

    Parameters
    ----------
    dist : Distribution
        A single-copy binary source ``(X)`` or ``(X, Y, ...)`` distribution.
    block_length : int
        The block length ``N``; must be a power of two.
    rv : int
        The index of the binary source variable. Default is 0.
    crvs : list, None
        Indices of the side-information variables available at the decoder.
    rank_by : str
        ``"entropy"`` (default) or ``"bhattacharyya"``.
    rate : float, None
        A fixed target rate in ``[0, 1]``. Mutually exclusive with ``size``.
    size : int, None
        The high-entropy set size. Defaults to the lossless set (see
        :class:`PolarSourceCode`).
    tol : float
        The conditional-entropy threshold used by the lossless default.
    max_states : int
        A guard on the number of enumerated joint outcomes.

    Returns
    -------
    code : PolarSourceCode
    """
    return PolarSourceCode(
        dist, block_length, rv=rv, crvs=crvs, rank_by=rank_by, rate=rate, size=size, tol=tol, max_states=max_states
    )
