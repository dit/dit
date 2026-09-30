"""
Surrogate time series for hypothesis testing.

:func:`whittle_surrogates` draws sequences uniformly from the set of sequences
sharing the observed ``(order + 1)``-gram counts and initial ``order``-word,
which is the exact null distribution of an ``order``-th order Markov chain
conditioned on its sufficient statistic :cite:`Pethel2014,Whittle1955`.
:func:`shift_surrogates` and :func:`block_surrogates` are cheaper nulls that
preserve more (or different) structure of a series.
"""

from collections import defaultdict

import numpy as np
from scipy.special import gammaln

from ._symbols import Trials, as_generator, decode, is_trials, standardize, standardize_trials, word_codes

__all__ = (
    "block_surrogates",
    "shift_surrogates",
    "stationary_bootstrap",
    "whittle_count",
    "whittle_surrogates",
)


def _word_graph(codes, order, M):
    """
    The multigraph whose vertices are `order`-words and whose edges are the
    ``(order + 1)``-grams of `codes`, in time order.
    """
    words = word_codes(codes, order, M)
    successors = defaultdict(list)
    for a, b, sym in zip(words[:-1], words[1:], codes[order:], strict=False):
        successors[int(a)].append((int(b), int(sym)))
    return int(words[0]), int(words[-1]), successors


def whittle_count(data, order=1):
    """
    The natural log of the number of sequences sharing the observed
    ``(order + 1)``-gram counts, first ``order``-word, and last ``order``-word.

    Parameters
    ----------
    data : iterable or Trials
        The observed sequence. Rows of a 2D array are joint symbols. For
        :class:`~dit.inference.Trials`, the log of the product of the per-trial
        counts.
    order : int
        The Markov order of the null hypothesis.

    Returns
    -------
    log_count : float
        The natural log of the size of the surrogate set.

    Notes
    -----
    This is Whittle's formula :cite:`Whittle1955,Billingsley1961`, as used by
    :cite:`Pethel2014`:

    .. math::

        N_{uv}(F) = \\frac{\\prod_i F_{i\\cdot}!}{\\prod_{ij} F_{ij}!} C_{vu}

    where :math:`F` counts transitions between observed ``order``-words and
    :math:`C_{vu}` is the :math:`(v, u)` cofactor of
    :math:`\\delta_{ij} - F_{ij} / F_{i\\cdot}`. For ``order=0`` it reduces to
    the multinomial coefficient of the symbol counts.
    """
    if is_trials(data):
        trials, alphabet = standardize_trials(data)
        return float(sum(_whittle_count_codes(c, order, len(alphabet)) for c in trials))
    codes, alphabet = standardize(data)
    return _whittle_count_codes(codes, order, len(alphabet))


def _whittle_count_codes(codes, order, M):
    if len(codes) <= order:
        return 0.0
    if order == 0:
        return float(gammaln(len(codes) + 1) - gammaln(np.bincount(codes) + 1).sum())
    start, end, successors = _word_graph(codes, order, M)
    vertices = sorted({start, end} | set(successors) | {b for succ in successors.values() for b, _ in succ})
    index = {v: i for i, v in enumerate(vertices)}
    F = np.zeros((len(vertices), len(vertices)))
    for a, succ in successors.items():
        for b, _ in succ:
            F[index[a], index[b]] += 1
    rows = F.sum(axis=1)
    Fstar = np.eye(len(vertices))
    nonzero = rows > 0
    Fstar[nonzero] -= F[nonzero] / rows[nonzero, None]
    u, v = index[start], index[end]
    minor = np.delete(np.delete(Fstar, v, axis=0), u, axis=1)
    sign, logdet = np.linalg.slogdet(minor) if minor.size else (1.0, 0.0)
    cofactor_sign = sign * (-1) ** (u + v)
    if cofactor_sign <= 0:
        return -np.inf
    return float(gammaln(rows + 1).sum() - gammaln(F + 1).sum() + logdet)


def _random_trail(start, end, successors, rng):
    """
    A uniformly random Eulerian trail from `start` to `end` in the multigraph
    `successors`, via a random last-exit arborescence :cite:`Kandel1996`.

    Returns the emitted symbols (the edge labels) in order.
    """
    # Wilson's algorithm: a random arborescence oriented toward `end`, with
    # probability proportional to the product of the edge multiplicities.
    last_exit = {}
    in_tree = {end}
    for v in successors:
        u = v
        while u not in in_tree:
            succ = successors[u]
            last_exit[u] = int(rng.integers(len(succ)))
            u = succ[last_exit[u]][0]
        u = v
        while u not in in_tree:
            in_tree.add(u)
            u = successors[u][last_exit[u]][0]

    order_out = {}
    for v, succ in successors.items():
        edges = list(succ)
        if v in last_exit:
            final = edges.pop(last_exit[v])
            rng.shuffle(edges)
            edges.append(final)
        else:
            rng.shuffle(edges)
        order_out[v] = edges

    pointer = dict.fromkeys(successors, 0)
    symbols = []
    v = start
    total = sum(len(succ) for succ in successors.values())
    for _ in range(total):
        b, sym = order_out[v][pointer[v]]
        pointer[v] += 1
        symbols.append(sym)
        v = b
    return symbols


def whittle_surrogates(data, order=1, n=1, prng=None):
    """
    Draw surrogates that exactly preserve the ``(order + 1)``-gram counts.

    Each surrogate is drawn uniformly from the set of sequences with the same
    first ``order`` symbols and the same count of every ``(order + 1)``-gram as
    `data` (and hence the same last ``order``-word). Under the null hypothesis
    that `data` is an ``order``-th order Markov chain, the observed sequence is
    itself a uniform draw from this set, so statistics computed on the
    surrogates give an exact null distribution at any sample size
    :cite:`Pethel2014`.

    Parameters
    ----------
    data : iterable or Trials
        The observed sequence. Rows of a 2D array are joint symbols. Each of
        several :class:`~dit.inference.Trials` is drawn independently from its
        own surrogate set.
    order : int
        The Markov order to preserve. ``order=0`` is a random permutation.
    n : int
        The number of surrogates.
    prng : None, int, Generator, RandomState
        Source of randomness.

    Returns
    -------
    surrogates : np.ndarray or list of Trials
        Shape ``(n,) + np.shape(data)``, or a list of `n` :class:`Trials`.

    Notes
    -----
    :cite:`Pethel2014` sample by weighting each step with Whittle's formula
    (:func:`whittle_count`), which costs a determinant per symbol. This
    implementation samples the same uniform distribution in linear time by
    drawing a random Eulerian trail of the ``order``-word de Bruijn multigraph
    from a random last-exit arborescence :cite:`Kandel1996`.
    """
    rng = as_generator(prng)
    if order < 0:
        raise ValueError("`order` must be non-negative.")
    if is_trials(data):
        trials, alphabet = standardize_trials(data)
        draws = _whittle_trials(trials, order, n, rng)
        return [Trials(decode(c, alphabet, t) for c, t in zip(draw, data, strict=True)) for draw in draws]
    codes, alphabet = standardize(data)
    return decode(_whittle_codes(codes, order, n, rng), alphabet, data)


def _whittle_codes(codes, order, n, rng):
    """
    `n` Whittle surrogates of one standardized sequence, as an ``(n, N)`` array.
    """
    out = np.empty((n, len(codes)), dtype=np.int64)
    if len(codes) <= order + 1:
        out[:] = codes
        return out
    start, end, successors = _word_graph(codes, order, int(codes.max()) + 1)
    for i in range(n):
        out[i, :order] = codes[:order]
        out[i, order:] = _random_trail(start, end, successors, rng)
    return out


def _whittle_trials(trials, order, n, rng):
    """
    `n` surrogates of standardized trials; each trial is drawn independently.

    Returns a list of `n` lists of code arrays.
    """
    per_trial = [_whittle_codes(c, order, n, rng) for c in trials]
    return [[draws[i] for draws in per_trial] for i in range(n)]


def shift_surrogates(data, n=1, prng=None):
    """
    Circularly shift `data` by uniformly random nonzero offsets.

    Shifting a source series relative to a target preserves each series'
    own dynamics while destroying their alignment, which gives a null for
    directed measures such as transfer entropy.

    Parameters
    ----------
    data : iterable or Trials
        The sequence to shift (along the first axis); each of several
        :class:`~dit.inference.Trials` is shifted independently.
    n : int
        The number of surrogates.
    prng : None, int, Generator, RandomState
        Source of randomness.

    Returns
    -------
    surrogates : np.ndarray or list of Trials
        Shape ``(n,) + np.shape(data)``, or a list of `n` :class:`Trials`.
    """
    rng = as_generator(prng)
    if is_trials(data):
        per_trial = [shift_surrogates(t, n, rng) for t in data]
        return [Trials(draws[i] for draws in per_trial) for i in range(n)]
    data = np.asarray(data)
    N = len(data)
    shifts = rng.integers(1, N, size=n) if N > 1 else np.zeros(n, dtype=int)
    return np.stack([np.roll(data, s, axis=0) for s in shifts])


def block_surrogates(data, block_length, n=1, prng=None):
    """
    Randomly permute contiguous blocks of `data`.

    Dependencies shorter than `block_length` are (mostly) kept, while longer
    ones are destroyed.

    Parameters
    ----------
    data : iterable or Trials
        The sequence to shuffle (along the first axis); blocks never move
        between :class:`~dit.inference.Trials`.
    block_length : int
        The length of each block; the final block may be shorter.
    n : int
        The number of surrogates.
    prng : None, int, Generator, RandomState
        Source of randomness.

    Returns
    -------
    surrogates : np.ndarray or list of Trials
        Shape ``(n,) + np.shape(data)``, or a list of `n` :class:`Trials`.
    """
    if block_length < 1:
        raise ValueError("`block_length` must be positive.")
    rng = as_generator(prng)
    if is_trials(data):
        per_trial = [block_surrogates(t, block_length, n, rng) for t in data]
        return [Trials(draws[i] for draws in per_trial) for i in range(n)]
    data = np.asarray(data)
    blocks = [data[i : i + block_length] for i in range(0, len(data), block_length)]
    return np.stack([np.concatenate([blocks[j] for j in rng.permutation(len(blocks))]) for _ in range(n)])


def stationary_bootstrap(data, n=1, mean_block_length=None, prng=None):
    """
    Resample a series with the stationary bootstrap :cite:`Politis1994`.

    Each resample concatenates blocks that start at uniformly random positions
    and have geometrically distributed lengths, wrapping around the end of the
    series. Rows of multivariate `data` are resampled together. For
    :class:`~dit.inference.Trials`, whole trials are resampled with replacement
    instead (a cluster bootstrap), so no resample joins two trials.

    Parameters
    ----------
    data : array_like or Trials
        The series (along the first axis), or independent trials.
    n : int
        The number of resamples.
    mean_block_length : float, None
        The mean block length; defaults to ``N ** (1 / 3)``.
    prng : None, int, Generator, RandomState
        Source of randomness.

    Returns
    -------
    resamples : np.ndarray or list of Trials
        Shape ``(n,) + np.shape(data)``, or a list of `n` :class:`Trials`.
    """
    rng = as_generator(prng)
    if is_trials(data):
        picks = rng.integers(0, len(data), size=(n, len(data)))
        return [Trials(data[j] for j in row) for row in picks]
    data = np.asarray(data)
    N = len(data)
    if mean_block_length is None:
        mean_block_length = N ** (1 / 3)
    new_block = rng.random((n, N)) < 1 / mean_block_length
    new_block[:, 0] = True
    starts = rng.integers(0, N, size=(n, N))
    positions = np.arange(N)
    block_start = np.maximum.accumulate(np.where(new_block, positions, 0), axis=1)
    index = (np.take_along_axis(starts, block_start, axis=1) + positions - block_start) % N
    return data[index]
