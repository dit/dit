"""
Shared helpers for sample-based estimators: symbol standardization, word counts,
and random number generator handling.
"""

import warnings

import numpy as np

__all__ = (
    "Trials",
    "UndersamplingWarning",
    "check_sampling",
    "as_generator",
    "decode",
    "is_trials",
    "standardize",
    "standardize_trials",
    "word_codes",
    "word_counts",
)


class Trials(list):
    """
    Independent realizations (trials) of the same process.

    Wrap a list of sequences in :class:`Trials` to tell the estimators in
    :mod:`dit.inference` that they are separate recordings: counts are pooled
    across trials, but no word ever spans the boundary between two trials.
    Trials may have different lengths.

    Examples
    --------
    >>> from dit.inference import Trials, block_entropy
    >>> block_entropy(Trials([[0, 1], [1, 0]]), 2)  # (1, 1) never straddles the trials
    1.0
    """


class UndersamplingWarning(UserWarning):
    """
    Warned when counts are too sparse for plug-in (and most bias-corrected)
    information estimates to be trusted.
    """


def check_sampling(n_distinct, n_windows, M=None, L=None, stacklevel=3):
    """
    Warn if `n_windows` samples of `n_distinct` distinct words are undersampled.

    Two conditions trigger an :class:`UndersamplingWarning`: more than
    ``n_windows / 5`` distinct words (on average fewer than five observations
    each), or a word space ``M ** L`` larger than ``n_windows``.

    Returns
    -------
    undersampled : bool
    """
    reasons = []
    if n_windows > 0 and n_distinct > n_windows / 5:
        reasons.append(f"{n_distinct} distinct words from {n_windows} windows (fewer than 5 per word)")
    if M is not None and L is not None and n_windows > 0 and L * np.log(max(M, 1)) > np.log(n_windows):
        reasons.append(f"{M}**{L} possible words exceeds {n_windows} windows")
    if reasons:
        warnings.warn(
            "Undersampled: " + "; ".join(reasons) + ". Estimates are biased; consider shorter words, "
            "a bias-corrected estimator, or more data.",
            UndersamplingWarning,
            stacklevel=stacklevel,
        )
    return bool(reasons)


def is_trials(data):
    """
    Whether `data` is a :class:`Trials` collection.
    """
    return isinstance(data, Trials)


def as_generator(prng=None):
    """
    Coerce `prng` into a :class:`numpy.random.Generator`.

    Parameters
    ----------
    prng : None, int, Generator, RandomState
        If None, a generator is seeded from ``dit.math.prng`` so that
        ``dit.math.prng.seed(...)`` makes results reproducible. Integers are
        used as seeds.

    Returns
    -------
    rng : numpy.random.Generator
    """
    if isinstance(prng, np.random.Generator):
        return prng
    if isinstance(prng, np.random.RandomState):
        return np.random.default_rng(prng.randint(2**32, dtype=np.uint64))
    if prng is None:
        from ..math import prng as default_prng

        return np.random.default_rng(default_prng.randint(2**32, dtype=np.uint64))
    return np.random.default_rng(prng)


def standardize(data):
    """
    Map a sequence of hashable symbols to integer codes ``0..M-1``.

    Rows of a 2D array are treated as single (joint) symbols.

    Parameters
    ----------
    data : iterable
        The sequence of symbols.

    Returns
    -------
    codes : np.ndarray
        Integer codes, one per time step.
    alphabet : list
        ``alphabet[c]`` is the symbol with code ``c``.
    """
    arr = np.asarray(data)
    symbols = [tuple(row) for row in arr] if arr.ndim > 1 else arr.tolist()
    alphabet = sorted(set(symbols))
    index = {s: i for i, s in enumerate(alphabet)}
    codes = np.fromiter((index[s] for s in symbols), dtype=np.int64, count=len(symbols))
    return codes, alphabet


def _symbols_of(data):
    arr = np.asarray(data)
    return [tuple(row) for row in arr] if arr.ndim > 1 else arr.tolist()


def standardize_trials(data):
    """
    Standardize one sequence or several :class:`Trials` with a shared alphabet.

    Parameters
    ----------
    data : iterable or Trials
        A single sequence, or independent trials.

    Returns
    -------
    trials : list of np.ndarray
        Integer codes for each trial (a single-element list for one sequence).
    alphabet : list
        ``alphabet[c]`` is the symbol with code ``c``.
    """
    sequences = [_symbols_of(t) for t in data] if is_trials(data) else [_symbols_of(data)]
    alphabet = sorted(set().union(*map(set, sequences)))
    index = {s: i for i, s in enumerate(alphabet)}
    trials = [np.fromiter((index[s] for s in seq), dtype=np.int64, count=len(seq)) for seq in sequences]
    return trials, alphabet


def decode(codes, alphabet, like):
    """
    Invert :func:`standardize`, returning an array shaped like `like`.
    """
    like = np.asarray(like)
    table = np.asarray(alphabet, dtype=like.dtype)
    return table[np.asarray(codes)]


def word_codes(codes, L, M):
    """
    Integer identifiers of the overlapping length-`L` words of `codes`.

    Parameters
    ----------
    codes : np.ndarray or list of np.ndarray
        Standardized symbols, or one array per trial. Words never span two trials.
    L : int
        The word length.
    M : int
        The alphabet size.

    Returns
    -------
    ids : np.ndarray
        One identifier per window (``len(codes) - L + 1`` per trial, concatenated
        in trial order). Identifiers are base-`M` integers when they fit in 64
        bits, and arbitrary but consistent integers otherwise.
    """
    if isinstance(codes, list):
        parts = [np.asarray(c, dtype=np.int64) for c in codes]
    else:
        parts = [np.asarray(codes, dtype=np.int64)]
    parts = [c for c in parts if len(c) - L + 1 > 0]
    if not parts:
        return np.zeros(0, dtype=np.int64)
    if L == 0:
        return np.zeros(sum(len(c) + 1 for c in parts), dtype=np.int64)
    if L * np.log2(max(M, 2)) < 62:
        pieces = []
        for c in parts:
            n = len(c) - L + 1
            ids = np.zeros(n, dtype=np.int64)
            for k in range(L):
                ids = ids * M + c[k : k + n]
            pieces.append(ids)
        return np.concatenate(pieces)
    windows = np.concatenate([np.lib.stride_tricks.sliding_window_view(c, L) for c in parts])
    _, ids = np.unique(windows, axis=0, return_inverse=True)
    return ids.ravel().astype(np.int64)


def word_counts(codes, L, M):
    """
    Counts of the distinct overlapping length-`L` words of `codes` (one array or
    a list of per-trial arrays).
    """
    ids = word_codes(codes, L, M)
    if len(ids) == 0:
        return np.zeros(0, dtype=float)
    _, counts = np.unique(ids, return_counts=True)
    return counts.astype(float)
