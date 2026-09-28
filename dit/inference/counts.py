"""
Non-cython methods for getting counts and distributions from data.
"""

import contextlib
from collections import Counter

import numpy as np
from boltons.iterutils import windowed_iter

from ._symbols import is_trials

__all__ = (
    "counts_from_data",
    "distribution_from_data",
    "get_counts",
)


try:  # cython
    from .pycounts import counts_from_data, distribution_from_data

except ImportError:  # no cython
    from collections import defaultdict
    from itertools import product

    def counts_from_data(data, hLength, fLength, marginals=True, alphabet=None, standardize=True):
        """
        Returns conditional counts from `data`.

        To obtain counts for joint distribution only, use fLength=0.

        Parameters
        ----------
        data : NumPy array
            The data used to calculate morphs. Note: `data` cannot be a generator.
            Also, if standardize is True, then data can be any indexable iterable,
            such as a list or tuple.
        hLength : int
            The maxmimum history word length used to calculate morphs.
        fLength : int
            The length of future words that defines the morph.
        marginals : bool
            If True, then the morphs for all histories words from L=0 to L=hLength
            are calculated.  If False, only histories of length L=hLength are
            calculated.
        alphabet : list
            The alphabet to use when creating the morphs. If `None`, then one is
            obtained from `data`. If not `None`, then the provided alphabet
            supplements what appears in the data.  So the data is always scanned
            through in order to get the proper alphabet.
        standardize : bool
            The algorithm requires that the symbols in data be standardized to
            a canonical alphabet consisting of integers from 0 to k-1, where k
            is the alphabet size.  If `data` is already standard, then an extra
            pass through the data can be avoided by setting `standardize` to
            `False`, but note: if `standardize` is False, then data MUST be a
            NumPy array.


        Returns
        -------
        histories : list
            A list of observed histories, corresponding to the rows in `cCounts`.
        cCounts : NumPy array
            A NumPy array representing conditional counts. The rows correspond to
            the observed histories, so this is sparse. The number of rows in this
            array cannot be known in advance, but the number of columns will be
            equal to the alphabet size raised to the `fLength` power.
        hCounts : NumPy array
            A 1D array representing the count of each history word.
        alphabet : tuple
            The ordered tuple representing the alphabet of the data. If `None`,
            the one is created from the data.


        Notes
        -----
        This requires three complete passes through the data. One to obtain
        the full alphabet. Another to standardize the data.  A final pass to
        obtain the counts.

        This is implemented densely.  So during the course of the algorithm,
        we work with a large array containing a row for each possible history.
        Only the rows corresponding to observed histories are returned.

        """
        with contextlib.suppress(TypeError):
            data = list(map(tuple, data))
        counts = Counter(windowed_iter(data, hLength + fLength))
        cond_counts = defaultdict(lambda: defaultdict(int))
        for word, count in counts.items():
            cond_counts[word[:hLength]][word[hLength:]] += count

        histories = sorted(counts.keys())
        alphabet = set(alphabet) if alphabet is not None else set()
        alphabet = tuple(sorted(alphabet.union(*[set(hist) for hist in histories])))

        cCounts = np.empty((len(histories), len(alphabet) ** fLength))
        for i, hist in enumerate(histories):
            for j, future in enumerate(product(alphabet, repeat=fLength)):
                cCounts[i, j] = cond_counts[hist][future]

        hCounts = cCounts.sum(axis=1)

        return histories, cCounts, hCounts, alphabet

    def distribution_from_data(d, L, trim=True, base=None):
        """
        Returns a distribution over words of length `L` from `d`.

        The returned distribution is the naive estimate of the distribution,
        which assigns probabilities equal to the number of times a particular
        word appeared in the data divided by the total number of times a word
        could have appeared in the data.

        Roughly, it corresponds to the stationary distribution of a maximum
        likelihood estimate of the transition matrix of an (L-1)th order Markov
        chain.

        Parameters
        ----------
        d : list
            A list of symbols to be converted into a distribution.
        L : integer
            The length of the words for the distribution.
        trim : bool
            If true, then words with zero probability are trimmed from the
            distribution.
        base : int or string
            The desired base of the returned distribution. If `None`, then the
            value of `dit.ditParams['base']` is used.

        """
        d = np.asarray(d)
        d = d.tolist() if d.ndim == 1 else [tuple(row) for row in d]
        alphabet = tuple(sorted(set(d)))
        return _distribution_from_word_counts(Counter(windowed_iter(d, L)), alphabet, L, trim, base)


def _distribution_from_word_counts(word_counts, alphabet, L, trim, base):
    """
    Build the plug-in distribution over length-`L` words from their counts.

    With ``trim=True`` only observed words are materialized, rather than all
    ``len(alphabet) ** L`` of them.
    """
    from itertools import product

    from dit.distribution import Distribution
    from dit.params import ditParams

    from .. import modify_outcomes
    from ..exceptions import ditException

    if base is None:
        base = ditParams["base"]

    all_words = sorted(word_counts) if trim else list(product(alphabet, repeat=L))
    counts_arr = np.array([word_counts.get(w, 0) for w in all_words], dtype=float)
    total = counts_arr.sum()
    pmf = counts_arr / total if total > 0 else np.zeros_like(counts_arr)

    # Flatten nested-tuple words: ((0,), (1,)) -> (0, 1)
    def _flatten_word(w):
        flat = []
        for elem in w:
            if isinstance(elem, tuple) and len(elem) != 1:
                flat.extend(elem)
            else:
                flat.append(elem[0] if isinstance(elem, tuple) else elem)
        return tuple(flat)

    words = [_flatten_word(w) for w in all_words]

    # Always build in linear space so zero probabilities are stored as 0.
    # set_base will then correctly convert 0 -> -inf for log bases.
    dist = Distribution(words, pmf, trim=trim, base="linear")

    if L == 1:
        with contextlib.suppress(ditException):
            # Only unwrap 1-tuples (e.g. (0,) -> 0), not multi-variable outcomes (e.g. (0,1,0))
            dist = modify_outcomes(dist, lambda o: o[0] if isinstance(o, tuple) and len(o) == 1 else o)

    # Call set_base after modify_outcomes: modify_outcomes creates a new Distribution
    # with base from the original; if we converted to log first, it would use np.zeros
    # for unfilled positions, giving 0 instead of -inf in log space (which becomes p=1).
    dist.set_base(base)

    return dist


_distribution_from_sequence = distribution_from_data


def distribution_from_data(d, L, trim=True, base=None):
    """
    Returns a distribution over words of length `L` from `d`.

    The returned distribution is the naive estimate of the distribution,
    which assigns probabilities equal to the number of times a particular
    word appeared in the data divided by the total number of times a word
    could have appeared in the data.

    Roughly, it corresponds to the stationary distribution of a maximum
    likelihood estimate of the transition matrix of an (L-1)th order Markov
    chain.

    Parameters
    ----------
    d : list or Trials
        A list of symbols to be converted into a distribution, or independent
        :class:`~dit.inference.Trials`, whose word counts are pooled without
        any word spanning two trials.
    L : integer
        The length of the words for the distribution.
    trim : bool
        If true, then words with zero probability are trimmed from the
        distribution.
    base : int or string
        The desired base of the returned distribution. If `None`, then the
        value of `dit.ditParams['base']` is used.

    Notes
    -----
    With ``trim=True`` only the observed words are built, rather than all
    ``len(alphabet) ** L`` of them. A :class:`~dit.Distribution` still indexes
    its outcomes densely by each variable's alphabet, so for large alphabets
    and word lengths use the count-level estimators
    (:func:`~dit.inference.block_entropy`,
    :func:`~dit.inference.conditional_entropy_rate`), which never build a
    distribution.
    """
    if not is_trials(d):
        return _distribution_from_sequence(d, L, trim=trim, base=base)
    trials = []
    for t in d:
        t = np.asarray(t)
        trials.append(t.tolist() if t.ndim == 1 else [tuple(row) for row in t])
    alphabet = tuple(sorted(set().union(*map(set, trials))))
    word_counts = Counter()
    for t in trials:
        word_counts.update(windowed_iter(t, L))
    return _distribution_from_word_counts(word_counts, alphabet, L, trim, base)


def get_counts(data, length):
    """
    Count the occurrences of all words of `length` in `data`.

    Parameters
    ----------
    data : iterable
        The sequence of samples
    length : int
        The length to group samples into.

    Returns
    -------
    counts : np.array
        Array with the count values.
    """
    hists, _, counts, _ = counts_from_data(data, length, 0)
    mask = np.array([len(h) == length for h in hists])
    counts = counts[mask]
    return counts
