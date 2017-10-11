
try: # cython

    from . import counts_from_data, distribution_from_data

except ImportError: # no cython

    from boltons.iterutils import windowed_iter
    from collections import Counter, defaultdict
    from itertools import product

    import numpy as np

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
        counts = Counter(windowed_iter(data, hLength+fLength))
        cond_counts = defaultdict(lambda: defaultdict(int))
        for word, count in counts.items():
            cond_counts[word[:hLength]][word[hLength:]] += count

        histories = sorted(counts.keys())
        alphabet = set(alphabet) if alphabet is not None else set()
        alphabet = tuple(sorted(alphabet.union(*[set(hist) for hist in histories])))

        cCounts = np.empty((len(histories), len(alphabet)**fLength))
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
        from dit import ditParams, Distribution

        if base is None:
            base = ditParams['base']

        words, _, counts, _ = counts_from_data(d, L, 0)

        # We turn the counts to probabilities
        pmf = counts/counts.sum()

        dist = Distribution(words, pmf, trim=trim)

        if base is not None:
            dist.set_base(base)

        return dist


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
    counts = counts_from_data(data, length, 0)[2]
    return counts
