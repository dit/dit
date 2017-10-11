
try: # cython

    from . import counts_from_data, distribution_from_data

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

        Notes
        -----
        This function utilizes the cython-implemented `counts_from_data`.
        """
        counts = counts_from_data(data, 0, length)[1][0]
        return counts

except ImportError: # no cython

    from collections import Counter
    from boltons.iterutils import windowed_iter

    import numpy as np

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

        Notes
        -----
        This function uses `collections.Counter` and `boltons.iterutils.windowed_iter`.
        """
        counts = Counter(windowed_iter(data, length))
        counts = np.array(list(counts.values()))
        return counts


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

        counts = get_counts(d, L)

        # We turn the counts to probabilities
        total = sum(counts)
        dist = {e: count/total for e, count in counts.items()}

        dist = Distribution(dist, trim=trim)

        if base is not None:
            dist.set_base(base)

        return dist
