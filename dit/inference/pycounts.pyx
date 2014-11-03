# encoding: utf-8
# cython: profile=False
# cython: embedsignature=True

from __future__ import division

import sys

import cython
cimport cython

from cpython cimport PyTuple_New, PyTuple_SET_ITEM, PyDict_New, PyDict_SetItem

import numpy as np
cimport numpy as np
from numpy cimport PyArray_DATA, import_array

import collections
import itertools
from collections import defaultdict
from itertools import product
from bisect import bisect_left

from segmentaxis import segment_axis

# Globals
BTYPE = np.bool
ctypedef bint BTYPE_t
ITYPE = np.int64
ctypedef np.int64_t ITYPE_t
empty = ()

cdef extern from "counts.h":
    void counts_st(ITYPE_t *data, int nObservations,
                   int hLength, int fLength, int nSymbols,
                   ITYPE_t *out, int marginalize)

# Required before using any NumPy C-API
import_array()

__all__ = [
    'standardize_data',
    'counts_from_data',
    'probabilities_from_counts',
    'distribution_from_data',
    ]

@cython.boundscheck(False)
@cython.wraparound(False)
def standardize_data(data, alphabet=None,
                     np.ndarray[ITYPE_t, ndim=1, mode="c"] out=None,
                     validate=False):
    """
    Returns `data` standardized to an integer alphabet.

    The data points should be sortable if the alphabet size is greater than
    four. This is because we use the `bisect` module to standardize the data
    whenever the alphabet size is five or larger.

    Parameters
    ----------
    data : iterable
        The data to be standardized.  The items in data must be hashable and
        also sortable. If, for example, the data consists of frozensets, then
        the alphabet can not (and will not) be sorted properly. The result
        will be that the data is incorrectly standardized.
    alphabet : list
        The ordered symbols in the alphabet. These symbols are standardized to
        integers from 0 to len(`alphabet`)-1. Caution: If the alphabet is not
        sorted, then the data will not be standardized correctly.
    out : NumPy array
        The output NumPy array whose length is at least as large as the
        length of `data`.
    validate : bool
        If `True`, then make sure that the data only contains symbols in the
        specified alphabet. If `alphabet` is None, this parameter is ignored.


    Returns
    -------
    out : NumPy array
        An integer NumPy array containing the standardized data.
    alphabet : list
        The ordered symbols in the alphabet. These symbols are standardized to
        integers from 0 to len(`alphabet`)-1.

    Notes
    -----
    Generally, only one pass through `data` is required, but if `alphabet` is
    `None`, then two passes are necessary.  The time-complexity, assuming
    an alphabet was provided is O( len(data) * log(len(alphabet)) ). When
    `validate` is `True`, then two passes are required.

    """
    cdef int L = len(data)

    if alphabet is None:
        alphabet = sorted(set(data))
    elif isinstance(alphabet, set) or isinstance(alphabet, frozenset):
        msg = "Alphabet is unordered, but it must be ordered."
        raise Exception(msg)
    elif validate:
        alph = set(data)
        diff = alph.difference(set(alphabet))
        if diff:
            msg = 'Data contains symbols in {0}, which are not in the alphabet {1}.'
            msg = msg.format(list(diff), alphabet)
            raise Exception(msg)

    if out is None:
        out = np.empty(L, dtype=ITYPE)
    else:
        assert( len(out) >= len(data), '`out` is not long enough.')

    cdef ITYPE_t *outPtr = <ITYPE_t *>PyArray_DATA(out)
    cdef int K = len(alphabet)
    cdef int i, symbol

    # Note data[i] will be slower for an array than for a list since NumPy
    # indexing is slower than pure Python indexing.

    if K == 1:
        # Unary alphabet
        for i in range(L):
            outPtr[i] = 0
    elif K == 2:
        # Binary alphabet
        first = alphabet[0]
        for i in range(L):
            if first == data[i]:
                outPtr[i] = 0
            else:
                outPtr[i] = 1
    elif K == 3:
        # Ternary alphabet
        first = alphabet[0]
        second = alphabet[1]
        for i in range(L):
            s = data[i]
            if s == first:
                outPtr[i] = 0
            elif s == second:
                outPtr[i] = 1
            else:
                outPtr[i] = 2
    elif K == 4:
        # 4-ary alphabet
        first = alphabet[0]
        second = alphabet[1]
        third = alphabet[2]
        for i in range(L):
            s = data[i]
            if s == first:
                outPtr[i] = 0
            elif s == second:
                outPtr[i] = 1
            elif s == third:
                outPtr[i] = 2
            else:
                outPtr[i] = 3
    else:
        # Otherwise, we use binary searches to determine the standard symbol.
        for i in range(L):
            # This is done in pure python so we can use pythons comparisons.
            symbol = bisect_left(alphabet, data[i])
            outPtr[i] = symbol

    return out, alphabet


def dec2base(int dec, alphabet, np.ndarray[ITYPE_t, ndim=1, mode="c"] offset):
    """Converts an integer into a tuple-based representation.

    For example,  3 -> (1,1)  if alphabet = [0,1].

    Assumptions:  The child nodes of the n-ary tree are obtained by prepending
                  each new symbol.  The encoding of the nodes is assigned by
                  a breadth-first traversal of the tree.

    """
    # The offsets are the first index of each level of the n-ary tree.
    # Ex: For binary alphabet and hLength = 3, we have offset = [0,1,3,7,15]
    cdef ITYPE_t *offsetPtr = <ITYPE_t *>PyArray_DATA(offset)
    cdef int L = 0
    cdef int size = offset.size
    while L < size:
        if dec < offsetPtr[L+1]:
            break
        else:
            L += 1
    if L == size - 1:
        # size == hLength + 2
        # So size - 1 == hlength + 1 ... meaning we are past the n-ary tree.
        raise Exception('Integer is too large for `offset`.')

    # The unoffset number
    cdef int n = dec - offsetPtr[L]

    # Create the tuple
    """cdef object symbol
    cdef object t = PyTuple_New(L)
    cdef int base = len(alphabet)
    cdef int div, mod
    cdef int i = 0
    while i < L:
        div = n // base
        mod = n - div * base
        symbol = alphabet[mod] // does cython automatically incref?
        Py_INCREF(symbol)
        PyTuple_SET_ITEM(t, i, symbol)
        n = div
        i += 1
    """
    t = []
    base = len(alphabet)
    for i in range(L):
        n, mod = divmod(n, base)
        t.append(alphabet[mod])
    t = tuple(t)

    return t

def counts_from_data(data, int hLength, int fLength, marginals=True, alphabet=None, standardize=True):
    """
    Returns conditional counts from `data`.

    ALERT: `data` must not be a generator.

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
    cdef int i,j

    # Scan through the data, grabbing unique elements
    alph = set(data)
    if alphabet is None:
        alphabet = sorted(alph)
    else:
        alphabet = sorted(alph.union(alphabet))

    # Standardize alphabet to 0 .. k where k = len(alphabet) - 1
    if standardize:
        data, _ = standardize_data(data, alphabet)

    # construct counts
    cdef int nSymbols = len(alphabet)
    if nSymbols == 1:
        shape = ( hLength+1, 1 )
    else:
        shape = ( (nSymbols**(hLength+1) - 1) // (nSymbols - 1), nSymbols**fLength )
    cdef np.ndarray[ITYPE_t, ndim=2, mode="c"] cCounts = np.zeros(shape, dtype=ITYPE)
    cdef ITYPE_t *dataPtr = <ITYPE_t *>PyArray_DATA(data)
    cdef ITYPE_t *cCountsPtr = <ITYPE_t *>PyArray_DATA(cCounts)
    counts_st(dataPtr, len(data), hLength, fLength, nSymbols, cCountsPtr, marginals)

    # first, construct array of offsets
    # The offset is the index of the first element of each level of the n-ary tree.
    # We go one level extra so that we can make sure we have the last element.
    cdef int nOffsets = hLength + 2
    cdef np.ndarray[ITYPE_t, ndim=1, mode="c"] offset = np.empty(nOffsets, dtype=ITYPE)
    cdef ITYPE_t *offsetPtr = <ITYPE_t *>PyArray_DATA(offset)
    if nSymbols == 1:
        for i in range(nOffsets):
            offsetPtr[i] = i
    else:
        for i in range(nOffsets):
            offsetPtr[i] = (nSymbols**i - 1) // (nSymbols - 1)

    # construct mapping from histories to rows
    cdef np.ndarray[ITYPE_t, ndim=1, mode="c"] hCounts = np.ascontiguousarray(cCounts.sum(axis=1, dtype=ITYPE))
    cdef ITYPE_t *hCountsPtr = <ITYPE_t *>PyArray_DATA(hCounts)
    cdef int nHistories = hCounts.size
    histories = []
    for i in range(nHistories):
        if hCountsPtr[i] > 0:
            hword = dec2base(i, alphabet, offset)
            histories.append(hword)

    allowed = hCounts > 0

    return histories, cCounts[allowed], hCounts[allowed], alphabet

def probabilities_from_counts(np.ndarray[ITYPE_t, ndim=2, mode="c"] cCounts,
                              np.ndarray[ITYPE_t, ndim=1, mode="c"] hCounts,
                              bint marginals):
    """
    Returns probabilities using the output of `counts_from_data`.

    Parameters
    ----------
    histories : list
        The list of observed histories.
    cCounts : NumPy array
        The conditional counts as output from `counts_from_data`.
    hCounts : NumPy array
        The history counts as output from `counts_from_data`.
    marginals : bool
        The value of `marginals` that was passed to `counts_from_data`.

    Returns
    -------
    cDists : NumPy array
        A NumPy array representing the conditional distributions. The rows
        correspond to the observed histories and the columns to the future
        words.
    hDist : NumPy array
        A 1D array representing the probability of each history word, where
        probabilities are taken with respect to histories of the same length.

    """
    cDists = cCounts.astype(float)
    norms = cCounts.sum(axis=1)
    cDists /= norms[:,np.newaxis]
    hDist = hCounts.astype(float)

    cdef long total
    if marginals:
        total = hCounts[0]
    else:
        total = hCounts.sum()
    hDist /= total

    return cDists, hDist

def morphs_from_data(data, hLength, fLength, marginals=True, probabilities=False, logs=False):
    """
    Alternative (pure Python) based implementation of morphs from data.

    """
    block_length = hLength + fLength

    assert len(data) > 0 and len(data) >= block_length
    data = np.asarray(data)
    counts = collections.defaultdict(int)

    if data.ndim == 1:
        blocks = segment_axis(data, length=block_length, overlap=block_length - 1)
    else:
        blocks = segment_axis(data, length=block_length, overlap=0)

    for block in blocks:
        counts[tuple(block)] += 1

    # if caller wants probabilities they don't get marginals
    if probabilities:
        total = sum(counts.itervalues())
        print total, len(blocks)
        for block in counts.iterkeys():
            counts[block] /= total
        if logs:
            for block, p in counts.iteritems():
                counts[block] = np.log2(p)
    elif marginals:
        counts_marginal = {block_length: counts}
        for length in range(block_length - 1, 0, -1):
            counts_marginal[length] = collections.defaultdict(int)
            for (block, count) in counts_marginal[length + 1].iteritems():
                counts_marginal[length][block[1:]] += count
            undercounted = tuple(data[:length])
            counts_marginal[length][undercounted] += 1
            counts.update(counts_marginal[length])
        counts[empty] = len(data)

    # Calculating all conditional distributions

    alphabet =  sorted(set(data))
    futures = list(itertools.product(alphabet, repeat=fLength))

    def normalize(x):
        return np.asarray(x, dtype=float) / sum(x)

    cdists = {}
    histories = filter(lambda past: len(past) <= block_length - fLength, counts.iterkeys())
    for hword in histories:
        def fprob(future):
            joint_word = hword + tuple(future)
            if joint_word not in counts:
                p = 0
            else:
                p = counts[joint_word] / counts[hword]
            return p
        dist = map(fprob, futures)
        cdists[hword] = normalize(map(fprob, futures))

    # dict(counts) prevents adding keys without exceptions
    return cdists, dict(counts), alphabet

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

    # The general function takes a history length and a future length.
    # The docstring for counts_from_data should explain the outputs.
    hLength = 0
    fLength = L
    marginals = False
    x = counts_from_data(d, hLength, fLength, marginals=marginals)
    histories, cCounts, hCounts, alphabet = x

    # We turn the counts to probabilities
    cProbs, hProbs = probabilities_from_counts(cCounts, hCounts, marginals)

    # There is only the empty history.
    pmf = cProbs[0]

    futures = list(product(alphabet, repeat=fLength))
    dist = Distribution(futures, pmf, trim=trim)

    if base is not None:
        dist.set_base(base)
    return dist

