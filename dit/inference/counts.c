/**
    An implementation of calculating counts from data.

    This implementation is mostly C90 compatible, but makes use of gcc's
    support for variable-length arrays (VLAs) which was accepted into C99.

    Discussion
    ----------
    windowSize = hLength + fLength
    nWindows = nObservations - windowSize

    In Python, we begin with a list. As we iterate through, we construct tuples
    of the history and future words.  This is O(windowSize). Then, we add
    to the count in the dictionary. This requires a computation of the hashes
    for the history and future words. This is also O(windowSize).  Overall
    then, we are probably at:  O( nWindows * 2 * windowSize ).

    In the C version, there is no need to construct the tuples. Since the data
    is standard, we can construct the hash directly. So this is only
    O(windowSize). Overall then, we are probably at: O(nWindows * windowSize).
    This is a slight improvement.

**/

#include <stdio.h>
#include <math.h>

#include "counts.h"

/**
    Calculates conditional counts of words (single-threaded version)

    Parameters
    ----------
    dataPtr : int *
        A pointer to an array of integer-valued observations.
    nObservations : int
        The number of observations and thus, the length of the data array.
    hLength : int
        The length of the history words.
    fLength : int
        The length of the future words.
    nSymbols : int
        The number of symbols in the alphabet.
    out : int *
        A pointer to an integer array which will contain the counts of future
        words given history words. The array's length should be equal to:
            \frac{nSymbols^{hLength+1} - 1}{nSymbols - 1} * nSymbols^fLength.
    marginalize : int
        If 1, then we marginalize all the counts to obtain conditional counts
        of the future words of length fLength for every history word of length
        0 to length hLength.

    Notes
    -----
    The counts are stored in a flattened row major array.  For each history
    word, the counts of the nSymbols^fLength future words are given by their
    k-ary encoding. For example, if nSymbols=2 and fLength=2, then there are
    four columns representing w=00,01,10,11.  The history words are given by
    their encoding in a k-ary tree.

**/
void counts_st(long* data, int nObservations,
               int hLength, int fLength, int nSymbols,
               long* out, int marginals) {

    int i,j,L,hist;

    /**
        The encoding for histories fixes the right side of the word, as
        that is closest to the cutpoint at t=0. So in breadth-first order,
        the words are:

          \lambda = 0

          0 = 1
          1 = 2

          00 = 3
          10 = 4
          01 = 5
          11 = 6

          000 = 7
          100 = 8
          010 = 9
          110 = 10
          001 = 11
          101 = 12
          011 = 13
          111 = 14

        The encoding for 110, for example, is given by:

            encoding(110) = offset(3) + 1 * 2^0
                                      + 1 * 2^1
                                      + 0 * 2^2
                          = 10
        where

            offset(L) = (nSymbols^L - 1) / (nSymbols - 1)

        When we marginalize the histories, we work with their index only.
        So if we start at w=110, then we want to chop from the left going
        to w=10, w=0, w=\lambda. Of course, we only work with the integers,
        so this corresponds from 10 -> 4, 1, 0.

        The formula describing this progression is:

            w' = offset(L-1) + (w - offset(L)) / nSymbols

        Futures are encoding by appending since the left-side (t=0) of the word
        is fixed. So words of length 2 over a binary alphabet are given the
        following assignments:

            00 -> 0,
            01 -> 1,
            10 -> 2,
            11 -> 3

    **/

    // VLA is allocated on the stack (C99)
    int offset[hLength+2];

    int index, hword, fword;
    int nFutures = pow(nSymbols, fLength);

    if (nSymbols == 1) {
        for (L=0; L < hLength+2; ++L) {
            offset[L] = L;
        }
    }
    else {
        for (L=0; L < hLength+2; ++L) {
            offset[L] = (pow(nSymbols,L) - 1) / (nSymbols - 1);
        }
    }

    /**
        When we parse the data, we only fill in the tail end of `out`, since
        this corresponds to filling out the leaves of the tree.
    **/
    for (i = hLength; i <= nObservations - fLength; ++i) {
        // Construct the index for the history word (from the left)
        hword = offset[hLength];
        L = 0;
        for (j = i - hLength; j < i; ++j) {
            hword += data[j] * pow(nSymbols, L);
            L += 1;
        }
        index = hword * nFutures;

        // Construct the index for the future word (from the right)
        fword = 0;
        L = 0;
        for (j = i + fLength - 1; j >= i; --j) {
            fword += data[j] * pow(nSymbols, L);
            L += 1;
        }
        out[index+fword] += 1;
    }

    if (marginals) {
        // Time-complexity for outer two loops: O( 2^(hLength+1) - 1 )
        int mhist, index_mhist, index_hist;
        for (L = hLength; L > 0; --L) {
            for (hist = offset[L]; hist < offset[L+1]; ++hist) {
                // determine the predecessor history
                mhist = offset[L-1] + (hist - offset[L]) / nSymbols;
                // add the counts for all future words
                index_mhist = mhist * nFutures;
                index_hist = hist * nFutures;
                for (j = 0; j < nFutures; ++j) {
                    out[index_mhist+j] += out[index_hist+j];
                }
            }
        }
    } // end marginalize
}
