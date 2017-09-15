#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Define miscellaneous utilities.
"""

from __future__ import absolute_import

from collections import Iterable
import functools
import inspect
from itertools import tee
import os
import sys
import subprocess
import warnings

import six
from six.moves import range, zip # pylint: disable=redefined-builtin

__all__ = (
    'Property',
    'abstract_method',
    'default_opener',
    'flatten',
    'get_fobj',
    'is_string_like',
    'quasilexico_key',
    'ordered_partitions',
    'OrderedDict',
    'partitions',
    'partition_set',
    'powerset',
    'product_maker',
    'require_keys',
    'str_product',
    'digits',
    'pairwise',
)

######################################################
# Hacks for simultaneous 2.x and 3.x compatibility.
#
try: # pragma: no cover
    # 2.7+
    from collections import OrderedDict
except ImportError: # pragma: no cover
    # 2.6
    from ordereddict import OrderedDict


def Property(fcn):
    """Simple property decorator.

    Usage:

        @Property
        def attr():
            doc = '''The docstring.'''
            def fget(self):
                pass
            def fset(self, value):
                pass
            def fdel(self)
                pass
            return locals()

    """
    return property(**fcn())

def abstract_method(f):
    """Simple decorator to designate an abstract method.

    Examples
    --------
    class A(object):
        @abstract_method
        def method(self):
            pass
    """
    def abstract_f(*args, **kwargs): # pylint: disable=unused-argument
        raise NotImplementedError("Abstract method.")
    abstract_f.__name__ = f.__name__
    abstract_f.__doc__ = f.__doc__
    return abstract_f

def default_opener(filename):
    """Opens `filename` using system's default program.

    Parameters
    ----------
    filename : str
        The path of the file to be opened.

    """
    cmds = {'darwin': ['open'],
            'linux2': ['xdg-open'], # Python 2.x
            'linux': ['xdg-open'],  # Python 3.x
            'win32': ['cmd.exe', '/c', 'start', '']}
    cmd = cmds[sys.platform] + [filename]
    subprocess.call(cmd)

def flatten(l):
    """Flatten an irregular list of lists.

    Parameters
    ----------
    l : iterable
       The object to be flattened.

    Yields
    -------
    el : object
        The non-iterable items in `l`.
    """
    for el in l:
        if isinstance(el, Iterable) and not (isinstance(el, six.string_types) and len(el) == 1):
            for sub in flatten(el):
                yield sub
        else:
            yield el

def get_fobj(fname, mode='w+'):
    """Obtain a proper file object.

    Parameters
    ----------
    fname : string, file object, file descriptor
        If a string or file descriptor, then we create a file object. If *fname*
        is a file object, then we do nothing and ignore the specified *mode*
        parameter.
    mode : str
        The mode of the file to be opened.

    Returns
    -------
    fobj : file object
        The file object.
    close : bool
        If *fname* was a string or file descriptor, then *close* will be *True*
        to signify that the file object should be closed. Otherwise, *close*
        will be *False* signifying that the user has opened the file object and
        that we should not close it.

    """
    if is_string_like(fname):
        fobj = open(fname, mode)
        close = True
    elif hasattr(fname, 'write'):
        # fname is a file-like object, perhaps a StringIO (for example)
        fobj = fname
        close = False
    else:
        # assume it is a file descriptor
        fobj = os.fdopen(fname, mode)
        close = True
    return fobj, close

def is_string_like(obj):
    """Returns *True* if *obj* is string-like, and *False* otherwise."""
    try:
        obj + ''
    except (TypeError, ValueError):
        return False
    return True

def quasilexico_key(x):
    """Returns a key suitable for a quasi-lexicographic sort [1]_.

    Objects are sorted by length first, then lexicographically.

    Examples
    --------
    >>> L = ['a', 'aa', 'b']
    >>> sorted(L, key=quasilexico_key)
    ['a', 'b', 'aa']

    References
    ----------
    .. [1] Calude, Cristian (1994). Information and randomness. An algorithmic
           perspective. EATCS Monographs on Theoretical Computer Science.
           Springer-Verlag. p. 1.

    """
    return (len(x), x)

def partition_set(elements, relation=None, innerset=False, reflexive=False,
                  transitive=False):
    """Returns the equivlence classes from `elements`.

    Given `relation`, we test each element in `elements` against the other
    elements and form the equivalence classes induced by `relation`.  By
    default, we assume the relation is symmetric.  Optionally, the relation
    can be assumed to be reflexive and transitive as well.  All three
    properties are required for `relation` to be an equivalence relation.

    However, there are times when a relation is not reflexive or transitive.
    For example, floating point comparisons do not have these properties. In
    this instance, it might be desirable to force reflexivity and transitivity
    on the elements and then, work with the resulting partition.

    Parameters
    ----------
    elements : iterable
        The elements to be partitioned.
    relation : function, None
        A function accepting two elements, which returns `True` iff the two
        elements are related. The relation need not be an equivalence relation,
        but if `reflexive` and `transitive` are not set to `False`, then the
        resulting partition will not be unique. If `None`, then == is used.
    innerset : bool
        If `True`, then the equivalence classes will be returned as frozensets.
        This means that duplicate elements (according to __eq__ not `relation`)
        will appear only once in the equivalence class. If `False`, then the
        equivalence classes will be returned as lists. This means that
        duplicate elements will appear multiple times in an equivalence class.
    reflexive : bool
        If `True`, then `relation` is assumed to be reflexive. If `False`, then
        reflexivity will be enforced manually. Effectively, a new relation
        is considered: relation(a,b) AND relation(b,a).
    transitive : bool
        If `True`, then `relation` is assumed to be transitive. If `False`, then
        transitivity will be enforced manually. Effectively, a new relation is
        considered: relation(a,b) for all b in the class.

    Returns
    -------
    eqclasses : list
        The collection of equivalence classes.
    lookup : list
        A list relating where lookup[i] contains the index of the eqclass
        that elements[i] was mapped to in `eqclasses`.

    """
    if relation is None:
        from operator import eq
        relation = eq

    lookup = []
    if reflexive and transitive:
        eqclasses = []
        for _, element in enumerate(elements):
            for eqclass_idx, (representative, eqclass) in enumerate(eqclasses):
                if relation(representative, element):
                    eqclass.append(element)
                    lookup.append(eqclass_idx)
                    # Each element can belong to *one* equivalence class
                    break
            else:
                lookup.append(len(eqclasses))
                eqclasses.append((element, [element]))


        eqclasses = [c for _, c in eqclasses]

    else:
        def belongs(element, eqclass):
            for representative in eqclass:
                if not relation(representative, element):
                    return False
                if not reflexive:
                    if not relation(element, representative):
                        return False
                if transitive:
                    return True
                else:
                    # Test against all members
                    continue

            # Then it equals all memembers symmetrically.
            return True

        eqclasses = []
        for _, element in enumerate(elements):
            for eqclass_idx, eqclass in enumerate(eqclasses):
                if belongs(element, eqclass):
                    eqclass.append(element)
                    lookup.append(eqclass_idx)
                    # Each element can belong to one equivalence class
                    break
            else:
                lookup.append(len(eqclasses))
                eqclasses.append([element])


    if innerset:
        eqclasses = [frozenset(c) for c in eqclasses]
    else:
        eqclasses = [tuple(c) for c in eqclasses]

    return eqclasses, lookup

def powerset(iterable):
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)

    """
    from itertools import chain, combinations
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def product_maker(func):
    """
    Returns a customized product function.

    itertools.product yields tuples.  Sometimes, one desires some other type
    of object---for example, strings.  This function transforms the output
    of itertools.product, returning a customized product function.

    Parameters
    ----------
    func : callable
        Any function which accepts an iterable as the single input argument.
        The iterates of itertools.product are transformed by this callable.

    Returns
    -------
    _product : callable
        A customized itertools.product function.

    """
    from itertools import product
    def _product(*args, **kwargs):
        for prod in product(*args, **kwargs):
            yield func(prod)
    return _product

str_product = product_maker(''.join)

def require_keys(keys, dikt):
    """Verifies that keys appear in the specified dictionary.

    Parameters
    ----------
    keys : list of str
        List of required keys.

    dikt : dict
        The dictionary that is checked for keys.

    Returns
    -------
    None

    Raises
    ------
    Exception
        Raised when a required key is not present.

    """
    dikt_keys = set(dikt)
    for key in keys:
        if key not in dikt_keys:
            msg = "'%s' is required." % (key,)
            raise Exception(msg)

def partitions1(set_):
    """
    Generates partitions of elements in `set_'.

    For set_ = range(12), this finishes in 52.37 seconds.

    Yields tuple of sets.

    """
    # Thomas Dybdahl Ahle (https://github.com/thomasahle)
    # Source:
    #   http://compprog.wordpress.com/2007/10/15/generating-the-partitions-of-a-set
    if not set_:
        yield ()
        return
    for i in range(2**len(set_) // 2): # 2**() is even, so using // is safe.
        parts = [set(), set()]
        for item in set_:
            parts[i&1].add(item)
            i >>= 1
        for b in partitions1(parts[1]):
            yield (parts[0],) + b

def partitions2(n):
    """
    Generates all partitions of {1,...,n}.

    For n=12, this finishes in 4.48 seconds.

    """
    # Original source: George Hutchinson [CACM 6 (1963), 613--614]
    #
    # This implementation is:
    #    Algorithm H (Restricted growth strings in lexicographic order)
    # from pages 416--417 of Knuth's The Art of Computer Programming, Vol 4A:
    # Combinatorial Problems, Part 1. 1st Edition (2011).
    # ISBN-13: 978-0-201-03804-0
    # ISBN-10:     0-201-03804-8
    #

    # To maintain notation with Knuth, we ignore the first element of
    # each array so that a[j] == a_j, b[j] == b_j for j = 1,...,n.

    # H1 [Initialize.]
    # Per above, make lists larger by one element to give 1-based indexing.
    if n == 0:
        yield [[]]
    elif n == 1:
        yield [[0]]
    else:
        a = [0] * (n+1)
        b = [1] * (n)
        m = 1

        while True:
            # H2 [Visit.]
            yield a[1:]
            if a[n] == m:
                # H4 [Find $j$.]
                j = n - 1
                while a[j] == b[j]:
                    j -= 1

                # H5 [Increase $a_j$.]
                if j == 1:
                    break
                else:
                    a[j] += 1

                # H6 [Zero out $a_{j+1} \ldots a_n$]
                m = b[j]
                if a[j] == b[j]: # Iverson braket
                    m += 1
                j += 1
                while j < n:
                    a[j] = 0
                    b[j] = m
                    j += 1
                a[n] = 0

            else:
                # H3
                a[n] += 1

def partitions(seq, tuples=False):
    """
    Generates all partitions of `seq`.

    Parameters
    ----------
    seq : iterable
        Any iterable.  Used to generate the partitions.
    tuples : bool
        If `True`, yields tuple of tuples. Otherwise, yields frozenset of
        frozensets.

    Yields
    ------
    partition : frozenset or tuple
        A frozenset of frozensets, or a sorted tuple of sorted tuples.

    """
    # Handle iterators.
    seq = list(seq)

    if tuples:
        for partition in partitions1(seq):
            # Convert the partition into a list of sorted tuples.
            partition = map(tuple, map(sorted, partition))

            # Convert the partition into a sorted tuple of sorted tuples.
            # Sort by smallest parts first, then lexicographically.
            partition = tuple(sorted(partition, key=quasilexico_key))

            yield partition

    else:
        for partition in partitions1(seq):
            partition = frozenset(map(frozenset, partition))
            yield partition

def ordered_partitions(seq, tuples=False):
    """
    Generates ordered partitions of elements in `seq`.

    Parameters
    ----------
    seq : iterable
        Any iterable.  Used to generate the partitions.
    tuples : bool
        If `True`, yields tuple of tuples. Otherwise, yields tuple of
        frozensets.

    Yields
    ------
    partition : tuple
        A tuple of frozensets, or a tuple of sorted tuples.

    """
    from itertools import permutations

    # Handle iterators.
    seq = list(seq)

    if tuples:
        for partition in partitions1(seq):
            # Convert the partition into a list of sorted tuples.
            partition = list(map(tuple, map(sorted, partition)))

            # Generate all permutations.
            for perm in permutations(partition):
                yield perm
    else:
        for partition in partitions1(seq):
            partition = list(map(frozenset, partition))
            for perm in permutations(partition):
                yield perm

def digits(n, base, alphabet=None, pad=0, big_endian=True):
    """
    Returns `n` as a sequence of indexes into an alphabet.

    Parameters
    ----------
    n : int
        The number to convert into a sequence of indexes.
    base : int
        The desired base of the sequence representation. The base must be
        greater than or equal to 2.
    alphabet : iterable
        If specified, then the indexes are converted into symbols using
        the specified alphabet.
    pad : int
        If  `True`, the resultant sequence is padded with zeros.
    big_endian : bool
        If `True`, then the resultant sequence has the least significant
        digits at the end. This is standard for Western culture.

    Returns
    -------
    sequence : list
        The digits representation of `n`.

    Examples
    --------
    >>> digits(6, base=2, pad=4)
    [0, 1, 1, 0]

    >>> ''.join(digits(6, base=2, pad=4, alphabet='xo'))
    'xoox'

    """
    # http://stackoverflow.com/a/2088440

    if base < 2 or int(base) != base:
        raise ValueError('`base` must be an integer greater than 2')

    if alphabet is not None:
        if len(alphabet) != base:
            raise ValueError('Length of `alphabet` must equal `base`.')

    sequence = []
    while True:
        sequence.append(n % base)
        if n < base:
            break
        n //= base

    if pad:
        zeros = [0] * (pad - len(sequence))
        sequence.extend(zeros)

    if big_endian:
        sequence.reverse()

    if alphabet is not None:
        sequence = [alphabet[i] for i in sequence]

    return sequence

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)
