#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Define miscellaneous utilities.
"""

from __future__ import absolute_import

import os
import sys
import subprocess
import warnings

__all__ = ('Property',
           'abstract_method',
           'default_opener',
           'deprecate',
           'get_fobj',
           'is_string_like',
           'len_cmp',
           'partition_set',
           'require_keys',
           'str_product',
           'product_maker',
           'xzip')

def Property( fcn ):
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
    return property( **fcn() )

def abstract_method(f):
    """Simple decorator to designate an abstract method.

    Examples
    --------
    class A(object):
        @abstract_method
        def method(self):
            pass
    """
    def abstract_f(*args, **kwargs):
        raise NotImplementedError("Abstract method.")
    abstract_f.func_name = f.func_name
    abstract_f.__doc__ = f.__doc__
    return abstract_f

def default_opener(filename):
    """Opens *filename* using system's default program.

    Parameters
    ----------
    filename : str
        The path of the file to be opened.

    """
    cmds = {'darwin': ['open'],
            'linux2': ['xdg-open'],
            'win32': ['cmd.exe', '/c', 'start', '']}
    cmd = cmds[sys.platform] + [filename]
    subprocess.call(cmd)

class deprecate(object):
    """Decorator for deprecating functions.

    Note: You must decorate with an instance like so:

        @deprecate(msg)
        def func_to_be_decorated:
            pass

    """
    def __init__(self, msg=''):
        """Initializes the decorator.

        Parameters
        ----------
        msg : str
            A string that is added to the deprecation warning.

        """
        self.msg = msg

    def __call__(self, f):
        """Return the modified function/method."""

        if hasattr(f, 'thefunc'):
            # handle numpy.vectorize() output
            name = None
            docname = 'Vectorized ' + f.thefunc.__name__
        else:
            name = f.__name__
            docname = name

        def new_f(*args, **kwargs):
            msg = "%s() is deprecated. %s"
            warnings.warn(msg % (docname, self.msg),
                          category=DeprecationWarning,
                          stacklevel=2)
            return f(*args, **kwargs)

        if name is not None:
            new_f.__name__ = name

        msg = "\n\n    %s() is deprecated.\n    %s" % (docname, self.msg)
        if f.__doc__ is None:
            new_f.__doc__ = msg
        else:
            new_f.__doc__ = f.__doc__ + msg

        new_f.__dict__.update(f.__dict__)

        return new_f

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

def xzip(*args):
    """This is just like the zip function, except it is a generator.
    
    list(xzip(seq1 [, seq2 [...]])) -> [(seq1[0], seq2[0] ...), (...)]
    
    Return a list of tuples, where each tuple contains the i-th element
    from each of the argument sequences.  The returned list is truncated
    in length to the length of the shortest argument sequence.

    """
    iters = [iter(a) for a in args]
    while 1:
        yield tuple([i.next() for i in iters])
    
def len_cmp(x,y):
    """A comparison function which sorts shorter objects first."""
    lenx, leny = len(x), len(y)
    if lenx < leny:
        return -1
    elif lenx > leny:
        return 1
    else:
        return cmp(x,y)
    
def partition_set(elements, relation=None, innerset=False, reflexive=False, transitive=False):
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
        for element_idx, element in enumerate(elements):
            for eqclass_idx, (representative, eqclass) in enumerate(eqclasses):
                if relation(representative, element):
                    eqclass.append(element)
                    lookup.append(eqclass_idx)
                    # Each element can belong to *one* equivalence class
                    break
            else:
                lookup.append(len(eqclasses))
                eqclasses.append( (element, [element]) )


        eqclasses = [c for r,c in eqclasses]
        
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
        for element_idx, element in enumerate(elements):
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
    def _product(*args):
        for prod in product(*args):
            yield func(prod)
    return _product

str_product = product_maker(''.join)

