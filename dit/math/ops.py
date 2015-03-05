#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Classes to contextualize math operations in log vs linear space.

"""
from types import MethodType

import numpy as np

from ..exceptions import InvalidBase
from .equal import close

__all__ = [
    'get_ops',
    'LinearOperations',
    'LogOperations'
]

# For 2.x, these are ascii strings. For 3.x these are unicode strings.
acceptable_base_strings = set(['linear', 'e'])

def get_ops(base):
    """
    Returns an *Operations instance, depending on the base.

    Parameters
    ----------
    base : float, 'linear', 'e'
        The base for the Operations instance.

    """
    # Let's not initialize unless we have to.
    if base in cache:
        ops = cache[base]
    else:
        # This assumes that 'linear' is in cache.
        ops = LogOperations(base)
        cache[base] = ops
    return ops

def exp_func(b):
    """
    Returns a base-`b` exponential function.

    Parameters
    ----------
    b : positive float or 'e'
        The base of the desired exponential function.

    Returns
    -------
    exp : function
        The base-`b` exponential function. The returned function will operate
        elementwise on NumPy arrays, but note, it is not a ufunc.

    Examples
    --------
    >>> exp2 = exp_func(2)
    >>> exp2(1)
    2.0
    >>> exp3 = exp_func(3)
    >>> exp3(1)
    3.0

    Raises
    ------
    InvalidBase
        If the base is less than zero or equal to one.

    """
    from dit.utils import is_string_like

    if is_string_like(b) and b not in acceptable_base_strings:
        raise InvalidBase(msg=b)

    if b == 'linear':
        exp = lambda x: x
    elif b == 2:
        exp = np.exp2
    elif b == 10:
        exp = lambda x: 10**x
    elif b == 'e' or close(b, np.e):
        exp = np.exp
    else:
        if b <= 0 or b == 1:
            raise InvalidBase(b)

        def exp(x, base=b):
            """
            Return `base`**`x`

            Parameters
            ----------
            x : float
                The number to exponentiate
            base : float
                The base of the exponential

            Returns
            -------
            p : float
                `base`**`x`
            """
            return base**np.asarray(x)

    return exp

def log_func(b):
    """
    Returns a base-`b` logarithm function.

    Parameters
    ----------
    b : positive float or 'e'
        The base of the desired logarithm function.

    Returns
    -------
    log : function
        The base-`b` logarithm function. The returned function will operate
        elementwise on NumPy arrays, but note, it is not a ufunc.

    Examples
    --------
    >>> log2 = log_func(2)
    >>> log2(2)
    1.0
    >>> log3 = log_func(3)
    >>> log3(3)
    1.0

    Raises
    ------
    InvalidBase
        If the base is less than zero or equal to one.

    """
    from dit.utils import is_string_like

    if is_string_like(b) and b not in acceptable_base_strings:
        raise InvalidBase(msg=b)

    if b == 'linear':
        log = lambda x: x
    elif b == 2:
        log = np.log2
    elif b == 10:
        log = np.log10
    elif b == 'e' or close(b, np.e):
        log = np.log
    else:
        if b <= 0 or b == 1:
            raise InvalidBase(b)

        Z = np.log(b)
        def log(x, func=np.log):
            """
            Return the log of `x`

            Parameters
            ----------
            x : float
                The value to take the log of
            func : function
                A logarithm function

            Returns
            -------
            log : float
                The logarithm of `x` in base `b` (from outer scope)
            """
            return func(x) / Z

    return log

class Operations(object):
    """
    Base class which implements certain math operations.

    For example, regular addition with log probabilities is handled specially.

    While we could implement many more operations, we do not.  Their usage
    is uncommon and their implementation would be slower as well.  For example,
    subtraction with log probabailities must go as:

        log_2(x-y) = log_2(x) + log_2(1 - 2^[ log_2(y) - log_2(x) ])

    Note that if y > x, then log(y) > log(x) and the inner term of the second
    logarithm will be less than 0, yielding NaN.

    """
    ### Do we allow base == 'e' or should we convert to its numerical value?
    ### Ans: We store whatever was specified but provide get_base() with an
    ###      option to return a numerical base.

    one = None
    zero = None
    base = None
    exp = None
    log = None

    def get_base(self, numerical=False):
        """
        Returns the base in which operations take place.

        For linear-based operations, the result is 'linear'.

        Parameters
        ----------
        numerical : bool
            If `True`, then if the base is 'e', it is returned as a float.

        """
        if numerical and self.base == 'e':
            base = np.exp(1)
        else:
            base = self.base
        return base

    def is_null(self, p):
        """
        Returns `True` if `p` is a null probability.

        Parameters
        ----------
        p : float
            The probability to be tested.

        """
        return close(self.zero, p)

    def is_null_exact(self, p):
        """
        Returns `True` if `p` is exactly a null probability.

        Parameters
        ----------
        p : float
            The probability to be tested.

        """
        return self.zero == p

    def add(self, x, y):
        """ Abstract base class """
        raise NotImplementedError
    def add_inplace(self, x, y):
        """ Abstract base class """
        raise NotImplementedError
    def add_reduce(self, x):
        """ Abstract base class """
        raise NotImplementedError
    def mult(self, x, y):
        """ Abstract base class """
        raise NotImplementedError
    def mult_inplace(self, x, y):
        """ Abstract base class """
        raise NotImplementedError
    def mult_reduce(self, x):
        """ Abstract base class """
        raise NotImplementedError
    def invert(self, x):
        """ Abstract base class """
        raise NotImplementedError
    def normalize(self, x):
        """ Abstract base class """
        raise NotImplementedError

class LinearOperations(Operations):
    """
    The class of operations on linear values.
    """

    one = 1
    zero = 0
    base = 'linear'

    # If the functions below are standard Python functions (as opposed to
    # NumPy ufuncs), then they will be treated as unbound methods for the class.
    # During instantiation, they are bound to the instance (since before
    # instantiation they are class methods) and thus, we are left with
    # bound methods (undesirably). If we had modified these attributes in the
    # __init__ function, then they would not be bound (or even unbound methods)
    # but functions instead (desirably).  This is precisely what LogOperations
    # does, which is why it does not have this issue. An alternative approach
    # is to explicitly declare these functions to be static methods, as we
    # do below.
    #
    exp = staticmethod(exp_func(base))
    log = staticmethod(log_func(base))

    def add(self, x, y):
        """
        Add the arrays element-wise.  Neither x nor y will be modified.

        Assumption: y >= 0.

        Operation:  z[i] = x[i] + y[i]

        Parameters
        ----------
        x, y : NumPy arrays, shape (n,)
            The arrays to add.

        Returns
        -------
        z : NumPy array, shape (n,)
            The resultant array.

        """
        z = x + y
        return z

    def add_inplace(self, x, y):
        """
        Adds `y` to `x`, in-place.  `x` will be modified, but `y` will not.

        Assumption: y >= 0.

        Operation: x[i] += y[i]

        Parameters
        ----------
        x, y : NumPy arrays, shape (n,)
            The arrays to add.

        Returns
        -------
        x : NumPy array, shape (n,)
            The resultant array.

        """
        x += y
        return x

    def add_reduce(self, x, axis=None):
        """
        Performs an `addition' reduction on `x`.

        Assumption: y >= 0.

        Operation: z = \sum_i x[i]

        Returns
        -------
        z : float
            The summation of the elements in `x`.

        """
        z = x.sum(axis=axis)
        return z

    def mult(self, x, y):
        """
        Multiplies the arrays element-wise.  Neither x nor y will be modified.

        Operation: z[i] = x[i] * y[i]

        Parameters
        ----------
        x, y : NumPy arrays, shape (n,)
            The arrays to multiply.

        Returns
        -------
        z : NumPy array, shape (n,)
            The resultant array.

        """
        z = x * y
        return z

    def mult_inplace(self, x, y):
        """
        Multiplies `y` to `x`, in-place. `x` will be modified, but `y` will not.

        Operation: x[i] *= y[i]

        Parameters
        ----------
        x, y : NumPy arrays, shape (n,)
            The arrays to multiply.

        Returns
        -------
        x : NumPy array, shape (n,)
            The resultant array.

        """
        x *= y
        return x

    def mult_reduce(self, x, axis=None):
        """
        Performs an `multiplication' reduction on `x`.

        Operation: z = \prod_i x[i]

        Returns
        -------
        z : float
            The product of the elements in `x`.

        """
        z = np.prod(x, axis=axis)
        return z

    def invert(self, x):
        """
        Returns the element-wise multiplicative inverse of x.

        Operation: z[i] = 1/x[i]

        Parameters
        ----------
        x : NumPy array, shape (n,)
            The array to invert.

        Returns
        -------
        z : NumPy array, shape (n,)
            The inverted array.

        """
        z = 1/x
        return z

    def normalize(self, x, axis=None):
        """
        Returns a normalized version of x.

        Operation: z[i] = x[i] / sum(x)

        If x is 2D and axis is None, then normalization is over all elements.
        Use axis=-1 to normalize each row of x.

        Parameters
        ----------
        x : NumPy array, shape (n,)
            The array to normalize.

        Returns
        -------
        z : NumPy array, shape (n,)
            The normalized array.

        """
        z = x / x.sum(axis=None)
        return z

def set_add(ops):
    """
    Set the add method on the LogOperations instance.

    """
    # To preserve numerical accuracy, we must make use of a logaddexp
    # function.  These functions only exist in Numpy for base-e and base-2.
    # For all other bases, we must convert and then convert back.

    # In each case, we use default arguments to make the function that we
    # are calling 'local'.
    base = ops.base
    if base == 2:
        def add(self, x, y, func=np.logaddexp2):
            return func(x, y)
    elif base == 'e' or close(base, np.e):
        def add(self, x, y, func=np.logaddexp):
            return func(x, y)
    else:
        # No need to optimize this...
        def add(self, x, y):
            # Convert log_b probabilities to log_2 probabilities.
            x2 = x * np.log2(base)
            y2 = y * np.log2(base)
            z = np.logaddexp2(x2, y2)
            # Convert log_2 probabilities to log_b probabilities.
            z *= self.log(2)
            return z

    add.__doc__ = """
    Add the arrays element-wise.  Neither x nor y will be modified.

    Assumption: y <= 0.

    Parameters
    ----------
    x, y : NumPy arrays, shape (n,)
        The arrays to add.

    Returns
    -------
    z : NumPy array, shape (n,)
        The resultant array.

    """
    ops.add = MethodType(add, ops)

def set_add_inplace(ops):
    """
    Set the add_inplace method on the LogOperations instance.

    """
    base = ops.base
    if base == 2:
        def add_inplace(self, x, y, func=np.logaddexp2):
            return func(x, y, x)
    elif base == 'e' or close(base, np.e):
        def add_inplace(self, x, y, func=np.logaddexp):
            return func(x, y, x)
    else:
        def add_inplace(self, x, y):
            x *= np.log2(base)
            y2 = y * np.log2(base)
            np.logaddexp2(x, y2, x)
            x *= self.log(2)
            return x

    add_inplace.__doc__ = """
    Adds `y` to `x`, in-place.  `x` will be modified, but `y` will not.

    Assumption: y <= 0.

    Parameters
    ----------
    x, y : NumPy arrays, shape (n,)
        The arrays to add.

    Returns
    -------
    x : NumPy array, shape (n,)
        The resultant array.

    """
    ops.add_inplace = MethodType(add_inplace, ops)

def set_add_reduce(ops):
    """
    Set the add_reduce method on the LogOperations instance.

    """
    # https://github.com/numpy/numpy/issues/4599
    base = ops.base
    if base == 2:
        def add_reduce(self, x, axis=None, func=np.logaddexp2):
            if len(x) == 0:
                # Since logaddexp.identity is None, we handle it separately.
                z = self.zero
            else:
                # Note, we are converting to a NumPy array, if necessary.
                z = func.reduce(x, axis=axis, dtype=float)
            return z

    elif base == 'e' or close(base, np.e):
        def add_reduce(self, x, axis=None, func=np.logaddexp):
            if len(x) == 0:
                # Since logaddexp.identity is None, we handle it separately.
                z = self.zero
            else:
                # Note, we are converting to a NumPy array, if necessary.
                z = func.reduce(x, axis=axis, dtype=float)
            return z

    else:
        def add_reduce(self, x, axis=None):
            if len(x) == 0:
                # Since logaddexp.identity is None, we handle it separately.
                z = self.zero
            else:
                # Note, we are converting to a NumPy array, if necessary.
                # Change the base-2, add, and then convert back.
                x2 = x * np.log2(base)
                z = np.logaddexp2.reduce(x2, axis=axis, dtype=float)
                z /= np.log2(base)
            return z

    add_reduce.__doc__ = """
    Performs an `addition' reduction on `x`.

    Assumption: y <= 0.

    Returns
    -------
    z : float
        The summation of the elements in `x`.

    """
    ops.add_reduce = MethodType(add_reduce, ops)

class LogOperations(Operations):

    one = None
    zero = None
    base = None
    exp = None
    log = None

    def __init__(self, base):
        """
        Initialize the log operation manager.

        Parameters
        ----------
        base : float
            The base of the logarithm.

        """
        self.set_base(base)

    def set_base(self, base):
        """
        Change the base of the logarithm.

        Parameters
        ----------
        base : float
            The base of the logarithm.

        """
        self.base = base
        self.exp = exp_func(base)
        self.log = log_func(base)
        # Note: When base < 1, zero == +inf. When base > 1, zero == -inf.
        self.one = self.log(1)
        self.zero = self.log(0)

        # Update the add methods.
        set_add(self)
        set_add_inplace(self)
        set_add_reduce(self)

    def mult(self, x, y):
        """
        Multiplies the arrays element-wise.  Neither x nor y will be modified.

        Parameters
        ----------
        x, y : NumPy arrays, shape (n,)
            The arrays to multiply.

        Returns
        -------
        z : NumPy array, shape (n,)
            The resultant array.

        """
        z = x + y
        return z

    def mult_inplace(self, x, y):
        """
        Multiplies `y` to `x`, in-place. `x` will be modified, but `y` will not.

        Parameters
        ----------
        x, y : NumPy arrays, shape (n,)
            The arrays to multiply.

        Returns
        -------
        x : NumPy array, shape (n,)
            The resultant array.

        """
        x += y
        return x

    def mult_reduce(self, x, axis=None):
        """
        Performs an `multiplication' reduction on `x`.

        Returns
        -------
        z : float
            The product of the elements in `x`.

        """
        # The identity for addition in NumPy is zero.
        # This corresponds to an identity of 1 for log operations, and this is
        # exactly the desired identity for multiplying probabilities.
        z = x.sum(axis=axis)
        return z

    def invert(self, x):
        """
        Returns the element-wise multiplicative inverse of x:  1/x.

        Parameters
        ----------
        x : NumPy array, shape (n,)
            The array to invert.

        Returns
        -------
        z : NumPy array, shape (n,)
            The inverted array.

        """
        z = -x
        return z

    def normalize(self, x, axis=None):
        """
        Returns a normalized version of x.

        Non-log equivalent operation: z[i] = x[i] / sum(x)

        If x is 2D and axis is None, then normalization is over all elements.
        Use axis=-1 to normalize each row of x.

        Parameters
        ----------
        x : NumPy array, shape (n,)
            The array to normalize.

        Returns
        -------
        z : NumPy array, shape (n,)
            The normalized array.

        """
        # The API way would be:  mult(x, invert( add_reduce(x) ))
        # We'll avoid some of those function calls.
        z = x - self.add_reduce(x, axis=axis)
        return z


cache = {
    'linear' : LinearOperations(),
    2: LogOperations(2),
    'e': LogOperations('e')
}
