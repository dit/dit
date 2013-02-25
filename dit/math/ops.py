#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Classes to contextualize math operations in log vs linear space.

"""

import numpy as np

from ..exceptions import InvalidBase
from .equal import close

__all__ = [
    'LinearOperations',
    'LogOperations'
]

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

    acceptable_strings = set(['linear', 'e'])
    if is_string_like(b) and b not in acceptable_strings:
        raise InvalidBase(msg=b)

    if b == 'linear':
        log = lambda x : x
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
        def log(x):
            return np.log(x) / Z

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

    one = None
    zero = None
    base = None
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

    def add(self, x, y):
        raise NotImplementedError
    def add_inplace(self, x, y):
        raise NotImplementedError
    def add_reduce(self, x):
        raise NotImplementedError
    def mult(self, x, y):
        raise NotImplementedError
    def mult_inplace(self, x, y):
        raise NotImplementedError
    def mult_reduce(self, x):
        raise NotImplementedError
    def invert(self, x):
        raise NotImplementedError

class LinearOperations(Operations):

    one = 1
    zero = 0
    base = 'linear'
    log = log_func(base)

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

    def add_reduce(self, x):
        """
        Performs an `addition' reduction on `x`.

        Assumption: y >= 0.

        Operation: z = \sum_i x[i]

        Returns
        -------
        z : float
            The summation of the elements in `x`.

        """
        z = x.sum()
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

    def mult_reduce(self, x):
        """
        Performs an `multiplication' reduction on `x`.

        Operation: z = \prod_i x[i]

        Returns
        -------
        z : float
            The product of the elements in `x`.

        """
        # The identity for addition in NumPy is zero.
        # This corresponds to an identity of 1 for log operations, and this is
        # exactly the desired identity for multiplying probabilities.
        z = np.prod(x)
        return z

    def invert(self, x):
        """
        Returns the element-wise multiplicative inverse of x.

        Operation: z = 1/x

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

class LogOperations(Operations):

    one = None
    zero = None
    base = None
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
        self.log = log_func(base)
        # Note: When base < 1, zero == +inf. When base > 1, zero == -inf.
        self.one = self.log(1)
        self.zero = self.log(0)

    def add(self, x, y):
        """
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
        # To preserve numerical accuracy, we must make use of a logaddexp
        # function.  These functions only exist in Numpy for base-e and base-2.
        # For all other bases, we must convert and then convert back.

        base = self.base
        if base == 2:
            z = np.logaddexp2(x,y)
        elif base == 'e' or close(base, np.e):
            z = np.logaddexp2(x,y)
        else:
            # Convert log_b probabilities to log_2 probabilities.
            x2 = x * np.log2(base)
            y2 = y * np.log2(base)
            z = np.logaddexp2(x2, y2)
            # Convert log_2 probabilities to log_b probabilities.
            z *= self.log(2)
        return z

    def add_inplace(self, x, y):
        """
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
        base = self.base
        if base == 2:
            np.logaddexp2(x,y,x)
        elif base == 'e' or close(base, np.e):
            np.logaddexp(x,y,x)
        else:
            x *= np.log2(base)
            y2 = y * np.log2(base)
            np.logaddexp2(x, y2, x)
            x *= self.log(2)
        return x

    def add_reduce(self, x):
        """
        Performs an `addition' reduction on `x`.

        Assumption: y <= 0.

        Returns
        -------
        z : float
            The summation of the elements in `x`.

        """
        if len(x) == 0:
            # Since logaddexp.identity is None, we must handle it separately.
            z = self.zero
        else:
            # Note, we are converting to a NumPy array, if necessary.
            base = self.base
            if base == 2:
                z = np.logaddexp2.reduce(x, dtype=float)
            elif base == 'e' or close(base, np.e):
                z = np.logaddexp.reduce(x, dtype=float)
            else:
                # Change the base-2, add, and then convert back.
                x2 = x * np.log2(base)
                z = np.logaddexp2.reduce(x2, dtype=float)
                z /= np.log2(base)

        return z

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

    def mult_reduce(self, x):
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
        z = x.sum()
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

