"""
Utilities for adding units to information values.
"""

from functools import wraps

import numpy as np

from .. import ditParams


try:

    import pint

    ureg = pint.UnitRegistry()
    ureg.define('nat = {} * bit'.format(np.log2(np.e)))
    ureg.define('dit = {} * bit'.format(np.log2(10)))


    def unitful(f):
        """
        Add units to the functions return value.

        Parameters
        ----------
        f : func
            The function to wrap.

        Returns
        -------
        wrapper : func
            A function which optionally adds units to return values.
        """
        @wraps(f)
        def wrapper(*args, **kwargs):
            value = f(*args, **kwargs)

            if ditParams['units']:
                value *= ureg.bit

            return value

        return wrapper


except ImportError:

    def unitful(f):
        """
        Non-op.

        Parameters
        ----------
        f : func
            The function to wrap.

        Returns
        -------
        wrapper : func
            A function which does exactly the same as f.
        """
        @wraps(f)
        def wrapper(*args, **kwargs):
            value = f(*args, **kwargs)
            return value

        return wrapper
