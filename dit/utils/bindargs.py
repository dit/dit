# -*- coding: utf-8 -*-

"""
Provides usable args and kwargs from inspect.getcallargs.

For Python 3.3 and above, this module is unnecessary and can be achieved using
features from PEP 362:

    http://www.python.org/dev/peps/pep-0362/

For example, to override a parameter of some function:

    >>> import inspect
    >>> def func(a, b=1, c=2, d=3):
    ...    return a, b, c, d
    ...
    >>> def override_c(*args, **kwargs):
    ...    sig = inspect.signature(override)
    ...    ba = sig.bind(*args, **kwargs)
    ...    ba['c'] = 10
    ...    return func(*ba.args, *ba.kwargs)
    ...
    >>> override_c(0, c=3)
    (0, 1, 10, 3)

Also useful:

    http://www.python.org/dev/peps/pep-3102/

"""

import inspect


def bindcallargs(_fUnCtIoN_, *args, **kwargs):
    # Should match functionality of bindcallargs_32 for Python > 3.3.
    sig = inspect.signature(_fUnCtIoN_)
    ba = sig.bind(*args, **kwargs)
    # Add in all default values
    for param in sig.parameters.values():
        if param.name not in ba.arguments:
            ba.arguments[param.name] = param.default
    return ba.args, ba.kwargs
