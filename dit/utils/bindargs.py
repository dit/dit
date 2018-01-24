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

import sys
import inspect

from inspect import getcallargs


try:
    from inspect import getfullargspec
except ImportError:
    # Python 2.X
    from collections import namedtuple
    from inspect import getargspec

    FullArgSpec = namedtuple('FullArgSpec',
    'args, varargs, varkw, defaults, kwonlyargs, kwonlydefaults, annotations')

    def getfullargspec(f):
        args, varargs, varkw, defaults = getargspec(f)
        kwonlyargs = []
        kwonlydefaults = None
        annotations = getattr(f, '__annotations__', {})
        return FullArgSpec(args, varargs, varkw, defaults,
                           kwonlyargs, kwonlydefaults, annotations)


def bindcallargs_leq32(_fUnCtIoN_, *args, **kwargs):
    """Binds arguments and keyword arguments to a function or method.

    Returns a tuple (bargs, bkwargs) suitable for manipulation and passing
    to the specified function.

    `bargs` consists of the bound args, varargs, and kwonlyargs from
    getfullargspec. `bkwargs` consists of the bound varkw from getfullargspec.
    Both can be used in a call to the specified function.  Any default
    parameter values are included in the output.

    Examples
    --------
    >>> def func(a, b=3, *args, **kwargs):
    ...    pass

    >>> bindcallargs(func, 5)
    ((5, 3), {})

    >>> bindcallargs(func, 5, 4, 3, 2, 1, hello='there')
    ((5, 4, 3, 2, 1), {'hello': 'there'})

    >>> args, kwargs = bindcallargs(func, 5)
    >>> kwargs['b'] = 5 # overwrite default value for b
    >>> func(*args, **kwargs)

    """
    # It is necessary to choose an unlikely variable name for the function.
    # The reason is that any kwarg by the same name will cause a TypeError
    # due to multiple values being passed for that argument name.
    func = _fUnCtIoN_

    callargs = getcallargs(func, *args, **kwargs)
    spec = getfullargspec(func)

    # Construct all args and varargs and use them in bargs
    bargs = [callargs[arg] for arg in spec.args]
    if spec.varargs is not None:
        bargs.extend(callargs[spec.varargs])
    bargs = tuple(bargs)

    # Start with kwonlyargs.
    bkwargs = dict((kwonlyarg, callargs[kwonlyarg]) for kwonlyarg in spec.kwonlyargs)
    # Add in kwonlydefaults for unspecified kwonlyargs only.
    if spec.kwonlydefaults is not None:
        bkwargs.update(dict([(k, v) for k, v in spec.kwonlydefaults.items()
                             if k not in bkwargs]))
    # Add in varkw.
    if spec.varkw is not None:
        bkwargs.update(callargs[spec.varkw])

    return bargs, bkwargs


def bindcallargs_geq33(_fUnCtIoN_, *args, **kwargs):
    # Should match functionality of bindcallargs_32 for Python > 3.3.
    sig = inspect.signature(_fUnCtIoN_)
    ba = sig.bind(*args, **kwargs)
    # Add in all default values
    for param in sig.parameters.values():
        if param.name not in ba.arguments:
            ba.arguments[param.name] = param.default
    return ba.args, ba.kwargs

if sys.version_info[0:2] < (3,3): # pragma: no cover
    bindcallargs = bindcallargs_leq32
else: # pragma: no cover
    bindcallargs = bindcallargs_geq33
