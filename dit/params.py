"""
    Defines `dit` configuration parameters.
"""

import warnings

from .exceptions import InvalidBase

__all__ = ['ditParams', 'reset_params']

def validate_boolean(b):
    """Convert b to a boolean or raise a ValueError."""
    try:
        b = b.lower()
    except AttributeError:
        pass
    if b in ('t', 'y', 'yes', 'on', 'true', '1', 1, True):
        return True
    elif b in ('f', 'n', 'no', 'off', 'false', '0', 0, False):
        return False
    else:
        raise ValueError('Could not convert "%s" to boolean' % b)

def validate_float(s):
    """Convert s to float or raise a ValueError."""
    try:
        return float(s)
    except TypeError:
        raise ValueError('Could not convert "%s" to float' % s)

def validate_base(b):
    """Convert s to a valid base or raise InvalidBase."""

    # String bases.
    if b == 'e' or b == 'linear':
        return b
    else:
        try:
            b + '' # pylint disable=pointless-statement
            raise InvalidBase(b)
        except TypeError:
            pass

    # Numerical bases.
    if b <= 0 or b == 1:
        raise InvalidBase(b)
    else:
        return b

def validate_choice(s, choices):
    try:
        s = s.lower()
    except AttributeError:
        pass
    if s not in choices:
        raise ValueError("%s is an invalid specification." % s)
    else:
        return s

def validate_text(s):
    choices = ['ascii', 'linechar']
    return validate_choice(s, choices)

class DITParams(dict):
    """
    A dictionary including validation, representing dit parameters.

    """
    # A dictionary relating params to validators.
    def __init__(self, *args, **kwargs):
        self.validate = dict([(key, converter) for key, (_, converter)
                              in defaultParams.items()])
        dict.__init__(self, *args, **kwargs)

    def __setitem__(self, key, val):
        if key in _deprecated_map.keys():
            alt = _deprecated_map[key]
            msg = "%r is deprecated. Use %r instead."
            warnings.warn(msg % (key, alt), DeprecationWarning, stacklevel=2)

        try:
            cval = self.validate[key](val)
        except KeyError:
            msg = '%r is not a valid dit parameter. ' % key
            msg += 'See ditParams.keys() for a list of valid parameters.'
            raise KeyError(msg)

        dict.__setitem__(self, key, cval)

    def __getitem__(self, key):
        if key in _deprecated_map.keys():
            alt = _deprecated_map[key]
            msg = "%r is deprecated. Use %r instead."
            warnings.warn(msg % (key, alt), DeprecationWarning, stacklevel=2)
        else:
            alt = key

        return dict.__getitem__(self, alt)

def reset_params():
    """Restore rcParams to defaults from when dit was originally imported."""
    ditParams.update(ditParamsDefault)

def set_params():
    """Return the default params, after updating from the .ditrc file."""
    ## Currently, we don't support a .ditrc file.
    ## So we just return the default parameters.
    ret = DITParams([(key, tup[0]) for key, tup in defaultParams.items()])
    return ret

## key -> (value, validator)
## TODO:  key -> (value, validator, info_string)
defaultParams = {'rtol': (1e-7, validate_float),
                 'atol': (1e-9, validate_float),
                 'logs': (True, validate_boolean),
                 'base': (2, validate_base),
                 'text.usetex': (False, validate_boolean),
                 'text.font': ('ascii', validate_text),
                 'print.exact': (False, validate_boolean),
                 'repr.print': (False, validate_boolean),
                }

## Dictionary relating deprecated parameter names to new parameter names.
_deprecated_map = {}

### This is what will be used by every dit instance.
ditParams = set_params()

ditParamsDefault = DITParams([(key, tup[0]) \
                             for key, tup in defaultParams.items()])
