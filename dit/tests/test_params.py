"""
Tests for dit.params.
"""

import pytest

from numpy import inf, nan

from dit.exceptions import InvalidBase
from dit.params import (validate_boolean, validate_float, validate_base,
                        validate_choice, validate_text)

def test_validate_boolean1():
    good = ['t', 'T', 'y', 'Y', 'yes', 'Yes', 'YES', 'on', 'On', 'ON', 'true',
            'True', 'TRUE', '1', 1, True]
    for value in good:
        assert validate_boolean(value)

def test_validate_boolean2():
    bad = ['f', 'F', 'n', 'N', 'no', 'No', 'NO', 'off', 'Off', 'OFF', 'false',
           'False', 'FALSE', '0', 0, False]
    for value in bad:
        assert not validate_boolean(value)

def test_validate_boolean3():
    not_valid = ['maybe', 2, 0.5]
    for value in not_valid:
        with pytest.raises(ValueError):
            validate_boolean(value)

def test_validate_float1():
    good = [0.5, 0, '0.123', 2, inf, nan]
    for value in good:
        val1 = validate_float(value)
        val2 = float(value)
        if not val1 is val2:
            assert val1 == pytest.approx(val2)

def test_validate_float2():
    bad = ['pants', float, []]
    for value in bad:
        with pytest.raises(ValueError):
            validate_float(value)

def test_validate_base1():
    good = ['e', 'linear', 0.5, 1.5, 2, 10]
    for value in good:
        assert validate_base(value) == value

def test_validate_base2():
    bad = ['nope', -0.5, 0, 1]
    for value in bad:
        with pytest.raises(InvalidBase):
            validate_base(value)

def test_validate_choice1():
    choices = [0, 1, '2', 3, 'four']
    for choice in choices:
        assert validate_choice(choice, choices) == choice

def test_validate_choice2():
    choices = [0, 1, 2]
    bads = ['0', '1', '2']
    for choice in bads:
        with pytest.raises(ValueError):
            validate_choice(choice, choices)

def test_validate_text1():
    for s in ['ascii', 'linechar']:
        assert validate_text(s) == s

def test_validate_text2():
    for s in ['no', 3]:
        with pytest.raises(ValueError):
            validate_text(s)
