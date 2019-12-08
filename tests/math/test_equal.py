# -*- coding: utf-8 -*-

"""
Tests for dit.math.equal.
"""

import warnings

from numpy import inf, nan

from dit.math.equal import allclose, close


def test_close1():
    x = 0
    y = 1
    with warnings.catch_warnings():
        assert not close(x, y)


def test_close2():
    x = 0
    y = inf
    with warnings.catch_warnings():
        assert not close(x, y)


def test_close3():
    x = -inf
    y = inf
    with warnings.catch_warnings():
        assert not close(x, y)


def test_close4():
    x = inf
    y = nan
    with warnings.catch_warnings():
        assert not close(x, y)


def test_close5():
    x = 0
    y = nan
    with warnings.catch_warnings():
        assert not close(x, y)


def test_close6():
    x = 1
    y = 1
    with warnings.catch_warnings():
        assert close(x, y)


def test_close7():
    x = 0.33333333333333
    y = 1/3
    with warnings.catch_warnings():
        assert close(x, y)


def test_close8():
    x = inf
    y = inf
    with warnings.catch_warnings():
        assert close(x, y)


def test_allclose1():
    x = [0, 0, -inf, inf, 0]
    y = [1, inf, inf, nan, nan]
    with warnings.catch_warnings():
        assert not allclose(x, y)


def test_allclose2():
    x = [1, 0.333333333333333, inf]
    y = [1, 1/3, inf]
    with warnings.catch_warnings():
        assert allclose(x, y)
