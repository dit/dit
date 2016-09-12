"""
Tests for dit.multivariate.functional_common_information.
"""

from __future__ import division

import pytest

from dit import Distribution, random_distribution
from dit.multivariate import (functional_common_information as F,
                              dual_total_correlation as B,
                              mss_common_information as M
                             )

def test_fci1():
    """
    Test known values.
    """
    d = Distribution(['000', '011', '101', '110'], [1/4]*4)
    assert F(d) == pytest.approx(2.0)
    assert F(d, [[0], [1]]) == pytest.approx(0.0)
    assert F(d, [[0], [1]], [2]) == pytest.approx(1.0)

def test_fci2():
    """
    Test known values w/ rv names.
    """
    d = Distribution(['000', '011', '101', '110'], [1/4]*4)
    d.set_rv_names('XYZ')
    assert F(d) == pytest.approx(2.0)
    assert F(d, ['X', 'Y']) == pytest.approx(0.0)
    assert F(d, ['X', 'Y'], 'Z') == pytest.approx(1.0)

def test_fci3():
    """
    Test against known values
    """
    outcomes = ['000',
                'a00',
                '00c',
                'a0c',
                '011',
                'a11',
                '101',
                'b01',
                '01d',
                'a1d',
                '10d',
                'b0d',
                '110',
                'b10',
                '11c',
                'b1c',]
    pmf = [1/16]*16
    d = Distribution(outcomes, pmf)
    assert F(d) == pytest.approx(2.0)
