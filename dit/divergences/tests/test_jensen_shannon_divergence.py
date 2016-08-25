"""
Tests for dit.divergences.jensen_shannon_divergence.
"""
import pytest

from dit import Distribution
from dit.exceptions import ditException
from dit.divergences.jensen_shannon_divergence import (
    jensen_shannon_divergence as JSD,
    jensen_shannon_divergence_pmf as JSD_pmf
)

def test_jsd0():
    """ Test the JSD of a distribution but with weights misspecified."""
    d1 = Distribution("AB", [0.5, 0.5])
    with pytest.raises(ditException):
        JSD(d1, d1)

def test_jsd1():
    """ Test the JSD of a distribution with itself """
    d1 = Distribution("AB", [0.5, 0.5])
    jsd = JSD([d1, d1])
    assert jsd == pytest.approx(0)

def test_jsd2():
    """ Test the JSD with half-overlapping distributions """
    d1 = Distribution("AB", [0.5, 0.5])
    d2 = Distribution("BC", [0.5, 0.5])
    jsd = JSD([d1, d2])
    assert jsd == pytest.approx(0.5)

def test_jsd3():
    """ Test the JSD with disjoint distributions """
    d1 = Distribution("AB", [0.5, 0.5])
    d2 = Distribution("CD", [0.5, 0.5])
    jsd = JSD([d1, d2])
    assert jsd == pytest.approx(1.0)

def test_jsd4():
    """ Test the JSD with half-overlapping distributions with weights """
    d1 = Distribution("AB", [0.5, 0.5])
    d2 = Distribution("BC", [0.5, 0.5])
    jsd = JSD([d1, d2], [0.25, 0.75])
    assert jsd == pytest.approx(0.40563906222956625)

def test_jsd5():
    """ Test that JSD fails when more weights than dists are given """
    d1 = Distribution("AB", [0.5, 0.5])
    d2 = Distribution("BC", [0.5, 0.5])
    with pytest.raises(ditException):
        JSD([d1, d2], [0.1, 0.6, 0.3])

def test_jsd_pmf1():
    """ Test the JSD of a distribution with itself """
    d1 = [0.5, 0.5]
    jsd = JSD_pmf([d1, d1])
    assert jsd == pytest.approx(0)

def test_jsd_pmf2():
    """ Test the JSD with half-overlapping distributions """
    d1 = [0.5, 0.5, 0.0]
    d2 = [0.0, 0.5, 0.5]
    jsd = JSD_pmf([d1, d2])
    assert jsd == pytest.approx(0.5)

def test_jsd_pmf3():
    """ Test the JSD with disjoint distributions """
    d1 = [0.5, 0.5, 0.0, 0.0]
    d2 = [0.0, 0.0, 0.5, 0.5]
    jsd = JSD_pmf([d1, d2])
    assert jsd == pytest.approx(1.0)

def test_jsd_pmf4():
    """ Test the JSD with half-overlapping distributions with weights """
    d1 = [0.5, 0.5, 0.0]
    d2 = [0.0, 0.5, 0.5]
    jsd = JSD_pmf([d1, d2], [0.25, 0.75])
    assert jsd == pytest.approx(0.40563906222956625)

def test_jsd_pmf5():
    """ Test that JSD fails when more weights than dists are given """
    d1 = [0.5, 0.5, 0.0]
    d2 = [0.0, 0.5, 0.5]
    with pytest.raises(ditException):
        JSD_pmf([d1, d2], [0.1, 0.6, 0.2, 0.1])
