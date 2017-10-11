"""
Tests for dit.pid.hcs.
"""

import pytest

import sys

from dit.pid.hcs import h_cs, PED_CS
from dit.pid.distributions import bivariates, trivariates
from dit import Distribution as D

def test_iccs1():
    """
    Test hcs on redundant distribution.
    """
    d = bivariates['redundant']
    red = h_cs(d, ((0,), (1,)), (2,))
    assert red == pytest.approx(1)

def test_iccs2():
    """
    Test hcs on synergistic distribution.
    """
    d = bivariates['synergy']
    red = h_cs(d, ((0,), (1,)), (2,))
    assert red == pytest.approx(0)

def test_iccs3():
    """
    Test hcs on two redundant bits.
    """
    d = D(['00','11'],[0.5]*2)
    red = h_cs(d, ((0,), (1,)))
    assert red == pytest.approx(1)

def test_iccs4():
    """
    Test hcs on two independent bits
    """
    d = D(['00','01','10','11'],[0.25]*4)
    red = h_cs(d, ((0,), (1,)))
    assert red == pytest.approx(0)

def test_iccs5():
    """
    Test hcs on two correlated bits
    """
    d = D(['00','01','10','11'],[0.4,0.1,0.1,0.4])
    red = h_cs(d, ((0,), (1,)))
    assert red == pytest.approx(0.542457524090110)

def test_iccs6():
    """
    Test hcs on triadic (required maxent)
    """
    triadic = D(['000', '111','022','133','202','313','220','331'], [1/8.0]*8)
    red = h_cs(triadic, ((0,), (1,), (2,)))
    assert red == pytest.approx(1)

def test_ped_cs1():
    """
    Test iccs on AND.
    """
    d = bivariates['and']
    pid = PED_CS(d)
    for atom in pid._lattice:
        if atom == ((0,), (1,), (2,)):
            assert pid[atom] == pytest.approx(0.10375937481971094)
        elif atom == ((0,),(1,)):
            assert pid[atom] == pytest.approx(-0.10375937481971098)
        elif atom in [((0,),(2,)), ((1,),(2,))]:
            assert pid[atom] == pytest.approx(0.35375937481971098)
        elif atom in [((0,),(1,2)), ((1,),(0,2))]:
            assert pid[atom] == pytest.approx(0.14624062518028902)
        elif atom in [((0,),), ((1,),)]:
            assert pid[atom] == pytest.approx(0.5)
        else:
            assert pid[atom] == pytest.approx(0.0)

def test_ped_cs2():
    """
    Test iccs on SUM. 
    """
    d = D(['000','011','101','112'], [1/4.0]*4)
    pid = PED_CS(d)
    for atom in pid._lattice:
        if atom in [((1,),(2,)), ((0,),(2,)), ((0,),(1,2)), ((1,),(0,2)), ((2,),(0,1))]:
            assert pid[atom] == pytest.approx(0.5)
        elif atom == ((0,1),(0,2),(1,2)):
            assert pid[atom] == pytest.approx(-0.5)
        else:
            assert pid[atom] == pytest.approx(0.0)

def test_ped_cs3():
    d = D(['00','01','10','11'],[0.4,0.1,0.1,0.4])
    ped = PED_CS(d)
    string = """\
+--------+--------+--------+
|  H_cs  |  H_r   |  H_d   |
+--------+--------+--------+
| {0:1}  | 1.7219 | 0.2644 |
|  {0}   | 1.0000 | 0.4575 |
|  {1}   | 1.0000 | 0.4575 |
| {0}{1} | 0.5425 | 0.5425 |
+--------+--------+--------+"""
    assert str(ped) == string
