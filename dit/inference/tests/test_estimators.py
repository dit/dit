"""
Tests for dit.inference.estimators.
"""

import pytest

from dit.inference import entropy_0, entropy_1, entropy_2

def test_entropy_0():
    data = [0]*7 + [1]*3
    h0 = entropy_0(data, 1)
    assert h0 == pytest.approx(0.8812908992306927)

def test_entropy_0():
    data = [0]*7 + [1]*3
    h0 = entropy_0(data, 2)
    assert h0 == pytest.approx(1.2243944454059861)

def test_entropy_1():
    data = [0]*7 + [1]*3
    h1 = entropy_1(data, 1)
    assert h1 == pytest.approx(0.95790370730770369)

def test_entropy_1():
    data = [0]*7 + [1]*3
    h1 = entropy_1(data, 2)
    assert h1 == pytest.approx(1.4043376727383439)

def test_entropy_2():
    data = [0]*7 + [1]*3
    h2 = entropy_2(data, 1)
    assert h2 == pytest.approx(1.1187360918572902)

def test_entropy_2():
    data = [0]*7 + [1]*3
    h2 = entropy_2(data, 2)
    assert h2 == pytest.approx(1.3303313645210046)
