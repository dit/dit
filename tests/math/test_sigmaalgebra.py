# -*- coding: utf-8 -*-

"""
Tests for dit.math.sigmaalgebra.
"""

import pytest

import numpy as np

from dit.math.sigmaalgebra import sets2matrix, is_sigma_algebra, is_sigma_algebra__brute, atom_set


sa = frozenset([frozenset([]), frozenset(['a']), frozenset(['b']), frozenset(['c']),
                frozenset(['a', 'b']), frozenset(['a', 'c']), frozenset(['b', 'c']),
                frozenset(['a', 'b', 'c'])])
not_sa1 = frozenset([frozenset([]), frozenset('c'), frozenset('b'), frozenset(['a', 'b'])])
not_sa2 = frozenset([frozenset([]), frozenset(['a']), frozenset(['b']), frozenset(['c']),
                frozenset(['a', 'b']), frozenset(['a', 'c']), frozenset(['b', 'c'])])
not_sa3 = frozenset([frozenset([]), frozenset('c'), frozenset(['a', 'b', 'c'])])


def test_s2m1():
    """
    Test matrix construction.
    """
    m, _ = sets2matrix(sa)
    true_m = set([(0, 1, 0),
                  (0, 0, 1),
                  (0, 0, 0),
                  (1, 0, 1),
                  (1, 0, 0),
                  (1, 1, 1),
                  (1, 1, 0),
                  (0, 1, 1)])
    assert set(map(tuple, m.tolist())) == true_m


def test_s2m2():
    """
    Test matrix construction.
    """
    m, _ = sets2matrix(sa, X=['a', 'b', 'c'])
    true_m = set([(0, 1, 0),
                  (0, 0, 1),
                  (0, 0, 0),
                  (1, 0, 1),
                  (1, 0, 0),
                  (1, 1, 1),
                  (1, 1, 0),
                  (0, 1, 1)])
    assert set(map(tuple, m.tolist())) == true_m


def test_s2m3():
    """
    Test matrix construction.
    """
    with pytest.raises(Exception):
        sets2matrix(sa, X=['a'])


def test_isa1():
    """
    Test SA.
    """
    assert is_sigma_algebra(sa)


def test_isa2():
    """
    Test not SA.
    """

    assert not is_sigma_algebra(not_sa1)


def test_isa3():
    """
    Test not SA.
    """

    assert not is_sigma_algebra(not_sa2)


def test_isa4():
    """
    Test SA brute.
    """
    assert is_sigma_algebra__brute(sa)


def test_isa5():
    """
    Test not SA brute.
    """
    assert not is_sigma_algebra__brute(not_sa1)


def test_isa6():
    """
    Test not SA brute.
    """
    assert not is_sigma_algebra__brute(not_sa2, X=['a', 'b', 'c'])


def test_isa7():
    """
    Test not SA brute.
    """
    assert not is_sigma_algebra__brute(not_sa3)


def test_atom_set1():
    """
    Find the atoms.
    """
    atoms = atom_set(sa)
    assert atoms == frozenset([frozenset('a'), frozenset('b'), frozenset('c')])


def test_atom_set2():
    """
    Find the atoms.
    """
    atoms = atom_set(not_sa1, method=1)
    assert atoms == frozenset([frozenset('c'), frozenset('b')])


def test_atom_set3():
    """
    Find the atoms.
    """
    atoms = atom_set(not_sa2, method=1)
    assert atoms == frozenset([frozenset('a'), frozenset('b'), frozenset('c')])


def test_atom_set4():
    """
    Find the atoms.
    """
    atoms = atom_set(sa, method=1)
    assert atoms == frozenset([frozenset('a'), frozenset('b'), frozenset('c')])

def test_atom_set5():
    """
    Test failure.
    """
    with pytest.raises(Exception):
        atom_set('pants')
