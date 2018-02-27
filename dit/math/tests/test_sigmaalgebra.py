"""
Tests for dit.math.sigmaalgebra.
"""
import pytest

from dit.math.sigmaalgebra import is_sigma_algebra, is_sigma_algebra__brute, atom_set


sa = frozenset([frozenset([]), frozenset(['a']), frozenset(['b']), frozenset(['c']), 
                frozenset(['a', 'b']), frozenset(['a', 'c']), frozenset(['b', 'c']),
                frozenset(['a', 'b', 'c'])])
not_sa1 = frozenset([frozenset([]), frozenset('c'), frozenset(['a', 'b'])])
not_sa2 = frozenset([frozenset([]), frozenset(['a']), frozenset(['b']), frozenset(['c']), 
                frozenset(['a', 'b']), frozenset(['a', 'c']), frozenset(['b', 'c'])])


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
    assert atoms == frozenset([frozenset('c')])


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
