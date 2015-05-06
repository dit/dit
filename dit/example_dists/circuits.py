"""
Distributions based on circuits with independent inputs.
"""

from __future__ import division

from dit import Distribution
import dit

def Unq():
    """
    A distribution with unique information.
    """
    pmf = [1/4] * 4
    outcomes = [
        ('a', 'b', 'ab'),
        ('a', 'B', 'aB'),
        ('A', 'b', 'Ab'),
        ('A', 'B', 'AB')
    ]

    d = Distribution(outcomes, pmf)
    return d

def Rdn():
    """
    A distribution with redundant information.
    """
    pmf = [1/2, 1/2]
    outcomes = ['000', '111']
    d = Distribution(outcomes, pmf)
    return d

def Xor():
    """
    A distribution with synergistic information, [0] xor [1] = [2]
    """
    pmf = [1/4] * 4
    outcomes = ['000', '011', '101', '110']
    d = Distribution(outcomes, pmf)
    return d

def And(k=2):
    """
    [0] and [1] = [2]
    """
    d = dit.uniform_distribution(k, ['01'])
    d = dit.distconst.modify_outcomes(d, lambda x: ''.join(x))
    d = dit.insert_rvf(d, lambda x: '1' if all(map(bool, map(int, x))) else '0')
    return d

def Or(k=2):
    """
    [0] or [1] = [2]
    """
    d = dit.uniform_distribution(k, ['01'])
    d = dit.distconst.modify_outcomes(d, lambda x: ''.join(x))
    d = dit.insert_rvf(d, lambda x: '1' if any(map(bool, map(int, x))) else '0')
    return d

def RdnXor():
    """
    Concatenation of Rdn() and Xor(). Distribution has both redundant and
    synergistic information.
    """
    pmf = [1/8] * 8
    outcomes = [
        ('r0', 'r0', 'r0'),
        ('r0', 'r1', 'r1'),
        ('r1', 'r0', 'r1'),
        ('r1', 'r1', 'r0'),
        ('R0', 'R0', 'R0'),
        ('R0', 'R1', 'R1'),
        ('R1', 'R0', 'R1'),
        ('R1', 'R1', 'R0'),
    ]

    d = Distribution(outcomes, pmf)
    return d

def ImperfectRdn():
    """
    Like Rdn() with a small off-term.
    """
    pmf = [.499, .5, .001]
    outcomes = [('0', '0', '0'), ('1', '1', '1'), ('0', '1', '0')]
    d = Distribution(outcomes, pmf)
    return d

def Subtle():
    """
    The Subtle distribution.
    """
    pmf = [1/3] * 3
    outcomes = [('0', '0', '00'), ('1', '1', '11'), ('0', '1', '01')]
    d = Distribution(outcomes, pmf)
    return d
