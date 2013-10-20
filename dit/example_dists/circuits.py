from __future__ import division

from dit import Distribution

def Unq():
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
    pmf = [1/2, 1/2]
    outcomes = ['000', '111']
    d = Distribution(outcomes, pmf)
    return d

def Xor():
    pmf = [1/4] * 4
    outcomes = ['000', '011', '101', '110']
    d = Distribution(outcomes, pmf)
    return d

def RdnXor():
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
    pmf = [.499, .5, .001]
    outcomes = [('0', '0', '0'), ('1', '1', '1'), ('0', '1', '0')]
    d = Distribution(outcomes, pmf)
    return d

def Subtle():
    pmf = [1/3] * 3
    outcomes = [('0', '0', '00'), ('1', '1', '11'), ('0', '1', '01')]
    d = Distribution(outcomes, pmf)
    return d
