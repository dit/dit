from __future__ import division
from __future__ import print_function

import pytest

from dit import Distribution, ScalarDistribution
from dit.distribution import BaseDistribution
from dit.exceptions import ditException, InvalidNormalization

def test_dist_iter1():
    outcomes = ['00', '01', '10', '11']
    pmf = [1/4]*4
    d = Distribution(outcomes, pmf)
    for o in d:
        assert o in outcomes
    for o1, o2 in zip(d, outcomes):
        assert o1 == o2


def test_dist_iter2():
    outcomes = ['00', '01', '10', '11']
    pmf = [1/4]*4
    d = Distribution(outcomes, pmf)
    for o in reversed(d):
        assert o in outcomes
    for o1, o2 in zip(reversed(d), reversed(outcomes)):
        assert o1 == o2


def test_numerical():
    outcomes = ['00', '01', '10', '11']
    pmf = [1/4]*4
    d = Distribution(outcomes, pmf)
    assert d.is_numerical()


@pytest.mark.parametrize('i', range(10))
def test_rand(i):
    outcomes = ['00', '01', '10', '11']
    pmf = [1/4]*4
    d = Distribution(outcomes, pmf)
    assert d.rand() in outcomes


def test_to_dict():
    outcomes = ['00', '01', '10', '11']
    pmf = [1/4]*4
    d = Distribution(outcomes, pmf)
    dd = d.to_dict()
    for o, p in dd.items():
        assert d[o] == pytest.approx(p)

def test_validate1():
    outcomes = ['00', '01', '10', '11']
    pmf = [1/4]*4
    d = Distribution(outcomes, pmf)
    assert d.validate()
    assert BaseDistribution.validate(d)

def test_validate2():
    outcomes = ['00', '01', '10', '11']
    pmf = [1/4]*4
    d = Distribution(outcomes, pmf)
    d['00'] = 0
    with pytest.raises(InvalidNormalization):
        d.validate()
    with pytest.raises(InvalidNormalization):
        BaseDistribution.validate(d)

def test_zipped1():
    outcomes = ['00', '01', '10', '11']
    pmf = [1/4]*4
    d = Distribution(outcomes, pmf)
    zipped = d.zipped(mode='pants')
    with pytest.raises(ditException):
        list(zipped)


def test_to_string1():
    # Basic
    outcomes = ['00', '01', '10', '11']
    pmf = [1/4]*4
    d = Distribution(outcomes, pmf)
    s = d.to_string()
    s_ = """Class:          Distribution
Alphabet:       ('0', '1') for all rvs
Base:           linear
Outcome Class:  str
Outcome Length: 2
RV Names:       None

x    p(x)
00   0.25
01   0.25
10   0.25
11   0.25"""
    assert s == s_


def test_to_string2():
    # Test with exact.
    outcomes = ['00', '01', '10', '11']
    pmf = [1/4]*4
    d = Distribution(outcomes, pmf)
    s = d.to_string(exact=True)
    s_ = """Class:          Distribution
Alphabet:       ('0', '1') for all rvs
Base:           linear
Outcome Class:  str
Outcome Length: 2
RV Names:       None

x    p(x)
00   1/4
01   1/4
10   1/4
11   1/4"""
    assert s == s_

def test_to_string3():
    # Test printing
    outcomes = ['00', '01', '10', '11']
    pmf = [1/4]*4
    d = Distribution(outcomes, pmf)
    s_ = """Class:          Distribution
Alphabet:       ('0', '1') for all rvs
Base:           linear
Outcome Class:  str
Outcome Length: 2
RV Names:       None

x    p(x)
00   0.25
01   0.25
10   0.25
11   0.25"""

    # context manager?
    import sys
    from six import StringIO
    sio = StringIO()
    try:
        old = sys.stdout
        sys.stdout = sio
        print(d, end='')
    finally:
        sys.stdout = old
    sio.seek(0)
    s = sio.read()
    assert s == s_

def test_to_string4():
    # Basic with marginal
    outcomes = ['00', '01', '10', '11']
    pmf = [1/4]*4
    d = Distribution(outcomes, pmf)
    d = d.marginal([0])
    s = d.to_string()
    s_ = """Class:          Distribution
Alphabet:       ('0', '1') for all rvs
Base:           linear
Outcome Class:  str
Outcome Length: 1
RV Names:       None

x   p(x)
0   0.5
1   0.5"""
    assert s == s_

def test_to_string5():
    # Basic with marginal and mask
    outcomes = ['00', '01', '10', '11']
    pmf = [1/4]*4
    d = Distribution(outcomes, pmf)
    d = d.marginal([0])
    s = d.to_string(show_mask=True)
    s_ = """Class:          Distribution
Alphabet:       ('0', '1') for all rvs
Base:           linear
Outcome Class:  str
Outcome Length: 1 (mask: 2)
RV Names:       None

x    p(x)
0*   0.5
1*   0.5"""
    assert s == s_

def test_to_string6():
    # Basic
    outcomes = ['00', '01', '10', '11']
    pmf = [1/4]*4
    d = Distribution(outcomes, pmf)
    s = d.to_string(digits=1)
    s_ = """Class:          Distribution
Alphabet:       ('0', '1') for all rvs
Base:           linear
Outcome Class:  str
Outcome Length: 2
RV Names:       None

x    p(x)
00   0.2
01   0.2
10   0.2
11   0.2"""
    assert s == s_

def test_to_string7():
    # Basic
    outcomes = ['00', '01', '10', '11']
    pmf = [1/4]*4
    d = ScalarDistribution(outcomes, pmf)
    s = d.to_string()
    s_ = """Class:    ScalarDistribution
Alphabet: ('00', '01', '10', '11')
Base:     linear

x    p(x)
00   0.25
01   0.25
10   0.25
11   0.25"""
    assert s == s_

def test_to_string8():
    outcomes = ['00', '01', '10', '11']
    pmf = [1/4]*4
    d = Distribution(outcomes, pmf)
    d = d.marginal([0])
    s = d.to_string(show_mask='!')
    s_ = """Class:          Distribution
Alphabet:       ('0', '1') for all rvs
Base:           linear
Outcome Class:  str
Outcome Length: 1 (mask: 2)
RV Names:       None

x    p(x)
0!   0.5
1!   0.5"""
    assert s == s_

def test_to_string9():
    # Basic
    outcomes = ['00', '01', '10', '11']
    pmf = [1/4]*4
    d = Distribution(outcomes, pmf)
    d.set_base(2)
    s = d.to_string()
    s_ = """Class:          Distribution
Alphabet:       ('0', '1') for all rvs
Base:           2
Outcome Class:  str
Outcome Length: 2
RV Names:       None

x    log p(x)
00   -2.0
01   -2.0
10   -2.0
11   -2.0"""
    assert s == s_

def test_to_string10():
    # Basic
    d = ScalarDistribution([], sample_space=[0, 1], validate=False)
    s = d.to_string()
    s_ = """Class:    ScalarDistribution
Alphabet: (0, 1)
Base:     2

x   log p(x)"""
    assert s == s_

def test_prepare_string1():
    # Basic
    outcomes = ['00', '01', '10', '11']
    pmf = [1/4]*4
    d = ScalarDistribution(outcomes, pmf)
    from dit.distribution import prepare_string
    with pytest.raises(ditException):
        prepare_string(d, show_mask=True)

def test_prepare_string2():
    # Basic
    outcomes = ['00', '01', '10', '11']
    pmf = [1/4]*4
    d = ScalarDistribution(outcomes, pmf)
    from dit.distribution import prepare_string
    with pytest.raises(ditException):
        prepare_string(d, str_outcomes=True)

def test_prepare_string3():
    outcomes = [(0, 0), (0, 1), (1, 0), (1, 1)]
    pmf = [1/4]*4
    d = Distribution(outcomes, pmf)
    s_ = """Class:          Distribution
Alphabet:       (0, 1) for all rvs
Base:           linear
Outcome Class:  tuple
Outcome Length: 2
RV Names:       None

x    p(x)
00   0.25
01   0.25
10   0.25
11   0.25"""
    s = d.to_string(str_outcomes=True)
    assert s == s_

def test_prepare_string4():
    class WeirdInt(int):
        def __str__(self):
            raise Exception
    outcomes = [(0, 0), (0, 1), (1, 0), (1, 1)]
    outcomes = [(WeirdInt(x), WeirdInt(y)) for (x, y) in outcomes]
    pmf = [1/4]*4
    d = Distribution(outcomes, pmf)
    s_ = """Class:          Distribution
Alphabet:       (0, 1) for all rvs
Base:           linear
Outcome Class:  tuple
Outcome Length: 2
RV Names:       None

x        p(x)
(0, 0)   0.25
(0, 1)   0.25
(1, 0)   0.25
(1, 1)   0.25"""
    s = d.to_string(str_outcomes=True)
    assert s == s_

def test_really_big_words():
    """
    Test to ensure that large but sparse outcomes are fast.
    """
    outcomes = ['01'*45, '10'*45]
    pmf = [1/2]*2
    d = Distribution(outcomes, pmf)
    d = d.coalesce([range(30), range(30, 60), range(60, 90)])
    new_outcomes = (('10'*15,)*3, ('01'*15,)*3)
    assert d.outcomes == new_outcomes
