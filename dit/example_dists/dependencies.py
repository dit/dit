"""
Distributions illustrating three times of dependency among two variables.
"""

from __future__ import division

from .. import Distribution

__all__ = ['stacked', 'mixed']


# the first four outcomes are conditional dependence
# the next two are conditional independence
# the last four are intrinsic dependence
_stacked_outcomes = [
    '000',
    '011',
    '101',
    '110',
    '222',
    '333',
    '444',
    '445',
    '554',
    '555',
]
_stacked_pmf = [1/12]*4 + [1/6]*2 + [1/12]*4
stacked = Distribution(_stacked_outcomes, _stacked_pmf)

# each var expands in binary as:
#    (conditional dependence, conditional independence, intrinsic dependence)
_mixed_outcomes = [
    '000',
    '044',
    '404',
    '440',
    '222',
    '266',
    '626',
    '662',
    '110',
    '154',
    '514',
    '550',
    '332',
    '376',
    '736',
    '772',
    '001',
    '045',
    '405',
    '441',
    '223',
    '267',
    '627',
    '663',
    '111',
    '155',
    '515',
    '551',
    '333',
    '377',
    '737',
    '773',
]
_mixed_pmf = [1/32]*32
mixed = Distribution(_mixed_outcomes, _mixed_pmf)
