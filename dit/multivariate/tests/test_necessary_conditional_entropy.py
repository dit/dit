"""

"""

import pytest

from dit.distconst import uniform
from dit.multivariate.necessary_conditional_entropy import necessary_conditional_entropy

def test_1():
    """
    """
    d = uniform(['0000', '0001', '0010', '0100', '0101', '1000', '1001', '1010'])
    assert necessary_conditional_entropy(d, [0,1], [2,3]) == pytest.approx(0.68872187554086695)
    assert necessary_conditional_entropy(d, [2,3], [0,1]) == pytest.approx(0.68872187554086695)
    assert necessary_conditional_entropy(d, [1], [2,3]) == pytest.approx(0.68872187554086695)