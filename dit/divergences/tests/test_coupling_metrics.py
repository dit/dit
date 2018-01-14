"""

"""

from __future__ import division

import pytest

from dit import Distribution
from dit.divergences.coupling_metrics import coupling_metric


@pytest.mark.flaky(reruns=5)
def test_cm_1():
    """
    Test that equivalent distributions have zero metric.
    """
    d1 = Distribution(['0', '1'], [1/3, 2/3])
    d2 = Distribution(['a', 'b'], [2/3, 1/3])
    cm = coupling_metric([d1, d2], p=1.0)
    assert cm == pytest.approx(0.0)
