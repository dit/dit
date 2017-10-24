"""

"""

from hypothesis import given

from dit.utils.testing import distributions

import numpy as np

from dit.divergences import hellinger_distance, variational_distance


@given(dist1=distributions(size=1, alphabet=10), dist2=distributions(size=1, alphabet=10))
def test_inequalities(dist1, dist2):
    """
    H^2(p||q) <= V(p||q) <= sqrt(2)*H(p||q)
    """
    h = hellinger_distance(dist1, dist2)
    v = variational_distance(dist1, dist2)
    assert h**2 <= v + 1e-10
    assert v <= np.sqrt(2)*h + 1e-10
