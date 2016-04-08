"""
"""

# from iterutils import pairwise

import numpy as np

from ..algorithms import marginal_maxent_dists
from ..shannon import entropy as H
# from ..divergences import kullback_leibler_divergence as D
from .base_profile import BaseProfile

class SchneidmanProfile(BaseProfile):
    """
    """

    def _compute(self):
        """
        """
        dists = marginal_maxent_dists(self.dist)
        diffs = -np.diff([ H(d) for d in dists ])
        # diffs = [ D(b,a) for a, b in pairwise(dists) ]
        self.profile = dict( (i, v) for i, v in enumerate(diffs) )
