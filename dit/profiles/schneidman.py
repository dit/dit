"""
"""

import numpy as np

from ..algorithms import marginal_maxent_dists
from ..shannon import entropy as H
from .base_profile import BaseProfile

class SchneidmanProfile(BaseProfile):
    """
    """

    def _compute(self):
        """
        """
        dists = marginal_maxent_dists(self.dist)
        diffs = -np.diff([ H(d) for d in dists ])
        self.profile = dict( (i, v) for i, v in enumerate(diffs) )
