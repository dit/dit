#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Schneidman's ``connected information'' decomposition: [Schneidman, Elad, et al.
"Network information and connected correlations." Physical review letters 91.23
(2003): 238701]
"""

# from boltons.iterutils import pairwise

import numpy as np

from ..algorithms import marginal_maxent_dists
from ..shannon import entropy as H
# from ..divergences import kullback_leibler_divergence as D
from .base_profile import BaseProfile, profile_docstring

class SchneidmanProfile(BaseProfile):
    __doc__ = profile_docstring.format(name='SchneidmanProfile',
                                       static_attributes='',
                                       attributes='',
                                       methods='')

    def _compute(self):
        """
        Compute the connected information decomposition.

        Implementation Notes
        --------------------
        This uses the Frank-Wolfe implementation to find the maxent
        distributions.
        """
        dists = marginal_maxent_dists(self.dist)
        diffs = -np.diff([H(d) for d in dists])
        # diffs = [ D(b,a) for a, b in pairwise(dists) ]
        self.profile = dict((i+1, v) for i, v in enumerate(diffs))
        self.widths = np.ones(len(self.profile))
