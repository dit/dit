"""
Schneidman's ``connected information'' decomposition: [Schneidman, Elad, et al.
"Network information and connected correlations." Physical review letters 91.23
(2003): 238701]
"""

# from boltons.iterutils import pairwise

import numpy as np

from .base_profile import BaseProfile, profile_docstring
from ..algorithms import marginal_maxent_dists
# from ..divergences import kullback_leibler_divergence as D
from ..multivariate import dual_total_correlation as B
from ..shannon import entropy as H


__all__ = [
    'ConnectedInformations',
    'ConnectedDualInformations',
    'SchneidmanProfile',
]


class ConnectedInformations(BaseProfile):  # noqa: D101
    __doc__ = profile_docstring.format(name='ConnectedInformations',
                                       static_attributes='',
                                       attributes='',
                                       methods='')

    def _compute(self):
        """
        Compute the connected information decomposition.
        """
        dists = marginal_maxent_dists(self.dist)
        diffs = -np.diff([H(d) for d in dists])
        # diffs = [ D(b,a) for a, b in pairwise(dists) ]
        self.profile = {i + 1: v for i, v in enumerate(diffs)}
        self.widths = np.ones(len(self.profile))


SchneidmanProfile = ConnectedInformations


class ConnectedDualInformations(BaseProfile):  # noqa: D101
    __doc__ = profile_docstring.format(name='ConnectedDualInformations',
                                       static_attributes='',
                                       attributes='',
                                       methods='')

    def _compute(self):
        """
        Compute the connected dual information decomposition.
        """
        dists = marginal_maxent_dists(self.dist)
        diffs = np.diff([B(d) for d in dists])
        self.profile = {i + 1: v for i, v in enumerate(diffs)}
        self.widths = np.ones(len(self.profile))
