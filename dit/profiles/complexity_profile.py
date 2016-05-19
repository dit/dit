"""
Implement the ``complexity profile'' from [Y. Bar-Yam. Multiscale
complexity/entropy. Advances in Complex Systems, 7(01):47-63, 2004].
"""

from collections import defaultdict

import numpy as np

from .information_partitions import ShannonPartition
from .base_profile import BaseProfile, profile_docstring

class ComplexityProfile(BaseProfile):
    __doc__ = profile_docstring.format(name='ComplexityProfile',
                                       static_attributes='',
                                       attributes='',
                                       methods='')

    def _compute(self):
        """
        Compute the complexity profile.

        Implementation Notes
        --------------------
        This make use of the ShannonPartition. There may be more efficient
        methods.
        """
        sp = ShannonPartition(self.dist)
        profile = defaultdict(float)
        for atom in sp.get_atoms(string=False):
            profile[len(atom[0])] += sp[atom]
        levels = reversed(sorted(profile))
        next(levels) # skip the middle
        for level in levels:
            profile[level] += profile[level+1]
        self.profile = dict(profile)
        self.widths = np.ones(len(self.profile))
