"""
Implement the ``complexity profile'' from [Y. Bar-Yam. Multiscale
complexity/entropy. Advances in Complex Systems, 7(01):47-63, 2004].
"""

from collections import defaultdict

import numpy as np

from .base_profile import BaseProfile, profile_docstring
from .information_partitions import ShannonPartition


__all__ = [
    'ComplexityProfile',
]


class ComplexityProfile(BaseProfile):  # noqa: D101
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
        levels = iter(sorted(profile, reverse=True))
        next(levels)  # skip the middle
        for level in levels:
            profile[level] += profile[level + 1]
        self.profile = dict(profile)
        self.widths = np.ones(len(self.profile))
