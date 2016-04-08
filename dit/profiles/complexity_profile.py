"""
Implement the ``complexity profile'' from [Y. Bar-Yam. Multiscale complexity/entropy. Advances in Complex Systems, 7(01):47-63, 2004].
"""

from collections import defaultdict

from ..algorithms import ShannonPartition
from .base_profile import BaseProfile

class ComplexityProfile(BaseProfile):
    """
    """

    def _compute(self):
        """
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
