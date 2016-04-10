"""
"""

from .base_profile import BaseProfile

from itertools import product
from iterutils import powerset

import numpy as np
from scipy.optimize import linprog

from dit.algorithms import ShannonPartition
from dit.multivariate import entropy as H
from dit.utils import flatten

__all__ = ['MUIProfile']

def get_lp_form(dist):
    """
    """
    pa = list(frozenset(s) for s in powerset(flatten(dist.rvs)))[1:]
    sp = sorted(ShannonPartition(dist).atoms.items())
    atoms = list(frozenset(flatten(a[0])) for a, _ in sp)

    A = []
    b = []
    c = []
    bounds = []

    for pa_W, pa_V in product(pa, pa):

        cond = np.zeros(len(atoms))
        for j, atom in enumerate(atoms):
            if pa_V & atom:
                cond[j] = 1
        A.append(cond)
        b.append(H(dist, pa_V))

        if pa_W < pa_V:
            cond = np.zeros(len(atoms))
            for j, atom in enumerate(atoms):
                if (pa_V & atom) and not (pa_W & atom):
                    cond[j] = 1
            A.append(cond)
            b.append(H(dist, pa_V) - H(dist, pa_W))

        cond = np.zeros(len(atoms))
        for j, atom in enumerate(atoms):
            if (pa_V & atom):
                cond[j] += 1
            if (pa_W & atom):
                cond[j] += 1
            if ((pa_V | pa_W) & atom):
                cond[j] -= 1
            if ((pa_V & pa_W) & atom):
                cond[j] -= 1
            A.append(cond)
            b.append(H(dist, pa_V) + H(dist, pa_W) - H(dist, pa_V | pa_W) - H(dist, pa_V & pa_W))

    A.append([1]*len(atoms))
    b.append(0)

    A = np.array(A)
    b = np.array(b)

    c = np.array([ -len(atom) for atom in atoms ])

    bounds = [ (0, val) if val > 0 else (val, 0) for _, val in sp ]

    return c, A, b, bounds

def max_util_of_info(c, A, b, bounds, y):
    """
    """
    b[-1] = y
    solution = linprog(c, A, b, bounds=bounds)
    maximum_utility_of_information = -solution.fun
    return maximum_utility_of_information

class MUIProfile(BaseProfile):
    """
    """

    xlabel = "scale [bits]"
    ylabel = "marginal utility of information"

    def _compute(self):
        """
        """
        c, A, b, bounds = get_lp_form(self.dist)
        ent = H(dist)
        pnts = np.linspace(0, ent, 100*ent)
        maxui = [ max_util_of_info(c, A, b, bounds, y) for y in pnts ]
        self.profile = dict(zip(pnts, np.gradient(maxui, np.diff(pnts)[0])))
