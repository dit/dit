#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Marginal Utility of Information, as defined here: http://arxiv.org/abs/1409.4708
"""

from .base_profile import BaseProfile, profile_docstring

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
    Construct the constraint matrix for computing the maximum utility of information in linear programming cononical form.

    Parameters
    ----------
    dist : Distribution
        The distribution from which to construct the constraints.

    Returns
    -------
    c : ndarray
        The utility function to minimize
    A : ndarray
        The lhs of the constraint equations
    b : ndarray
        The rhs of the constraint equations
    bounds : list of pairs
        The bounds on the individual elements of `x`
    """
    pa = list(frozenset(s) for s in powerset(flatten(dist.rvs)))[1:]
    sp = sorted(ShannonPartition(dist).atoms.items())
    atoms = list(frozenset(flatten(a[0])) for a, _ in sp)

    A = []
    b = []
    c = []
    bounds = []

    for pa_W, pa_V in product(pa, pa):
        # constraint (i)
        cond = np.zeros(len(atoms))
        for j, atom in enumerate(atoms):
            if pa_V & atom:
                cond[j] = 1
        A.append(cond)
        b.append(H(dist, pa_V))
        # constraint (ii)
        if pa_W < pa_V:
            cond = np.zeros(len(atoms))
            for j, atom in enumerate(atoms):
                if (pa_V & atom) and not (pa_W & atom):
                    cond[j] = 1
            A.append(cond)
            b.append(H(dist, pa_V) - H(dist, pa_W))
        # constraint (iii)
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

    c = np.array([ -len(atom) for atom in atoms ]) # negative for minimization

    bounds = [ (0, val) if val > 0 else (val, 0) for _, val in sp ]

    return c, A, b, bounds

def max_util_of_info(c, A, b, bounds, y):
    """
    Compute the maximum utility of information at scale `y`.

    Parameters
    ----------
    c : ndarray
        A list of atom-weights.
    A : ndarray
        The lhs of the various constraints.
    b : ndarray
        The rhs of the various constraints.
    bounds : list of pairs
        Each part of `x` must be between the atom's value and 0.
    y : float
        The total mutual information captured.
    """
    b[-1] = y
    solution = linprog(c, A, b, bounds=bounds)
    maximum_utility_of_information = -solution.fun
    return maximum_utility_of_information

class MUIProfile(BaseProfile):
    __docstring__ = profile_docstring.format(name='MUIProfile',
                                             static_attributes='',
                                             attributes='',
                                             methods='')

    xlabel = "scale [bits]"
    ylabel = "marginal utility of information"

    def _compute(self):
        """
        Compute the Marginal Utility of Information.
        """
        c, A, b, bounds = get_lp_form(self.dist)
        ent = H(dist)
        pnts = np.linspace(0, ent, 100*ent)
        maxui = [ max_util_of_info(c, A, b, bounds, y) for y in pnts ]
        self.profile = dict(zip(pnts, np.gradient(maxui, np.diff(pnts)[0])))
