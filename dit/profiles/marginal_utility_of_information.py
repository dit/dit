#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Marginal Utility of Information, as defined here: http://arxiv.org/abs/1409.4708
"""

from .base_profile import BaseProfile, profile_docstring

from itertools import product

import numpy as np

from .information_partitions import ShannonPartition
from ..math import close
from ..multivariate import entropy as H
from ..utils import flatten, powerset

__all__ = ['MUIProfile']

def get_lp_form(dist, ents):
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
    sp = sorted(ents.atoms.items())
    atoms = list(frozenset(flatten(a[0])) for a, v in sp if not close(v, 0))

    A = []
    b = []

    for pa_V, pa_W in product(pa, pa):
        if pa_V == pa_W:
            # constraint (i)
            cond = np.zeros(len(atoms))
            for j, atom in enumerate(atoms):
                if pa_V & atom:
                    cond[j] = 1
            A.append(cond)
            b.append(ents[([pa_V], [])])

        else:
            # constraint (ii)
            if pa_W < pa_V:
                cond = np.zeros(len(atoms))
                for j, atom in enumerate(atoms):
                    if (pa_V & atom) and not (pa_W & atom):
                        cond[j] = 1
                A.append(cond)
                b.append(ents[([pa_V], [])] - ents[([pa_W], [])])
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
                b.append(ents[([pa_V], [])] +
                         ents[([pa_W], [])] -
                         ents[([pa_V | pa_W], [])] -
                         ents[([pa_V & pa_W], [])])

    A.append([1]*len(atoms))
    b.append(0) # placeholder for y

    A = np.array(A)
    b = np.array(b)

    c = np.array([-len(atom) for atom in atoms]) # negative for minimization

    bounds = [(min(0, val), max(0, val)) for _, val in sp if not close(val, 0)]

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
    from scipy.optimize import linprog

    b[-1] = y
    solution = linprog(c, A, b, bounds=bounds)
    maximum_utility_of_information = -solution.fun
    return maximum_utility_of_information

class MUIProfile(BaseProfile):
    __doc__ = profile_docstring.format(name='MUIProfile',
                                       static_attributes='',
                                       attributes='',
                                       methods='')

    xlabel = "scale [bits]"
    ylabel = "marginal utility of information"
    align = 'edge'

    def _compute(self):
        """
        Compute the Marginal Utility of Information.
        """
        sp = ShannonPartition(self.dist)
        c, A, b, bounds = get_lp_form(self.dist, sp)
        ent = sum(sp.atoms.values())

        atoms = sp.atoms.values()
        ps = powerset(atoms)
        pnts = np.unique(np.round([sum(ss) for ss in ps], 7))
        pnts = [v for v in pnts if 0 <= v <= ent]

        maxui = [max_util_of_info(c, A, b, bounds, y) for y in pnts]
        mui = np.round(np.diff(maxui)/np.diff(pnts), 7)
        vals = np.array(np.unique(mui, return_index=True))
        self.profile = dict((pnts[int(row[1])], row[0]) for row in vals.T)
        self.widths = np.diff(list(sorted(self.profile.keys())) + [ent])

    def draw(self, ax=None): # pragma: no cover
        ax = super(MUIProfile, self).draw(ax=ax)
        pnts = np.arange(int(max(self.profile.keys()) + self.widths[-1]) + 1)
        ax.set_xticks(pnts)
        ax.set_xticklabels(pnts)
        return ax

    draw.__doc__ = BaseProfile.draw.__doc__
