# coding: utf-8
"""
A catch-all module for miscellaneous pmf-based operations.

Eventually, we will need to reorganize.

"""
from __future__ import division
from __future__ import print_function

from itertools import product

import dit
import numpy as np

__all__ = [
    'perturb',
    'convex_combination',
    'downsample',
]

def perturb(pmf, eps=.1, prng=None):
    """
    Returns a new distribution with all probabilities perturbed.

    Probabilities which are zero in the pmf cannot be perturbed by this method.
    All other probabilities are perturbed via the following process:

    0. Initial pmf ``p`` lives on the ``n``-simplex.
    1. Transform ``p`` via ilr (inverse logarithmic ratio) transform.
    2. Uniformly draw ``n`` random numbers between ``[0,1]``.
    3. Construct new transformed pmf: `p2_ilr = p1_ilr + eps * rand`
    4. Apply inverse ilr transformation.

    Practically, a large value of `eps` means that there is a better chance
    the perturbation will take the distribution closer to the simplex boundary.
    Large distributions (with more than 60 elements) fail, due to some
    underflow issue with the ilr transformation.

    Parameters
    ----------
    pmf : NumPy array
        The distribution to be perturbed. Assumes `pmf` represents linearly
        distributed probabilities.
    eps : float
        The scaling factor used for perturbing. Values of `10` correspond
        to large perturbations for the ``1``-simplex.
    prng : NumPy RandomState
        A random number generator.

    Returns
    -------
    out : NumPy array
        The perturbed distribution.

    """
    if prng is None:
        prng = dit.math.prng

    idx = pmf > 0
    p1 = pmf[idx]

    p1_ilr = dit.math.aitchison.ilr(p1)
    delta = eps * (prng.rand(len(p1_ilr)) - .5)
    p2_ilr = p1_ilr + delta
    p2 = dit.math.aitchison.ilr_inv(p2_ilr)

    out = np.zeros(len(pmf))
    out[idx] = p2

    return out

def convex_combination(pmfs, weights=None):
    """
    Forms the convex combination of the pmfs.

    Assumption: All pmf probabilities and weights are linearly distributed.

    Parameters
    ----------
    pmfs : NumPy array, shape (n,k)
        The `n` distributions, each of length `k` that will be mixed.
    weights : NumPy array, shape (n,)
        The weights applied to each pmf. This array will be normalized
        automatically.

    """
    # Possibly could be used to speed up dit.mixture_distribution2().
    pmfs = np.atleast_2d(pmfs)
    if weights is None:
        weights = np.ones(pmfs.shape[0], dtype=float) / pmfs.shape[0]
    else:
        weights = np.asarray(weights, dtype=float)
        weights /= weights.sum()
    mixture = (pmfs * weights[:, np.newaxis]).sum(axis=0)
    return mixture

def downsample(pmf, depth, base=2, method='componentL1'):
    """
    Returns the nearest pmf on a triangular grid.

    When multiple grid points are equally close, only one of them is returned.
    The particular grid point returned is arbitrary and determined by the
    method that compares distances.

    Parameters
    ----------
    pmf : NumPy array, shape (n,) or (k, n)
        The pmf on the ``(n-1)``-simplex.
    depth : int
        Controls the density of the grid.  The number of points on the simplex
        is given by: (base**depth + length - 1)! / (base**depth)! / (length-1)!
        At each depth, the number of points is exponentially increased.
    base : int
        The rate at which we divide probabilities..
    method : str
        The algorithm used to determine what `nearest` means. The default
        method, 'componentL1', moves each component to its nearest grid
        value using the L1 norm.

    Returns
    -------
    d : NumPy array, shape (n,)
        The downsampled pmf.

    See Also
    --------
    dit.simplex_grid

    """
    if method in _methods:
        return _methods[method](pmf, depth, base)
    else:
        raise NotImplementedError('Unknown method.')

def downsample_componentL1(pmf, depth, base=2):
    """
    Clamps each component, one-by-one.
    Renormalizes and uses updated insert indexes as you go.

    """
    N = base**depth
    locs = np.linspace(0, 1, N + 1)

    out = np.atleast_2d(pmf).transpose().copy()
    # Go through each component.
    for i in range(out.shape[0] - 1):
        # Find insertion indexes
        insert_index = np.searchsorted(locs, out[i])
        # Define the indexes of clamped region for each component.
        clamps = np.array([insert_index - 1, insert_index])
        # Actually get the clamped region
        gridvals = locs[clamps]
        # Calculate distance to each point, per component.
        distances = np.abs(gridvals - out[i])
        # Determine which index each component was closest to.
        desired = np.argmin(distances, axis=0)
        # Pull those indexes from the clamping indexes
        locations = np.where(desired, insert_index, insert_index - 1)
        out[i] = locs[locations]
        # Now renormalize the other components of the distribution...
        temp = out.transpose() # View
        prev_Z = temp[..., :i+1].sum(axis=-1)
        zeros = np.isclose(prev_Z, 1)
        Z = (1 - prev_Z) / temp[..., i+1:].sum(axis=-1)
        temp[..., i+1:] *= Z[..., np.newaxis]
        # This assumes len(shape) == 2.
        temp[zeros, i+1:] = 0

    out = out.transpose()
    out[...,-1] = 1 - out[...,:-1].sum(axis=-1)
    if len(pmf.shape) == 1:
        out = out[0]
    return out

def clamped_indexes(pmf, depth, base=2):
    """
    Returns the indexes of the component values that clamp the pmf.

    Returns
    -------
    clamps : NumPy array, shape (2,n) or (2,k,n)

    """
    N = base**depth
    locs = np.linspace(0, 1, N + 1)
    # Find insertion indexes
    insert_index = np.searchsorted(locs, pmf)
    # Define the indexes of clamped region for each component.
    clamps = np.array([insert_index - 1, insert_index])

    return clamps, locs

def projections(pmf, depth, base=2, method=None):
    """
    Returns the projections on the way to the nearest grid point.

    The original pmf is included in the final output.

    Parameters
    ----------
    pmf : NumPy array, shape (n,)
        The pmf on the ``(n-1)``-simplex.
    depth : int
        Controls the density of the grid.  The number of points on the simplex
        is given by: (base**depth + length - 1)! / (base**depth)! / (length-1)!
        At each depth, the number of points is exponentially increased.
    base : int
        The rate at which we divide probabilities..
    method : str
        The algorithm used to determine what `nearest` means. The default
        method, 'componentL1', moves each component to its nearest grid
        value using the L1 norm.

    Returns
    -------
    d : NumPy array, shape (n,n)
        The projections leading to the downsampled pmf.

    See Also
    --------
    downsample, dit.simplex_grid


    """
    # We can only have 1 pmf.
    assert(len(pmf.shape) == 1)

    N = base**depth
    locs = np.linspace(0, 1, N + 1)

    out = pmf.copy()
    # Go through each component.

    projs = [out.copy()]
    for i in range(out.shape[0] - 1):
        # Find insertion indexes
        insert_index = np.searchsorted(locs, out[i])
        # Define the indexes of clamped region for each component.
        clamps = np.array([insert_index - 1, insert_index])
        # Actually get the clamped region
        gridvals = locs[clamps]
        # Calculate distance to each point, per component.
        distances = np.abs(gridvals - out[i])
        # Determine which index each component was closest to.
        desired = np.argmin(distances, axis=0)
        # Pull those indexes from the clamping indexes
        locations = np.where(desired, insert_index, insert_index - 1)
        out[i] = locs[locations]
        # Now renormalize the other components of the distribution...
        prev_Z = out[:i+1].sum(axis=-1)
        zeros = np.isclose(prev_Z, 1)
        if zeros:
            Z = 0
        else:
            Z = (1 - prev_Z) / out[i+1:].sum(axis=-1)
        out[i+1:] *= Z
        projs.append(out.copy())

    return np.asarray(projs)


_methods = {
    'componentL1': downsample_componentL1
}

