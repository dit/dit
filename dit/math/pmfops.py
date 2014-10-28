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
    pmf : NumPy array, shape (n,) or (k,n)
        The distribution to be perturbed. Assumes `pmf` represents linearly
        distributed probabilities. One may pass in `k` such distributions.
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

    pmf_2d = np.atleast_2d(pmf)
    out = np.zeros_like(pmf_2d)
    for i, row in enumerate(pmf_2d):
        # We must treat each row differently because their supports
        # could live on different simplices.
        idx = row > 0
        p1 = row[idx]
        p1_ilr = dit.math.aitchison.ilr(p1)
        delta = eps * (prng.rand(*p1_ilr.shape) - .5)
        p2_ilr = p1_ilr + delta
        p2 = dit.math.aitchison.ilr_inv(p2_ilr)
        out[i,idx] = p2

    if len(pmf.shape) == 1:
        out = out[0]

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

def downsample(pmf, subdivisions, method='componentL1'):
    """
    Returns the nearest pmf on a triangular grid.

    When multiple grid points are equally close, only one of them is returned.
    The particular grid point returned is arbitrary and determined by the
    method that compares distances.

    Parameters
    ----------
    pmf : NumPy array, shape (n,) or (k, n)
        The pmf on the ``(n-1)``-simplex.
    subdivisions : int
        The number of subdivisions for the interval [0, 1]. The grid considered
        is such that each component will take on values at the boundaries of
        the subdivisions. For example, subdivisions corresponds to
        :math:`[[0, 1/2], [1/2, 1]]` and thus, each component can take the
        values 0, 1/2, or 1. So one possible pmf would be (1/2, 1/2, 0).
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
        return _methods[method](pmf, subdivisions)
    else:
        raise NotImplementedError('Unknown method.')

def _downsample_componentL1(pmf, i, op, locs):
    """
    Low-level function to incrementally project a pmf.

    Parameters
    ----------
    pmf : NumPy array, shape (n, k)
        A 2D NumPy array that is modified in-place. The columns represent
        the various pmfs. The rows represent each component.
    i : int
        The component to be projected.
    op : callable
        This is np.argmin or np.argmax. It determines the projection.
    locs : NumPy array
        The subdivisions for each component.

    """
    # Find insertion indexes
    insert_index = np.searchsorted(locs, pmf[i])
    # Define the indexes of clamped region for each component.
    lower = insert_index - 1
    upper = insert_index
    clamps = np.array([lower, upper])
    # Actually get the clamped region
    gridvals = locs[clamps]
    # Calculate distance to each point, per component.
    distances = np.abs(gridvals - pmf[i])
    # Determine which index each component was closest to.
    # desired[i] == 0 means that the lower index was closer
    # desired[i] == 1 means that the upper index was closer
    desired = op(distances, axis=0)
    # Pull those indexes from the clamping indexes
    # So when desired[i] == 1, we want to pull the upper index.
    locations = np.where(desired, upper, lower)
    pmf[i] = locs[locations]
    # Now renormalize the other components of the distribution...
    temp = pmf.transpose() # View
    prev_Z = temp[..., :i+1].sum(axis=-1)
    zeros = np.isclose(prev_Z, 1)
    Z = (1 - prev_Z) / temp[..., i+1:].sum(axis=-1)
    temp[..., i+1:] *= Z[..., np.newaxis]
    # This assumes len(shape) == 2.
    temp[zeros, i+1:] = 0
    return locations

def downsample_componentL1(pmf, subdivisions):
    """
    Clamps each component, one-by-one.
    Renormalizes and uses updated insert indexes as you go.

    """
    locs = np.linspace(0, 1, subdivisions + 1)

    out = np.atleast_2d(pmf).transpose().copy()
    # Go through each component and move to closest component.
    op = np.argmin
    for i in range(out.shape[0] - 1):
        locations = _downsample_componentL1(out, i, op, locs)

    out = out.transpose()
    if len(pmf.shape) == 1:
        out = out[0]
    return out

def clamped_indexes(pmf, subdivisions):
    """
    Returns the indexes of the component values that clamp the pmf.

    Returns
    -------
    clamps : NumPy array, shape (2,n) or (2,k,n)

    """
    locs = np.linspace(0, 1, subdivisions + 1)
    # Find insertion indexes
    insert_index = np.searchsorted(locs, pmf)
    # Define the indexes of clamped region for each component.
    clamps = np.array([insert_index - 1, insert_index])

    return clamps, locs

def projections(pmf, subdivisions, ops=None):
    """
    Returns the projections on the way to the nearest grid point.

    The original pmf is included in the final output.

    Parameters
    ----------
    pmf : NumPy array, shape (n,) or (k, n)
        The pmf on the ``(n-1)``-simplex. Optionally, provide `k` pmfs.
    subdivisions : int
        The number of subdivisions for the interval [0, 1]. The grid considered
        is such that each component will take on values at the boundaries of
        the subdivisions. For example, subdivisions corresponds to
        :math:`[[0, 1/2], [1/2, 1]]` and thus, each component can take the
        values 0, 1/2, or 1. So one possible pmf would be (1/2, 1/2, 0).
    method : str
        The algorithm used to determine what `nearest` means. The default
        method, 'componentL1', moves each component to its nearest grid
        value using the L1 norm.

    Other Parameters
    ----------------
    ops : list
        A list of `n-1` callables, where `n` the number of components in the
        pmf. Each element in the list is a callable the determines how the
        downsampled pmf's are constructed by specifying which of the lower
        and upper clamped location indexes should be chosen. If `None`, then
        `ops` is a list of `np.argmin` and will select the closest grid point.

    Returns
    -------
    d : NumPy array, shape (n,n) or (n,k,n)
        The projections leading to the downsampled pmf.

    See Also
    --------
    downsample, dit.simplex_grid

    """
    locs = np.linspace(0, 1, subdivisions + 1)

    out = np.atleast_2d(pmf).transpose().copy()
    projs = [out.copy()]

    if ops is None:
        # Take closest point in regional cell.
        ops = [np.argmin] * (out.shape[0] - 1)

    # Go through each component and move to closest component.
    for i, op in zip(range(out.shape[0] - 1), ops):
        _downsample_componentL1(out, i, op, locs)
        projs.append(out.copy())

    projs = np.asarray(projs)
    projs = np.swapaxes(projs, 1, 2)
    if len(pmf.shape) == 1:
        projs = projs[:,0,:]
    return projs

_methods = {
    'componentL1': downsample_componentL1
}

