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
    'convex_combination',
    'downsample',
    'jittered',
    'perturb_support',
    'replace_zeros',
]

def perturb_support(pmf, eps=.1, shape='ball', prng=None):
    """
    Returns a new distribution with all nonzero probabilities perturbed.

    Probabilities which are zero in the pmf cannot be perturbed by this method.
    All other probabilities are perturbed via the following process:

    0. Initial pmf ``p`` lives on the ``n``-simplex.
    1. Transform ``p`` via ilr (isometric logarithmic ratio) transform.
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
    shape : str
        The type of neighborhood to draw from. Valid options are 'square' or
        'ball'. For 'square', a point is chosen uniformly from a unit square
        centered around `pmf` in ilr coordinates. For 'ball' a point is chosen
        uniformly from the unit circle centered around the pmf in ilr
        coordinates. In both cases, the region is then scaled by `eps`.
    prng : NumPy RandomState
        A random number generator. If `None`, then `dit.math.prng` is used.

    Returns
    -------
    out : NumPy array
        The perturbed distribution.

    References
    ----------
    For details on the isometric log-ratio transformation see [1]_.

    .. [1] J. J. Egozcue, V. Pawlowsky-Glahn, G. Mateu-Figueras, C.
    Barceló-Vidal. Isometric Logratio Transformations for Compositional
    Data Analysis, Mathematical Geology. April 2003, Volume 35, Issue 3,
    pp 279-300. http://dx.doi.org/10.1023/A:1023818214614

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
        if shape == 'square':
            delta = eps * (prng.rand(*p1_ilr.shape) - .5)
        elif shape == 'ball':
            delta = eps * dit.math.ball(p1_ilr.shape[0], prng=prng)
        p2_ilr = p1_ilr + delta
        p2 = dit.math.aitchison.ilr_inv(p2_ilr)
        out[i,idx] = p2

    if len(pmf.shape) == 1:
        out = out[0]

    return out

def replace_zeros(pmf, delta, rand=True, prng=None):
    """
    Replaces zeros in a pmf with values smaller than `eps`.

    Note that when considering the Aitchison geometry, the boundaries of the
    simplex are infinitely far away from the uniform distribution. So this
    operation, while numerically small, moves to a new distribution that
    is significantly closer to the uniform distribution (relatively speaking).

    The replacement strategy is done in a multiplicative fashion according
    to the following formula [1]_, but optionally, with randomly determined
    replacement values:

        x_i^\prime =
            \begin{cases}
                \delta_i & x_i = 0 \\
                x_i (1 - \sum_i \delta_i) & x_i \neq 0
            \end{cases}

    where :math:`\delta_i` is the replacement value for each zero element.
    This approach is preferred since it preserves the ratios of nonzero
    elements, an important feature of distributions. Simply adding some values
    to the zero elements and then renormalizing would not preserve the ratios.

    Parameters
    ----------
    pmf : NumPy array, shape (n,) or (k, n)
        The distribution or `k` distributions living on the `(n-1)`-simplex.
    delta : float
        A small value for all the zero elements in the pmf.
    rand : bool
        When `True`, the replacement values for zero elements are random
        numbers less than `delta`. When `False`, the replacement values are
        equal to `delta`.
    prng : NumPy RandomState
        A random number generator used to select replacement values when
        `rand` is `True`. If None, then `dit.math.prng` is used.

    Returns
    -------
    d : NumPy array, shape (n,) or (k, n)
        The distributions with zeros replaced.

    Examples
    --------
    >>> d = np.array([.25, .75, 0])
    >>> replace_zeros(d, .01, rand=False)
    array([ 0.2475,  0.7425,  0.01  ])

    Notes
    -----
    When the distribution is determined by counts and the total number of
    counts is known, this method can be modified so that the value of `delta`
    is chosen according to a Bayesian inferential procedure. Not implemented.

    References
    ----------
    .. [1] Martín-Fernández JA and Thió-Henestrosa S, 2006. Rounded zeros: some
    practical aspects for compositional data. In Compositional Data Analysis
    in the Geosciences: From Theory to Practice, vol. 264. Geological Society,
    London, pp. 191–201.

    """
    if prng is None:
        prng = dit.math.prng

    nonzero = pmf == 0

    replacements = np.zeros(nonzero.sum(), dtype=float)
    if rand:
        replacements = delta * prng.rand(len(replacements))
    else:
        replacements += delta

    d = pmf.copy()
    d[nonzero] += replacements
    d[~nonzero] *= 1 - replacements.sum()

    return d

def jittered(pmf, jitter=1e-5, zeros=True, prng=None):
    """
    Jitters the elements of `pmf`.

    Parameters
    ----------
    pmf : NumPy array, shape (n,) or (k, n)
        The pmf or `k` pmfs to jitter.
    jitter : float
        The jitter amount. The value is used for both zero and nonzero
        elements in the pmf.
    zeros : bool
        If `True`, the zeros in `pmf` are first replaced using `jitter` as
        the `delta` parameter in :meth:`replace_zeros`. If `False`, only the
        nonzero elements in `pmf` are jittered.
    prng : NumPy RandomState
        A random number generator used to select replacement values when
        `rand` is `True`. If None, then `dit.math.prng` is used.

    Returns
    -------
    d : NumPy array, shape (n,) or (k, n)
        The jittered pmf(s).

    Examples
    --------
    >>> d = np.array([.5, .5, 0, 0])
    >>> jittered(d)
    array([4.99999999e-01, 4.99999999e-01, 6.54894572e-10, 5.49417792e-10])

    See Also
    --------
    replace_zeros, perturb_support

    """
    if prng is None:
        prng = dit.math.prng

    if zeros:
        d = replace_zeros(pmf, jitter, prng=prng)
    else:
        d = pmf

    d = perturb_support(d, jitter, prng=prng)

    return d

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
    clamps = clamped_indexes(pmf[i], locs)
    lower, upper = clamps
    # Actually get the clamped region
    gridvals = locs[clamps]
    # Calculate distance to each point, per component.
    distances = np.abs(gridvals - pmf[i])

    # Determine which index each component was closest to.
    # desired[i] == 0 means that the lower index was closer
    # desired[i] == 1 means that the upper index was closer
    # If there are any symmetries in the distribution, it could happen
    # that some of the distances are equal. The op() will select only
    # one of these branches---this prevents an exhaustive listing of
    # all possible neighbors. A small jitter is recommended. This will
    # have a marginal effect on any binning we might do.
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

def clamped_indexes(pmf, locs):
    """
    Returns the indexes of the component values that clamp the pmf.

    If the component value is equal to a grid point, then the lower and upper
    clamps are equal to one another.

    Returns
    -------
    clamps : NumPy array, shape (2,n) or (2,k,n)

    Examples
    --------
    >>> locs = np.linspace(0, 1, 3) # [0, 1/2, 1]
    >>> d = np.array([.25, .5, .25])
    >>> clamped_indices(d, locs)
    array([[0, 1, 0],
           [1, 1, 1]])

    """
    # Find insertion indexes
    left_index = np.searchsorted(locs, pmf, 'left')
    right_index = np.searchsorted(locs, pmf, 'right')

    # If the left index does not equal the right index, then the coordinate
    # is equal to an element of `locs`. This means we want its upper and lower
    # clamped indexes to be equal.
    #
    # If the left and right indexes are equal, then, the (left) index specifies
    # the upper clamp. We subtract one for the lower clamp. There is no concern
    # for the lower clamp dropping to -1 since already know that the coordinate
    # is not equal to an element in `locs`.
    upper = left_index
    lower = upper - 1
    mask = left_index != right_index
    lower[mask] = upper[mask]
    clamps = np.array([lower, upper])
    return clamps

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
        locations = _downsample_componentL1(out, i, op, locs)
        projs.append(out.copy())

    projs = np.asarray(projs)
    projs = np.swapaxes(projs, 1, 2)
    if len(pmf.shape) == 1:
        projs = projs[:,0,:]
    return projs

_methods = {
    'componentL1': downsample_componentL1
}
