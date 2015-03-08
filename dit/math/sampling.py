#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Functions related to sampling from discrete distributions.

"""
from __future__ import division

import numpy as np

import dit.exceptions

__all__ = (
    'sample',
    'ball',
    'norm',
)

def sample(dist, size=None, rand=None, prng=None):
    """Returns a sample from a discrete distribution.

    Parameters
    ----------
    dist : Dit distribution
        The distribution from which the sample is drawn.
    size : int or None
        The number of samples to draw from the distribution. If `None`, then
        a single sample is returned.  Otherwise, a list of samples if returned.
    rand : float or NumPy array
        When `size` is `None`, `rand` should be a random number drawn uniformly
        from the interval [0,1]. When `size` is not `None`, then this should be
        a NumPy array of random numbers.  In either situation, if `rand` is
        `None`, then the random number(s) will be drawn from a pseudo random
        number generator.
    prng : random number generator
        A random number generator with a `rand' method that returns a random
        number between 0 and 1 when called with no arguments. If unspecified,
        then we use the random number generator on the distribution. Thus,
        the `dist` must have a `prng` attribute if `prng` is None.

    Returns
    -------
    s : sample
        The sample drawn from the distribution.

    """
    ### This works for NumPy-base distributions (in npdist.py)
    if size is None:
        n = 1
    else:
        n = size

    if rand is None:
        if prng is None:
            prng = dist.prng
        try:
            rand = prng.rand(n)
        except AttributeError:
            msg = "The random number generator must support a `rand()' call."
            e = dit.exceptions.ditException(msg)
            raise e
    else:
        if size is None:
            rand = np.array([rand])
        elif n != len(rand):
            msg = "The number of random numbers must equal n."
            e = dit.exceptions.ditException(msg)
            raise e

    # We need linear probabilities.
    if dist.is_log():
        pmf = dist.ops.exp(dist.pmf)
    else:
        pmf = dist.pmf

    indexes = _samples(pmf, rand)
    outcomes = dist.outcomes
    s = [outcomes[i] for i in indexes]
    if size is None:
        s = s[0]

    return s

def _sample_discrete__python(pmf, rand):
    """Returns a sample from a discrete distribution.

    Note: This version has no bells or whistles.

    Parameters
    ----------
    pmf : list of floats
        A list of floats representing the probability mass function. The events
        will be the indices of the list. The floats should represent
        probabilities (and not log probabilities).
    rand : float
        The sample is drawn using the passed number.

    Returns
    -------
    s : int
        The index of the sampled event.

    """
    total = 0
    for i, prob in enumerate(pmf):
        total += prob
        if rand < total:
            return i

def _samples_discrete__python(pmf, rands, out=None):
    """Returns samples from a discrete distribution.

    Note: This version has no bells or whistles.

    Parameters
    ----------
    pmf : NumPy float array, shape (n,)
        An array of floats representing the probability mass function. The
        events will be the indices of the list. The floats should represent
        probabilities (and not log probabilities).
    rand : NumPy float array, shape (k,)
        The k samples are drawn using the passed in random numbers.  Each
        element should be between 0 and 1, inclusive.
    out : NumPy int array, shape (k,)
        The samples from `pmf`.  Each element will be filled with an integer i
        representing a sampling of the event with probability `pmf[i]`.

    Returns
    -------
    out : NumPy int array, shape (k,)

    """
    L = rands.shape[0]
    if out is None:
        out = np.empty(L, dtype=int)

    n = pmf.shape[0]
    for i in range(L):
        rand = rands[i]
        total = 0
        for j in range(n):
            total += pmf[j]
            if rand < total:
                out[i] = j
                break

    return out

def ball(n, size=None, prng=None):
    """
    Return random samples within an n-dimensional standard ball of radius 1.

    Parameters
    ----------
    size : int
        The number of samples to draw from the unit n-ball.
    prng : NumPy RandomState
        A random number generator. If `None`, then `dit.math.prng` is used.

    Returns
    -------
    samples : NumPy array, shape (`size`, `n`)
        Points within the unit `n`-ball.

    """
    if size is None:
        s = 1
    else:
        s = size

    if prng is None:
        prng = dit.math.prng

    # The alternative versions _2ball and _3ball_cylinder were slower in
    # timings. So we do not use them.
    samples = _ball(n, s, prng)

    if size is None:
        samples = samples[0]

    return samples

def _ball(n, size, prng):
    """
    Return `size` samples from the unit n-ball.

    Parameters
    ----------
    size : int
        The number of samples to draw from the unit `n`-ball.

    Returns
    -------
    samples : NumPy array, shape (`size`, `n`)
        Points within the unit `n`-ball.

    References
    ----------
    .. [1] http://math.stackexchange.com/a/87238

    """
    R = prng.rand(size, 1)**(1/n)
    X = prng.randn(size, n)
    norm = np.sqrt( (X**2).sum(axis=1) )[..., np.newaxis]
    return (R * X) / norm

def _2ball(size, prng):
    """
    Return `size` samples from the unit 2-ball.

    Parameters
    ----------
    size : int
        The number of samples to draw from the unit 2-ball.

    Returns
    -------
    samples : NumPy array, shape (`size`, 2)
        Points within the unit 2-ball.

    References
    ----------
    .. [1] http://stackoverflow.com/a/5838055

    """
    # This ends up not being faster due to the fancy indexing.
    # In Cython, we'd do better.
    theta = 2 * np.pi * prng.rand(size)
    u = prng.rand(size) + prng.rand(size)
    cond = u > 1
    u[cond] = 2 - u[cond]
    x = u * np.cos(theta)
    y = u * np.sin(theta)
    return np.array([x, y]).transpose()

def _3ball_cylinder(size, prng):
    """
    Return `size` samples from the unit 3-ball.

    Parameters
    ----------
    size : int
        The number of samples to draw from the unit 3-ball.

    Returns
    -------
    samples : NumPy array, shape (`size`, 3)
        Points within the unit 3-ball.

    References
    ----------
    .. [1] http://math.stackexchange.com/a/87243

    """
    # This also ends up being slower than _ball.
    R = prng.rand(size)**(1/3)
    z = 2 * prng.rand(size) - 1
    theta = 2 * np.pi * prng.rand(size)
    r = np.sqrt(R ** 2 - z ** 2)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.array([x, y, z]).transpose()

def norm(pmf, ilrcov=None, size=None, prng=None):
    """
    Returns normally distributed mass functions with mean equal to `pmf`.

    Parameters
    ----------
    pmf : NumPy array, shape (n,)
        The probability mass function about which samples are drawn.
    ilrcov : float, NumPy array with shape (n-1,) or (n-1,n-1)
        The covariance matrix in isometric log-ratio coordinates. If a float,
        then a covariance matrix is constructed as a scalar multiple of the
        identity matrix---e.g. spherical covariance. If `ilrcov` is a 1D
        NumPy array, then it specifies the diagonal of the covariance matrix,
        all non-diagonal elements are set equal to zero. Otherwise, `ilrcov`
        should be an (n-1, n-1) 2D symmetric and positive semi-definite matrix.
        If `None`, then an identity matrix is used as the covariance matrix.
    size : int
        The number of samples to draw. Default is to return a single sample.
    prng : NumPy RandomState
        A random number generator. If `None`, then `dit.math.prng` is used.

    Return
    ------
    samples : NumPy array, shape (n,) or (`size`, n)
        The samples. If `size` is `None`, then a single sample is returned.

    """
    if prng is None:
        prng = dit.math.prng

    pmf = np.asarray(pmf)
    if len(pmf.shape) != 1:
        raise dit.exceptions.ditException('`pmf` must be a 1D array.')

    ilrmean = dit.math.aitchison.ilr(pmf)
    n = len(pmf)

    # Determine ilr covariance matrix. Include some simple checks since
    # its shape (n-1, n-1) can be a common source of error. But note, we are
    # not checking symmetry or positive semi-definiteness.
    if ilrcov is None:
        # unit covariance
        ilrcov = np.eye(n-1)
    else:
        ilrcov = np.asarray(ilrcov)
        D = len(ilrcov.shape)
        if D == 0:
            # spherical covariance
            ilrcov = ilrcov * np.eye(n-1)
        elif D == 1:
            # diagonal covariance
            if ilrcov.shape != (n-1,):
                msg = '`ilrcov` must have shape (n-1,)'
                raise dit.exceptions.ditException(msg)

            x = np.eye(n - 1)
            x[np.diag_indices(n-1)] = ilrcov
            ilrcov = x
        elif D == 2:
            # user specified covariance
            if ilrcov.shape != (n-1, n-1):
                msg = '`ilrcov` must have shape (n-1, n-1)'
                raise dit.exceptions.ditException(msg)
        else:
            raise dit.exceptions.ditException('`ilrcov` must be a 2D array.')

    if size is None:
        k = 1
    else:
        k = size

    samples = _norm(ilrmean, ilrcov, k, prng)
    if size is None:
        samples = samples[0]
    return samples

def _norm(ilrmean, ilrcov, size, prng):
    """
    Low-level sampling of normally distributed pmfs about a mean pmf.

    """
    samples = prng.multivariate_normal(ilrmean, ilrcov, size)
    samples = dit.math.aitchison.ilr_inv(samples)
    return samples


def _annulus2(rmin, rmax, size=None, prng=None):
    """
    Return samples uniformly distributed within an annulus of a 2-sphere.

    Parameters
    ----------
    rmin : float
        The minimum radius.
    rmax : float
        The maximum radius.
    size : int
        The number of samples to draw.
    prng : NumPy RandomState
        A random number generator. If `None`, then `dit.math.prng` is used.

    Returns
    -------
    samples : NumPy array, shape (`size`, 2)
        Points within the annulus of a 2-ball.

    References
    ----------
    .. [1] http://stackoverflow.com/a/9048443

    """
    if size is None:
        s = 1
    else:
        s = size

    if prng is None:
        prng = dit.math.prng

    U = prng.rand(s)
    r = np.sqrt(U * (rmax**2 - rmin**2) + rmin**2)

    theta = prng.rand(s) * 2 * np.pi

    samples = np.array([r * np.cos(theta), r * np.sin(theta)])
    samples = samples.transpose()

    if size is None:
        samples = samples[0]

    return samples

def annulus2(pmf, rmin, rmax, size=None, prng=None):
    """
    Returns pmfs uniformly distributed in an annulus around `pmf`.

    `pmf` must live on the 2-simplex.

    Parameters
    ----------
    pmf : NumPy array, shape (3,)
        The probability mass function about which samples are drawn.
    rmin : float
        The minimum radius.
    rmax : float
        The maximum radius.
    size : int
        The number of samples to draw.
    prng : NumPy RandomState
        A random number generator. If `None`, then `dit.math.prng` is used.

    Returns
    -------
    samples : NumPy array, shape (`size`, 2)
        Points within the annulus around `pmf`.

    """
    samples = _annulus2(rmin, rmax, size, prng)
    ilrpmf = dit.math.aitchison.ilr(np.asarray(pmf))
    samples += ilrpmf
    samples = dit.math.aitchison.ilr_inv(samples)

    if size is None:
        samples = samples[0]

    return samples

# Load the cython function if possible

try: # pragma: no cover
    from ._samplediscrete import sample as _sample_discrete__cython
    _sample = _sample_discrete__cython
except ImportError: # pragma: no cover
    _sample = _sample_discrete__python

try: # pragma: no cover
    from ._samplediscrete import samples as _samples_discrete__cython
    _samples = _samples_discrete__cython
except ImportError: # pragma: no cover
    _samples = _samples_discrete__python
