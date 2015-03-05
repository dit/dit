"""
Functions for manipulating compositions using the Aitchison geometry.

Throughout, we assume the compositions are defined such that the sum
of the components is 1.

http://www.springerlink.com/content/wx1166n56n685v82/
"""

from __future__ import division
from __future__ import absolute_import

import math
import numpy as np

from dit.exceptions import ditException
from dit.math import LogOperations

__all__ = ('closure',
           'subcomposition',
           'perturbation',
           'power',
           'add',
           'sub',
           'inner',
           'norm',
           'dist',
           'metric',
           'clr',
           'alr',
           'ilr',
           'basis',
           'clr_inv',
           'alr_inv',
           'ilr_inv',)

ops = LogOperations(2)
exp2 = ops.exp
log2 = ops.log

def _gm(x):
    """Returns the geometric means of the rows in `x`.

    Parameters
    ----------
    x : NumPy array, shape (k,n)
        The k compositions whose geometric means are to be computed.

    Returns
    -------
    x_gm : NumPy array, shape (k,)
        The geometric means for the k compositions in `x`.

    """
    last_axis = -1
    x_gm = x.prod(axis=last_axis) ** (1/x.shape[last_axis])

    return x_gm

def _log2_gm(x):
    """
    Returns the log of the geometric means for the rows in `x`.

    Parameters
    ----------
    x : NumPy array, shape (k,n)
        The k compositions whose geometric means are to be computed.

    Returns
    -------
    x_loggm : NumPy array, shape (k,)
        The log geometric means for the k compositions in `x`.

    """
    last_axis = -1
    x_loggm = 1 / x.shape[last_axis] * np.log2(x).sum(axis=last_axis)

    return x_loggm

def closure(x):
    """Returns the closure operation applied to the composition x.

    The closure operation renormalizes `x` so that its components sum to one.

    Parameters
    ----------
    x : NumPy array, shape (n,) or (k,n)
        The array can be one- or two-dimensional.  If one-dimensional, it is
        treated as a single composition. If two-dimensional, each row is
        treated as a composition and will be normalized individually.

    Returns
    -------
    cx : NumPy array, shape (n,) or (k,n)
        The closure of `x`.

    """
    s = x.sum(axis=-1, dtype=float)
    if np.any(s == 0.0):
        raise ditException("x contains an unnormalizable distribution.")
    cx = x / s[..., np.newaxis]
    return cx

def subcomposition(x, indexes):
    """Returns the subcomposition over the specified indexes.

    The subcomposition is the closure of a subset of events in the composition.

    Parameters
    ----------
    x : NumPy array, shape (n,) or (k,n)
        The composition(s) that will be truncated and renormalized.

    Returns
    -------
    xsub : NumPy array, shape (len(`indexes`),) or (k, len(`indexes`))
        The subcompositions of `x`.

    """
    xsub = closure(x[..., indexes])

    return xsub

def perturbation(x, dx):
    """Returns the perturbation of `x` by `dx`.

    Perturbation is the closure of the element-wise product. It is equivalent
    to translation (inner sum) in standard Euclidean space.

    Parameters
    ----------
    x : NumPy array, shape (n,) or (k,n)
        The composition (or k compositions) to be perturbed.
    dx : NumPy array
        The perturbation composition or (k perturbation compositions).

    Returns
    -------
    px : NumPy array, shape (n,) or (k,n)
        The perturbation of `x` by `dx`.

    """
    px = closure(x * dx)

    return px

def power(x, a):
    """Returns the result of powering `x` by `a`.

    The power transformation is the closure of raising each element to the
    `a`th power. It is equivalent to scalar multiplication (outer product).

    Parameters
    ----------
    x : NumPy array, shape (n,) or (k,n)
        The composition (or k compositions) which will be powered.
    a : NumPy array, shape () or (k,)
        The power (or k powers) to which the composition(s) is raised.

    Returns
    -------
    px : NumPy array, shape (n,) or (k,n)
        The result of powering `x` by `a`.

    """
    a = np.ravel(a)[..., np.newaxis]
    px = closure(x**a)
    if len(x.shape) == 1:
        px = px[0]

    return px

add = perturbation

def sub(x, y):
    """Returns the difference of compositions.

    Parameters
    ----------
    x : NumPy array, shape (n,) or (k,n)
        The composition that will be subtracted from.
    y : NumPy array, shape (n,) or (k,n)
        The composition to be subtracted.

    Returns
    -------
    z : NumPy array, shape (n,) or (k,n)
        The result of y subtracted from x.

    """
    z = perturbation(x, power(y, -1.0)) # 1.0 and not 1 forces coercion
    return z

def inner(x, y):
    """Returns the Aitchison inner product between `x` and `y`.

    Parameters
    ----------
    x,y : NumPy array, shape (n,) or (k,n)
        Compositions to be used in the inner product.

    Returns
    -------
    z : NumPy array, shape () or (k,)
        The inner product of x and y.  If `x` and `y` are 2D arrays, then the
        inner product is done row-wise and `z` is a 1D array of floats.
        Otherwise, a float is returned.

    """
    if len(x.shape) == 1 and len(y.shape) == 1:
        single = True
    else:
        single = False

    x = np.atleast_2d(x)
    y = np.atleast_2d(y)

    x_loggm = _log2_gm(x)[:, np.newaxis]
    y_loggm = _log2_gm(y)[:, np.newaxis]

    z = (log2(x) - x_loggm) * (log2(y) - y_loggm)
    z = z.sum(axis=1)

    if single:
        z = z[0]

    return z

def norm(x):
    """Returns the norm of `x`.

    Parameters
    ----------
    x : NumPy array, shape (n,) or (k,n)
        The composition(s) to be normed.

    Returns
    -------
    n : NumPy array, shape () or (k,)
        The norm(s) of the composition(s).

    """
    n = np.sqrt(inner(x, x))
    return n

def dist(x, y):
    """Returns the distance between `x` and `y`.

    Parameters
    ----------
    x, y : NumPy array, shape (n,) or (k,n)
        The compositions whose distance is computed.

    Returns
    -------
    d : NumPy array, shape () or (k,)
        The distance between `x` and `y`.

    """
    d = norm(sub(x, y))
    return d

metric = dist

def clr(x):
    """Returns the centered log-ratio transformation of `x`.

    The centered log-ratio transformation of `x` is defined as:

        clr(x) = \\log_2( \\frac{x}{g(x)} )
               = \\log_2(x) - <\\log_2 x_i>

    where g(x) is the geometric mean of `x`.

    Parameters
    ----------
    x : NumPy array, shape (n,) or (k,n)
        Composition(s) to be transformed by clr.

    Returns
    -------
    y : NumPy array, shape (n,) or (k,n)
        Centered log-ratio transformation(s) of `x`.

    """
    if len(x.shape) == 1:
        single = True
    else:
        single = False

    x = np.atleast_2d(x)
    x_loggm = _log2_gm(x)[:, np.newaxis]
    y = log2(x) - x_loggm

    if single:
        y = y[0]

    return y

def alr(x):
    """Returns the additive log-ratio transformation of `x`.

    The additive log-ratio transformation of `x` (with respect to the last
    component in the composition) is defined as:

        alr(x) = [ \\log_2 x_1 / x_D, \\ldots, \\log_2 \\frac{x_{D-1}}{x_D} ]

    where `x` is a composition of length D.  Essentially, take the first D-1
    components and divide them by the Dth component.

    Parameters
    ----------
    x : NumPy array, shape (n,) or (k,n)
        Composition(s) to be transformed by alr.

    Returns
    -------
    y : NumPy array, shape (n-1,) or (k,n-1)
        Additive log-ratio transformation(s) of `x`.

    """
    if len(x.shape) == 1:
        single = True
    else:
        single = False

    x = np.atleast_2d(x)

    y = log2(x[:, :-1]) - log2(x[:, -1][:, np.newaxis])

    if single:
        y = y[0]

    return y

def ilr(x):
    """Returns the isometric log-ratio transformation of `x`.

    The isometric log-ratio transformation of `x`, with respect to the
    canonical, orthonormal basis defined on the simplex (equation 18 in the
    paper), is defined as:

        y_i = ilr(x)_i = k_i \\log_2 \\frac{g(x_1,\\ldots,x_i)}{x_{i+1}}

    where

        k_i = \\sqrt{ \\frac{i}{i+1} }

        g_i = \\bigl( \\prod_{k=1}^i x_k \\bigr)^{1/i}

    for i = 1, 2, ..., D-1.

    Parameters
    ----------
    x : NumPy array, shape (n,) or (k,n)
        Composition(s) to be transformed by ilr.

    Returns
    -------
    y : NumPy array, shape (n-1,) or (k, n-1)
        Isometric log-ratio transformation(s) of `x`.

    """
    if len(x.shape) == 1:
        single = True
    else:
        single = False

    x = np.atleast_2d(x)

    rng = np.arange(1, x.shape[1])
    #gm = (x.cumprod(axis=1)[:, :-1])**(1/rng)
    loggm = 1 / rng * log2(x).cumsum(axis=1)[:, :-1]
    y = loggm - log2(x[:, 1:])
    y *= np.sqrt([i/(i+1) for i in rng]) # same coefficient for each column

    if single:
        y = y[0]

    return y

def ubasis(n):
    """Returns an orthonormal basis wrt the ordinary Euclidean inner product.

    The vectors constitute a basis of the `n`-dimensional linear subspace
    V_S.  There are `n` elements in the basis, each of which lives in an
    (`n`+1)-dimensional space.  From equation (17), each u_i is defined as:

        u_i = \\sqrt{i}{i+1} ( 1/i, ..., 1/i, -1, 0, ..., 0 )

    where there are i elements in the sequence of 1/i fractions.

    Parameters
    ----------
    n : int
        The dimensionality of the basis.

    Returns
    -------
    b : NumPy array, shape (`n`, `n`+1)
        The orthonormal basis.

    """
    # Upper triangle above main diagonal is zero. Everything else is 1.
    u = np.tri(N=n, M=n+1, k=1)

    # Set the lower triangle to 1/i for each row and apply coefficent
    rng = np.arange(1, n+1)
    u *= np.array([1/i for i in rng])[:, np.newaxis]

    # the 1st diag is set to -1
    u.flat[1::n+2] = -1

    # scale everything
    u *= np.array([math.sqrt(i/(i+1)) for i in rng])[:, np.newaxis]

    return u

def basis(n):
    """Returns an orthonormal basis wrt the Aitchison inner product.

    Parameters
    ----------
    n : int
        The dimensionality of the basis.  For example, the 2-simplex has a
        two-dimensional basis.

    Returns
    -------
    b : NumPy array, shape (`n`, `n`+1)
        The basis for the `n`-simplex, consisting of vectors of length `n`+1.

    """
    u = ubasis(n)
    b = clr_inv(u)

    return b

def clr_inv(xclr):
    """"Returns the inverse centered log-ratio transformation of x.

    Parameters
    ----------
    xclr : NumPy array, shape (n,) or (k,n)
        The centered log-ratio transformations of x.

    Returns
    -------
    x : NumPy array, shape (n,) or (k,n)
        The original compositions.

    """
    x = closure(exp2(xclr))
    return x

def alr_inv(xalr):
    """Returns the inverse additive log-ratio transformation of x.

    Parameters
    ----------
    xalr : NumPy array, shape (n,) or (k,n)
        The additive log-ratio transformations of x.

    Returns
    -------
    x : NumPy array, shape (n+1,) or (k,n+1)
        The original compositions

    Notes
    -----
    The sum of the composition is assumed to be 1.

    """
    if len(xalr.shape) == 1:
        single = True
    else:
        single = False

    xalr = np.atleast_2d(xalr)

    newshape = list(xalr.shape)
    newshape[1] += 1
    x = np.empty(newshape)

    x[:, :-1] = exp2(xalr)

    ### Now we can exactly solve for the last element, and
    ### then we can unscale each of the other components.
    #x[:, -1] = 1 / (1 + x[:, :-1].sum(axis=1))
    #x[:, :-1] *= x[:, -1][:, np.newaxis]

    ### Or we can set the last element equal to 1 and apply closure.
    ### This is quicker so we do that.
    x[:, -1] = 1
    x = closure(x)

    if single:
        x = x[0]

    return x

def ilr_inv(xilr):
    """Returns the inverse isometric log-ratio transformation of x.

    Parameters
    ----------
    xilr : NumPy array, shape (n,) or (k,n)
        The isometric log-ratio transformations of x.

    Returns
    -------
    x : NumPy array, shape (n+1,) or (k,n+1)
        The original compositions.

    """
    if len(xilr.shape) == 1:
        single = True
    else:
        single = False

    xilr = np.atleast_2d(xilr)

    newshape = list(xilr.shape)
    newshape[1] += 1
    x = np.empty(newshape)

    b = basis(xilr.shape[1])
    for i in range(xilr.shape[0]):
        # Here is what you'd normally do:
        #
        # closure(power(b, xilr[i]).prod(axis=0))
        #
        # but the product is multiplying a bunch a small numbers and it will
        # overflow to zero. This makes the closure operation fail.
        # Instead, we need to do everything with logs.
        #
        poww = power(b, xilr[i])
        logprods = ops.mult_reduce(log2(poww), axis=0)
        logprobs = ops.normalize(logprods)
        x[i] = ops.exp(logprobs)

    if single:
        x = x[0]

    return x


