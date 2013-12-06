
from __future__ import division

from nose.tools import (assert_almost_equal, assert_equal, assert_true)

import numpy as np

from numpy import allclose

from dit.math import LogOperations
ops = LogOperations(2)
log2 = ops.log
exp2 = ops.exp

from dit.math.aitchison import (alr, alr_inv, basis, closure, clr, clr_inv,
                                dist, ilr, ilr_inv, inner, norm, perturbation,
                                power)

def assert_almost_equal_array(x, y):
    for i, j in zip(x.ravel(), y.ravel()):
        assert_almost_equal(i, j)

def assert_equal_shape(x, y):
    assert_equal(x.shape, y.shape)

def test_closure():
    # 1D
    x = np.array([1.0, 1.0])
    y = np.array([0.5, 0.5])
    cl_x = closure(x)
    assert_equal_shape(x, cl_x)
    assert_almost_equal_array(cl_x, y)

    # 2D with multiple rows
    x = np.array([[1.0, 1.0], [1.0, 2.0]])
    y = np.array([[0.5, 0.5], [1/3.0, 2/3.0]])
    cl_x = closure(x)
    assert_equal_shape(x, cl_x)
    assert_almost_equal_array(cl_x, y)

    # 2D with one row
    x = np.array([[1.0, 1.0]])
    y = np.array([[0.5, 0.5]])
    cl_x = closure(x)
    assert_equal_shape(x, cl_x)
    assert_almost_equal_array(cl_x, y)

def test_closure_invariance():
    x = np.array([1, 2, 3, 4]) / 10
    y = x*2
    z = x*.5
    xcl = closure(x)
    ycl = closure(y)
    zcl = closure(z)

    assert_almost_equal_array(xcl, ycl)
    assert_almost_equal_array(ycl, zcl)
    assert_almost_equal_array(zcl, xcl)

def test_perturbation():
    # 1D
    x = np.array([1/3.0, 2/3.0])
    y = np.array([3/4.0, 1/4.0])
    pxy_ = np.array([3/5.0, 2/5.0])
    pxy = perturbation(x, y)
    assert_true(x.shape == pxy.shape)
    assert_true(allclose(pxy_, pxy))

    # 2D with multiple rows
    x = np.array([[1/3.0, 2/3.0], [3/4.0, 1/4.0]])
    y = np.array([[3/4.0, 1/4.0], [1/6.0, 5/6.0]])
    pxy_ = np.array([[3/5.0, 2/5.0], [3/8.0, 5/8.0]])
    pxy = perturbation(x, y)
    assert_true(x.shape == pxy.shape)
    assert_true(allclose(pxy_, pxy))

    # 2D with one row
    x = np.array([[1/3.0, 2/3.0]])
    y = np.array([[3/4.0, 1/4.0]])
    pxy_ = np.array([[3/5.0, 2/5.0]])
    pxy = perturbation(x, y)
    assert_true(x.shape == pxy.shape)
    assert_true(allclose(pxy_, pxy))

def test_power():
    # 1D
    x = np.array([1/3.0, 2/3.0])
    y = 3
    px_ = np.array([1/9.0, 8/9.0])
    px = power(x, y)
    assert_true(px_.shape == px.shape)
    assert_true(allclose(px_, px))
    x = np.array([1/4.0, 3/4.0])
    y = 3
    px_ = np.array([1/28.0, 27/28.0])
    px = power(x, y)
    assert_true(px_.shape == px.shape)
    assert_true(allclose(px_, px))

    # 2D with multiple rows
    x = np.array([[1/3.0, 2/3.0], [1/4.0, 3/4.0]])
    y = [3, 3]
    px_ = np.array([[1/9.0, 8/9.0], [1/28.0, 27/28.0]])
    px = power(x, y)
    assert_true(px_.shape == px.shape)
    assert_true(allclose(px_, px))

    # 2D with one row
    x = np.array([[1/3.0, 2/3.0]])
    y = [3]
    px_ = np.array([[1/9.0, 8/9.0]])
    px = power(x, y)
    assert_true(px_.shape == px.shape)
    assert_true(allclose(px_, px))

def test_inner():
    # 1D
    x = np.array([1/3.0, 2/3.0])
    y = np.array([1/4.0, 3/4.0])
    z = np.array([2/5.0, 3/5.0])

    xy_ = 0.79248125036057804
    xy = inner(x, y)
    yx = inner(y, x)
    assert_almost_equal(xy_, xy)
    assert_almost_equal(xy_, yx)

    yz_ = 0.46357181398555231
    yz = inner(y, z)
    zy = inner(z, y)
    assert_almost_equal(yz_, yz)
    assert_almost_equal(yz_, zy)

    xz_ = 0.29248125036057804
    xz = inner(x, z)
    zx = inner(z, x)
    assert_almost_equal(xz_, xz)
    assert_almost_equal(xz_, zx)

    # 2D with multiple rows
    xx = np.array([x, y, z])
    yy = np.array([y, z, x])
    xxyy = inner(xx, yy)
    xxyy_ = np.array([xy_, yz_, xz_])
    assert_true(allclose(xxyy_, xxyy))

    # 2D with multiple rows in x and single row in y
    yy_ = 1.2560530643461303
    xx = np.array([x, y, z])
    yy = np.array([y])
    xxyy = inner(xx, yy)
    xxyy_ = np.array([xy_, yy_, yz_])
    assert_true(allclose(xxyy_, xxyy))
    yyxx = inner(yy, xx)
    assert_true(allclose(xxyy_, yyxx))

    # 2D with single rows
    x2d = np.atleast_2d(x)
    y2d = np.atleast_2d(y)
    xy2d = inner(x2d, y2d)
    xy2d_ = np.array([xy_])
    assert_true(allclose(xy2d_, xy2d))

def test_clr():
    x = np.array([1/3.0, 2/3.0])
    y = np.array([1/4.0, 3/4.0])

    x_clr = [-0.5, 0.5]
    y_clr = [-0.79248125036057782, 0.79248125036057815]

    # 1D
    x1d_clr = clr(x)
    x1d_clr_ = np.array(x_clr)
    assert_equal(x1d_clr.shape, x1d_clr_.shape)
    assert_true(allclose(x1d_clr, x1d_clr_))

    # 2D
    xy = np.array([x, y])
    xy_clr = clr(xy)
    xy_clr_ = np.array([x_clr, y_clr])
    assert_equal(xy_clr.shape, xy_clr_.shape)
    assert_true(allclose(xy_clr, xy_clr_))

def test_alr():
    x = np.array([1/3.0, 2/3.0])
    y = np.array([1/4.0, 3/4.0])

    x_alr = [-1]
    y_alr = [-1.5849625007211563]

    # 1D
    x1d_alr = alr(x)
    x1d_alr_ = np.array(x_alr)
    assert_equal(x1d_alr.shape, x1d_alr_.shape)
    assert_true(allclose(x1d_alr, x1d_alr_))

    # 2D
    xy = np.array([x, y])
    xy_alr = alr(xy)
    xy_alr_ = np.array([x_alr, y_alr])
    assert_equal(xy_alr.shape, xy_alr_.shape)
    assert_true(allclose(xy_alr, xy_alr_))

def test_ilr():
    # 1D
    x = np.array([1, 2, 3, 4])
    lg = log2( [(x[0]              )**(1/1) / x[1],
                (x[0] * x[1]       )**(1/2) / x[2],
                (x[0] * x[1] * x[2])**(1/3) / x[3]] )
    coeff = np.sqrt([1/2, 2/3, 3/4])
    ilrx_ = coeff * lg
    ilrx = ilr(x)

    assert_equal(ilrx.shape, ilrx_.shape)
    for i, j in zip(ilrx, ilrx_):
        assert_almost_equal(i, j)

    # 2D with multiple rows
    x = np.array([1, 2, 3, 4])
    y = np.array([2, 3, 4, 5])
    ilrx_ = np.array([-0.70710678, -0.88586817, -0.98583641])
    ilry_ = np.array([-0.41363095, -0.57768664, -0.68728405])
    xy = np.array([x, y])
    ilrxy_ = np.array([ilrx_, ilry_])
    ilrxy = ilr(xy)

    assert_equal(ilrxy.shape, ilrxy_.shape)
    for i, j in zip(ilrxy.ravel(), ilrxy_.ravel()):
        assert_almost_equal(i, j)

    # 2D with single rows
    xy = np.array([x])
    ilrxy_ = np.array([ilrx_])
    ilrxy = ilr(xy)

    assert_equal(ilrxy.shape, ilrxy_.shape)
    for i, j in zip(ilrxy.ravel(), ilrxy_.ravel()):
        assert_almost_equal(i, j)

def test_equiv_inner1():
    x = np.array([1, 2, 3, 4])
    y = np.array([3, 5, 1, 1])
    aitchison_inner = inner(x, y)
    euclidean_inner = np.inner(ilr(x), ilr(y))
    assert_almost_equal(aitchison_inner, euclidean_inner)

def test_equiv_inner2():
    x = np.array([1, 2, 3, 4])
    y = np.array([3, 5, 1, 1])
    clrx = clr(x)
    clry = clr(y)
    D = len(x)
    M = -1 * np.ones((D, D))
    M.flat[::D+1] = D - 1

    z1 = inner(x, y)
    z2 = 1/D * np.dot(np.dot(clrx, M), clry[:, np.newaxis])
    assert_almost_equal(z1, z2)

def test_equiv_norm():
    x = np.array([1, 2, 3, 4])
    aitchison_norm = norm(x)
    euclidean_norm = np.linalg.norm(ilr(x))
    assert_almost_equal(aitchison_norm, euclidean_norm)

def test_equiv_dist():
    x = np.array([1, 2, 3, 4])
    y = np.array([3, 5, 1, 1])
    aitchison_dist = dist(x, y)
    euclidean_dist = np.linalg.norm(ilr(x)-ilr(y))
    assert_almost_equal(aitchison_dist, euclidean_dist)

def test_clr_prop():
    x1 = np.array([1, 2, 3, 4])
    x2 = np.array([3, 5, 1, 1])
    a1, a2 = 0.3, 0.9

    z1 = clr( perturbation( power(x1, a1), power(x2, a2) ) )
    z2 = a1 * clr(x1) + a2 * clr(x2)

    assert_almost_equal_array(z1, z2)

def test_basis():
    b = np.array([[1/1.,   -1,    0,    0,  0],
                  [1/2., 1/2.,   -1,    0,  0],
                  [1/3., 1/3., 1/3.,   -1,  0],
                  [1/4., 1/4., 1/4., 1/4., -1]])
    b *= np.sqrt([i/(i+1) for i in range(1, 5)])[:, np.newaxis]
    b = closure(exp2(b))
    b_ = basis(4)
    assert_equal_shape(b, b_)
    assert_almost_equal_array(b, b_)

def test_basis2():
    # test orthonormality in Aitchison inner product
    b = basis(4)
    for i in range(len(b)):
        ii = inner(b[i], b[i])
        assert_almost_equal(ii, 1.0)
        for j in range(i+1, len(b)):
            ij = inner(b[i], b[j])
            assert_almost_equal(ij, 0.0)

def test_clr_inv():
    x = np.array([1, 2, 3, 4]) / 10
    x_clr = clr(x)
    y = clr_inv(x_clr)
    assert_equal_shape(x, y)
    assert_almost_equal_array(x, y)

    x = np.array([[1, 2, 3, 4], [2, 2, 2, 4]]) / 10
    x_clr = clr(x)
    y = clr_inv(x_clr)
    assert_equal_shape(x, y)
    assert_almost_equal_array(x, y)

def test_alr_inv():
    x = np.array([1, 2, 3, 4]) / 10
    x_alr = alr(x)
    y = alr_inv(x_alr)
    assert_equal_shape(x, y)
    assert_almost_equal_array(x, y)

    x = np.array([[1, 2, 3, 4], [2, 2, 2, 4]]) / 10
    x_clr = clr(x)
    y = clr_inv(x_clr)
    assert_equal_shape(x, y)
    assert_almost_equal_array(x, y)


def test_ilr_inv():
    x = np.array([1, 2, 3, 4]) / 10
    x_ilr = ilr(x)
    y = ilr_inv(x_ilr)
    assert_equal_shape(x, y)
    assert_almost_equal_array(x, y)

    x = np.array([[1, 2, 3, 4], [2, 2, 2, 4]]) / 10
    x_clr = clr(x)
    y = clr_inv(x_clr)
    assert_equal_shape(x, y)
    assert_almost_equal_array(x, y)




