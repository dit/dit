.. w_information.rst
.. py:currentmodule:: dit.multivariate

.. _w_information:

*******************************
W-Information and M-Information
*******************************

Liardi et al. :cite:`liardi2025scalable` split the mutual information between sources :math:`X = (X_0, \ldots, X_{n-1})` and targets :math:`Y = (Y_0, \ldots, Y_{m-1})` into the part that the source-target pairs account for and the rest. The W-information is the least mutual information among the distributions that share every pairwise marginal of a source with a target:

.. math::

   \mathcal{W}(X : Y) = \min_{q \in \Delta_p} \I{X : Y}_q, \qquad
   \Delta_p = \left\{ q : q(x_i, y_j) = p(x_i, y_j) \text{ for all } i, j \right\},

and the M-information is the rest:

.. math::

   \mathcal{M}(X : Y) = \I{X : Y} - \mathcal{W}(X : Y) \geq 0.

Only the source-target pairs are kept: the dependencies among the sources, and among the targets, are free. With a single target, :math:`\mathcal{W}` is the union information of Bertschinger et al. and Griffith & Koch :cite:`bertschinger2014quantifying,griffith2014quantifying`. For two sources it is :math:`\I{X_0 : Y} + \I{X_1 : Y} - \Ibroja{X_0, X_1 : Y}`, and :math:`\mathcal{M}` is the BROJA synergy.

Liardi et al. read :math:`\mathcal{W}` as the lower-order (pairwise) information and :math:`\mathcal{M}` as the higher-order information. The reading is exact when some distribution in :math:`\Delta_p` carries no higher-order information. Otherwise :math:`\mathcal{W}` overestimates the lower-order information and :math:`\mathcal{M}` underestimates the higher-order information.

Copying a bit forward is entirely pairwise, while the parity of two bits is invisible to every pair:

.. ipython::

   In [1]: from dit.multivariate import m_information, w_information

   In [2]: d = dit.Distribution(['0000', '1111'], [1/2, 1/2])

   @doctest float
   In [3]: w_information(d)
   Out[3]: 1.0

   In [4]: d = dit.Distribution(['000', '011', '101', '110'], [1/4] * 4)

   @doctest float
   In [5]: m_information(d, [[0], [1]], [[2]])
   Out[5]: 1.0

By default the first half of the variables are the sources and the second half the targets. ``sources`` and ``targets`` take groups of variables, each counted as one source or target.

The BROJA-:math:`\Phi\mathrm{ID}` of the same paper, which splits :math:`\mathcal{W}` and :math:`\mathcal{M}` into the atoms of the integrated information decomposition, is :class:`~dit.pid.PhiID` with ``measure='broja'``.

Computation
===========

The minimum is found by alternating minimization :cite:`csiszar1984information`. Given marginals :math:`r_X` and :math:`r_Y`, the distribution :math:`q` is the I-projection of :math:`r_X r_Y` onto :math:`\Delta_p`, computed by Newton's method on its dual, and :math:`r_X, r_Y` are then replaced by the marginals of :math:`q`. Each step lowers :math:`\I{X : Y}_q`. Outcomes that the pairwise marginals allow but that no distribution in :math:`\Delta_p` can give positive probability are first found by linear programs and left out. A minimum that leaves some values of :math:`X` or :math:`Y` without mass is only approached sublinearly, so such values are dropped once their mass is small and still shrinking, and restored if they would grow again.

When the sources or the targets form a single group, that marginal is fixed and the problem is convex. Then

.. math::

   \max_x \log \frac{q(x)}{r_X(x)} - \DKL{q \| r_X},

and its counterpart for :math:`Y`, bound the distance to the minimum, by duality. The iteration stops once they are below ``tol``.

Otherwise the problem is not convex. On four uniform bits, for instance, both :math:`p` and the distribution with :math:`X_0 = X_1` and :math:`Y_0 = Y_1` attain :math:`\mathcal{W} = 0`, but their mixtures do not, so the minimizers form a nonconvex set. There can also be local minima that are not global, e.g. for three binary sources and three binary targets.

The minimization therefore starts from the uniform marginals and from ``niter`` random ones. Each run first descends by L-BFGS in the logarithms of the marginals, which crosses the flat valleys that the alternating projections only creep along, and then alternates briefly. The best run then alternates until the same quantity, now a measure of stationarity, is below ``tol``. Its value is returned: an upper bound on :math:`\mathcal{W}`, so that :func:`m_information` is then a lower bound on :math:`\mathcal{M}`. In nearly flat landscapes, for larger systems or near-independent variables, the stationarity measure can stay above ``tol``; a warning then reports it, although the value is usually much more accurate.

The Gaussian estimator of :cite:`liardi2025scalable` is not provided, since ``dit`` handles discrete distributions only.

API
===

.. autofunction:: w_information
   :no-index:

.. autofunction:: m_information
   :no-index:
