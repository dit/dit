.. cross_mutual_information.rst
.. py:module:: dit.multivariate.cross_mutual_information

.. _cross_mutual_information:

************************
Cross Mutual Information
************************

The cross mutual information :cite:`gohil2025cross`, denoted :math:`CI_{pq}`, quantifies how strongly an :math:`X`-:math:`Y` dependence defined by a *reference* distribution :math:`q` is expressed in test data sampled from :math:`p`. In analogy to the cross entropy, the pointwise information of each outcome is evaluated using :math:`q` while the expectation is taken over :math:`p`:

.. math::

   CI_{pq} = \sum_{x, y} p(x, y) \log_2 \frac{q(x, y)}{q(x) q(y)}

When :math:`p = q` the cross mutual information reduces to the ordinary :ref:`mutual_information`. If the reference distribution factorizes (:math:`X` and :math:`Y` are independent under :math:`q`) then the cross mutual information is zero for any test distribution. Unlike the conventional mutual information, the cross mutual information can be negative: this occurs when the dependence in the test data is surprising relative to the reference.

.. ipython::

   In [1]: from dit import Distribution

   In [2]: from dit.multivariate import cross_coinformation as CI

   In [3]: p = Distribution(['00', '01', '10', '11'], [0.4, 0.1, 0.1, 0.4])

   In [4]: q = Distribution(['00', '01', '10', '11'], [0.1, 0.4, 0.4, 0.1])

   @doctest float
   In [5]: CI(p, p)
   Out[5]: 0.2780719051126379

   @doctest float
   In [6]: CI(p, q)
   Out[6]: -0.9219280948873623

Generalizations
===============

Each of the multivariate mutual informations is a signed sum of joint entropies, and so admits a cross generalization by replacing every entropy with the analogous cross entropy between :math:`p` and :math:`q`. In addition to the cross co-information (:func:`cross_coinformation`, the multivariate generalization of the cross mutual information above), ``dit`` provides the cross :ref:`total_correlation`, the cross :ref:`dual_total_correlation`, and the cross :ref:`caekl_mutual_information`. Each reduces to its conventional counterpart when :math:`p = q`.

.. ipython::

   In [7]: from dit.multivariate import cross_total_correlation as CT

   In [8]: from dit.multivariate import cross_dual_total_correlation as CB

   In [9]: from dit.multivariate import cross_caekl_mutual_information as CJ

API
===

.. autofunction:: cross_coinformation
.. autofunction:: cross_total_correlation
.. autofunction:: cross_dual_total_correlation
.. autofunction:: cross_caekl_mutual_information
