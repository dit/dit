.. rate_distortion.rst
.. py:module:: dit.rate_distortion

**********************
Rate Distortion Theory
**********************

Rate-distortion theory is a framework for studying optimal lossy compression. Given a distribution :math:`p(x)`, we wish to find :math:`q(\hat{x}|x)` which compresses :math:`X` as much as possible while limiting the amount of user-defined distortion:

.. math::

   R(D) = \min_{q(\hat{x}|x), \langle d(x, \hat{x}) \rangle = D} \I{X : \hat{X}}



Information Bottleneck
======================