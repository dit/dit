.. functional_common_information.rst
.. py:currentmodule:: dit.multivariate

*****************************
Functional Common Information
*****************************

The functional common information captures the minimum amount of information neccessary to capture all of a distribution's share information using a function of that information. In other words:

.. math::

   \F{X_{0:n} \mid Y_{0:m}} = \min_{\substack{\ind X_{0:n} \mid Y_{0:m}, W \\ W = f(X_{0:n}, Y_{0:m})}} \H{W}

It was introduced in :cite:`james2017multivariate`.

Computation
===========

A function :math:`W` is a partition of the support, and :math:`X_{0:n}` are independent given :math:`W = w` (and :math:`Y_{0:m}`) exactly when the outcomes in that block carry a product distribution on a combinatorial rectangle. Because :math:`\H{W}` is a sum of per-block costs :math:`-p(w) \log_2 p(w)`, the functional common information is a minimum-weight exact cover of the support by such blocks, which ``dit`` solves exactly as an integer program. Without conditioning, every block lies within one connected component of the support (the blocks of the Gács-Körner common variable), so each component is solved independently.

Relationship To Other Measures of Common Information
====================================================

Since this is an additional constraint on the Exact common information, it is generally larger than it, and since its constraint is weaker than that of the :doc:`/measures/common_informations/mss_common_information`, it is generally less than it:

.. math::

   \G{X_{0:n}} \leq \F{X_{0:n}} \leq \M{X_{0:n}}

Example
=======

For the distribution uniform over :math:`\{00, 01, 11\}`, the support is
connected, so the Gács–Körner common information is zero, yet any variable
rendering :math:`X` and :math:`Y` independent must still carry the binary
split :math:`\{00, 01\} \mid \{11\}`. The functional common information is
the entropy of that split, :math:`\H{1/3}`, while the MSS common information
needs the full :math:`\log_2 3` bits:

.. ipython::

   In [1]: from dit.multivariate import functional_common_information, mss_common_information

   In [2]: d = dit.Distribution(['00', '01', '11'], [1/3] * 3)

   @doctest float
   In [3]: functional_common_information(d)
   Out[3]: 0.9182958340544896

   @doctest float
   In [4]: mss_common_information(d)
   Out[4]: 1.584962500721156

API
===

.. autofunction:: functional_common_information
   :no-index:

.. autofunction:: functional_variable
   :no-index:
