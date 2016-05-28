.. functional_common_information.rst
.. py:module:: dit.multivariate.functional_common_information

*****************************
Functional Common Information
*****************************

The functional common information captures the minimum amount of information neccessary to capture all of a distribution's share information using a function of that information. In other words:

.. math::

   \F[X_{0:n} \mid Y_{0:m}] = \min_{\substack{\ind X_{0:n} \mid Y_{0:m}, W \\ W = f(X_{0:n}, Y_{0:m})}} \H[W]

Relationship To Other Measures of Common Information
====================================================

Since this is an additional constraint on the Exact common information, it is generally larger than it, and since its constraint is weaker than that of the :doc:`mss_common_information`, it is generally less than it:

.. math::

   \G[X_{0:n}] \leq \F[X_{0:n}] \leq \M[X_{0:n}]

API
===

.. autofunction:: functional_common_information
