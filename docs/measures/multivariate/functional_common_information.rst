.. functional_common_information.rst
.. py:module:: dit.multivariate.functional_common_information

*****************************
Functional Common Information
*****************************

The functional common information captures the minimum amount of information neccessary to capture all of a distribution's share information using a function of that information. In other words:

.. math::
   \F[X_{0:n}] = \min_{\substack{W = f(X_{0:n} \\ \B[X_{0:n}|W] = 0} \H[W]
