.. wyner_common_information.rst
.. py:module:: dit.multivariate.wyner_common_information

************************
Wyner Common Information
************************

The Wyner common information :cite:`wyner1975common,liu2010common` measures the minimum amount of information which 

.. math::

   \C[X_{0:n}|Y_{0:m}] = \min_{\ind X_{0:n} \mid Y_{0:m}, V} \I[X_{0:n} : V | Y_{0:m}]

API
===

.. autofunction:: wyner_common_information
