.. lautum_information.rst
.. py:module:: dit.multivariate.lautum_information

******************
Lautum Information
******************

The lautum information :cite:`palomar2008lautum` is, in a sense, the mutual information in reverse (*lautum* is *mutual* backwards):

.. math::

   \L{X_{0:n}} = \DKL{X_0 \cdot X_1 \cdot \ldots \cdot X_n || X_{0:n}}


API
===

.. autofunction:: lautum_information
