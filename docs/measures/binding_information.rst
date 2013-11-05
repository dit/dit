.. binding_information.rst

*******************
Binding Information
*******************

The binding information :cite:`Abdallah2012`, or dual total correlation, is yet
another generalization of the mutual information. It is defined as:

.. math::

   \B[X_{0:n}] = \H[X_{0:n}] - \sum \H[X_i | X_{\{0..n\}/i}]

.. autofunction:: dit.algorithms.binding_information

.. bibliography:: ../references.bib
   :cited:
