.. interaction_information.rst
.. py:currentmodule:: dit.multivariate

***********************
Interaction Information
***********************

The interaction information is equal in magnitude to the :doc:`/measures/multivariate/coinformation`, but has the opposite sign when taken over an odd number of variables:

.. math::

    \II{X_{0:n}} = (-1)^{n} \cdot \I{X_{0:n}}

Interaction information was first studied in the 3-variable case which, for :math:`X_{0:3} = X_0X_1X_2`, takes the following form:

.. math::

   \II{X_0 : X_1 : X_2} = \I{X_0 : X_1 | X_2} - \I{X_0 : X_1}

The extension to :math:`n > 3` proceeds recursively. For example,

.. math::

   \II{X_0 : X_1 : X_2 : X_3}
      &= \II{X_0 : X_1 : X_2 | X_3} - \II{X_0 : X_1 : X_2} \\
      &= \I{X_0 : X_1 | X_2, X_3} - \I{X_0 : X_1 | X_3} \\
      &\qquad - \I{X_0 : X_1 | X_2} + \I{X_0 : X_1}

For three variables the sign flip means synergy is positive and redundancy
negative, the opposite of the co-information convention:

.. ipython::

   In [1]: from dit.multivariate import coinformation, interaction_information

   In [2]: xor = dit.example_dists.Xor()

   @doctest float
   In [3]: coinformation(xor)
   Out[3]: -1.0

   @doctest float
   In [4]: interaction_information(xor)
   Out[4]: 1.0

.. seealso::

   For more information, see :doc:`/measures/multivariate/coinformation`.


API
===

.. autofunction:: interaction_information
   :no-index:
