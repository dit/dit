.. interaction_information.rst

***********************
Interaction Information
***********************

The interaction information is equal in magnitude to the co-information, but
has the opposite sign when taken over an odd number of variables:

.. math::

    \II(X_{0:n}) &= (-1)^{n} \cdot \I(X_{0:n})

Interaction information was first studied in the 3-variable case which, for
:math:`X_{0:3} = X_0X_1X_2`, takes the following form:

.. math::

   \II(X_0:X_1:X_2) = \I(X_0:X_1|X_2) - \I(X_0:X_1)

The extension to :math:`n>3` proceeds recursively. For example,

.. math::

   \II(X_0:X_1:X_2:X_3)
      &= \II(X_0:X_1:X_2|X_3) - \II(X_0:X_1:X_2) \\
      &= \I(X_0:X_1|X_2,X_3) - \I(X_0:X_1|X_3) \\
      &\qquad - \I(X_0:X_1|X_2) + \I(X_0:X_1)

.. autofunction:: dit.algorithms.interaction_information.interaction_information
