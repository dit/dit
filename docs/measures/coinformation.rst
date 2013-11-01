**************
Co-Information
**************

The co-information is one generalization of the mutual information to multiple
variables.

.. autofunction:: dit.algorithms.coinformation

***********************
Interaction Information
***********************

The interaction information is equal in magnitude to the co-information, but
has the opposite sign when taken over an odd number of variables:

.. math:: \operatorname{II}(X) = (-1)^{|X|} \cdot \operatorname{I}(X)

.. autofunction:: dit.algorithms.interaction_information
