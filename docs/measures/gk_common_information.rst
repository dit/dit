.. gk_common_information.rst

******************************
Gács-Körner Common Information
******************************

The Gács-Körner common information take a very direct approach to the idea of
common information. Consider a joint distribution over :math:`X_0` and
:math:`X_1`. Given any particular outcome from that joint, we want a function
:math:`f(X_0)` and a function :math:`g(X_1)` such that :math:`\forall x_0x_1 =
X_0X_1, f(x_0) = g(x_1) = v`. Of all possible pairs of functions :math:`f(X_0) =
g(X_1) = V`, there exists a "largest" one, and it is known as the common random
variable. The entropy of that common random variable is the Gács-Körner common
information:

.. math::

   \K[X_0, X_1] = \max_{f(X_0) = g(X_1) = V} \H[V]

.. autofunction:: dit.algorithms.common_information
