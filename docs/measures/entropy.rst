.. entropy.rst
.. py:module:: dit.algorithms.entropy2

*******
Entropy
*******

This is a general entropy function, handling conditional joint entropy.

.. math::

   \H[X_{0:n} | Y_{0:m}] =  -\sum_{\substack{x_{0:n} \in X_{0:n} \\ y_{0:m} \in Y_{0:m}}} p(x_{0:n}, y_{0:m}) \log_2 p(x_{0:n}|y_{0:m})

.. image:: ../images/idiagrams/h_xy.png
   :alt: The entropy :math:`\H[X,Y]`
   :width: 342px
   :align: center

.. image:: ../images/idiagrams/h_xyz.png
   :alt: The entropy :math:`\H[X,Y,Z]`
   :width: 357px
   :align: center

.. todo::

   Add some clever examples.

.. autofunction:: dit.algorithms.entropy2.entropy
