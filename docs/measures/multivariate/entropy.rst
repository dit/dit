.. entropy.rst
.. py:module:: dit.multivariate.entropy

*******
Entropy
*******

The entropy measures the total amount of information contained in a set of
random variables, :math:`X_{0:n}`, potentially excluding the information contain
in others, :math:`Y_{0:m}`.

.. math::

   \H[X_{0:n} | Y_{0:m}] =  -\sum_{\substack{x_{0:n} \in \mathcal{X}_{0:n} \\ y_{0:m} \in \mathcal{Y}_{0:m}}} p(x_{0:n}, y_{0:m}) \log_2 p(x_{0:n}|y_{0:m})

.. image:: ../../images/idiagrams/h_xy.png
   :alt: The entropy :math:`\H[X,Y]`
   :width: 342px
   :align: center

.. image:: ../../images/idiagrams/h_xyz.png
   :alt: The entropy :math:`\H[X,Y,Z]`
   :width: 357px
   :align: center

.. todo::

   Add some clever examples.

.. autofunction:: entropy
