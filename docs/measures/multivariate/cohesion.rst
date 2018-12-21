.. cohesion.rst
.. py:module:: dit.multivariate.cohesion

********
Cohesion
********

The cohesion is a paramitrized multivariate mutual information which spans from the :ref:`Total Correlation` to the :ref:`Dual Total Correlation`:

.. math::

   C_k[X_0 : X_1 : \dotsc : X_n] = \sum_{\substack{A \subset [n]}{|A| = k}} \H{X_A} - \binom{n - 1}{k - 1} \H{X_0, X_1, \dotsc, X_n}
