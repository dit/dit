.. cohesion.rst
.. py:module:: dit.multivariate.cohesion

********
Cohesion
********

The cohesion :cite:`rosas2016understanding` is a parameterized multivariate mutual information which spans from the :ref:`Total Correlation` to the :ref:`Dual Total Correlation`:

.. note::

   TODO: confirm whether the canonical reference for cohesion is :cite:`rosas2016understanding` (Rosas et al. 2016) or a later Rosas et al. 2019 paper on high-order interdependencies.

.. math::

   C_k[X_0 : X_1 : \dotsc : X_n] = \sum_{\substack{A \subset [n]}{|A| = k}} \H{X_A} - \binom{n - 1}{k - 1} \H{X_0, X_1, \dotsc, X_n}
