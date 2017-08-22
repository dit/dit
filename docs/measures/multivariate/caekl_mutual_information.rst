.. caekl_mutual_information.rst
.. py:module:: dit.multivariate.caekl_mutual_information

************************
CAEKL Mutual Information
************************

The Chan-AlBashabsheh-Ebrahimi-Kaced-Liu mutual information :cite:`chan2015multivariate` is one possible generalization of the :ref:`mutual_information`.

:math:`\J{X_{0:n}}` is the smallest :math:`\gamma` such that:

.. math::

   \H{X_{0:n}} - \gamma = \sum_{C \in \mathcal{P}} \left[ \H{X_C} - \gamma \right]

for some non-trivial partition :math:`\mathcal{P}` of :math:`\left{0:n\right}`. For example, the CAEKL mutual information for the ``xor`` distribution is :math:`\frac{1}{2}`, because the joint entropy is 2 bits, each of the three marginals is 1 bit, and :math:`2 - \frac{1}{2} = 3 (1 - \frac{1}{2})`.

.. ipython::

   In [1]: from dit.multivariate import caekl_mutual_information as J

   In [2]: d = dit.example_dists.Xor()

   @doctest float
   In [3]: J(d)
   Out[3]: 0.5

A more concrete way of defining the CAEKL mutual information is:

.. math::

   \J{X_{0:n}} = \min_{\mathcal{P} \in \Pi} ~ \operatorname{I}_\mathcal{P}\left[X_{0:n}\right]

where :math:`\operatorname{I}_\mathcal{P}` is the :ref:`total_correlation` of the partition:

.. math::

   \operatorname{I}_\mathcal{P}\left[X_{0:n}\right] = \sum_{C \in \mathcal{P}} \H{X_C} - \H{X_{0:n}}

and :math:`\Pi` is the set of all non-trivial partitions of :math:`\left{0:n\right}`.

.. todo::

   Include a nice i-diagram of this quantity, if possible.

API
===

.. autofunction:: caekl_mutual_information
