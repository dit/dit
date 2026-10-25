.. necessary_conditional_entropy.rst
.. py:currentmodule:: dit.multivariate.necessary_conditional_entropy

*****************************
Necessary Conditional Entropy
*****************************

The necessary conditional entropy :cite:`cuff2010coordination` quantifies the amount of information that a random variable :math:`X` necessarily must carry above and beyond the mutual information :math:`\I{X : Y}` to actually contain that mutual information:

.. math::

   \H{X \dagger Y} = \H{ X \mss Y | Y}

It is not re-exported from :mod:`dit.multivariate`; import it from its module.
For the distribution uniform over :math:`\{00, 01, 11\}`, the two values of
:math:`X` induce different conditional distributions of :math:`Y`, so
:math:`X \mss Y = X` and the necessary conditional entropy is
:math:`\H{X | Y} = 2/3` bits, well above :math:`\I{X : Y} \approx 0.25`:

.. ipython::

   In [1]: from dit.multivariate.necessary_conditional_entropy import necessary_conditional_entropy

   In [2]: d = dit.Distribution(['00', '01', '11'], [1/3] * 3)

   @doctest float
   In [3]: necessary_conditional_entropy(d, [0], [1])
   Out[3]: 0.6666666666666665

API
===

.. autofunction:: necessary_conditional_entropy
   :no-index:
