.. hypothesis.rst
.. py:currentmodule:: dit.utils.testing

*********************************
Finding Examples with Hypothesis
*********************************

What if you'd like to find a distribution that has a particular property? For
example, what if I'd like to find a distribution with a :ref:`coinformation`
less than :math:`-0.5`? This is where `Hypothesis
<https://hypothesis.readthedocs.io>`_ comes in (install it with the ``test``
extra, ``pip install "dit[test]"``):

.. ipython::

   In [1]: from hypothesis import find

   In [2]: from dit.utils.testing import distributions

   In [3]: find(distributions(alphabets=3), lambda d: dit.multivariate.coinformation(d) < -0.5)

What hypothesis has done is use the :func:`distributions` *strategy* to randomly
test distributions. Once it finds a distribution satisfying the criteria we
specified (coinformation less than :math:`-0.5`) it then simplifies the example
as much as possible. Here, we see that even though it could have found any
distribution, it found the exclusive or distribution, and simplified the
probabilities to be uniform.

The same strategies drive ``dit``'s own property-based test suite, and are
equally useful for testing conjectured inequalities between measures.

Strategies
==========

:mod:`dit.utils.testing` provides the following Hypothesis strategies:

.. autofunction:: distributions

.. autofunction:: distribution_structures

.. autofunction:: markov_chains

.. autofunction:: channel_inputs

.. autofunction:: channel_pairs

.. autofunction:: degraded_channel_pairs
