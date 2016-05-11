.. profiles.rst
.. py:module:: dit.profiles

********************
Information Profiles
********************

There are several ways to decompose the information contained in a joint distribution. Here, we will demonstrate their behavior using four examples drawn from :cite:`Allen2014`:

.. ipython::

   In [1]: from dit.profiles import *

   In [2]: ex1 = dit.Distribution(['000', '001', '010', '011', '100', '101', '110', '111'], [1/8]*8)

   In [3]: ex2 = dit.Distribution(['000', '111'], [1/2]*2)

   In [4]: ex3 = dit.Distribution(['000', '001', '110', '111'], [1/4]*4)

   In [5]: ex4 = dit.Distribution(['000', '011', '101', '110'], [1/4]*4)

The I-diagrams for these four examples can be computed thusly:

.. ipython::

   In [6]: from dit.algorithms import ShannonPartition

   @doctest
   In [7]: print(ShannonPartition(ex1))
   +----------+--------+
   | measure  |  bits  |
   +----------+--------+
   | H[0|1,2] |  1.000 |
   | H[1|0,2] |  1.000 |
   | H[2|0,1] |  1.000 |
   | I[0:1|2] |  0.000 |
   | I[0:2|1] |  0.000 |
   | I[1:2|0] |  0.000 |
   | I[0:1:2] |  0.000 |
   +----------+--------+

   @doctest
   In [8]: print(ShannonPartition(ex2))
   +----------+--------+
   | measure  |  bits  |
   +----------+--------+
   | H[0|1,2] |  0.000 |
   | H[1|0,2] |  0.000 |
   | H[2|0,1] |  0.000 |
   | I[0:1|2] |  0.000 |
   | I[0:2|1] |  0.000 |
   | I[1:2|0] |  0.000 |
   | I[0:1:2] |  1.000 |
   +----------+--------+

   @doctest
   In [9]: print(ShannonPartition(ex3))
   +----------+--------+
   | measure  |  bits  |
   +----------+--------+
   | H[0|1,2] |  0.000 |
   | H[1|0,2] |  0.000 |
   | H[2|0,1] |  1.000 |
   | I[0:1|2] |  1.000 |
   | I[0:2|1] |  0.000 |
   | I[1:2|0] |  0.000 |
   | I[0:1:2] |  0.000 |
   +----------+--------+

   @doctest
   In [10]: print(ShannonPartition(ex4))
   +----------+--------+
   | measure  |  bits  |
   +----------+--------+
   | H[0|1,2] |  0.000 |
   | H[1|0,2] |  0.000 |
   | H[2|0,1] |  0.000 |
   | I[0:1|2] |  1.000 |
   | I[0:2|1] |  1.000 |
   | I[1:2|0] |  1.000 |
   | I[0:1:2] | -1.000 |
   +----------+--------+

.. py:class:: dit.profiles.ComplexityProfile

Complexity Profile
==================

The complexity profile is simply the amount of information at scale :math:`\geq k` of each "layer" of the I-diagram :cite:`Baryam2004`.

Consider example 1, which contains three independent bits. Each of these bits are in the outermost "layer" of the i-diagram, and so the information in the complexity profile is all at layer 1:

.. ipython::

   @savefig complexity_profile_example_1.png width=500 align=center
   In [11]: ComplexityProfile(ex1).draw();

Whereas in example 2, all the information is in the center, and so each scale of the complexity profile picks up that one bit:

.. ipython::

   @savefig complexity_profile_example_2.png width=500 align=center
   In [12]: ComplexityProfile(ex2).draw();

Both bits in example 3 are at a scale of at least 1, but only the shared bit persists to scale 2:

.. ipython::

   @savefig complexity_profile_example_3.png width=500 align=center
   In [13]: ComplexityProfile(ex3).draw();

Finally, example 4 (where each variable is the ``exclusive or`` of the other two):

.. ipython::

   @savefig complexity_profile_example_4.png width=500 align=center
   In [14]: ComplexityProfile(ex4).draw();

.. py:class:: dit.profiles.MUIProfile

Marginal Utility of Information
===============================

The marginal utility of information (MUI) :cite:`Allen2014` takes a different approach. It asks, given an amount of information :math:`\I[d : \{X\}] = y`, what is the maximum amount of information one can extract using an auxilliary variable :math:`d` as measured by the sum of the pairwise mutual informations, :math:`\sum \I[d : X_i]`. The MUI is then the rate of this maximum as a function of :math:`y`.

For the first example, each bit is independent and so basically must be extracted independently. Thus, as one increases :math:`y` the maximum amount extracted grows equally:

.. ipython::

   @savefig mui_profile_example_1.png width=500 align=center
   In [15]: MUIProfile(ex1).draw();

In the second example, there is only one bit total to be extracted, but it is shared by each pairwise mutual information. Therefore, for each increase in :math:`y` we get a threefold increase in the amount extracted:

.. ipython::

   @savefig mui_profile_example_2.png width=500 align=center
   In [16]: MUIProfile(ex2).draw();

For the third example, for the first one bit of :math:`y` we can pull from the shared bit, but after that one must pull from the independent bit, so we see a step in the MUI profile:

.. ipython::

   @savefig mui_profile_example_3.png width=500 align=center
   In [17]: MUIProfile(ex3).draw();

Lastly, the ``xor`` example:

.. ipython::

   @savefig mui_profile_example_4.png width=500 align=center
   In [18]: MUIProfile(ex4).draw();

.. py:class:: dit.profiles.SchneidmanProfile

Schneidman Profile
==================

Also known as the *connected information* or *network informations*, the Schneidman profile exposes how much information is learned about the distribution when considering :math:`k`-way dependencies :cite:`Amari2001,Schneidman2003`. In all the following examples, each individual marginal is already uniformly distributed, and so the connected information at scale 1 is 0.

In the first example, all the random variables are independent already, so fixing marginals above :math:`k=1` does not result in any change to the inferred distribution:

.. ipython::

   @savefig schneidman_profile_example_1.png width=500 align=center
   In [19]: SchneidmanProfile(ex1).draw();

   @suppress
   In [20]: plt.ylim((0, 1))

In the second example, by learning the pairwise marginals, we reduce the entropy of the distribution by two bits (from three independent bits, to one giant bit):

.. ipython::

   @savefig schneidman_profile_example_2.png width=500 align=center
   In [20]: SchneidmanProfile(ex2).draw();

For the third example, learning pairwise marginals only reduces the entropy by one bit:

.. ipython::

   @savefig schneidman_profile_example_3.png width=500 align=center
   In [21]: SchneidmanProfile(ex3).draw();

And for the ``xor``, all bits appear independent until fixing the three-way marginals at which point one bit about the distribution is learned:

.. ipython::

   @savefig schneidman_profile_example_4.png width=500 align=center
   In [22]: SchneidmanProfile(ex4).draw();

.. py:class:: dit.profiles.EntropyTriangle

Entropy Triangle
================

The entropy triangle :cite:`valverde2016multivariate` is a method of visualizing how the information in the distribution is distributed among deviation from uniformity, independence, and dependence. The deviation from independence is measured by considering the difference in entropy between a independent variables with uniform distributions, and independent variables with the same marginal distributions as the distribution in question. Independence is measured via the :doc:`measures/multivariate/residual_entropy`, and dependence is measured by the sum of the :doc:`measures/multivariate/total_correlation` and :doc:`measures/multivariate/dual_total_correlation`.

All four examples lay along the left axis because their distributions are uniform over the events that have non-zero probability.

In the first example, the distribution is all independence because the three variables are, in fact, independent:

.. ipython::

   @savefig entropy_triangle_example_1.png width=500 align=center
   In [23]: EntropyTriangle(ex1).draw();

In the second example, the distribution is all dependence, because the three variables are perfectly entwined:

.. ipython::

   @savefig entropy_triangle_example_2.png width=500 align=center
   In [24]: EntropyTriangle(ex2).draw();

Here, there is a mix of independence and dependence:

.. ipython::

   @savefig entropy_triangle_example_3.png width=500 align=center
   In [25]: EntropyTriangle(ex3).draw();

And finally, in the case of ``xor``, the variables are completely dependent again:

.. ipython::

   @savefig entropy_triangle_example_4.png width=500 align=center
   In [26]: EntropyTriangle(ex4).draw();

We can also plot all four on the same entropy triangle:

.. ipython::

   @savefig entropy_triangle_all_examples.png width=500 align=center
   In [27]: EntropyTriangle([ex1, ex2, ex3, ex4]).draw();

.. ipython::

   In [28]: dists = [ dit.random_distribution(3, 2, alpha=(0.5,)*8) for _ in range(250) ]

   @savefig entropy_triangle_example.png width=500 align=center
   In [29]: EntropyTriangle(dists).draw();
