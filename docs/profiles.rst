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

.. py:module:: dit.profiles.information_partitions

Shannon Partition and Extropy Partition
=======================================

The I-diagrams, or :class:`ShannonPartition`, for these four examples can be computed thusly:

.. ipython::
   :doctest:

   In [6]: ShannonPartition(ex1)
   +----------+--------+
   | measure  |  bits  |
   +----------+--------+
   | H[0|1,2] |  0.103 |
   | H[1|0,2] |  0.103 |
   | H[2|0,1] |  0.103 |
   | I[0:1|2] |  0.142 |
   | I[0:2|1] |  0.142 |
   | I[1:2|0] |  0.142 |
   | I[0:1:2] |  0.613 |
   +----------+--------+

   In [7]: ShannonPartition(ex2)
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

   In [8]: ShannonPartition(ex3)
   +----------+--------+
   | measure  |  bits  |
   +----------+--------+
   | H[0|1,2] |  0.000 |
   | H[1|0,2] |  0.000 |
   | H[2|0,1] |  0.245 |
   | I[0:1|2] |  0.245 |
   | I[0:2|1] |  0.000 |
   | I[1:2|0] |  0.000 |
   | I[0:1:2] |  0.755 |
   +----------+--------+

   In [9]: ShannonPartition(ex4)
   +----------+--------+
   | measure  |  bits  |
   +----------+--------+
   | H[0|1,2] |  0.000 |
   | H[1|0,2] |  0.000 |
   | H[2|0,1] |  0.000 |
   | I[0:1|2] |  0.245 |
   | I[0:2|1] |  0.245 |
   | I[1:2|0] |  0.245 |
   | I[0:1:2] |  0.510 |
   +----------+--------+

And their X-diagrams, or :class:`ExtropyDiagram`, can be computed like so:

.. ipython::
   :doctest:

   In [10]: ExtropyPartition(ex1)
   +----------+--------+
   | measure  | exits  |
   +----------+--------+
   | X[0|1,2] |  1.000 |
   | X[1|0,2] |  1.000 |
   | X[2|0,1] |  1.000 |
   | X[0:1|2] |  0.000 |
   | X[0:2|1] |  0.000 |
   | X[1:2|0] |  0.000 |
   | X[0:1:2] |  0.000 |
   +----------+--------+

   In [11]: ExtropyPartition(ex2)
   +----------+--------+
   | measure  | exits  |
   +----------+--------+
   | X[0|1,2] |  0.000 |
   | X[1|0,2] |  0.000 |
   | X[2|0,1] |  0.000 |
   | X[0:1|2] |  0.000 |
   | X[0:2|1] |  0.000 |
   | X[1:2|0] |  0.000 |
   | X[0:1:2] |  1.000 |
   +----------+--------+

   In [12]: ExtropyPartition(ex3)
   +----------+--------+
   | measure  | exits  |
   +----------+--------+
   | X[0|1,2] |  0.000 |
   | X[1|0,2] |  0.000 |
   | X[2|0,1] |  1.000 |
   | X[0:1|2] |  1.000 |
   | X[0:2|1] |  0.000 |
   | X[1:2|0] |  0.000 |
   | X[0:1:2] |  0.000 |
   +----------+--------+

   In [13]: ExtropyPartition(ex4)
   +----------+--------+
   | measure  | exits  |
   +----------+--------+
   | X[0|1,2] |  0.000 |
   | X[1|0,2] |  0.000 |
   | X[2|0,1] |  0.000 |
   | X[0:1|2] |  1.000 |
   | X[0:2|1] |  1.000 |
   | X[1:2|0] |  1.000 |
   | X[0:1:2] | -1.000 |
   +----------+--------+

.. py:module:: dit.profiles.complexity_profile

Complexity Profile
==================

The complexity profile, implimented by :class:`ComplexityProfile` is simply the amount of information at scale :math:`\geq k` of each "layer" of the I-diagram :cite:`Baryam2004`.

Consider example 1, which contains three independent bits. Each of these bits are in the outermost "layer" of the i-diagram, and so the information in the complexity profile is all at layer 1:

.. ipython::

   @savefig complexity_profile_example_1.png width=500 align=center
   In [14]: ComplexityProfile(ex1).draw();

Whereas in example 2, all the information is in the center, and so each scale of the complexity profile picks up that one bit:

.. ipython::

   @savefig complexity_profile_example_2.png width=500 align=center
   In [15]: ComplexityProfile(ex2).draw();

Both bits in example 3 are at a scale of at least 1, but only the shared bit persists to scale 2:

.. ipython::

   @savefig complexity_profile_example_3.png width=500 align=center
   In [16]: ComplexityProfile(ex3).draw();

Finally, example 4 (where each variable is the ``exclusive or`` of the other two):

.. ipython::

   @savefig complexity_profile_example_4.png width=500 align=center
   In [17]: ComplexityProfile(ex4).draw();

.. py:module:: dit.profiles.marginal_utility_of_information

Marginal Utility of Information
===============================

The marginal utility of information (MUI) :cite:`Allen2014`, implimented by :class:`MUIProfile` takes a different approach. It asks, given an amount of information :math:`\I{d : \left\{X\right\}} = y`, what is the maximum amount of information one can extract using an auxilliary variable :math:`d` as measured by the sum of the pairwise mutual informations, :math:`\sum \I{d : X_i}`. The MUI is then the rate of this maximum as a function of :math:`y`.

For the first example, each bit is independent and so basically must be extracted independently. Thus, as one increases :math:`y` the maximum amount extracted grows equally:

.. ipython::

   @savefig mui_profile_example_1.png width=500 align=center
   In [18]: MUIProfile(ex1).draw();

In the second example, there is only one bit total to be extracted, but it is shared by each pairwise mutual information. Therefore, for each increase in :math:`y` we get a threefold increase in the amount extracted:

.. ipython::

   @savefig mui_profile_example_2.png width=500 align=center
   In [19]: MUIProfile(ex2).draw();

For the third example, for the first one bit of :math:`y` we can pull from the shared bit, but after that one must pull from the independent bit, so we see a step in the MUI profile:

.. ipython::

   @savefig mui_profile_example_3.png width=500 align=center
   In [20]: MUIProfile(ex3).draw();

Lastly, the ``xor`` example:

.. ipython::

   @savefig mui_profile_example_4.png width=500 align=center
   In [21]: MUIProfile(ex4).draw();

.. py:module:: dit.profiles.schneidman

Schneidman Profile
==================

Also known as the *connected information* or *network informations*, the Schneidman profile (:class:`SchneidmanProfile`) exposes how much information is learned about the distribution when considering :math:`k`-way dependencies :cite:`Amari2001,Schneidman2003`. In all the following examples, each individual marginal is already uniformly distributed, and so the connected information at scale 1 is 0.

In the first example, all the random variables are independent already, so fixing marginals above :math:`k=1` does not result in any change to the inferred distribution:

.. ipython::

   @savefig schneidman_profile_example_1.png width=500 align=center
   In [22]: SchneidmanProfile(ex1).draw();

   @suppress
   In [22]: plt.ylim((0, 1))

In the second example, by learning the pairwise marginals, we reduce the entropy of the distribution by two bits (from three independent bits, to one giant bit):

.. ipython::

   @savefig schneidman_profile_example_2.png width=500 align=center
   In [23]: SchneidmanProfile(ex2).draw();

For the third example, learning pairwise marginals only reduces the entropy by one bit:

.. ipython::

   @savefig schneidman_profile_example_3.png width=500 align=center
   In [24]: SchneidmanProfile(ex3).draw();

And for the ``xor``, all bits appear independent until fixing the three-way marginals at which point one bit about the distribution is learned:

.. ipython::

   @savefig schneidman_profile_example_4.png width=500 align=center
   In [25]: SchneidmanProfile(ex4).draw();

.. py:module:: dit.profiles.mflat

M-Flat Connected Informations
=============================

The Schneidman profile above walks the *e-flat* MaxEnt ladder (match
:math:`k`-way marginals, set higher-order natural parameters to zero). Amari's
dual construction walks the *m-flat* mixture hierarchy instead
:cite:`Amari2001`: distributions whose ANOVA / Hoeffding expansion of the
*pmf itself* has no interactions above order :math:`k`,

.. math::

   \mathcal{M}_k = \Bigl\{ Q : Q(x) = \sum_{|S|\le k} h_S(x_S) \Bigr\}.

:class:`MFlatConnectedInformations` projects :math:`P` onto each
:math:`\mathcal{M}_k` under a chosen divergence ``criterion``:

* ``'jsd'`` (default) — Jensen–Shannon; finite on sparse supports with no smoothing
* ``'forward_kl'`` — :math:`D(P \Vert Q)`
* ``'reverse_kl'`` — Amari's true m-projection :math:`D(Q \Vert P)` (uses
  symmetric :math:`P_\varepsilon` / ``eps_schedule`` on sparse supports)

Profile atoms are consecutive drops in the residual to :math:`P` (for
``reverse_kl``, consecutive reverse-KL gaps between rungs, recovering the
Pythagorean decomposition).

Giant Bit / Copy saturate at order 2; XOR / W need order 3.

.. ipython::

   In [25a]: from dit.profiles import MFlatConnectedInformations

   In [25b]: from dit.algorithms import m_projection, mflat_mprojection_dists

   In [25c]: print(sorted(MFlatConnectedInformations(ex4, criterion='jsd').profile))
   [1, 2, 3]

Helpers :func:`~dit.algorithms.m_projection` /
:func:`~dit.algorithms.m_projection_from_subsets` and
:func:`~dit.algorithms.mflat_mprojection_dists` live under
:doc:`optimization`. The full dependency-lattice version with reverse KL is
:class:`~dit.profiles.DualDependencyDecomposition`.

Marginal Lift Profile
=====================

A complementary *fixed-block* construction: at order :math:`k`, form lifts of
the data's own marginals :math:`P_S` (:math:`|S|\le k`) plus the uniform, and
fit a convex combination by least squares
(:class:`~dit.profiles.MarginalLiftProfile`). Coefficients show which
marginals carry the approximation (Copy puts all mass on the copied pair at
order 2). Unlike Amari's free ANOVA tables, Giant Bit / XOR are *not* recovered
from pair lifts alone — only when the full joint is an allowed block.

.. ipython::

   In [25m]: from dit.profiles import MarginalLiftProfile

   In [25n]: copy = dit.Distribution(['000', '001', '110', '111'], [1/4]*4)

   In [25o]: print(round(MarginalLiftProfile(copy).residuals[2], 8))
   0.0

Binding Mixture Profile
=======================

Rosas et al. :cite:`rosas2019quantifying` distinguish *collective constraints*
(total correlation :math:`T`) from *shared randomness* (dual total correlation /
binding entropy :math:`B`). The MaxEnt Schneidman ladder decomposes the
constraint face. The shared-randomness face is captured by mixtures of product
distributions (latent-class / naive-Bayes models underlying Wyner common
information):

.. math::

   \mathcal{F}_k = \Bigl\{
       Q : Q(x) = \sum_{\alpha=1}^{k} \pi_\alpha \prod_i Q_i(x_i \mid \alpha)
   \Bigr\}.

:class:`BindingMixtureProfile` fits the MLE
:math:`Q^{(k)} = \arg\min_{Q\in\mathcal{F}_k} D(P\Vert Q)` by EM and reports

.. math::

   \Delta B_k = B\bigl(Q^{(k)}\bigr) - B\bigl(Q^{(k-1)}\bigr)

(with :math:`B(Q^{(0)}) := 0`). These atoms are nonnegative and sum to
:math:`B(P)` once the fit saturates. Giant bit concentrates at :math:`k=2`;
XOR needs :math:`k=4`; copy :math:`X=Y \perp Z` saturates at :math:`k=2`.

This is distinct from :class:`ConnectedDualInformations` (still the MaxEnt
ladder, only the *measure* is :math:`B`) and from
:class:`MFlatConnectedInformations` (Amari additive m-flat geometry).

.. ipython::

   In [25d]: from dit.profiles import BindingMixtureProfile

   In [25e]: from dit.algorithms import fit_mixture_of_products, mixture_of_products_dists

   In [25f]: gb = BindingMixtureProfile(ex2, k_max=4, n_init=8, seed=0)

   In [25g]: print(round(gb.profile[2], 6))
   1.0

   In [25h]: xor_prof = BindingMixtureProfile(ex4, k_max=4, n_init=10, seed=0, early_stop=False)

   In [25i]: print(round(sum(xor_prof.profile.values()), 6))
   2.0

Shared Randomness Decomposition
===============================

:class:`SharedRandomnessDecomposition` is the dependency-lattice counterpart:
at each antichain :math:`\pi` the reconstruction is the product of exact block
marginals (the *saturated* mixture-of-products model within each block, with
independence across blocks), and the default atom measure is :math:`B`.
It shares reconstructions with :class:`DependencyDecomposition` but reports
binding rather than entropy / total correlation.

.. ipython::

   In [25j]: from dit.profiles import SharedRandomnessDecomposition

   In [25k]: print('B' in next(iter(SharedRandomnessDecomposition(ex2).atoms.values())))
   True

.. py:module:: dit.profiles.entropy_triangle

Entropy Triangle and Entropy Triangle2
======================================

The entropy triangle, :class:`EntropyTriangle`, :cite:`valverde2016multivariate` is a method of visualizing how the information in the distribution is distributed among deviation from uniformity, independence, and dependence. The deviation from independence is measured by considering the difference in entropy between a independent variables with uniform distributions, and independent variables with the same marginal distributions as the distribution in question. Independence is measured via the :doc:`measures/multivariate/residual_entropy`, and dependence is measured by the sum of the :doc:`measures/multivariate/total_correlation` and :doc:`measures/multivariate/dual_total_correlation`.

All four examples lay along the left axis because their distributions are uniform over the events that have non-zero probability.

In the first example, the distribution is all independence because the three variables are, in fact, independent:

.. ipython::

   @savefig entropy_triangle_example_1.png width=500 align=center
   In [26]: EntropyTriangle(ex1).draw();

In the second example, the distribution is all dependence, because the three variables are perfectly entwined:

.. ipython::

   @savefig entropy_triangle_example_2.png width=500 align=center
   In [27]: EntropyTriangle(ex2).draw();

Here, there is a mix of independence and dependence:

.. ipython::

   @savefig entropy_triangle_example_3.png width=500 align=center
   In [28]: EntropyTriangle(ex3).draw();

And finally, in the case of ``xor``, the variables are completely dependent again:

.. ipython::

   @savefig entropy_triangle_example_4.png width=500 align=center
   In [29]: EntropyTriangle(ex4).draw();

We can also plot all four on the same entropy triangle:

.. ipython::

   @savefig entropy_triangle_all_examples.png width=500 align=center
   In [30]: EntropyTriangle([ex1, ex2, ex3, ex4]).draw();

.. ipython::

   In [31]: dists = [ dit.random_distribution(3, 2, alpha=(0.5,)*8) for _ in range(250) ]

   @savefig entropy_triangle_example.png width=500 align=center
   In [32]: EntropyTriangle(dists).draw();

We can plot these same distributions on a slightly different entropy triangle as well, :class:`EntropyTriangle2`, one comparing the :doc:`measures/multivariate/residual_entropy`, :doc:`measures/multivariate/total_correlation`, and :doc:`measures/multivariate/dual_total_correlation`:

.. ipython::
   :okwarning:

   @savefig entropy_triangle2_example.png width=500 align=center
   In [33]: EntropyTriangle2(dists).draw();

.. _dependency decomposition:

Dependency Decomposition
========================

Using :class:`DependencyDecomposition`, one can discover how an arbitrary information measure varies as marginals of the distribution are fixed. In our first example, each variable is independent of the others, and so constraining marginals makes no difference:

.. ipython::
   :doctest:

   In [34]: DependencyDecomposition(ex1)
   +------------+--------+
   | dependency |  bits  |
   +------------+--------+
   |    012     |  3.000 |
   |  01:02:12  |  3.000 |
   |   02:12    |  3.000 |
   |   01:12    |  3.000 |
   |   01:02    |  3.000 |
   |    12:0    |  3.000 |
   |    02:1    |  3.000 |
   |    01:2    |  3.000 |
   |   0:1:2    |  3.000 |
   +------------+--------+

In the second example, we see that fixing any one of the pairwise marginals reduces the entropy by one bit, and by fixing a second we reduce the entropy down to one bit:

.. ipython::
   :doctest:

   In [35]: DependencyDecomposition(ex2)
   +------------+--------+
   | dependency |  bits  |
   +------------+--------+
   |    012     |  1.000 |
   |  01:02:12  |  1.000 |
   |   02:12    |  1.000 |
   |   01:12    |  1.000 |
   |   01:02    |  1.000 |
   |    12:0    |  2.000 |
   |    02:1    |  2.000 |
   |    01:2    |  2.000 |
   |   0:1:2    |  3.000 |
   +------------+--------+

In the third example, only constraining the 01 marginal reduces the entropy, and it reduces it by one bit:

.. ipython::
   :doctest:

   In [36]: DependencyDecomposition(ex3)
   +------------+--------+
   | dependency |  bits  |
   +------------+--------+
   |    012     |  2.000 |
   |  01:02:12  |  2.000 |
   |   02:12    |  3.000 |
   |   01:12    |  2.000 |
   |   01:02    |  2.000 |
   |    12:0    |  3.000 |
   |    02:1    |  3.000 |
   |    01:2    |  2.000 |
   |   0:1:2    |  3.000 |
   +------------+--------+

And finally in the case of the exclusive or, only constraining the 012 marginal reduces the entropy.

.. ipython::
   :doctest:

   In [37]: DependencyDecomposition(ex4)
   +------------+--------+
   | dependency |  bits  |
   +------------+--------+
   |    012     |  2.000 |
   |  01:02:12  |  3.000 |
   |   02:12    |  3.000 |
   |   01:12    |  3.000 |
   |   01:02    |  3.000 |
   |    12:0    |  3.000 |
   |    02:1    |  3.000 |
   |    01:2    |  3.000 |
   |   0:1:2    |  3.000 |
   +------------+--------+

Dual Dependency Decomposition
=============================

:class:`DualDependencyDecomposition` uses the *same* dependency lattice, but
reconstructs each node by an m-flat reverse-KL m-projection rather than MaxEnt
:cite:`Amari2001`. At node :math:`\pi` the model is

.. math::

   \mathcal{M}_\pi = \Bigl\{ Q : Q(x) = \sum_{S \subseteq T,\ T\in\pi} h_S(x_S) \Bigr\},

with target the symmetrically smoothed :math:`P_\varepsilon=(1-\varepsilon)P+\varepsilon U`
and :math:`\varepsilon\downarrow 0` along ``eps_schedule`` (default
``(1e-4, 1e-6, 1e-8)``). The default atom is
:math:`D(Q_\pi \Vert P_\varepsilon)` at the final ``eps``. Symmetric
smoothing preserves permutation symmetries (e.g. the W distribution),
unlike random support jitter. The order-chain profile
:class:`MFlatConnectedInformations` is the rank-aggregated special case
(all blocks of size :math:`\le k`).

For the giant bit (``ex2``), the full pairwise cover ``01:02:12`` already
achieves reverse KL zero. For ``xor`` (``ex4``), every node short of the full
triple keeps a large reverse KL:

.. ipython::

   In [38]: from dit.profiles import DualDependencyDecomposition

   In [39]: gb = DualDependencyDecomposition(ex2, nrestarts=4)

   In [40]: pairs = frozenset([frozenset([0, 1]), frozenset([0, 2]), frozenset([1, 2])])

   In [41]: print(round(gb[pairs]['rKL'], 6))
   0.0

   In [42]: xor = DualDependencyDecomposition(ex4, nrestarts=4)

   In [43]: print(round(xor[pairs]['rKL'], 3) > 1)
   True
