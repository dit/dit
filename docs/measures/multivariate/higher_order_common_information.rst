.. higher_order_common_information.rst
.. py:currentmodule:: dit.multivariate

.. _higher_order_common_information:

*******************************
Higher-Order Common Information
*******************************

The higher-order common information :cite:`ostergaard2024higher` measures how
much statistical information survives a sequence of information bottlenecks,
one per variable. Starting from a reference variable :math:`T_1 = X_i`, each
stage processes another variable :math:`X_a`, keeping as much information about
it as possible while passing on no more than :math:`T_\ell` already shares with
it:

.. math::

   T_{\ell+1} \in \arg\max \left\{ \I{S : X_a} : S - T_\ell - X_{0:n},\ \I{S : T_\ell} \leq \I{T_\ell : X_a} \right\}

:math:`T_{\ell+1}` must also be target-minimal: it keeps only what is
statistically relevant to :math:`X_a`. After :math:`n - 2` stages, the final
auxiliary variable is scored by the least it shares with any variable. Each
reference takes its best ordering :math:`\pi` of the other variables, and the
worst reference is reported:

.. math::

   \operatorname{HCI}[X_{0:n}] = \min_i \max_\pi \sup_{T_{n-1}} \min_j \I{T_{n-1} : X_j}

where the supremum runs over the chains of stage optimizers. Each stage is a
rate-constrained information bottleneck, computed with
:func:`~dit.rate_distortion.information_bottleneck_rate`.

For two variables it is the :ref:`mutual_information`. In general it lies
between the :doc:`Gács–Körner common information
</measures/common_informations/gk_common_information>` and the least pairwise
mutual information. That in turn is at most the
:ref:`caekl_mutual_information`, since the total correlation among the
:math:`k` parts of a partition is a sum of :math:`k - 1` mutual informations,
each at least a pairwise one:

.. math::

   \K{X_{0:n}} \leq \operatorname{HCI}[X_{0:n}] \leq \min_{i \neq j} \I{X_i : X_j} \leq \J{X_{0:n}}

Unlike the Gács–Körner common information, it is positive for noisy copies of
a common bit. For :math:`n` copies with crossover :math:`q` it is
:math:`1 - h\left(\tfrac{1}{2}\left(1 - (1 - 2q)^{2^{n-1}}\right)\right)`:

.. ipython::

   In [1]: from dit.example_dists import noisy_copies

   In [2]: from dit.multivariate import higher_order_common_information as HCI

   In [3]: HCI(noisy_copies(0.1, 3), rng=0)

Like the Gács–Körner common information, it tells the dyadic distribution from
the triadic one, which no Shannon measure does:

.. ipython::

   In [4]: from dit.example_dists import dyadic, triadic

   @doctest float
   In [5]: HCI(dyadic)
   Out[5]: 0.0

   @doctest float
   In [6]: HCI(triadic)
   Out[6]: 1.0

Each pair of dyadic variables shares a bit, but stage-wise optimality makes the
first stage keep only the bit shared with the processed variable, which the
remaining variable knows nothing about.

Caveats
=======

The value counts processing stages rather than distinct variables, and it
fails two of the properties of :cite:`chan2015multivariate`:

* It is not consistent: a duplicated variable is processed again and costs
  information. Let :math:`X_0 \in \{0, 1, 2\}` have probabilities
  :math:`(\tfrac{1}{2}, \tfrac{1}{4}, \tfrac{1}{4})`, and let :math:`X_1` erase
  it with probability :math:`\tfrac{2}{5}`. Then
  :math:`\operatorname{HCI}[X_0 : X_1] = \I{X_0 : X_1} = \tfrac{9}{10}`, but
  :math:`\operatorname{HCI}[X_0 : X_1 : X_1] = \tfrac{27}{50}`.
* It does not satisfy the data processing inequality: coarse-graining a
  variable can raise it. For independent fair bits :math:`A, B, C`,
  :math:`\operatorname{HCI}[A \wedge B : A \wedge C : BC] = 0`, but
  :math:`\operatorname{HCI}[A \wedge B : A \wedge C : B \wedge C] \approx 0.0088`.
  With reference :math:`BC`, every stage optimizer is a degraded :math:`B` or
  :math:`C`, which is independent of the remaining conjunction.

Computation
===========

When the Gács–Körner common information equals the least pairwise mutual
information, that value is returned. Otherwise every ordering of every
reference is searched, so the cost grows as :math:`n!`, though orderings that
share a prefix share its stages. A stage whose relevant part is a deterministic
function of the processed variable is solved exactly. The others are solved
numerically, and then a second optimization breaks ties among the stage
optimizers. It prefers the one sharing the most with the least related other
variable. That choice is exact for three variables and a heuristic for more.

API
===

.. autofunction:: higher_order_common_information
   :no-index:
