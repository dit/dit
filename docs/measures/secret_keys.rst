.. secret_keys.rst
.. py:module:: dit.multivariate.secret_key_agreement

.. _secret key agreement:

********************
Secret Key Agreement
********************

One of the only methods of encrypting a message from Alice to Bob such that no third party (Eve) can possibly decrypt it is a one-time pad.
This technique requires that Alice and Bob have a secret sequence of bits, :math:`S`, which Alice then encrypts by computing the exclusive-or of it with the plaintext, :math:`P`, to produce the cyphertext, :math:`C`: :math:`C = S \oplus P`.
Bob can then decrypt by ``xor``\ ing again: :math:`P = S \oplus C`.

In order to pull this off, Alice and Bob need to construct :math:`S` out of some sort joint randomness, :math:`p(x, y, z)`, and public communication, :math:`V`, which is assumed to have perfect fidelity.
The maximum rate at which :math:`S` can be constructed in the *secret key agreement rate*.

Background
==========

Given :math:`N` IID copies of a joint distribution governed by :math:`p(x, y, z)`, let :math:`X^N` denote the random variables observed by Alice, :math:`Y^N` denote the random variables observed by Bob, and :math:`Z^N` denote the random variables observed by Eve.
A *secret key agreement scheme* consists of functions :math:`f` and :math:`g`, as well as a protocol for public communication (producing :math:`V`), and is considered :math:`R`-achievable if:

.. math::

   S_X = f(X^N, V) \\
   S_Y = g(Y^N, V) \\
   p(S_X = S_Y = S) \geq 1 - \epsilon \\
   \I{S : V Z^N} \leq \epsilon \\
   \frac{1}{N} \H{S} \geq R - \epsilon

The maximum rate :math:`R` such that there exists a :math:`R`-achievable scheme is known as the *secret key agreement rate*.
Intuitively, this means there exists some procedure such that, for every :math:`N` observations, Alice and Bob can publicly converse and then construct :math:`S` bits which agree almost surely, and are almost surely independent of everything Eve has access to.
:math:`S` is then known as a *secret key*.

There are three general classes of secret key agreement rates, depending on which parties are permitted to communicate. We discuss them below.

.. py:module:: dit.multivariate.secret_key_agreement.no_communication

No Communication
================

In the case that neither Alice nor Bob are permitted communication, the no-communication secret key agreement rate is given by:

.. math::

   \operatorname{S}[X : Y || Z] = \I{X \meet Y | Z}

where :math:`X \meet Y` is the :ref:`Gács-Körner Common Information` variable.

.. py:module:: dit.multivariate.secret_key_agreement.secrecy_capacity

Secrecy Capacity
----------------

Consider the situation that no party is allowed to communication, but rather than passively observing :math:`p(X, Y, Z)` Alice has full access to driving the channel :math:`p(Y, Z | X)`.
In this case we arrive at a maximum secret-key agreement rate known as the *secrecy capacity*, which is given by:

.. math::

   \operatorname{S_C}[X \rightarrow Y || Z] = \displaystyle \max_{U - X - YZ} \I{U : Y} - \I{U : Z}

.. py:module:: dit.multivariate.secret_key_agreement.one_way_skar

One-Way Communication
=====================

If only Alice is allowed to publicly broadcast information, the secret key agreement rate is given by:

.. math::

   \operatorname{S}[X \rightarrow Y || Z] = \displaystyle \max_{V - U - X - YZ} \I{U : Y | V} - \I{U : Z | V}

.. py:module:: dit.multivariate.secret_key_agreement.two_way_skar

Two-Way Communication
=====================

When both Alice and Bob are permitted communication, the secret key agreement rate, :math:`\operatorname{S}[X \leftrightarrow Y || Z]`, is much more difficult to compute, and in fact only upper and lower bounds on this rate are known.

Lower Bounds
------------

The first few lower bounds on two-way secret key agreement rate are simply symmetrized forms of the more restricted secret key agreement rates.

.. py:module:: dit.multivariate.secret_key_agreement.trivial_bounds

Lower Intrinsic Mutual Information
**********************************

The first lower bound on the secret key agreement rate is known in ``dit`` as the :py:func:`lower_intrinsic_mutual_information`, and is given by:

.. math::

   \I{X : Y \uparrow Z} = \max
      \begin{cases}
         \I{X : Y} - \I{X : Z} \\
         \I{X : Y} - \I{Y : Z} \\
         0
      \end{cases}

.. py:module:: dit.multivariate.secret_key_agreement.skar_lower_bounds

Secrecy Capacity
****************

Next is the secrecy capacity:

.. math::

   \I{X : Y \uparrow\uparrow Z} = \max
      \begin{cases}
         \displaystyle \max_{U - X - YZ} \I{U : Y} - \I{U : Z} \\
         \displaystyle \max_{U - Y - XZ} \I{U : X} - \I{U : Z}
      \end{cases}

This gives the secret key agreement rate when communication is not allowed.

Necessary Intrinsic Mutual Information
**************************************

A tighter bound is given by the :py:func:`necessary_intrinsic_mutual_information` :cite:`gohari2017achieving`, which is the maximum of the two one-way secret key agreement rates:

.. math::

   \I{X : Y \uparrow\uparrow\uparrow Z} = \max
      \begin{cases}
         \displaystyle \max_{V - U - X - YZ} \I{U : Y | V} - \I{U : Z | V} \\
         \displaystyle \max_{V - U - Y - XZ} \I{U : X | V} - \I{U : Z | V}
      \end{cases}

.. py:module:: dit.multivariate.secret_key_agreement.interactive_intrinsic_mutual_informations

Interactive Intrinsic Mutual Information
****************************************

.. math::

   \I{X : Y \uparrow\uparrow\uparrow\uparrow Z} = \max
      \sum_{i \textrm{even}} \I{U_i : Y | U_{0 \ldots i}} - \I{U_i : Z | U_{0 \ldots i}} + \\
      \sum_{i \textrm{odd}}  \I{U_i : X | U_{0 \ldots i}} - \I{U_i : Z | U_{0 \ldots i}}

.. py:module:: dit.multivariate.secret_key_agreement.iterated_discarding

Iterated Public Discarding
**************************

The :py:func:`iterated_discarding_skar` restricts the interactive bound to protocols in which every auxiliary variable is a public keep/discard flag :cite:`maurer1993secret`.
Alice and Bob alternate rounds; in each round the speaker announces, independently for each position, whether to keep it, with a keep probability :math:`k_i(\cdot)` that depends only on their own symbol.
Discarded positions are abandoned, and on the surviving positions whichever party fares better sends their variable as a one-way key, yielding the final term :math:`\max\{\I{X : Y} - \I{X : Z}, \I{X : Y} - \I{Y : Z}\}` evaluated on the post-selected distribution.
Each round contributes :math:`\I{U_i : Y} - \I{U_i : Z}` (Alice speaking) or :math:`\I{U_i : X} - \I{U_i : Z}` (Bob speaking), weighted by the probability that a position survives the earlier rounds.

Because each round needs only one keep probability per symbol, this bound remains tractable for many more rounds than :py:func:`interactive_intrinsic_mutual_information`.
For example, the W distribution, :math:`\{100, 010, 001\}` uniformly, has a one-way secret key agreement rate of zero in both directions, while iterated discarding reaches approximately :math:`0.138`, :math:`0.164`, and :math:`0.174` bits with one, two, and three discarding rounds, approaching roughly :math:`0.186` bits as the number of rounds grows.


Upper Bounds
------------

.. py:module:: dit.multivariate.secret_key_agreement.trivial_bounds
   :no-index:

Upper Intrinsic Mutual Information
***********************************
The secret key agreement rate is trivially upper bounded by:

.. math::

   \min\{ \I{X : Y}, \I{X : Y | Z} \}

.. py:module:: dit.multivariate.secret_key_agreement.intrinsic_mutual_informations

.. _intrinsic mutual information:

Intrinsic Mutual Information
****************************

The :py:func:`intrinsic_mutual_information` :cite:`maurer1997intrinsic` is defined as:

.. math::

   \I{X : Y \downarrow Z} = \min_{p(\overline{z} | z)} \I{X : Y | \overline{Z}}

It is straightforward to see that :math:`p(\overline{z} | z)` being a constant achieves :math:`\I{X : Y}`, and :math:`p(\overline{z} | z)` being the identity achieves :math:`\I{X : Y | Z}`.

.. py:module:: dit.multivariate.secret_key_agreement.reduced_intrinsic_mutual_informations

.. _reduced intrinsic mutual information:

Reduced Intrinsic Mutual Information
************************************

This bound can be improved, producing the :py:func:`reduced_intrinsic_mutual_information` :cite:`renner2003new`:

.. math::

   \I{X : Y \downarrow\downarrow Z} = \min_{U} \I{X : Y \downarrow ZU} + \H{U}

This bound improves upon the :ref:`Intrinsic Mutual Information` when a small amount of information, :math:`U`, can result in a larger decrease in the amount of information shared between :math:`X` and :math:`Y` given :math:`Z` and :math:`U`.

Although written as a nested minimization, the inner intrinsic mutual information is itself a minimization over :math:`p(\overline{z} | z u)`, and :math:`\H{U}` does not depend on :math:`\overline{Z}`. The two minimizations therefore combine into a single joint minimization over two chained channels, :math:`p(u | x y z)` and :math:`p(\overline{z} | z u)`:

.. math::

   \I{X : Y \downarrow\downarrow Z} = \min_{p(u | x y z),\, p(\overline{z} | z u)} \I{X : Y | \overline{Z}} + \H{U}

This is how ``dit`` computes it, with :math:`|\overline{Z}| \leq |Z| |U|` from the corresponding bound for the intrinsic mutual information :cite:`renner2003new`.

.. py:module:: dit.multivariate.secret_key_agreement.minimal_intrinsic_mutual_informations

.. _minimal intrinsic mutual information:

Minimal Intrinsic Mutual Information
************************************

The :ref:`Reduced Intrinsic Mutual Information` can be further reduced into the :py:func:`minimal_intrinsic_total_correlation` :cite:`gohari2017comments`:

.. math::

   \I{X : Y \downarrow\downarrow\downarrow Z} = \min_{U} \I{X : Y | U} + \I{XY : U | Z}

.. py:module:: dit.multivariate.secret_key_agreement.two_part_intrinsic_mutual_informations

Two-Part Intrinsic Mutual Information
*************************************

The :py:func:`two_part_intrinsic_mutual_information` :cite:`gohari2010information,gohari2017comments` is:

.. math::

   \I{X : Y \downarrow\downarrow\downarrow\downarrow Z} = \inf_{J} \max_{V - U - XY - ZJ} \I{X : Y | J} + \I{U : J | V} - \I{U : Z | V}

The inner maximization is the one-way secret key agreement rate from :math:`XY` to :math:`J` with :math:`Z` eavesdropping.
No cardinality bound on :math:`J` is known, so the bound cannot be computed exactly :cite:`gohari2017comments`; any particular :math:`J` still yields a valid upper bound, provided the inner maximization is solved exactly.

.. py:module:: dit.multivariate.secret_key_agreement.relaxed_two_part_intrinsic_mutual_informations

Relaxed Two-Part Intrinsic Mutual Information
*********************************************

Because the inner maximization above is a one-way secret key agreement rate, it is bounded from above by the :ref:`Intrinsic Mutual Information` :math:`\I{XY : J \downarrow Z}` :cite:`maurer1997intrinsic`.
Substituting this gives the :py:func:`relaxed_two_part_intrinsic_mutual_information`:

.. math::

   \I{X : Y \downarrow\downarrow\downarrow\downarrow_r Z} = \min_{p(j | x y z),\, p(\overline{z} | z)} \I{X : Y | J} + \I{XY : J | \overline{Z}}

Since :math:`\I{XY : J \downarrow Z} \leq \I{XY : J | Z}`, it is never larger than the :ref:`Minimal Intrinsic Mutual Information`, and since it relaxes only the inner maximization, it is never smaller than the two-part intrinsic mutual information.
Unlike the latter, it is a single joint minimization over two channels, with :math:`|\overline{Z}| \leq |Z|` :cite:`christandl2003property`.
Restricting the size of :math:`J` or stopping at a local optimum can therefore only loosen it, never invalidate it, which is why :py:func:`two_way_skar` uses it in place of the two-part bound.

.. py:module:: dit.multivariate.secret_key_agreement.less_noisy_intrinsic_mutual_information

Less-Noisy Intrinsic Mutual Information
***************************************

If Eve's channel :math:`p(z | x y)` is *less noisy* than a channel :math:`p(j | x y)`, meaning :math:`\I{U : Z} \geq \I{U : J}` for every :math:`U - XY - ZJ` and every input distribution, then no protocol can distill more key against :math:`Z` than against :math:`J` :cite:`gohari2017achieving`. The :py:func:`less_noisy_intrinsic_mutual_information` :cite:`pauwels2026bipartite` optimizes over all such :math:`J`:

.. math::

   \I{X : Y \downarrow_{\mathrm{ln}} Z} = \inf_{p(j | x y) \,:\, p(z | x y) \succeq_{\mathrm{ln}} p(j | x y)} \I{X : Y | J}

Every degradation :math:`p(\overline{z} | z)` is dominated, so this never exceeds the :ref:`Intrinsic Mutual Information`, and it is never smaller than the inf-max upper bound of Gohari and Anantharam (2010), stated as Eq. (11) of :cite:`abin2026source`. It is not comparable in general to the reduced or minimal intrinsic mutual informations.

A channel is less noisy than another iff the difference of the two output entropies is concave in the input distribution :cite:`vandijk1997special`. That forces every dominated channel to factor as :math:`p(j | x y) = \sum_z p(z | x y) K(z, j)` for a real, possibly signed, matrix :math:`K`; ``dit`` optimizes over :math:`K`, imposing the concavity condition on a finite sample of input distributions.

The distribution ``bound_information`` of :cite:`pauwels2026bipartite` has :math:`\I{X : Y \downarrow Z} > 0` but :math:`\I{X : Y \downarrow_{\mathrm{ln}} Z} = 0`, and so a secret key agreement rate of zero. When :math:`X` and :math:`Y` are binary and :math:`Z = X \oplus Y`, every known upper bound, this one included, equals :math:`\I{X : Y}` :cite:`abin2026source`.

All Together Now
----------------

Taken together, we see the following structure:

.. math::

   \begin{aligned}
     &\min\{ \I{X : Y}, \I{X : Y | Z} \} \\
     &\quad \geq \I{X : Y \downarrow Z} \\
     &\quad\quad \geq \I{X : Y \downarrow\downarrow Z} \\
     &\quad\quad\quad \geq \I{X : Y \downarrow\downarrow\downarrow Z} \\
     &\quad\quad\quad\quad \geq \I{X : Y \downarrow\downarrow\downarrow\downarrow_r Z} \\
     &\quad\quad\quad\quad \geq \I{X : Y \downarrow\downarrow\downarrow\downarrow Z} \\
     &\quad\quad\quad\quad\quad \geq S[X \leftrightarrow Y || Z] \\
     &\quad\quad\quad\quad\quad\quad \geq \I{X : Y \uparrow\uparrow\uparrow\uparrow Z} \\
     &\quad\quad\quad\quad\quad\quad\quad \geq \I{X : Y \uparrow\uparrow\uparrow Z} \\
     &\quad\quad\quad\quad\quad\quad\quad\quad \geq \I{X : Y \uparrow\uparrow Z} \\
     &\quad\quad\quad\quad\quad\quad\quad\quad\quad \geq \max\{ \I{X : Y \uparrow Z}, S[X : Y || Z] \} \\
     &\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad \geq 0.0
   \end{aligned}

The secrecy capacity dominates both of the final two bounds: choosing :math:`U = X` (or :math:`U = Y`) recovers :math:`\I{X : Y \uparrow Z}`, and choosing :math:`U = X \meet Y` recovers :math:`S[X : Y || Z]`.
The final two bounds, however, are incomparable.
For example, let :math:`W`, :math:`A`, and :math:`B` be independent uniform bits, and let :math:`X = (W, A)`, :math:`Y = (W, B)`, and :math:`Z = (A, B)`.
Then :math:`S[X : Y || Z] = \H{W | Z} = 1` bit, since Alice and Bob share :math:`W` and Eve knows nothing about it, while :math:`\I{X : Y} = \I{X : Z} = \I{Y : Z} = 1` bit, so :math:`\I{X : Y \uparrow Z} = 0`.

Generalizations
---------------

Most of the above bounds have straightforward multivariate generalizations. These are not necessarily bounds on the multiparty secret key agreement rate. For example, one could compute the :py:func:`minimal_intrinsic_dual_total_correlation`:

.. math::

   \B{X_0 : \ldots : X_n \downarrow\downarrow\downarrow Z} = \min_{U} \B{X_0 : \ldots : X_n | U} + \I{X_0, \ldots, X_n : U | Z}

Examples
--------

Let us consider a few examples:

.. ipython::

   In [1]: from dit.multivariate.secret_key_agreement import *

   In [2]: from dit.example_dists.intrinsic import intrinsic_1, intrinsic_2, intrinsic_3

First, we consider the distribution ``intrinsic_1``:

.. ipython::

   In [3]: print(intrinsic_1)
   Class:          Distribution
   Alphabet:       ('0', '1', '2', '3') for all rvs
   Base:           linear
   Outcome Class:  str
   Outcome Length: 3
   RV Names:       None
   x     p(x)
   000   1/8
   011   1/8
   101   1/8
   110   1/8
   222   1/4
   333   1/4

With upper bounds:

.. ipython::

   @doctest float
   In [4]: upper_intrinsic_mutual_information(intrinsic_1, [[0], [1]], [2])
   Out[4]: 0.5

We see that the trivial upper bound is 0.5, because without conditioning on :math:`Z`, :math:`X` and :math:`Y` can agree when the observe either a :math:`2` or a :math:`3`, which results in :math:`\I{X : Y} = 0.5`. Given :math:`Z`, however, that information is no longer private. But, given :math:`Z`, a conditional dependence is induced between :math:`X` and :math:`Y`: :math:`Z` knows that if she is a :math:`0` that :math:`X` and :math:`Y` agree, and if she is a :math:`1` they disagree. This results :math:`\I{X : Y | Z} = 0.5`. In either case, however, :math:`X` and :math:`Y` can not agree upon a secret key: in the first case the eavesdropper knows their correlation, while in the second they are actually independent.

The :py:func:`intrinsic_mutual_information`, however can detect this:

.. ipython::

   @doctest float
   In [5]: intrinsic_mutual_information(intrinsic_1, [[0], [1]], [2])
   Out[5]: 0.0

Next, let's consider the distribution ``intrinsic_2``:

.. ipython::

   In [7]: print(intrinsic_2)
   Class:          Distribution
   Alphabet:       (('0', '1', '2', '3'), ('0', '1', '2', '3'), ('0', '1'))
   Base:           linear
   Outcome Class:  str
   Outcome Length: 3
   RV Names:       None
   x     p(x)
   000   1/8
   011   1/8
   101   1/8
   110   1/8
   220   1/4
   331   1/4

In this case, :math:`Z` no longer can distinguish between the case where :math:`X` and :math:`Y` can agree on a secret bit, and when they can not, because she can not determine when they are in the :math:`01` regime or in the :math:`23` regime:

.. ipython::

   @doctest float
   In [8]: intrinsic_mutual_information(intrinsic_2, [[0], [1]], [2])
   Out[8]: 1.5

This seems to imply that :math:`X` and :math:`Y` can adopt a scheme such as: if they observe either a :math:`0` or a :math:`1`, write down :math:`0`, and if they observe either a :math:`2` or a :math:`3`, write that down. This has a weakness, however: what if :math:`Z` were able to distinguish the two regimes? This costs her :math:`1` bit, but reduces the secrecy of :math:`X` and :math:`Y` to nil. Thus, the secret key agreement rate is actually only :math:`1` bit:

.. ipython::

   @doctest float
   In [9]: minimal_intrinsic_mutual_information(intrinsic_2, [[0], [1]], [2], bounds=(3,))
   Out[9]: 1.0
