.. pid.rst
.. py:module:: dit.pid

*********************************
Partial Information Decomposition
*********************************

The *partial information decomposition* (PID), put forth by Williams & Beer :cite:`williams2010nonnegative`, is a framework for decomposing the information shared between a set of variables we will refer to as *inputs*, :math:`X_0, X_1, \ldots`, and another random variable we will refer to as the *output*, :math:`Y`. This decomposition seeks to partition the information :math:`\I[X_0,X_1,\ldots : Y]` among the antichains of the inputs.

Background
==========

It is often desirable to determine how a set of inputs influence the behavior of an output. Consider the exclusive or logic gates, for example:

.. ipython::

    In [1]: from dit.pid.distributions import bivariates, trivariates

    In [2]: xor = bivariates['synergy']

    In [3]: print(xor)
    Class:          Distribution
    Alphabet:       ('0', '1') for all rvs
    Base:           linear
    Outcome Class:  str
    Outcome Length: 3
    RV Names:       None

    x     p(x)
    000   1/4
    011   1/4
    101   1/4
    110   1/4

We can see from inspection that either input (the first two indexes) is independent of the output (the final index), yet the two inputs together determine the output. One could call this "synergistic" information. Next, consider the giant bit distribution:

.. ipython::

    In [4]: gb = bivariates['redundant']

    In [5]: print(gb)
    Class:          Distribution
    Alphabet:       ('0', '1') for all rvs
    Base:           linear
    Outcome Class:  str
    Outcome Length: 3
    RV Names:       None

    x     p(x)
    000   1/2
    111   1/2

Here, we see that either input informs us of exactly what the output is. One could call this "redundant" information. Furthermore, consider the :ref:`coinformation` of these distributions:

.. ipython::

    In [6]: from dit.multivariate import coinformation as I

    In [7]: I(xor)
    Out[7]: -1.0

    In [8]: I(gb)
    Out[8]: 1.0

This could lead one to intuit that negative values of the coinformation correspond to synergistic effects in a distribution, while positive values correspond to redundant effects. This intuition, however, is at best misleading: the coinformation of a 4-variable giant bit and 4-variable parity distribution are both positive:

.. ipython::

    In [9]: I(dit.example_dists.giant_bit(4, 2))
    Out[9]: 1.0

    In [10]: I(dit.example_dists.n_mod_m(4, 2))
    Out[10]: 1.0

This, as well as other issues, lead Williams & Beer :cite:`williams2010nonnegative` to propose the *partial information decomposition*.

Framework
=========

The goal of the partial information is to assign to each some non-negative portion of :math:`\I[\{X_i\} : Y]` to each antichain over the inputs. An antichain over the inputs is a set of sets, where each of those sets is not a subset of any of the others. For example, :math:`\{ \{X_0, X_1\}, \{X_1, X_2\} \}` is an antichain, but :math:`\{ \{X_0, X_1\}, \{X_0 X_1, X_2\} \}` is not.

The antichains for a lattice based on this partial order:

.. math::

    \alpha \leq \beta \iff \forall \mathbf{b} \in \beta, \exists \mathbf{a} \in \alpha, \mathbf{a} \subseteq \mathbf{b}

From here, we wish to find a redundancy measure, :math:`\I_\cap` which would assign a fraction of :math:`\I[\{X_i\} : Y]` to each antichain intuitively quantifying what portion of the information in the output could be learned by observing any of the sets of variables within the antichain. In order to be a viable measure of redundancy, there are several axioms a redundancy measure must satisfy.

Bivariate Lattice
-----------------

Let us consider the special case of two inputs. The lattice consists of four elements: :math:`\{\{X_0\}, \{X_1\}\}`, :math:`\{\{X_0\}\}`, :math:`\{\{X_1\}\}`, and :math:`\{\{X_0, X_1\}\}`. We can interpret these elements as the *redundancy* provided by both inputs, the information *uniquely* provided by :math:`X_0`, the information *uniquely* provided by :math:`X_1`, and the information *synergistically* provided only by both inputs together. Together these for elements decompose the input-output mutual information:

.. math::

    \I[X_0, X_1 : Y] = \I_\cap[\{X_0\}, \{X_1\} : Y] + \I_\cap[\{X_0\} : Y] + \I_\cap[\{X_1\} : Y] + \I_\cap[\{X_0, X_1\} : Y]

Furthermore, due to the self-redundancy axiom (described ahead), the single-input mutual informations decomposed in the following way:

.. math::

    \I[X_0 : Y] = \I_\cap[\{X_0\}, \{X_1\} : Y] + \I_\cap[\{X_0\} : Y]

    \I[X_1 : Y] = \I_\cap[\{X_0\}, \{X_1\} : Y] + \I_\cap[\{X_1\} : Y]

Colloquially, from input :math:`X_0` one can learn what is redundantly provided by either input, plus what is uniquely provided by :math:`X_0`, but not what is uniquely provided by :math:`X_1` or what can only be learned synergistically from both inputs.

Axioms
------

The following three axioms were provided by Williams & Beer.

Symmetry
^^^^^^^^

The redundancy :math:`\I_\cap[\{X_i\} : Y]` is invariant under reorderings of :math:`X_i`.

Self-Redundancy
^^^^^^^^^^^^^^^

The redundancy of a single input is its mutual information with the output:

.. math::

    \I_\cap[X_i : Y] = \I[X_i : Y]

Monotonicity
^^^^^^^^^^^^

The redundancy should only decrease with in inclusion of more inputs:

.. math::

    \I_\cap[\mathcal{A}_1, \ldots, \mathcal{A}_{k-1}, \mathcal{A}_k : Y] \leq \I_\cap[\mathcal{A}_1, \ldots, \mathcal{A}_{k-1} : Y]

with equality if :math:`\mathcal{A}_{k-1} \subseteq \mathcal{A}_k`.

There have been other axioms proposed following from those of Williams & Beer.

Identity
^^^^^^^^

The identity axiom :cite:`harder2013bivariate` states that if the output is identical to the inputs, then the redundancy is the mutual information between the inputs:

.. math::

    \I_\cap[X_0, X_1 : (X_0, X_1)] = \I[X_0 : X_1]

Target (output) Monotonicity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This axiom states that redundancy can not increase when replacing the output by a function of itself.

.. math::

    \I_\cap[\{X_i\} : Y] \ge I_\cap[\{X_i\} : f(Y)]

It first appeared in :cite:`bertschinger2013shared` and was expanded upon in :cite:`rauh2017extractable`.

Measures
========

We now turn our attention a variety of methods proposed to flesh out this partial information decomposition.

.. ipython::

    In [11]: from dit.pid import *

.. py:module:: dit.pid.imin
:math:`\I_{min}`
----------------

:math:`\I_{min}`:cite:`williams2010nonnegative` was Williams & Beer's initial proposal for a redundancy measure. It is given by:

.. math::

    \I_{min}(\mathcal{A}_1, \mathcal{A}_2, \ldots : Y) = \sum_{y \in Y} p(y) \min_{\mathcal{A}_i} \I[\mathcal{A}_i : Y=y]

However, this measure has been criticized for acting in an unintuitive manner :cite:`griffith2014quantifying`:

.. ipython::

    In [12]: d = dit.Distribution(['000', '011', '102', '113'], [1/4]*4)

    In [13]: PID_WB(d)
    ╔════════╤════════╤════════╗
    ║ I_min  │  I_∩   │   π    ║
    ╟────────┼────────┼────────╢
    ║ {0:1}  │ 2.0000 │ 1.0000 ║
    ║  {0}   │ 1.0000 │ 0.0000 ║
    ║  {1}   │ 1.0000 │ 0.0000 ║
    ║ {0}{1} │ 1.0000 │ 1.0000 ║
    ╚════════╧════════╧════════╝

We have constructed a distribution whose inputs are independent random bits, and whose output is the concatenation of those inputs. Intuitively, the output should then be informed by one bit of unique information from :math:`X_0` and one bit of unique information from :math:`X_1`. However, :math:`\I_{min}` assesses that there is one bit of redundant information, and one bit of synergistic information. This is because :math:`\I_{min}` quantifies redundancy as the least amount of information one can learn about an output given any single input. Here, however, the one bit we learn from :math:`X_0` is, in a sense, orthogonal from the one bit we learn from :math:`X_1`. This observation has lead to much of the follow-on work.

.. py:module:: dit.pid.immi
:math:`\I_{MMI}`
----------------

One potential measure of redundancy is the *minimum mutual information* :cite:`bertschinger2013shared`:

.. math::

    \I_\cap[\{X_i\} : Y] = \min_{i} \I{X_i : Y}

This measure, though crude, is known to be correct for multivariate gaussian variables :cite:`olbrich2015information`.

.. py:module:: dit.pid.idownarrow
:math:`\I_{\downarrow}`
-----------------------

Drawing inspiration from information-theoretic cryptography, this PID quantifies unique information using the :ref:`Intrinsic Mutual Information`:

.. math::

    \I_{\pi}[\{X_i\} : Y] = \I{X_i : Y \downarrow \{X_0, X_1, \ldots\} - \{X_i\}}

While this seems intuitively plausible, it turns out that this leads to an inconsistent decomposition :cite:`bertschinger2013shared`; namely, in the bivariate case, if one were to compute redundancy using either unique information subtracted from that inputs mutual information with the output the value should be the same. There are examples where this is not the case:

.. ipython::

    In [14]: d = bivariates['prob 1']

    In [15]: PID_downarrow(d)
    ╔════════╤════════╤════════╗
    ║  I_da  │  I_∩   │   π    ║
    ╟────────┼────────┼────────╢
    ║ {0:1}  │ 1.0000 │ 0.1887 ║
    ║  {0}   │ 0.3113 │ 0.1887 ║
    ║  {1}   │ 0.5000 │ 0.5000 ║
    ║ {0}{1} │ 0.1226 │ 0.1226 ║
    ╚════════╧════════╧════════╝

Interestingly, compared to other measures the intrinsic mutual information seems to *overestimate* unique information. Since :math:`I{X_0 : Y \downarrow X_1} \leq \min\{ \I{X_0 : Y | X_1}, \I{X_0 : Y} \} = \min\{ U_0 + S, U_0 + R\}`, where :math:`R` is redundancy, :math:`U_0` is unique information from input :math:`X_0`, and :math:`S` is synergy, this implies that the optimization performed in computing the intrinsic mutual information is unable to completely remove either redundancy, synergy, or both.

.. py:module:: dit.pid.iwedge
:math:`\I_{\wedge}`
------------------

Redundancy seems to intuitively be related to common information :ref:`Common Informations`. This intuition lead to the development of :math:`\I_{\wedge}` :cite:`griffith2014intersection`:

.. math::

    \I_\cap[\{X_i\} : Y] = \I[ X_0 \meet X_1 \meet \ldots \meet X_n : Y]

That is, redundancy is the information the :ref:`Gács-Körner Common Information` of the inputs shares with the output. This measure is known to produce negative partial information values in some instances.

.. py:module:: dit.pid.iproj
:math:`\I_{proj}`
-----------------

Utilizing information geometry, Harder et al :cite:`harder2013bivariate` have developed a strictly bivariate measure of redundancy, :math:`\I_{proj}`:

.. math::

    \I_{proj} = \min \{ I^\pi_Y[X_0 \mss X_1], I^\pi_Y[X_1 \mss X_0] \}

where

.. math::

    I^\pi_Y[X_0 \mss X_1] = \sum_{x_0, y} p(x_0, y) \log \frac{p_{(x_0 \mss X_1)}(y)}{p(y)}

    p_{(x_0 \mss X_1)}(Y) = \pi_{C_{cl}(\langle X_1 \rangle_Y)}(p(Y | x_0)

    \pi_B(p) = \arg \min_{r \in B} \DKL[p || r]

    C_{cl}(\langle X_1 \rangle_Y) = C_{cl}([{p(Y | x_1) : x_1 \in X_1 \})

where :math:`C_{cl}(\bullet)` denotes closure. Intuitively, this measures seeks to quantify redundancy as the minimum of how much :math:`p(Y | X_0)` can be expressed when :math:`X_0` is projected on to :math:`X_1`, and vice versa.

.. py:module:: dit.pid.ibroja
:math:`\I_{BROJA}`
------------------

In a very intuitive effort, Bertschinger et al (henceforth BROJA) :cite:`bertschinger2014quantifying,griffith2014quantifying` defined unique information as the minimum conditional mutual informations obtainable while holding the input-output marginals fixed:

.. math::

    \Delta = \{ Q : \forall i : p(x_i, y) = q(x_i, y) \}

    \I_\partial[\{X_i\} : Y] = \min_{Q \in \Delta} \I_{Q}[X_i : Y | \{X_0, \ldots\} - \{X_i\}]

.. note::

    In the bivariate case, Griffith independently suggested the same decomposition but from the viewpoint of synergy :cite:`griffith2014quantifying`.

The BROJA measure has recently been criticized for behaving in an unintuitive manner on some examples. Consider the *reduced or* distribution:

.. ipython::

    In [16]: bivariates['reduced or']
    Out[16]:
    Class:          Distribution
    Alphabet:       ('0', '1') for all rvs
    Base:           linear
    Outcome Class:  str
    Outcome Length: 3
    RV Names:       None

    x     p(x)
    000   1/2
    011   1/4
    101   1/4

    In [17]: print(PID_BROJA(bivariates['reduced or']))
    ╔═════════╤════════╤════════╗
    ║ I_broja │  I_∩   │   π    ║
    ╟─────────┼────────┼────────╢
    ║  {0:1}  │ 1.0000 │ 0.6887 ║
    ║   {0}   │ 0.3113 │ 0.0000 ║
    ║   {1}   │ 0.3113 │ 0.0000 ║
    ║  {0}{1} │ 0.3113 │ 0.3113 ║
    ╚═════════╧════════╧════════╝

We see that in this instance BROJA assigns no partial information to either unique information. However, it is not difficult to argue that in the case that either input is a 1, that input then has unique information regarding the output.

:math:`\I_{proj}` and :math:`\I_{BROJA}` are Distinct
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the BROJA paper :cite:`bertschinger2014quantifying` the only example given where their decomposition differs from that of Harder et al. is the :py:func:`dit.example_dists.summed_dice`. We can find a simpler example where they differ using hypothesis:

.. ipython::

   In [18]: from hypothesis import find

   In [19]: from dit.utils.testing import distributions

   In [20]: def broja_not_proj(d):
   ...:     return PID_BROJA(d) != PID_Proj(d)
   ...:

   In [21]: find(distributions(3, 2, True), broja_not_proj)
   Out[21]:
   Class:          Distribution
   Alphabet:       (0, 1) for all rvs
   Base:           linear
   Outcome Class:  tuple
   Outcome Length: 3
   RV Names:       None

   x           p(x)
   (0, 0, 0)   0.25
   (0, 0, 1)   0.25
   (0, 1, 1)   0.25
   (1, 0, 0)   0.25

.. py:module:: dit.pid.iccs
:math:`\I_{ccs}`
----------------

Taking a pointwise point of view, Ince has proposed a measure of redundancy based on the :ref:`coinformation` :cite:`ince2017measuring`:

.. math::

    \I_\cap[\{X_i\} : Y] = \sum p(x_0, \ldots, x_n, y) \I[x_0 : \ldots : x_n : y] \textrm{if} \op{sign}(\I[x_i : y]) = \op{sign}(\I[x_0 : \ldots : x_n : y])

While this measure behaves intuitively in many examples, it also assigns negative values to some partial information atoms in some instances.

This decomposition also displays an interesting phenomena, that of *subadditive redundancy*. The **gband** distribution is an independent mix of a giant bit (redundancy of 1 bit) and the **and** distribution (redundancy of 0.1038 bits), and yet **gband** has 0.8113 bits of redundancy:

.. ipython::

    In [21]: PID_CCS(bivariates['gband'])
    Out[21]:
    ╔════════╤════════╤════════╗
    ║ I_ccs  │  I_∩   │   π    ║
    ╟────────┼────────┼────────╢
    ║ {0:1}  │ 1.8113 │ 0.0000 ║
    ║  {0}   │ 1.3113 │ 0.5000 ║
    ║  {1}   │ 1.3113 │ 0.5000 ║
    ║ {0}{1} │ 0.8113 │ 0.8113 ║
    ╚════════╧════════╧════════╝
