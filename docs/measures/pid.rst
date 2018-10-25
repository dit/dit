.. pid.rst
.. py:module:: dit.pid

*********************************
Partial Information Decomposition
*********************************

The *partial information decomposition* (PID), put forth by Williams & Beer :cite:`williams2010nonnegative`, is a framework for decomposing the information shared between a set of variables we will refer to as *inputs*, :math:`X_0, X_1, \ldots`, and another random variable we will refer to as the *output*, :math:`Y`. This decomposition seeks to partition the information :math:`\I{X_0,X_1,\ldots : Y}` among the antichains of the inputs.

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

The goal of the partial information is to assign to each some non-negative portion of :math:`\I{\{X_i\} : Y}` to each antichain over the inputs. An antichain over the inputs is a set of sets, where each of those sets is not a subset of any of the others. For example, :math:`\left\{ \left\{X_0, X_1\right\}, \left\{X_1, X_2\right\} \right\}` is an antichain, but :math:`\left\{ \left\{X_0, X_1\right\}, \left\{X_0 X_1, X_2\right\} \right\}` is not.

The antichains for a lattice based on this partial order:

.. math::

   \alpha \leq \beta \iff \forall \mathbf{b} \in \beta, \exists \mathbf{a} \in \alpha, \mathbf{a} \subseteq \mathbf{b}

From here, we wish to find a redundancy measure, :math:`\Icap{\bullet}` which would assign a fraction of :math:`\I{\{X_i\} : Y}` to each antichain intuitively quantifying what portion of the information in the output could be learned by observing any of the sets of variables within the antichain. In order to be a viable measure of redundancy, there are several axioms a redundancy measure must satisfy.

Bivariate Lattice
-----------------

Let us consider the special case of two inputs. The lattice consists of four elements: :math:`\left\{\left\{X_0\right\}, \left\{X_1\right\}\right\}`, :math:`\left\{\left\{X_0\right\}\right\}`, :math:`\left\{\left\{X_1\right\}\right\}`, and :math:`\left\{\left\{X_0, X_1\right\}\right\}`. We can interpret these elements as the *redundancy* provided by both inputs, the information *uniquely* provided by :math:`X_0`, the information *uniquely* provided by :math:`X_1`, and the information *synergistically* provided only by both inputs together. Together these for elements decompose the input-output mutual information:

.. math::

   \I{X_0, X_1 : Y} = \Ipart{\left\{X_0\right\}, \left\{X_1\right\} : Y} + \Ipart{\left\{X_0\right\} : Y} + \Ipart{\left\{X_1\right\} : Y} + \Ipart{\left\{X_0, X_1\right\} : Y}

Furthermore, due to the self-redundancy axiom (described ahead), the single-input mutual informations decomposed in the following way:

.. math::

   \I{X_0 : Y} = \Ipart{\left\{X_0\right\}, \left\{X_1\right\} : Y} + \Ipart{\left\{X_0\right\} : Y}

   \I{X_1 : Y} = \Ipart{\left\{X_0\right\}, \left\{X_1\right\} : Y} + \Ipart{\left\{X_1\right\} : Y}

Colloquially, from input :math:`X_0` one can learn what is redundantly provided by either input, plus what is uniquely provided by :math:`X_0`, but not what is uniquely provided by :math:`X_1` or what can only be learned synergistically from both inputs.

Axioms
------

The following three axioms were provided by Williams & Beer.

Symmetry
^^^^^^^^

The redundancy :math:`\Icap{X_{0:n} : Y}` is invariant under reorderings of :math:`X_i`.

Self-Redundancy
^^^^^^^^^^^^^^^

The redundancy of a single input is its mutual information with the output:

.. math::

   \Icap{X_i : Y} = \I{X_i : Y}

Monotonicity
^^^^^^^^^^^^

The redundancy should only decrease with in inclusion of more inputs:

.. math::

   \Icap{\mathcal{A}_1, \ldots, \mathcal{A}_{k-1}, \mathcal{A}_k : Y} \leq \Icap{\mathcal{A}_1, \ldots, \mathcal{A}_{k-1} : Y}

with equality if :math:`\mathcal{A}_{k-1} \subseteq \mathcal{A}_k`.

There have been other axioms proposed following from those of Williams & Beer.

Identity
^^^^^^^^

The identity axiom :cite:`harder2013bivariate` states that if the output is identical to the inputs, then the redundancy is the mutual information between the inputs:

.. math::

   \Icap{X_0, X_1 : \left(X_0, X_1\right)} = \I{X_0 : X_1}

Target (output) Monotonicity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This axiom states that redundancy can not increase when replacing the output by a function of itself.

.. math::

   \Icap{X_{0:n} : Y} \ge \Icap{X_{0:n} : f(Y)}

It first appeared in :cite:`bertschinger2013shared` and was expanded upon in :cite:`rauh2017extractable`.

Measures
========

We now turn our attention a variety of methods proposed to flesh out this partial information decomposition.

.. ipython::

   In [11]: from dit.pid import *

.. py:module:: dit.pid.imin
:math:`\Imin{\bullet}`
----------------------

:math:`\Imin{\bullet}`:cite:`williams2010nonnegative` was Williams & Beer's initial proposal for a redundancy measure. It is given by:

.. math::

   \Imin{\mathcal{A}_1, \mathcal{A}_2, \ldots : Y} = \sum_{y \in Y} p(y) \min_{\mathcal{A}_i} \I{\mathcal{A}_i : Y=y}

However, this measure has been criticized for acting in an unintuitive manner :cite:`griffith2014quantifying`:

.. ipython::

   In [12]: d = dit.Distribution(['000', '011', '102', '113'], [1/4]*4)

   In [13]: PID_WB(d)
   ╔════════╤════════╤════════╗
   ║ I_min  │  I_r   │   pi   ║
   ╟────────┼────────┼────────╢
   ║ {0:1}  │ 2.0000 │ 1.0000 ║
   ║  {0}   │ 1.0000 │ 0.0000 ║
   ║  {1}   │ 1.0000 │ 0.0000 ║
   ║ {0}{1} │ 1.0000 │ 1.0000 ║
   ╚════════╧════════╧════════╝

We have constructed a distribution whose inputs are independent random bits, and whose output is the concatenation of those inputs. Intuitively, the output should then be informed by one bit of unique information from :math:`X_0` and one bit of unique information from :math:`X_1`. However, :math:`\Imin{\bullet}` assesses that there is one bit of redundant information, and one bit of synergistic information. This is because :math:`\Imin{\bullet}` quantifies redundancy as the least amount of information one can learn about an output given any single input. Here, however, the one bit we learn from :math:`X_0` is, in a sense, orthogonal from the one bit we learn from :math:`X_1`. This observation has lead to much of the follow-on work.

.. py:module:: dit.pid.immi
:math:`\Immi{\bullet}`
----------------------

One potential measure of redundancy is the *minimum mutual information* :cite:`bertschinger2013shared`:

.. math::

   \Immi{X_{0:n} : Y} = \min_{i} \I{X_i : Y}

This measure, though crude, is known to be correct for multivariate gaussian variables :cite:`olbrich2015information`.

.. py:module:: dit.pid.iwedge
:math:`\Iwedge{\bullet}`
------------------------

Redundancy seems to intuitively be related to common information :ref:`Common Informations`. This intuition lead to the development of :math:`\Iwedge{\bullet}` :cite:`griffith2014intersection`:

.. math::

   \Iwedge{X_{0:n} : Y} = \I{ \meet X_i : Y}

That is, redundancy is the information the :ref:`Gács-Körner Common Information` of the inputs shares with the output.

.. warning::

   This measure can result in a negative PID.

.. py:module:: dit.pid.iproj
:math:`\Iproj{\bullet}`
-----------------------

Utilizing information geometry, Harder et al :cite:`harder2013bivariate` have developed a strictly bivariate measure of redundancy, :math:`\Iproj{\bullet}`:

.. math::

   \Iproj{\left\{X_0\right\}\left\{X_1\right\} : Y} = \min \{ I^\pi_Y[X_0 \mss X_1], I^\pi_Y[X_1 \mss X_0] \}

where

.. math::

   I^\pi_Y[X_0 \mss X_1] = \sum_{x_0, y} p(x_0, y) \log \frac{p_{(x_0 \mss X_1)}(y)}{p(y)}

   p_{(x_0 \mss X_1)}(Y) = \pi_{C_{cl}(\langle X_1 \rangle_Y)}(p(Y | x_0)

   \pi_B(p) = \arg \min_{r \in B} \DKL{p || r}

   C_{cl}(\langle X_1 \rangle_Y) = C_{cl}(\left\{p(Y | x_1) : x_1 \in X_1 \right\})

where :math:`C_{cl}(\bullet)` denotes closure. Intuitively, this measures seeks to quantify redundancy as the minimum of how much :math:`p(Y | X_0)` can be expressed when :math:`X_0` is projected on to :math:`X_1`, and vice versa.

.. py:module:: dit.pid.ibroja
:math:`\Ibroja{\bullet}`
------------------------

In a very intuitive effort, Bertschinger et al (henceforth BROJA) :cite:`bertschinger2014quantifying,griffith2014quantifying` defined unique information as the minimum conditional mutual informations obtainable while holding the input-output marginals fixed:

.. math::

   \Delta = \{ Q : \forall i : p(x_i, y) = q(x_i, y) \}

   \Ibroja{X_{0:n} : Y} = \min_{Q \in \Delta} \I{X_i : Y | X_\overline{\{i\}}}

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
   ║ I_broja │  I_r   │   pi   ║
   ╟─────────┼────────┼────────╢
   ║  {0:1}  │ 1.0000 │ 0.6887 ║
   ║   {0}   │ 0.3113 │ 0.0000 ║
   ║   {1}   │ 0.3113 │ 0.0000 ║
   ║  {0}{1} │ 0.3113 │ 0.3113 ║
   ╚═════════╧════════╧════════╝

We see that in this instance BROJA assigns no partial information to either unique information. However, it is not difficult to argue that in the case that either input is a 1, that input then has unique information regarding the output.

:math:`\Iproj{\bullet}` and :math:`\Ibroja{\bullet}` are Distinct
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the BROJA paper :cite:`bertschinger2014quantifying` the only example given where their decomposition differs from that of Harder et al. is the :py:func:`dit.example_dists.summed_dice`. We can find a simpler example where they differ using hypothesis:

.. ipython::

   In [17]: from hypothesis import find

   In [18]: from dit.utils.testing import distribution_structures

   In [19]: find(distribution_structures(3, 2, True), lambda d: PID_Proj(d) != PID_BROJA(d))
   Out[19]:
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
:math:`\Iccs{\bullet}`
----------------------

Taking a pointwise point of view, Ince has proposed a measure of redundancy based on the :ref:`coinformation` :cite:`ince2017measuring`:

.. math::

   \Iccs{X_{0:n} : Y} = \sum p(x_0, \ldots, x_n, y) \I{x_0 : \ldots : x_n : y}~~\textrm{if}~~\operatorname{sign}(\I{x_i : y}) = \operatorname{sign}(\I{x_0 : \ldots : x_n : y})

While this measure behaves intuitively in many examples, it also assigns negative values to some partial information atoms in some instances.

This decomposition also displays an interesting phenomena, that of *subadditive redundancy*. The **gband** distribution is an independent mix of a giant bit (redundancy of 1 bit) and the **and** distribution (redundancy of 0.1038 bits), and yet **gband** has 0.8113 bits of redundancy:

.. ipython::

   In [20]: PID_CCS(bivariates['gband'])
   Out[20]:
   ╔════════╤════════╤════════╗
   ║ I_ccs  │  I_r   │   pi   ║
   ╟────────┼────────┼────────╢
   ║ {0:1}  │ 1.8113 │ 0.0000 ║
   ║  {0}   │ 1.3113 │ 0.5000 ║
   ║  {1}   │ 1.3113 │ 0.5000 ║
   ║ {0}{1} │ 0.8113 │ 0.8113 ║
   ╚════════╧════════╧════════╝

.. warning::

   This measure can result in a negative PID.

.. py:module:: dit.pid.idep
:math:`\Idep{\bullet}`
----------------------

James et al :cite:`james2017unique` have developed a method of quantifying unique information based on the :ref:`Dependency Decomposition`. Unique information from variable :math:`X_i` is evaluated as the least change in sources-target mutual information when adding the constraint :math:`X_i Y`.

.. ipython::

   In [21]: PID_dep(bivariates['not two'])
   Out[21]:
   ╔════════╤════════╤════════╗
   ║ I_dep  │  I_r   │   pi   ║
   ╟────────┼────────┼────────╢
   ║ {0:1}  │ 0.5710 │ 0.5364 ║
   ║  {0}   │ 0.0200 │ 0.0146 ║
   ║  {1}   │ 0.0200 │ 0.0146 ║
   ║ {0}{1} │ 0.0054 │ 0.0054 ║
   ╚════════╧════════╧════════╝

.. py:module:: dit.pid.ipm
:math:`\Ipm{\bullet}`
---------------------

Also taking a pointwise view, Finn & Lizier's :math:`\Ipm{\bullet}` :cite:`finn2017` instead splits the pointwise mutual information into two components:

.. math::

   i(s, t) = h(s) - h(s|t)

They then define two partial information lattices, one quantified locally by :math:`h(s)` and the other by :math:`h(s|t)`. By averaging these local lattices and then recombining them, we arrive at a standard Williams & Beer redundancy lattice.

.. ipython::

   In [22]: PID_PM(bivariates['pnt. unq'])
   Out[22]:
   ╔════════╤════════╤════════╗
   ║ I_pm   │  I_r   │   pi   ║
   ╟────────┼────────┼────────╢
   ║ {0:1}  │ 1.0000 │ 0.0000 ║
   ║  {0}   │ 0.5000 │ 0.5000 ║
   ║  {1}   │ 0.5000 │ 0.5000 ║
   ║ {0}{1} │ 0.0000 │ 0.0000 ║
   ╚════════╧════════╧════════╝

.. warning::

   This measure can result in a negative PID.

.. py:module:: dit.pid.irav
:math:`\Irav{\bullet}`
---------------------

Taking a functional perspective as in :math:`\Iwedge`, :math:`\Irav` defines bivariate redundancy as the maximum coinformation between the two sources :math:`X_0, X_1', a target :math:`Y`, and a deterministic function of the inputs :math:`f(X_0,X_1)`.

.. math::

   \Irav{X_{0:2} : Y} = \max_f\left(\I{X_0\!:\!X_1\!:\!Y\!:\!f(X_0,X_1)}

This measure is designed to exploit the conflation of synergy and redundancy in the three variable coinformation: :math:`\I{X_0\!:\!X_1\!:\!Y} = R - S`.

.. ipython::

   In [23]: PID_RAV(bivariates['pnt. unq'])
   Out[23]:
   ╔════════╤════════╤════════╗
   ║ I_pm   │  I_r   │   pi   ║
   ╟────────┼────────┼────────╢
   ║ {0:1}  │ 1.0000 │ 0.0000 ║
   ║  {0}   │ 0.5000 │ 0.5000 ║
   ║  {1}   │ 0.5000 │ 0.5000 ║
   ║ {0}{1} │ 0.0000 │ 0.0000 ║
   ╚════════╧════════╧════════╝


.. py:module:: dit.pid.irr
:math:`\Irr{\bullet}`
---------------------

In order to combine :math:`\Immi{\bullet}` with the coinformation, Goodwell and Kumar :cite:`goodwell2017temporal` have introduced their *rescaled redundancy*:

.. math::

   \Irr{X_0 : X_1} = R_{\text{min}} + I_{S} (\Immi{X_{0:2} : Y} - R_{\text{min}}

   R_{\text{min}} = \max\{ 0, \I{X_0 : X_1 : Y} \}

   I_{S} = \frac{\I{X_0 : X_1}}{\min\{ \H{X_0}, \H{X_1} \}}

.. ipython::

   In [24]: PID_RR(bivariates['pnt. unq'])
   Out[24]:
   ╔════════╤════════╤════════╗
   ║ I_rr   │  I_r   │   pi   ║
   ╟────────┼────────┼────────╢
   ║ {0:1}  │ 1.0000 │ 0.3333 ║
   ║  {0}   │ 0.5000 │ 0.1667 ║
   ║  {1}   │ 0.5000 │ 0.1667 ║
   ║ {0}{1} │ 0.3333 │ 0.3333 ║
   ╚════════╧════════╧════════╝

.. py:module:: dit.pid.ira
:math:`\Ira{\bullet}`
---------------------

Drawing from the reconstructability analysis work of Zwick :cite:`zwick2004overview`, we can define :math:`Ira{\bullet}` as a restricted form of :math:`\Idep{\bullet}`.

.. warning::

   This measure can result in a negative PID.

.. py:module:: dit.pid.iskar
Secret Key Agreement Rates
--------------------------

One can associate :ref:`Secret Key Agreement` rates with unique informations by considering the rate at which one source and the target can agree upon a secret key while the other source eavesdrops. This results in four possibilities:
- neither source nor target communicate
- only the source communicates
- only the target communicates
- both the source and the target communicate

No Communication
~~~~~~~~~~~~~~~~

.. math::

   \Ipart{X_i \rightarrow Y \setminus X_j} = \operatorname{S}[X_i : Y || X_j]

.. warning::

   This measure can result in an inconsistent PID.


One-Way Communication
~~~~~~~~~~~~~~~~~~~~~

Camel
^^^^^

.. math::

   \Ipart{X_i \rightarrow Y \setminus X_j} = \operatorname{S}[X_i \rightarrow Y || X_j]

Elephant
^^^^^^^^

.. math::

   \Ipart{X_i \rightarrow Y \setminus X_j} = \operatorname{S}[X_i \leftarrow Y || X_j]

.. warning::

   This measure can result in an inconsistent PID.


Two-Way Communication
~~~~~~~~~~~~~~~~~~~~~

.. math::

   \Ipart{X_i \rightarrow Y \setminus X_j} = \operatorname{S}[X_i \leftrightarrow Y || X_j]

.. warning::

   This measure can result in an inconsistent PID.


Partial Entropy Decomposition
=============================

Ince :cite:`ince2017partial` proposed applying the PID framework to decompose multivariate entropy (without considering information about a separate target variable). This *partial entropy decomposition* (PED), seeks to partition a mutlivariate entropy :math:`\H{X_0,X_1,\ldots}` among the antichains of the variables. The PED perspective shows that bivariate mutual information is equal to the difference between redundant entropy and synergistic entropy.

.. math::

   \I{X_0 : X_1} = \Hpart{\left\{X_0\right\}, \left\{X_1\right\}} - \Hpart{\left\{X_0,X_1\right\}}

.. py:module:: dit.pid.hcs
:math:`\Hcs{\bullet}`
----------------------

Taking a pointwise point of view, following :math:`\Iccs{\bullet}`, Ince has proposed a measure of redundant entropy based on the :ref:`coinformation` :cite:`ince2017partial`:

.. math::

   \Hcs{X_{0:n}} = \sum p(x_0, \ldots, x_n) \I{x_0 : \ldots : x_n}~~\textrm{if}~~(\I{x_0 : \ldots : x_n} > 0)

While this measure behaves intuitively in many examples, it also assigns negative values to some partial entropy atoms in some instances. However, Ince :cite:`ince2017partial` argues that concepts such as mechanistic information redundnacy (non-zero information redundancy between independent predictors, c.f. AND) necessitate negative partial entropy terms.

Like :math:`\Iccs{\bullet}`,  :math:`\Hcs{\bullet}` is also subadditive.

.. ipython::

   In [25]: PED_CS(dit.Distribution(['00','01','10','11'],[0.25]*4))
   Out[25]:
   ╔════════╤════════╤════════╗
   ║  H_cs  │  H_r   │  H_d   ║
   ╟────────┼────────┼────────╢
   ║ {0:1}  │ 2.0000 │ 0.0000 ║
   ║  {0}   │ 1.0000 │ 1.0000 ║
   ║  {1}   │ 1.0000 │ 1.0000 ║
   ║ {0}{1} │ 0.0000 │ 0.0000 ║
   ╚════════╧════════╧════════╝
