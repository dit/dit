.. pid.rst
.. py:currentmodule:: dit.pid

*********************************
Partial Information Decomposition
*********************************

The *partial information decomposition* (PID), put forth by Williams & Beer :cite:`williams2010nonnegative`, is a framework for decomposing the information shared between a set of variables we will refer to as *inputs*, :math:`X_0, X_1, \ldots`, and another random variable we will refer to as the *output*, :math:`Y`. This decomposition seeks to partition the information :math:`\I{X_0,X_1,\ldots : Y}` among the antichains of the inputs.

Background
==========

It is often desirable to determine how a set of inputs influence the behavior of an output. :mod:`dit.pid.distributions` ships two dictionaries of standard test cases from the PID literature: ``bivariates`` (two inputs; e.g. ``'and'``, ``'synergy'``, ``'redundant'``, ``'unique 1'``, ``'rdn xor'``, ``'reduced or'``, ``'sum'``, ``'cat'``, ``'boom'``) and ``trivariates`` (three inputs; e.g. ``'synergy'``, ``'redundant'``, ``'dbl xor'``, ``'giant xor'``, ``'anddup'``); ``sorted(bivariates)`` lists them all. Consider the exclusive or logic gates, for example:

.. ipython::

   In [1]: from dit.pid.distributions import bivariates, trivariates

   In [2]: xor = bivariates['synergy']

   In [3]: print(xor)

We can see from inspection that either input (the first two indexes) is independent of the output (the final index), yet the two inputs together determine the output. One could call this "synergistic" information. Next, consider the giant bit distribution:

.. ipython::

   In [4]: gb = bivariates['redundant']

   In [5]: print(gb)

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

Source Chain Rule
^^^^^^^^^^^^^^^^^

If a context :math:`C` of sources is known, collections that each contain it share all of its information, plus what the remaining collections share once it is known :cite:`gutknecht2026partial`:

.. math::

   \Icap{\mathcal{A}_1 C, \ldots, \mathcal{A}_k C : Y} = \I{C : Y} + \Icap{\mathcal{A}_1, \ldots, \mathcal{A}_k : Y | C}

where the conditional redundancy averages the redundancy under each :math:`p(\cdot | C = c)`. It holds exactly when conditioning on the context regroups the atoms by the further sources that unlock them, that is, when the conditional and contextual decompositions agree (see `Conditional and Contextual Decompositions`_). With three or more sources it is incompatible with nonnegative atoms.

These and the other properties proposed in the literature can be checked on any decomposition; see `Checking Properties`_.

Measures
========

We now turn our attention a variety of methods proposed to flesh out this partial information decomposition.

.. ipython::

   In [11]: from dit.pid import *

Each measure is a class, constructed from a distribution and, optionally, the sources and the target: by default every variable but the last is a source, and the last is the target. Measures differ in what they quantify directly:

* a *redundancy* gives every node of the lattice, so the decomposition is complete for any number of sources;
* a *pointwise* redundancy is the average of a redundancy of each outcome; ``pointwise=True`` also gives the decomposition of each outcome;
* a *bivariate* redundancy is defined for two collections of sources;
* a *unique information* is the information a source provides beyond all the others;
* a *union information* is what any of several collections of sources provide; the redundancies follow by inclusion-exclusion.

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Measures
     - Quantity
     - Computed by
   * - :class:`PID_WB`, :class:`PID_MMI`, :class:`PID_GK`
     - redundancy
     - closed form
   * - :class:`PID_PM`, :class:`PID_SX`
     - pointwise
     - closed form
   * - :class:`PID_CCS`
     - redundancy
     - maximum entropy
   * - :class:`PID_Deg`, :class:`PID_Prec`, :class:`PID_GH`
     - redundancy
     - linear programs over extreme rays
   * - :class:`PID_MC`, :class:`PID_LN`
     - redundancy
     - optimization under sampled constraints (approximate)
   * - :class:`PID_RAV`
     - redundancy
     - search over partitions of the outcomes (small distributions)
   * - :class:`PID_RDR`
     - union information
     - concave envelopes of ROC curves
   * - :class:`PID_CI`
     - union information
     - optimization
   * - :class:`PID_MES`, :class:`PID_Do`, :class:`PID_RR`
     - bivariate
     - closed form
   * - :class:`PID_CT`
     - paths through the sources
     - exact
   * - :class:`PID_Proj`, :class:`PID_IG`, :class:`PID_Delta`
     - bivariate
     - convex optimization
   * - :class:`PID_IPID`, :class:`PID_DeltaLambda`
     - bivariate
     - nonconvex optimization from several starting points
   * - :class:`PID_BROJA`
     - unique information
     - convex optimization (exponential cone or admUI)
   * - :class:`PID_dep`, :class:`PID_RA`
     - unique information
     - maximum entropy
   * - ``PID_SKAR_*``
     - unique information
     - secret key agreement rates and their bounds

Decompositions by bivariate redundancies and unique informations are complete for two sources. For more, only some redundancies are computed. The atoms they determine are exact. Others follow from self-redundancy and the monotonicity of redundancy on the lattice, provided the atoms so obtained are nonnegative, as monotonicity assumes. The remaining atoms are ``nan``. ``pid.complete`` tells whether every atom is known, and ``pid.consistent`` whether the known values satisfy the Möbius inversion; some unique informations, such as the secret key agreement rates and :math:`\Ira{\bullet}`, give inconsistent decompositions. Here the bivariate :math:`\Ido{\bullet}` leaves the redundancy of the three sources, and the atoms that depend on it, unknown:

.. ipython::

   In [52]: pid = PID_Do(trivariates['anddup'])

   In [53]: pid.get_red(((0,), (1,), (2,))), pid.get_pi(((1,),)), pid.complete

Options of a measure are passed to its class, and an option the measure does not take is an error.


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

.. autoclass:: PID_WB
   :no-index:


:math:`\Immi{\bullet}`
----------------------

One potential measure of redundancy is the *minimum mutual information* :cite:`bertschinger2013shared`:

.. math::

   \Immi{X_{0:n} : Y} = \min_{i} \I{X_i : Y}

This measure, though crude, coincides with every decomposition satisfying BROJA's assumption ``*``, such as :math:`\Ibroja{\bullet}` and :math:`\Imin{\bullet}`, for jointly Gaussian sources and a univariate target :cite:`barrett2015exploration,olbrich2015information`.

It is also the largest redundancy of any nonnegative decomposition: monotonicity bounds the redundancy of every collection by its smallest mutual information, and :math:`\Immi{\bullet}` attains that bound at every node at once :cite:`vansanten2026between`, Theorem 1. Its synergy and its overlap of redundancy and synergy (see `Overlap and Bounds`_) are therefore the largest, and its unique informations the smallest, of any nonnegative decomposition: an upper bound, not a conservative estimate. Its atoms have closed forms: the synergy is :math:`\min_i \I{X_i : Y | X_{\text{rest}}}`, and the unique information of :math:`X_i` is :math:`\max\{0, \I{X_i : Y} - \I{X_{\text{rest}} : Y}\}`, where :math:`X_{\text{rest}}` are the other sources.

.. autoclass:: PID_MMI
   :no-index:


:math:`\Iwedge{\bullet}`
------------------------

Redundancy seems to intuitively be related to common information :ref:`Common Informations`. This intuition lead to the development of :math:`\Iwedge{\bullet}` :cite:`griffith2014intersection`:

.. math::

   \Iwedge{X_{0:n} : Y} = \I{ \meet X_i : Y}

That is, redundancy is the information the :ref:`Gács-Körner Common Information` of the inputs shares with the output.

.. warning::

   This measure can result in a negative PID.

.. autoclass:: PID_GK
   :no-index:


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

.. autoclass:: PID_Proj
   :no-index:


:math:`\Ibroja{\bullet}`
------------------------

In a very intuitive effort, Bertschinger et al (henceforth BROJA) :cite:`bertschinger2014quantifying,griffith2014quantifying` defined unique information as the minimum conditional mutual informations obtainable while holding the input-output marginals fixed:

.. math::

   \Delta = \{ Q : \forall i : p(x_i, y) = q(x_i, y) \}

   \Ibroja{X_{0:n} : Y} = \min_{Q \in \Delta} \I{X_i : Y | X_{\overline{\{i\}}}}

For bivariate sources (two inputs), ``PID_BROJA`` accepts a ``method`` keyword selecting how the marginal-matching optimization is solved:

* ``'scipy'`` — SLSQP on the free joint pmf parameters. It can stop well short of the minimum, overstating the unique information, so ``'auto'`` does not use it.
* ``'admui'`` — alternating divergence minimization :cite:`banerjee2017computing`. If it has not converged within ``maxiter`` iterations, its last iterate, which matches the marginals, is used.
* ``'cone'`` — exponential cone program solved with ECOS :cite:`makkeh2018broja` (requires ``pip install dit[broja]``).
* ``'auto'`` (the default) — picks ``cone`` if ECOS is installed and ``admui`` otherwise, falling back to the other methods if a solver fails.

Every result is checked against bounds that each BROJA solution satisfies. The distribution itself and :math:`q = p(y)\,p(x_0|y)\,p(x_1|y)` share its source-target marginals, so the larger of their co-informations bounds the redundancy from below :cite:`vansanten2026between`; a result below it is raised to it. A redundancy above :math:`\min_i \I{X_i : Y}`, or one that differs between the two sources, is treated as a failure of that solver, and the next one is tried.

.. note::

   In the bivariate case, Griffith independently suggested the same decomposition but from the viewpoint of synergy :cite:`griffith2014quantifying`.

The BROJA measure has recently been criticized for behaving in an unintuitive manner on some examples. Consider the *reduced or* distribution:

.. ipython::

   In [16]: bivariates['reduced or']

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the BROJA paper :cite:`bertschinger2014quantifying` the only example given where their decomposition differs from that of Harder et al. is the :py:func:`dit.example_dists.summed_dice`. We can find a simpler example where they differ using hypothesis:

.. ipython::
   :okexcept:

   In [17]: from hypothesis import find

   In [18]: from dit.utils.testing import distribution_structures

   In [19]: find(distribution_structures(3, 2, True), lambda d: PID_Proj(d) != PID_BROJA(d))

.. autoclass:: PID_BROJA
   :no-index:


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

.. autoclass:: PID_CCS
   :no-index:


:math:`\Idep{\bullet}`
----------------------

James et al :cite:`james2017unique` have developed a method of quantifying unique information based on the :ref:`Dependency Decomposition`. Unique information from variable :math:`X_i` is evaluated as the least change in sources-target mutual information when adding the constraint :math:`X_i Y`.

For two sources this four-edge minimum has the equivalent reference-family form
:cite:`vansanten2026between`, Appendix A:

.. math::

   R_{\mathrm{dep}} = \max\{C^{(2)}, I_{Q_{\mathrm{CI}}}(X_0;X_1)\},
   \qquad U_i=\I{X_i:Y}-R_{\mathrm{dep}}.

Here :math:`P^{(2)}` is the maximum-entropy joint matching all three pairwise
marginals, :math:`C^{(2)}` is its co-information, and
:math:`Q_{\mathrm{CI}}(x_0,x_1,y)=p(y)p(x_0|y)p(x_1|y)` makes the sources
conditionally independent given the target. The second term is the redundancy
of :class:`PID_MES` (equivalently :class:`PID_Do`), and the first is that of
:class:`PID_RA`, so for two sources :math:`\Idep{\bullet}` is the larger of the
redundancies of :math:`\Ira{\bullet}` and :math:`\Imes{\bullet}`. ``dit`` uses this
form to avoid constructing the dependency lattice: only one pairwise
maximum-entropy fit is required. Composite source groups and the target are
coalesced before fitting. With more than two sources, each source is paired with
the union of the other sources, and the same formula gives its unique
information.

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

.. autoclass:: PID_dep
   :no-index:


:math:`\Ipm{\bullet}`
---------------------

Also taking a pointwise view, Finn & Lizier's :math:`\Ipm{\bullet}` :cite:`finn2018pointwise` instead splits the pointwise mutual information into two components:

.. math::

   i(s, t) = h(s) - h(s|t)

They then define two partial information lattices, one quantified locally by :math:`h(s)` and the other by :math:`h(s|t)`. By averaging these local lattices and then recombining them, we arrive at a standard Williams & Beer redundancy lattice.

.. ipython::

   In [22]: PID_PM(bivariates['pnt. unq.'])
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

.. autoclass:: PID_PM
   :no-index:


:math:`\Isx{\bullet}`
---------------------
Shared Exclusions PID (:math:`I^{\mathrm{sx}}`) by Makkeh et al.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Makkeh et al. propose a pointwise redundancy measure built on the notion of shared exclusions in probability space (:cite:`makkeh2021introducing`). Intuitively, the measure quantifies how much probability mass for the target Y is *commonly excluded* by observing sets of sources simultaneously — that common excluded mass is taken as the redundant (shared) contribution.

.. math::

   \Isx{X_{0:1} : Y} \;=\; \sum_{x_0,x_1,y} p(x_0,x_1,y)\; \log_2 \frac{p(x_0 \cup x_1 | y)}{p(x_0 \cup x_1)}

For the general multivariate definition (which is implemented in dit), refer to :cite:`makkeh2021introducing`.
For large datasets with up to five source variables, refer to the reference implementation at https://github.com/Abzinger/SxPID

.. warning::

   This measure can result in a negative PID.

.. autoclass:: PID_SX
   :no-index:


:math:`\Irav{\bullet}`
----------------------

Taking a functional perspective as in :math:`\Iwedge{\bullet}`, :math:`\Irav{\bullet}` defines bivariate redundancy as the maximum coinformation between the two sources :math:`X_0, X_1`, a target :math:`Y`, and a deterministic function of the inputs :math:`f(X_0,X_1)`.

.. math::

   \Irav{X_{0:2} : Y} = \max_f\left(\I{X_0\!:\!X_1\!:\!Y\!:\!f(X_0,X_1)}\right)

This measure is designed to exploit the conflation of synergy and redundancy in the three variable coinformation: :math:`\I{X_0\!:\!X_1\!:\!Y} = R - S`.

The measure was contributed to ``dit`` by J. Emenheiser and has not been published separately :cite:`vansanten2026between`. For two sources its redundancy lies between :math:`\max\{0, \I{X_0 : X_1 : Y}\}` and that of :math:`\Ibroja{\bullet}`, so it is a nonnegative decomposition with no more overlap than :math:`\Ibroja{\bullet}` (:cite:`vansanten2026between`, Proposition 10; see `Overlap and Bounds`_).

.. ipython::

   In [23]: PID_RAV(bivariates['pnt. unq.'])
   Out[23]:
   ╔════════╤════════╤════════╗
   ║ I_RAV  │  I_r   │   pi   ║
   ╟────────┼────────┼────────╢
   ║ {0:1}  │ 1.0000 │ 0.0000 ║
   ║  {0}   │ 0.5000 │ 0.5000 ║
   ║  {1}   │ 0.5000 │ 0.5000 ║
   ║ {0}{1} │ 0.0000 │ 0.0000 ║
   ╚════════╧════════╧════════╝

.. autoclass:: PID_RAV
   :no-index:



:math:`\Irr{\bullet}`
---------------------

In order to combine :math:`\Immi{\bullet}` with the coinformation, Goodwell and Kumar :cite:`goodwell2017temporal` have introduced their *rescaled redundancy*:

.. math::

   \Irr{X_0 : X_1} = R_{\text{min}} + I_{S} (\Immi{X_{0:2} : Y} - R_{\text{min}})

   R_{\text{min}} = \max\{ 0, \I{X_0 : X_1 : Y} \}

   I_{S} = \frac{\I{X_0 : X_1}}{\min\{ \H{X_0}, \H{X_1} \}}

:math:`R_{\text{min}}` and :math:`\Immi{\bullet}` are the two ends of the segment of nonnegative bivariate decompositions, so :math:`I_S` is the position of :math:`\Irr{\bullet}` on it (see `Overlap and Bounds`_) :cite:`vansanten2026between`: it moves from the zero-overlap end for independent sources to :math:`\Immi{\bullet}` when one source is a function of the other.

.. ipython::

   In [24]: PID_RR(bivariates['pnt. unq.'])
   Out[24]:
   ╔════════╤════════╤════════╗
   ║ I_rr   │  I_r   │   pi   ║
   ╟────────┼────────┼────────╢
   ║ {0:1}  │ 1.0000 │ 0.3333 ║
   ║  {0}   │ 0.5000 │ 0.1667 ║
   ║  {1}   │ 0.5000 │ 0.1667 ║
   ║ {0}{1} │ 0.3333 │ 0.3333 ║
   ╚════════╧════════╧════════╝

.. autoclass:: PID_RR
   :no-index:


:math:`\Ihf{\bullet}`
---------------------

Hi-Fi :cite:`ontiveroortega2025assessing` decomposes the dependence of a target on a driver :math:`X_i` by conditioning on subsets :math:`W` of the other variables. In its information-theoretic form :cite:`lazic2026information`, the unique part is :math:`\min_W \I{X_i : Y | W}`, which is the optimally conditioned transfer entropy of :cite:`korenek2025higher`. The redundant part is the drop from :math:`\I{X_i : Y}` to that minimum, and the synergistic part is the largest rise above it. For two sources this is a decomposition, the zero-overlap end of the segment described in `Overlap and Bounds`_ :cite:`vansanten2026between`:

.. math::

   \Ihf{X_0 : X_1 : Y} = \max\{0, \I{X_0 : X_1 : Y}\}, \qquad
   S = \max\{0, -\I{X_0 : X_1 : Y}\}, \qquad
   U_i = \min\{\I{X_i : Y}, \I{X_i : Y | X_j}\}.

It has the smallest redundancy and synergy, and the largest unique informations, of any nonnegative decomposition; :cite:`goodwell2017temporal` gave these bounds as :math:`R_{\text{min}}`. It is symmetric, satisfies self-redundancy, monotonicity, and the identity property, and vanishes for independent sources. Its redundancy and synergy do not depend on which variable is the target. It depends on more than the source-target marginals and lacks the Blackwell property. On ``bivariates['rdn xor']`` it reports neither redundancy nor synergy: one bit of each cancels in the co-information.

.. ipython::

   In [25]: print(PID_HiFi(bivariates['and']))

.. warning::

   With more than two sources Hi-Fi is not a decomposition. Like other two-source measures, :class:`PID_HiFi` computes the redundancy of every pair of disjoint groups of sources, so every such pair has zero overlap, and fills in the lattice from those values. Zero overlap on all pairs at once is often impossible :cite:`vansanten2026between`. For :math:`X_0 = (a, b)`, :math:`X_1 = a`, :math:`X_2 = c` and :math:`Y = (a, b \oplus c)`, every nonnegative decomposition has a bit of overlap between :math:`X_0` and :math:`(X_1, X_2)`, so :class:`PID_HiFi` has a negative atom. For more sources use :func:`hifi`, which gives Hi-Fi's bounds for each source.

.. autoclass:: PID_HiFi
   :no-index:


:math:`\Ira{\bullet}`
---------------------

Drawing from the reconstructability analysis work of Zwick :cite:`zwick2004overview`, we can define :math:`\Ira{\bullet}` as a restricted form of :math:`\Idep{\bullet}`. It keeps only the edge of the dependency lattice that adds the source-target constraint to the other two pairwise constraints. For two sources its redundancy is therefore the co-information :math:`C^{(2)}` of the maximum-entropy distribution with the three pairwise marginals, and its synergy is the third-order connected information :math:`I^{(3)}` :cite:`vansanten2026between` (see `Pairwise-reference PID`_).

.. warning::

   This measure can result in a negative PID.

.. autoclass:: PID_RA
   :no-index:


Pairwise-reference PID
----------------------

:class:`PID_PR` (formerly ``PID_dep_c``) is the pairwise-reference PID of
:cite:`vansanten2026between`, Definition 2, for **two sources**. Let
:math:`P^{(2)}` be the maximum-entropy joint with the three pairwise marginals
of :math:`P`, and write

.. math::

   C^{(2)}=\operatorname{coI}(P^{(2)}), \qquad
   I^{(3)}=H(P^{(2)})-H(P)=D_{\mathrm{KL}}(P\|P^{(2)}).

All terms except the joint entropy agree in the co-information expansion, so
:math:`C=C^{(2)}-I^{(3)}`. The pairwise-reference atoms are

.. math::

   R=[C^{(2)}]_+, \qquad U_i=\I{X_i:Y}-R, \qquad
   S=I^{(3)}+[-C^{(2)}]_+,

where :math:`[x]_+=\max(0,x)`. They form a nonnegative bivariate PID. In
contrast, :class:`PID_RA` uses the unclipped :math:`R=C^{(2)}` and
:math:`S=I^{(3)}`, and can have negative redundancy.

This separates pairwise-generated net synergy from the third-order connected
information without erasing coexisting redundancy and synergy. On AND,
:math:`I^{(3)}=0` and :class:`PID_PR` has zero redundancy and about 0.189 bits
of synergy. On ``bivariates['rdn xor']``, :math:`C=0` but
:math:`C^{(2)}=I^{(3)}=1`: it retains one bit each of redundancy and synergy,
whereas the zero-overlap endpoint reports neither.

For two sources it is symmetric, satisfies self-redundancy and the identity
property, and its redundancy and synergy do not depend on which variable is
the target. Its redundancy depends on more than the source-target marginals
:cite:`vansanten2026between`, Proposition 6. These properties are recorded in
:meth:`~dit.pid.pid.BasePID.claims`, marked where they are shown only for two
sources, together with those the paper leaves open. Whenever one source is a
function of the other, or of the target, or is a copy of the target, the pairwise
maximum-entropy distribution is the distribution itself, so subset equality,
target equality and strong monotonicity hold; target monotonicity, the target
chain rule, the lower bound, the Blackwell property and additivity fail.

.. warning::

   The equivalence and nonnegativity guarantee concern two source groups,
   each of which may itself contain several variables. The full
   dependency-lattice extension of :class:`PID_PR` to more sources remains
   experimental and can be inconsistent. Connected information characterizes
   an observed distribution, not an identified causal mechanism.

.. autoclass:: PID_PR
   :no-index:


:math:`\Irdr{\bullet}`
----------------------

The measure of Mages & Rohner :cite:`mages2023measure` can be interpreted as a pointwise version of :math:`\Ibroja{\bullet}` to provide a non-negative partial information decomposition for an arbitrary number of sources.
It obtains its operational interpretation from valuating the reachable decision regions (achievable type I/II error pairs) for each state of the target variable.
For each target outcome :math:`y`, each source defines a binary-input channel for the event :math:`Y = y`, and its reachable decision region is bounded by its ROC curve.
The union information of a collection of sources is the pointwise information of the convex hull of their regions (their join), averaged over :math:`y`, and redundancies follow from the union information of sub-collections by inclusion-exclusion :cite:`mages2023measure,mages2024nonnegative`.
``dit`` computes each join as the upper concave envelope of the ROC curves, and computes the union information of each collection once for the whole lattice.

.. autoclass:: PID_RDR
   :no-index:


:math:`\Ict{\bullet}`
---------------------

Sigtermans :cite:`sigtermans2020path` defines redundancy through causal tensors. The tensor of a path through several variables is the product of the transition tensors :math:`p(b \mid a)` of its links, and its path-based mutual information is that of its first variable and the target along the resulting Markov chain. The redundancy of a collection of sources is the least path-based mutual information over the paths that pass through every source and end at the target, leaving out those that contain a link which does not exist (Definition 2): a link does not exist if an indirect path reproduces its tensor (Proposition 1). ``dit`` leaves out the paths whose link to the target an indirect path through the other sources reproduces, unless every path is left out, and computes every node of the lattice this way, for any number of sources. For two sources this is the least of the paths :math:`\{X_0\}\{X_1\}\{Y\}` and :math:`\{X_1\}\{X_0\}\{Y\}` whose last link exists.

.. autoclass:: PID_CT
   :no-index:


:math:`\Ideg{\bullet}`
----------------------

The degradation intersection information of Kolchinsky :cite:`kolchinsky2022novel` (further developed by Gomes & Figueiredo :cite:`gomes2023orders`) defines redundancy as the maximum mutual information carried by a channel that is a Blackwell degradation of every source channel:

.. math::

   \Ideg{X_{0:n} \to Y} = \max_{Q :\; Q \preceq_d X_i \;\forall i} \I{Q : Y}

This is the Blackwell redundancy :math:`\Iprec{\bullet}` in the notation of Gomes & Figueiredo, for any number of sources.
The feasible degradation channels form a polytope and :math:`\I{Q : Y}` is convex, but the maximum can be found without enumerating the vertices of that polytope.
Each outcome :math:`q` of :math:`Q` corresponds to the vector of its degradation probabilities :math:`(p(q \mid x_0), \ldots, p(q \mid x_{n-1}))`, which lies in a polyhedral cone.
These vectors add up to the all-ones vector, and :math:`\I{Q : Y}` is a sum over the outcomes of a convex, positively homogeneous function of them.
Splitting an outcome along the extreme rays of the cone therefore never decreases :math:`\I{Q : Y}`, so the redundancy is the value of a linear program over the extreme rays, whose solution is also an optimal :math:`Q`.
This formulation is due to André F. C. Gomes.
``dit`` enumerates the rays with cddlib when it is installed (from the ``optional`` extra), and otherwise with an exact implementation of the double description method.
Passing ``method='local'`` instead runs a local search over the vertices of the polytope, which only gives a lower bound.

.. autoclass:: PID_Deg
   :no-index:


:math:`\Ici{\bullet}`
---------------------

Gomes & Figueiredo :cite:`gomes2024measure` define a *union* information rather than a redundancy. Each source :math:`A_i` contributes its channel :math:`p(a_i \mid y)`; by sharing channels (and marginals thereof) the sources can build any distribution :math:`q(a, y) = p(y) \prod_{B \in \pi} p(a_B \mid y)`, where :math:`\pi` partitions the variables of :math:`A = \bigcup_i A_i` into blocks that each lie within some source. The union information is the best such :math:`q` can do, capped at the true mutual information:

.. math::

   \Ici{A_0, \ldots, A_{m-1} \to Y} = \min\left\{ \I{A : Y},\; \max_{q} \I{A : Y}_q \right\}

Synergy is the information lost by assuming the sources are conditionally independent given the target, :math:`S^{CI} = \I{A : Y} - \Ici{A_0, \ldots, A_{m-1} \to Y} \geq 0`. Sources contained in another source, or a deterministic function of another source, are removed first. Each source is tested against the others one at a time: a source that only several others determine together, such as :math:`X_0 = (X_1, X_2)` in ``trivariates['xor shared']``, is kept, which keeps the union information monotone and independent of the order of the sources.

``PID_CI`` is built on ``BaseSynergyPID``, which turns a synergy (equivalently, union information) into redundancies by inclusion-exclusion over sub-collections of each antichain, as in the parthood formulation of Gutknecht et al. :cite:`gutknecht2021bits`:

.. math::

   I_\cap(\alpha) = \sum_{\emptyset \neq \beta \subseteq \alpha} (-1)^{|\beta| + 1} I_\cup(\beta)

.. ipython::

   In [1]: from dit.pid import PID_CI

   In [2]: PID_CI(bivariates['and'])

   In [3]: PID_CI(bivariates['and']).get_syn(((0,), (1,)))

.. autoclass:: PID_CI
   :no-index:


:math:`\Idelta{\bullet}`
------------------------

The :math:`\delta`-PID of Banerjee, Olbrich, Jost & Rauh :cite:`banerjee2018unique` quantifies unique information via the *weighted output KL deficiency*, the cost of approximating one source channel from the other via output randomization:

.. math::

   \delta(Y : X_0 \setminus X_1) = \inf_{P(X_0'|X_1)} \mathbb{E}_{Y}\!\left[ \DKL{P(X_0|Y) \;\|\; P(X_0'|X_1) \circ P(X_1|Y)} \right]

Redundancy is then min-symmetrized across the two sources.

.. autoclass:: PID_Delta
   :no-index:


:math:`\IdeltaLambda{\bullet}`
------------------------------

Venkatesh, Gurushankar & Schamberg :cite:`venkatesh2023capturing` introduce a Lagrangian generalization that smoothly interpolates between the :math:`\delta`-PID of :cite:`banerjee2018unique` and the BROJA PID of :cite:`bertschinger2014quantifying`:

.. math::

   \delta_{\lambda}(Y : X_0 \setminus X_1) = \inf_{P(X_0'|X_1 Y)} \mathbb{E}_{Y}\!\left[\DKL{P(X_0|Y) \;\|\; P(X_0'|Y)}\right] + \lambda \, \I{Y : X_0' | X_1}

As :math:`\lambda \to \infty` the measure recovers :math:`\Idelta{\bullet}`. As :math:`\lambda \to 0`, :math:`\delta_\lambda / \lambda` tends to the unique information of :math:`\Ibroja{\bullet}`; :math:`\delta_\lambda` itself vanishes, so the redundancy tends to that of :math:`\Immi{\bullet}`.

.. autoclass:: PID_DeltaLambda
   :no-index:


:math:`\Igh{\bullet}`
---------------------

Griffith & Ho :cite:`griffith2015quantifying` define redundancy, their :math:`I_\alpha`, as the most information about the target carried by a variable that is independent of the target given each source:

.. math::

   \Igh{X_{0:n} \to Y} = \max_{Q :\; Q - X_i - Y \;\forall i} \I{Q : Y}

where :math:`Q` may depend on the sources and the target jointly.
Each condition :math:`Q - X_i - Y`, that is :math:`p(q \mid x_i, y) = p(q \mid x_i)`, is linear in the channel :math:`p(q \mid x_{0:n}, y)` :cite:`kolchinsky2022novel`, so as for :math:`\Ideg{\bullet}` the redundancy is the value of a linear program over the extreme rays of a polyhedral cone.
The optimal :math:`Q` can need more values than the sources have; on 'erase' it needs four.
Passing ``method='optimizer'`` instead runs a local optimization of a bounded auxiliary variable, which only gives a lower bound.

Conditional independence implies the Blackwell order and is implied by a common function, so :math:`\Iwedge{\bullet} \leq \Igh{\bullet} \leq \Iprec{\bullet}`.
Unlike the Blackwell order it is not transitive: :math:`Q - X_1 - Y` does not imply :math:`Q - (X_1, X_2) - Y`, so adding a source that contains another can lower the redundancy, and with three sources the redundancies need not be monotone on the lattice.

.. autoclass:: PID_GH
   :no-index:


:math:`\Iig{\bullet}`
---------------------

Niu & Quinn :cite:`niu2019measure` propose an information-geometric PID: the synergy is defined as the KL divergence from the true joint distribution to the nearest distribution satisfying :math:`X_0 - X_1 - Y` or :math:`X_1 - X_0 - Y` Markov chains, minimized along the geodesic :math:`q_\lambda \propto p(x_0, x_1)\, p(y \mid x_0)^\lambda p(y \mid x_1)^{1 - \lambda}` between them; the minimizing :math:`\lambda` need not lie in :math:`[0, 1]`.

.. autoclass:: PID_IG
   :no-index:


:math:`\Iipid{\bullet}`
-----------------------

The I-PID of Venkatesh, Gurushankar & Schamberg :cite:`venkatesh2023capturing` defines unique information via an *information deficiency* that maximizes the mutual-information gap attainable by a Markov test channel :math:`T - Y - (X_0, X_1)`:

.. math::

   \delta_I(Y : X_0 \setminus X_1) = \sup_{P(T|Y)} \left[ \I{T : X_0} - \I{T : X_1} \right]

Redundancy is obtained by min-symmetrization, analogously to :math:`\Idelta{\bullet}`.

.. autoclass:: PID_IPID
   :no-index:


:math:`\Imc{\bullet}`
---------------------

The more-capable intersection information of Gomes & Figueiredo :cite:`gomes2023orders` (building on :cite:`kolchinsky2022novel`) replaces the Blackwell-degradation order in :math:`\Ideg{\bullet}` with the weaker *more-capable* channel preorder:

.. math::

   \Imc{X_{0:n} \to Y} = \max_{Q :\; Q \preceq_{\mathrm{mc}} X_i \;\forall i} \I{Q : Y}

where :math:`Q \preceq_{\mathrm{mc}} X_i` iff :math:`\I{Q : Y} \leq \I{X_i : Y}` for every input distribution on :math:`Y`.

.. autoclass:: PID_MC
   :no-index:


:math:`\Iln{\bullet}`
---------------------

The less-noisy intersection information of Gomes & Figueiredo :cite:`gomes2023orders` uses the *less noisy* channel preorder, which sits between the Blackwell-degradation and more-capable orders, so :math:`\Ideg{\bullet} \leq \Iln{\bullet} \leq \Imc{\bullet}`:

.. math::

   \Iln{X_{0:n} \to Y} = \max_{Q :\; Q \preceq_{\mathrm{ln}} X_i \;\forall i} \I{Q : Y}

where :math:`Q \preceq_{\mathrm{ln}} X_i` iff :math:`\I{U : Q} \leq \I{U : X_i}` for every :math:`U` with :math:`U - Y - (Q, X_i)`.
Equivalently, the one-way secret key agreement rate from :math:`Y` to :math:`Q` with :math:`X_i` eavesdropping vanishes, so :math:`\Iln{\bullet}` is the most information about the target that cannot support secret key agreement against any single source.

By :cite:`vandijk1997special`, :math:`Q \preceq_{\mathrm{ln}} X_i` iff :math:`\I{Y : X_i} - \I{Y : Q}` is concave in the distribution of :math:`Y`.
``dit`` imposes the equivalent pointwise condition on the Hessians of the output entropies at sampled distributions and directions, so the computed value is approximate.
No bound on :math:`|Q|` is known; it defaults to the largest source alphabet.

.. warning::

   This measure appears to result in a negative PID. For :math:`p_Y = (0.3, 0.55, 0.15)` with :math:`p(x_0 | y)` rows :math:`(0.2, 0.5, 0.3), (0, 0, 1), (0.1, 0.9, 0)` and :math:`p(x_1 | y)` rows :math:`(0.1, 0, 0.9), (0.05, 0.5, 0.45), (0.05, 0.5, 0.45)`, :math:`\Iln{\bullet} \approx 0.021` bits, below the coinformation :math:`0.6925 + 0.2341 - 0.8836 = 0.043` bits, so the synergy is about :math:`-0.022` bits. More generally, since :math:`\Iln{\bullet}` depends only on the source-target marginals, it gives a nonnegative synergy on every distribution with those marginals only if it is at least :math:`\Ibroja{\bullet}` :cite:`bertschinger2014quantifying` (Lemma 3), which here is :math:`0.050` bits.

.. autoclass:: PID_LN
   :no-index:


:math:`\Imes{\bullet}`
----------------------

:math:`\Imes{\bullet}` is a maximum-entropy PID inspired by BROJA's "``*``" marginal-consistency assumption: the redundancy is the coinformation of the sources and the target under the maximum entropy distribution with the same source-target marginals,

.. math::

   q(x_0, \ldots, x_{n-1}, y) = p(y) \prod_i p(x_i \mid y).

The sources are independent given the target under :math:`q`, so for two sources the redundancy is :math:`I_q(X_0 : X_1)`, the same as that of :math:`\Ido{\bullet}`. James, Emenheiser & Crutchfield :cite:`james2019unique` discuss this distribution as the maximum entropy reading of assumption ``*``, and Liardi et al. :cite:`liardi2026landscape` attribute the measure to them; it was not proposed as a complete decomposition, and it does not satisfy subset equality.

.. warning::

   This measure can result in a negative PID. The distribution :math:`p` itself is not among the reference distributions, so the synergy, :math:`I_q(X_0 : X_1)` minus the co-information of :math:`p`, is negative whenever the latter is larger :cite:`vansanten2026between`. For example, two identical sources and a target copying them through a binary symmetric channel with crossover 0.1 give redundancy 0.320 bits and synergy -0.211 bits.

.. autoclass:: PID_MES
   :no-index:


:math:`\Iprec{\bullet}`
-----------------------

Kolchinsky :cite:`kolchinsky2022novel` defines a redundancy measure using the Blackwell order on channels:

.. math::

   \Iprec{X_{0:n} \to Y} = \max_{s_{Q|Y}} \I{Y : Q} \quad \text{s.t.} \quad s_{Q|Y} \preceq p_{X_i|Y} \;\;\forall i

This is the same quantity as :math:`\Ideg{\bullet}`, for any number of sources, and by default it is computed by the same linear program over the extreme rays of the cone of common degradations.
Passing ``n_q`` fixes the cardinality of :math:`Q`.
Every outcome of :math:`Q` is a combination of extreme rays of that cone, and the objective is convex, so the maximum is attained at a vertex of the linear program's feasible polytope with the rays in its support grouped into the outcomes of :math:`Q`.
Splitting an outcome never lowers the objective, so only the partitions of each support into ``n_q`` groups need be checked, and from the cardinality of Theorem A1 on, the value is that of the linear program.
The vertices are enumerated in exact arithmetic.

.. autoclass:: PID_Prec
   :no-index:


:math:`\Ido{\bullet}`
---------------------

Lyu, Clark & Raviv :cite:`lyu2024explicit` give an explicit bivariate redundancy via Pearl-style do-calculus. An intervened source :math:`X'_i` is defined by

.. math::

   p(X'_i, X_j, Y) = p(X_j)\, p(Y \mid X_j)\, p(X_i \mid Y)

and the redundancy is :math:`I(X'_1 : X_2) = I(X'_2 : X_1)`. Only defined for two sources.

.. warning::

   This measure can result in a negative PID. For two sources its redundancy is that of :math:`\Imes{\bullet}`, so the example given there has negative synergy here as well :cite:`vansanten2026between`.

.. ipython::

   In [26]: from dit.pid import PID_Do

   In [27]: print(PID_Do(bivariates['synergy']))

.. autoclass:: PID_Do
   :no-index:


Synergistic Disclosure
----------------------

Rosas, Mediano, Rassouli & Barrett :cite:`rosas2020operational` decompose :math:`I(X;Y)` via :math:`\alpha`-synergistic channels (see also :doc:`/measures/multivariate/synergistic_disclosure`). :class:`~dit.pid.SynDisc` inverts on the extended constraint lattice; :class:`~dit.pid.ModifiedSynDisc` is the backbone coarse-graining.

.. autoclass:: ModifiedSynDisc
   :no-index:

.. autoclass:: SynDisc
   :no-index:


Secret Key Agreement Rates
--------------------------

One can associate :ref:`Secret Key Agreement` rates with unique informations :cite:`banerjee2015synergy` by considering the rate at which one source and the target can agree upon a secret key while the other source eavesdrops. The four classes are :class:`~dit.pid.PID_SKAR_nw` (no communication), :class:`~dit.pid.PID_SKAR_owa` (source to target), :class:`~dit.pid.PID_SKAR_owb` (target to source), and :class:`~dit.pid.PID_SKAR_tw` (two-way). This results in four possibilities:

- neither source nor target communicate
- only the source communicates
- only the target communicates
- both the source and the target communicate

No Communication
^^^^^^^^^^^^^^^^

.. math::

   \Ipart{X_i \rightarrow Y \setminus X_j} = \operatorname{S}[X_i : Y || X_j]

.. warning::

   This measure can result in an inconsistent PID.


One-Way Communication
^^^^^^^^^^^^^^^^^^^^^

Camel
~~~~~

.. math::

   \Ipart{X_i \rightarrow Y \setminus X_j} = \operatorname{S}[X_i \rightarrow Y || X_j]

Elephant
~~~~~~~~

.. math::

   \Ipart{X_i \rightarrow Y \setminus X_j} = \operatorname{S}[X_i \leftarrow Y || X_j]

.. warning::

   This measure can result in an inconsistent PID.


Two-Way Communication
^^^^^^^^^^^^^^^^^^^^^

.. math::

   \Ipart{X_i \rightarrow Y \setminus X_j} = \operatorname{S}[X_i \leftrightarrow Y || X_j]

.. warning::

   This measure can result in an inconsistent PID.

.. autoclass:: PID_SKAR_nw
   :no-index:

.. autoclass:: PID_SKAR_owa
   :no-index:

.. autoclass:: PID_SKAR_owb
   :no-index:

.. autoclass:: PID_SKAR_tw
   :no-index:


Experimental Measures
---------------------

Three variants of :math:`\Idep{\bullet}` take the unique information of a source from the dependency lattice of all the sources and the target, rather than of the source, the other sources together, and the target. With two sources :class:`PID_dep_a` is :math:`\Idep{\bullet}`, and :class:`PID_dep_b` only considers the edges that never constrain the sources among themselves, and gives inconsistent decompositions. The third only considers the edges that keep the joint distribution of the sources; it is the extension of :class:`PID_PR` to more sources, whose two-source case is the `Pairwise-reference PID`_ described above. These full dependency-lattice extensions remain experimental, and their lattices grow quickly with the number of sources.

.. autoclass:: PID_dep_a
   :no-index:

.. autoclass:: PID_dep_b
   :no-index:


Conditional and Contextual Decompositions
=========================================

When some variables are already known, the information the sources provide about the target beyond them can be decomposed in two ways :cite:`gutknecht2026partial`. The *conditional* decomposition decomposes the sources under each conditional distribution given the known variables, and averages:

.. math::

   \Ipart{\alpha : Y | Z} = \sum_z p(z) \, \Ipart{\alpha : Y}_{p(\cdot | Z = z)}

When the known variables are some of the sources, the *context*, the *contextual* decomposition instead regroups the atoms of the decomposition of all the sources: the atoms the context provides by itself are dropped, and the others are grouped by the remaining sources that give access to them once the context is known. Its redundancies are

.. math::

   \Icap{\mathcal{A}_1, \ldots, \mathcal{A}_k : Y \setminus C} = \Icap{\mathcal{A}_1 C, \ldots, \mathcal{A}_k C : Y} - \I{C : Y}

Both decompose :math:`\I{\mathcal{A} : Y | C}`, and they agree atom by atom exactly when the measure satisfies the source chain rule. On the distribution where a third source decides which of the first two copies the target, :math:`\Imin{\bullet}` gives half a bit of contextual redundancy and no conditional redundancy, while :math:`\Isx{\bullet}`, which satisfies the rule, gives the same decomposition both ways:

.. ipython::

   In [54]: from dit.pid import ConditionalPID, ContextualPID

   In [55]: d = trivariates['max min']

   In [56]: print(ContextualPID(PID_WB(d), context=[2]))

   In [57]: print(ConditionalPID(PID_WB, d, [[0], [1]], [3], crvs=[2]))

   @doctest
   In [58]: ContextualPID(PID_SX(d), context=[2]) == ConditionalPID(PID_SX, d, [[0], [1]], [3], crvs=[2])
   Out[58]: True

``crvs`` need not be sources: conditioning on any other variables gives the decomposition of a conditional mutual information, such as a transfer entropy.

.. autoclass:: ConditionalPID
   :no-index:

.. autoclass:: ContextualPID
   :no-index:


Partial Entropy Decomposition
=============================

Ince :cite:`ince2017partial` proposed applying the PID framework to decompose multivariate entropy (without considering information about a separate target variable). This *partial entropy decomposition* (PED), seeks to partition a mutlivariate entropy :math:`\H{X_0,X_1,\ldots}` among the antichains of the variables. The PED perspective shows that bivariate mutual information is equal to the difference between redundant entropy and synergistic entropy.

.. math::

   \I{X_0 : X_1} = \Hpart{\left\{X_0\right\}, \left\{X_1\right\}} - \Hpart{\left\{X_0,X_1\right\}}


:math:`\Hcs{\bullet}`
---------------------

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

.. autoclass:: PED_CS
   :no-index:


:math:`H_{\mathrm{mos}}`
------------------------

Finn & Lizier :cite:`finn2020generalised` define a partial entropy
decomposition whose redundancy functional is the expected minimum of the
source-group surprisals:

.. math::

   H_{\mathrm{mos}}(\alpha) = \mathbb{E}\bigl[\min_i h(A_i)\bigr]

.. ipython::

   In [28]: from dit.pid import PED_MOS

   In [29]: print(PED_MOS(dit.Distribution(['00','01','10','11'], [0.25]*4)))

This is the decomposition that later work, such as :cite:`varley2023partial`, calls :math:`h_{\min}`.

.. autoclass:: PED_MOS
   :no-index:


:math:`H_{\mathrm{sx}}`
-----------------------

Varley et al. :cite:`varley2023partial` define the redundant entropy of an antichain :math:`\alpha = \{A_1, \ldots, A_k\}` through shared probability mass exclusions: the surprisal of the event that *at least one* source group takes its observed value,

.. math::

   H_{\mathrm{sx}}(\alpha) = \mathbb{E}\left[-\log P\left(\bigcup_i \{A_i = a_i\}\right)\right]

This is the entropy analogue of :math:`\Isx{\bullet}`. Like :class:`PED_MOS`, it accepts ``pointwise=True`` to keep the per-outcome atoms.

.. ipython::

   In [30]: from dit.pid import PED_SX

   In [31]: print(PED_SX(dit.Distribution(['00','01','10','11'], [0.25]*4)))
   +--------------------------+
   |           H_sx           |
   +--------+--------+--------+
   |  H_sx  |  H_r   |  H_d   |
   +--------+--------+--------+
   | {0:1}  | 2.0000 | 0.4150 |
   |  {0}   | 1.0000 | 0.5850 |
   |  {1}   | 1.0000 | 0.5850 |
   | {0}{1} | 0.4150 | 0.4150 |
   +--------+--------+--------+

Unlike :math:`H_{\mathrm{cs}}`, which gives these independent bits no redundant entropy, :math:`H_{\mathrm{sx}}` counts the :math:`\log_2 4/3` bits it takes to exclude the outcome that matches neither observed value.

.. autoclass:: PED_SX
   :no-index:



Generalized Information Decomposition
=====================================

The local Kullback-Leibler divergence :math:`\log q(x)/p(x)` of a posterior :math:`q` from a prior :math:`p` is the difference :math:`h_p(x) - h_q(x)` of two local entropies. Varley :cite:`varley2024generalized` therefore decomposes :math:`\DKL{q || p}` over the redundancy lattice as the posterior expectation of the difference between the prior's and the posterior's pointwise partial entropy atoms. With the product of marginals as prior this decomposes the total correlation; with the uniform distribution, the negentropy. :class:`GID` takes ``measure='sx'`` (:class:`PED_SX`) or ``measure='mos'`` (:class:`PED_MOS`).

.. ipython::

   In [32]: from dit.pid import GID

   In [33]: from dit.distconst import product_distribution

   In [34]: d = dit.example_dists.Xor()

   In [35]: gid = GID(d, product_distribution(d))

   @doctest float
   In [36]: sum(gid[node] for node in gid._lattice)
   Out[36]: 1.0

.. autoclass:: GID
   :no-index:
   :members: get_pi, get_red, get_pw_pi, to_string



Integrated Information Decomposition
====================================

The integrated information decomposition (:math:`\Phi\mathrm{ID}`) :cite:`mediano2019beyond,mediano2025toward` decomposes the mutual information between a set of sources (typically a system's past) and a set of targets (its future) over the *product* of the two redundancy lattices. An atom :math:`\alpha \to \beta` is information carried by the sources in the mode :math:`\alpha` and by the targets in the mode :math:`\beta`. For two sources and two targets there are 16 atoms, including persistent redundancy :math:`\{0\}\{1\} \to \{2\}\{3\}`, information transfer :math:`\{0\} \to \{3\}`, and downward causation :math:`\{0:1\} \to \{2\}`.

The atoms are the Mobius inversion of a *double redundancy* :math:`I_\partial(\alpha \to \beta)`. :class:`PhiID` provides four:

* ``measure='mmi'``: :math:`\min_{i,j} \I{A_i : B_j}` :cite:`mediano2019beyond`.
* ``measure='pm'``: the expectation of :math:`\min_i h(a_i) + \min_j h(b_j) - \min_{i,j} h(a_i, b_j)`, after :cite:`finn2020generalised`.
* ``measure='sx'``: the expectation of :math:`h_{\mathrm{sx}}(\alpha) + h_{\mathrm{sx}}(\beta) - h_{\mathrm{sx}}(\{A_i \cup B_j\})` :cite:`varley2023decomposing`.
* ``measure='gk'``: :math:`\I{\wedge_i A_i : \wedge_j B_j}`, the mutual information between the Gács–Körner common parts of the two sides, extending :math:`\Iwedge{\bullet}` :cite:`griffith2014intersection`. Two independent bits copied forward give only the atoms :math:`\{0\} \to \{2\}` and :math:`\{1\} \to \{3\}`. But groups with full joint support share no common part, so on noisy data there is no redundancy between disjoint groups.

The atoms always sum to :math:`\I{X : Y}`, but individual atoms can be negative.

.. ipython::

   In [37]: from dit.pid import PhiID

   In [38]: d = dit.Distribution(['0000', '1111'], [1/2, 1/2])

   @doctest float
   In [39]: PhiID(d)[(((0,), (1,)), ((2,), (3,)))]
   Out[39]: 1.0

.. autoclass:: PhiID
   :no-index:
   :members: get_pi, get_red, get_pw_pi, get_pw_red, nodes, to_string


Representational Complexity
===========================

Ehrlich et al. :cite:`ehrlich2023measure` summarize a decomposition by the atom-weighted average degree of synergy: each atom is assigned the number of sources in its smallest group, so purely redundant (or unique) information has degree 1 and information available only from all :math:`n` sources jointly has degree :math:`n`.

.. ipython::

   In [40]: from dit.pid.helpers import representational_complexity

   @doctest float
   In [41]: representational_complexity(PED_SX(dit.example_dists.Xor()))
   Out[41]: 1.1225562489182657

.. autofunction:: dit.pid.helpers.representational_complexity

Comparing Measures
==================

The helpers in :mod:`dit.pid.helpers` print side-by-side tables.
:func:`~dit.pid.helpers.compare_measures` runs several PID measures on one
distribution (by default, every measure in ``dit.pid.measures.__all_pids``), and
:func:`~dit.pid.helpers.pointwise_pid_table` lists the per-outcome atoms of a
pointwise measure such as :math:`\Isx{\bullet}`:

.. ipython::

   In [42]: from dit.pid.helpers import compare_measures, pointwise_pid_table

   In [43]: compare_measures(bivariates['and'], pids=[PID_WB, PID_MMI, PID_CCS])

   In [44]: pointwise_pid_table(bivariates['and'], pid_class=PID_SX)

For two sources, ``overlap=True`` appends each decomposition's overlap of redundancy
and synergy and its position on the segment of nonnegative decompositions (see
`Overlap and Bounds`_):

.. ipython::

   In [45]: compare_measures(bivariates['and'], pids=[PID_WB, PID_dep, PID_CCS, PID_RA], overlap=True)

.. autofunction:: dit.pid.helpers.compare_measures

.. autofunction:: dit.pid.helpers.pointwise_pid_table


Overlap and Bounds
==================

For two sources the Shannon equations leave one degree of freedom. If
:math:`I_i=\I{X_i:Y}` and :math:`C=I_0+I_1-\I{X_0,X_1:Y}`, every consistent
PID, including a signed one, has the form :cite:`vansanten2026between`

.. math::

   R=[C]_+ + \delta, \qquad S=[-C]_+ + \delta, \qquad
   U_i=I_i-[C]_+-\delta.

The *overlap* :math:`\delta=\min(R,S)` is the common excess of redundancy and
synergy above their net balance. The four atoms are nonnegative exactly on

.. math::

   0\leq\delta\leq\delta_{\max}, \qquad
   \delta_{\max}=\min(I_0,I_1)-[C]_+.

The lower endpoint is the information-theoretic Hi-Fi decomposition, the upper
endpoint MMI; the corresponding sharp atom bounds were given by
:cite:`goodwell2017temporal`. :func:`pid_overlap` reports this segment and a
measure's position :math:`\lambda=\delta/\delta_{\max}` without performing any
additional maximum-entropy optimization:

.. ipython::

   In [70]: from dit.pid import pid_overlap

   In [71]: overlap = pid_overlap(PID_MMI(bivariates['and']))

   @doctest
   In [72]: round(overlap.delta, 6), round(overlap.position, 6)
   Out[72]: (0.311278, 1.0)

   @doctest
   In [73]: tuple(round(x, 6) for x in overlap.synergy_bounds)
   Out[73]: (0.188722, 0.5)

   @doctest
   In [74]: pid_overlap(PID_PM(bivariates['and'])).position > 1
   Out[74]: True

A signed PID may leave the segment: negative redundancy or synergy gives
:math:`\delta<0`, and negative unique information can give
:math:`\delta>\delta_{\max}`. These values are **not clipped**. If the segment
collapses within ``atol``, ``position`` is ``nan`` rather than an arbitrarily
chosen endpoint. All information values are plain numbers in the PID's
logarithmic units; ``position`` is dimensionless. ``unique_bounds`` follows the
supplied source order.

The nested reference families also give useful checks for these particular
bivariate measures:

.. math::

   R_{\mathrm{HF}}\leq R_{\mathrm{pairwise}}\leq R_{\mathrm{dep}}
   \leq R_{\mathrm{BROJA}}\leq R_{\mathrm{MMI}}.

This is not an ordering of every PID, and neither endpoint is universally
preferred. The bounds concern **nonnegative** PIDs, not constraints on signed
measures. With more sources, pairwise overlap choices need not be jointly
feasible, so this helper deliberately accepts only two-source PIDs. Conditional,
contextual, and entropy decompositions are not supported.

.. autofunction:: pid_overlap
   :no-index:

.. autoclass:: PIDOverlap
   :no-index:

Hi-Fi for each source
---------------------

With any number of sources, :func:`hifi` decomposes the dependence of the target on each source by exhaustive search over the groups of other sources it is conditioned on :cite:`lazic2026information`. Its unique information :math:`U_i` bounds that of every nonnegative decomposition from above, and one nonnegative decomposition attains all of these bounds at once. Its redundancy :math:`R_i` and synergy :math:`S_i` bound, from below, the redundancy and synergy between the source and the group attaining them :cite:`vansanten2026between`, Theorem 1. The two are generally attained with different groups, so they are not parts of one decomposition. A greedy search over partners can miss a whole bit: no single partner reveals the parity of three bits.

.. ipython::

   In [75]: from dit import Distribution

   In [76]: from dit.pid import hifi

   In [77]: parity = Distribution(['0000', '0011', '0101', '0110', '1001', '1010', '1100', '1111'], [1/8]*8)

   @doctest
   In [78]: hifi(parity)[(0,)].synergy_partners
   Out[78]: ((1,), (2,))

.. autofunction:: hifi
   :no-index:

.. autoclass:: HiFi
   :no-index:

.. autoclass:: HiFiDriver
   :no-index:

Triplet report
--------------

For two sources :cite:`vansanten2026between` recommend reporting the co-information :math:`C`, its split :math:`C = C^{(2)} - I^{(3)}` into the co-information of the pairwise maximum-entropy model and the third-order connected information :cite:`Amari2001,Schneidman2003`, and :math:`\delta_{\max}`. These give the Hi-Fi and pairwise-reference decompositions and the range of every nonnegative one. :func:`triplet_report` computes them, together with the decomposition of :math:`\Idep{\bullet}`:

.. ipython::

   In [79]: from dit.pid import triplet_report

   In [80]: print(triplet_report(bivariates['rdn xor']))

.. autofunction:: triplet_report
   :no-index:

.. autoclass:: TripletReport
   :no-index:

Bounds on every nonnegative decomposition
-----------------------------------------

The Shannon equations fix the mutual information :math:`\I{X_A : Y}` of each group of sources. With two sources they leave one free parameter, the overlap above; with three they leave 11, with four 151 and with five 7548. Local positivity, :math:`\pi \geq 0`, turns this space into a polytope of nonnegative decompositions. :class:`PIDBounds` reports the range of every atom and every redundancy over it, without choosing a redundancy measure :cite:`vansanten2026between`. Each interval is a projection. Different intervals generally reach their ends at different decompositions, so not every combination of values within them is possible.

.. ipython::

   In [81]: from dit.pid import PIDBounds

   In [82]: print(PIDBounds(trivariates['anddup']))

The upper end of every redundancy is that of :math:`\Immi{\bullet}`, and the upper end of every unique information is Hi-Fi's :math:`U_i`. In each case one decomposition attains every bound at once. :meth:`PIDBounds.positions` locates a decomposition's atoms within their intervals. The ``fixed`` and ``zero_overlap`` arguments add constraints, such as the pairwise redundancies of a two-source measure or Hi-Fi's zero overlaps. Each interval takes up to two linear programs, so five sources, with 7579 atoms, are evaluated one node at a time. For more sources, see the layer cake below.

.. autoclass:: PIDBounds
   :no-index:

.. autoclass:: Interval
   :no-index:

Forced overlap
--------------

Merging two disjoint groups of sources gives a two-source decomposition with its own overlap :math:`\delta_{A,B}`. Hi-Fi takes every such overlap to be zero. The pairs share atoms, though, so with three or more sources this is often impossible :cite:`vansanten2026between`.

Take a copy and a partner: :math:`X_0 = (a, b)`, :math:`X_1 = a`, :math:`X_2 = c` and :math:`Y = (a, b \oplus c)`. Redundancy with the copy and synergy with the partner cancel against :math:`(X_1, X_2)`, so every nonnegative decomposition has a bit of overlap there. :func:`forced_overlap` finds the least total overlap by linear programming, and returns a minimal-overlap decomposition as a witness. :func:`zero_overlap_conditions` lists the conditions (M) and (S) that rule out zero overlap; for three sources they decide it exactly.

.. ipython::

   In [83]: from dit.pid import forced_overlap, zero_overlap_conditions

   @doctest
   In [84]: round(forced_overlap(trivariates['copy partner']).total, 6)
   Out[84]: 1.0

   In [85]: zero_overlap_conditions(trivariates['copy partner'])

On Rauh et al.'s distribution, ``trivariates['xor cat']``, condition (S) forces the overlaps of three pairs to add up to a bit. :func:`pid_overlaps` reports a decomposition's overlap on each pair.

.. autofunction:: forced_overlap
   :no-index:

.. autoclass:: ForcedOverlap
   :no-index:

.. autofunction:: zero_overlap_conditions
   :no-index:

.. autoclass:: OverlapCondition
   :no-index:

.. autofunction:: pid_overlaps
   :no-index:

Layer cake
----------

The redundancy :math:`\min_{B \in \beta} g(B)` of a nondecreasing set function :math:`g` has nonnegative atoms, all on one chain of nodes :cite:`vansanten2026between`, Lemma 2. With :math:`g(A) = \I{X_A : Y}` it is :math:`\Immi{\bullet}`, and :func:`mmi_atoms` returns its nonzero atoms. :func:`max_unique_atoms` gives every source Hi-Fi's largest unique information :math:`U_i`, then fills in the rest with the layer cake of :math:`\I{X_A : Y} - \sum_{i \in A} U_i`. Both need only the :math:`2^n - 1` mutual informations, not the lattice, so they handle more sources than :class:`PIDBounds`.

.. ipython::

   In [86]: from dit.pid import mmi_atoms

   In [87]: {node: round(atom, 4) for node, atom in mmi_atoms(trivariates['sum']).items()}

.. autofunction:: mmi_atoms
   :no-index:

.. autofunction:: max_unique_atoms
   :no-index:


Checking Properties
===================

Liardi et al. :cite:`liardi2026landscape` catalogue the properties proposed for redundancy measures, and which measures satisfy them. :meth:`~dit.pid.pid.BasePID.check` tests one property on a computed decomposition. Its result is truthy when the property holds, and lists the comparisons that fail. Here :math:`\Imin{\bullet}` violates the source chain rule on the distribution of Gutknecht :cite:`gutknecht2026partial`, in which a third source decides which of the first two copies the target, while :math:`\Isx{\bullet}` satisfies it:

.. ipython::

   In [45]: from dit.pid.distributions import trivariates

   In [46]: d = trivariates['max min']

   In [47]: print(PID_WB(d).check('SCR'))

   @doctest
   In [48]: bool(PID_SX(d).check('SCR'))
   Out[48]: True

:meth:`~dit.pid.pid.BasePID.check_properties` checks every property, next to the properties claimed for the measure; :meth:`~dit.pid.pid.BasePID.claims` returns just the claims, and :func:`~dit.pid.helpers.property_table` prints those of several measures:

.. ipython::

   In [49]: print(PID_WB(bivariates['and']).check_properties())

   In [50]: from dit.pid.helpers import property_table

   In [51]: property_table(pids=[PID_WB, PID_MMI, PID_GK, PID_SX], props=['LP', 'TC', 'SCR'])

Properties are named by the keys of :cite:`liardi2026landscape`, or by name (e.g. ``'source_chain_rule'``):

.. list-table::
   :header-rows: 1
   :widths: 8 22 70

   * - Key
     - Property
     - Check
   * - SR
     - self-redundancy
     - The redundancy of each single collection is its mutual information with the target.
   * - S0
     - weak symmetry
     - Recompute with the sources reordered (``orders``).
   * - M0
     - weak monotonicity
     - Redundancy does not decrease up the lattice, and dropping a collection that determines another changes nothing.
   * - GP
     - global positivity
     - Every redundancy is nonnegative.
   * - LP0, LP1, LP
     - local positivity
     - Every atom is nonnegative; LP0 concerns two sources, LP1 three or more.
   * - IID, ID
     - (independent) identity
     - The redundancy of two sources about a target copying them, on fixed copy distributions; a property of the measure, not of this distribution.
   * - TM
     - target monotonicity
     - Recompute with each target variable dropped.
   * - TC
     - target chain rule
     - Recompute with the first target variable, and with the rest given it (``split``).
   * - SE
     - subset equality
     - Dropping a collection that determines another changes nothing.
   * - LB
     - lower bound
     - Each redundancy is at least the information the Gács-Körner meet of its collections carries about the target.
   * - TE
     - target equality
     - Adding a copy of the target to the sources changes nothing.
   * - M1
     - strong monotonicity
     - M0 and TE, and dropping a collection that determines the target changes nothing.
   * - S1
     - strong symmetry
     - Exchanging the target with any source changes nothing.
   * - \*
     - assumption (\*)
     - Recompute under :math:`p(y) \prod_i p(x_i | y)`, which has the same source-target marginals.
   * - BP
     - Blackwell property
     - A unique information vanishes exactly when its source is Blackwell inferior to the other (two sources).
   * - AD
     - additivity
     - Recompute on two independent copies of the system, or on it and ``other``.
   * - CO, DI
     - continuity, differentiability
     - Not checkable on a single distribution.
   * - EI
     - equivalence-class invariance
     - Recompute with every alphabet relabeled.
   * - SCR
     - source chain rule
     - Recompute the decomposition of the other sources given each context (``context``).

Comparisons use the absolute tolerance ``atol`` (1e-5 by default); measures computed by numerical optimization may need a looser one. Checks that recompute the decomposition run the measure again, several times; additivity squares every alphabet, which is prohibitive for measures that search exhaustively or optimize large auxiliary variables, and slow for :math:`\Iprec{\bullet}` and :math:`\Ideg{\bullet}` with three or more sources, whose cones of common degradations then have very many extreme rays, so leave it out of ``props`` for them. Two-source measures do not define the redundancy of three or more collections, so comparisons that need one are undetermined, and their claims concern two sources only.

The claims transcribe Table 5 of :cite:`liardi2026landscape`, whose footnote marks they keep († two sources only, § full support only, ‡ the pointwise version holds, ∥ global, \* supported by simulations), and the source chain rule results of :cite:`gutknecht2026partial`. :math:`\Ideg{\bullet}` is the same measure as :math:`\Iprec{\bullet}`, so it takes that column. Entries marked ! correct Table 5: by its own appendix, :math:`I_{\prec}` satisfies (\*); identical sources show that :math:`\Iwedge{\bullet}` violates (\*); 'rdn xor' shows that :math:`\Irr{\bullet}` violates the lower bound; the correlated two-bit copy shows that :math:`I_{rdr}` satisfies only one direction of the Blackwell property; and 'max min' shows that :math:`\Igh{\bullet}` violates weak monotonicity, and so strong monotonicity, since its order :math:`Q - X - Y` is not transitive. Theorem 4.2 of Niu & Quinn, which Table 5 follows for :math:`I_{IG}`, does not hold: its full proof (:cite:`niu2021multivariate`, Theorem 3.4.3) stops at a stationary point of a functional that is unbounded below, and the redundancy is :math:`\log_2(1 + \sqrt{2}) - 4/3 = -0.062` on 'w', and -0.011 on a mixture of 'w' with a uniform distribution, which has full support; so :math:`I_{IG}` violates global and weak local positivity and the lower bound. The sign condition of :math:`\Iccs{\bullet}` drops outcomes where a source and a function of it disagree in sign, so it violates subset equality: for :math:`X_0 = a`, :math:`X_1 = (a, b)` and :math:`T = a \wedge b`, its redundancy is 0.458, while :math:`\I{X_0 : T} = 0.311`. Table 5's target equality for :math:`\Irav{\bullet}` concerns a single source and the target (its Proposition 37); with two sources it fails, as on 'uniform'. The monotonicity of :math:`\Ict{\bullet}` covers adding a source, not enlarging one: on 'max min', :math:`\Ict{\{0\}\{1\}} = 0.027` but :math:`\Ict{\{0\}\{1{:}2\}} = 0`. Nor does it satisfy target equality, since a copy of the target added as a source is a source like the others, through which every path must pass; on 'prob. 1', adding it changes the redundancy by 0.017. Both hold whichever paths are left out, so strong monotonicity fails too.

The properties the literature leaves open, for the measures it does not cover and the source chain rule for those it does, are determined here; such claims have no source. Each failure rests on a counterexample, which its caveat names and :func:`~dit.pid.properties.check_property` reproduces, and each success on a short argument, which its caveat gives. For example, the redundancy of :class:`PID_RA` is the co-information of the pairwise maximum-entropy distribution, which is the distribution itself when one variable is a function of another, so it satisfies subset equality, and it is additive, since the maximum-entropy distribution of two independent systems is the product of theirs; and since every channel is less capable than the target itself, :math:`\Imc{\bullet}` satisfies target equality.

.. autoclass:: dit.pid.properties.PropertyCheck
   :no-index:
   :members:

.. autofunction:: dit.pid.helpers.property_table
