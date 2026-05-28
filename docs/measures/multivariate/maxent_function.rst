.. maxent_function.rst
.. py:module:: dit.multivariate.common_informations.maxent_function

*************************
Maximum Entropy Function
*************************

The Maximum Entropy Function of Salamatian, Cohen, and Médard :cite:`salamatian2016maxent` is an *approximate* :ref:`gács-körner common information` that admits a small "helper" rate. It captures the largest entropy of a binary function :math:`\phi_X(X) \in \{-1, +1\}` of one source whose agreement with a binary function :math:`\phi_Y(Y) \in \{-1, +1\}` of the other source costs at most :math:`\epsilon` bits to repair:

.. math::

   M_\epsilon(X; Y) =
       \max_{\phi_X, \phi_Y}
       \H{\phi_X(X)}
       \quad \text{s.t.} \quad
       \H{\phi_X(X) \mid \phi_Y(Y)} \le \epsilon.

When the bipartite graph of :math:`p_{X,Y}` has disjoint components, the optimal pair is the connected-component indicator and :math:`M_0 = \K{X : Y}`. Allowing :math:`\epsilon > 0` lets the optimizer commit to a "near"-Gács-Körner partition that misses agreement on a small fraction of joint outcomes (an "almost" common variable that a helper of rate :math:`\epsilon` can patch up).


Algorithms
==========

``dit`` exposes both the exact optimization (Definition 3 in the paper) and the spectral approximation (Section IV-A) through a single function with a ``method`` argument.

* ``method="spectral"`` (default) — the paper's spectral algorithm. Compute :math:`Q = D_X^{-1/2}\, P\, D_Y^{-1/2}` where :math:`D_X, D_Y` are diagonal matrices of marginals. Subtract the trivial rank-one component :math:`\sqrt{p_X}\sqrt{p_Y}^T` (which always carries the unit singular value of :math:`Q`) and take the leading left/right singular vectors :math:`u, v` of the residual. For a real threshold :math:`t`, set :math:`\phi_X(i) = \mathrm{sign}(u_i - t)` and :math:`\phi_Y(j) = \mathrm{sign}(v_j - t)`. Sweep :math:`t` across every distinct cut of :math:`u` and return the largest feasible :math:`\H{\phi_X(X)}`.

* ``method="exact"`` — brute force over every binary partition of :math:`\mathcal{X}` and :math:`\mathcal{Y}`. Refuses to run when the alphabets would generate more than :math:`2^{20}` partition pairs.


Worked example
==============

The paper's two leading examples both live on a :math:`4 \times 4` joint with two disjoint blocks, optionally connected by a small "leak" edge.

The clean block-diagonal joint has Gács-Körner :math:`= 1` bit; the binary indicator :math:`\phi_X(i) = +1` iff :math:`i \in \{0, 1\}` (and the same for :math:`\phi_Y`) achieves :math:`\H{\phi_X(X)} = 1` and :math:`\H{\phi_X(X) \mid \phi_Y(Y)} = 0`.

.. ipython::

   In [1]: from dit import Distribution as D

   In [2]: from dit.multivariate import maxent_function

   In [3]: outcomes = ['00', '01', '10', '11', '22', '23', '32', '33']

   In [4]: pmf = [1/8] * 8

   In [5]: d = D(outcomes, pmf)

   @doctest float
   In [6]: maxent_function(d, epsilon=0.0, method='exact')
   Out[6]: 1.0

   @doctest float
   In [7]: maxent_function(d, epsilon=0.0, method='spectral')
   Out[7]: 1.0

Moving the :math:`(1, 1)` atom to :math:`(1, 2)` joins the two blocks; Gács-Körner drops to :math:`0` and :math:`M_0` follows it. But once the helper is allowed enough rate to repair the leaked outcome, the original block partition becomes feasible again and :math:`M_\epsilon` jumps back to :math:`1` bit.

.. ipython::

   In [8]: from dit.multivariate import gk_common_information

   In [9]: leak = D(['00', '01', '10', '12', '22', '23', '32', '33'], [1/8] * 8)

   @doctest float
   In [10]: gk_common_information(leak)
   Out[10]: 0.0

   @doctest float
   In [11]: maxent_function(leak, epsilon=0.0, method='exact')
   Out[11]: 0.0

   @doctest float
   In [12]: maxent_function(leak, epsilon=0.6, method='exact')
   Out[12]: 1.0


Threshold sweep
===============

The spectral threshold :math:`t` parameterises the binary partitions returned by the algorithm; :func:`plot_maxent_function` plots :math:`\H{\phi_X(X))}` at every distinct threshold position.

.. ipython::

   In [13]: from dit.multivariate import plot_maxent_function

   In [14]: ax = plot_maxent_function(leak)


API
===

.. autofunction:: maxent_function
.. autofunction:: plot_maxent_function
