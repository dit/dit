.. kamath_common_information.rst
.. py:module:: dit.multivariate.common_informations.kamath_common_information

*************************************
Kamath-Anantharam Common Information
*************************************

The Kamath-Anantharam common information :cite:`kamath2010dual` is a "dual" to the :ref:`gács-körner common information` constructed from the viewpoint of the Gray-Wyner source-coding system: it is the *infimum* of common-rate values :math:`R_0 \ge I(X;Y)` for which every rate triple :math:`(R_0, R_1, R_2)` allowed by the elementary outer bound to the Gray-Wyner region is achievable.

Kamath and Anantharam show this quantity has a strikingly simple closed form. Define, for each :math:`y` in the support of :math:`Y`, the conditional law :math:`\Phi^X_Y(y) = p(\cdot \mid Y=y)`, viewed as a random variable on :math:`Y`. Then

.. math::

   G(Y \to X) &= \H{\Phi^X_Y} \\
   G(X \to Y) &= \H{\Phi^Y_X} \\
   U(X; Y)    &= \max\{ G(Y \to X), G(X \to Y) \}

The asymmetric quantity :math:`\Phi^X_Y` is exactly the **minimal sufficient statistic** of :math:`Y` about :math:`X` (Kamath & Anantharam 2010, Lemma 3.5(5)) — two values of :math:`Y` collapse iff they induce the same conditional distribution over :math:`X`. So :math:`G(Y \to X)` is the entropy of the partition of the alphabet of :math:`Y` into "conditional-distribution-equivalent" classes.

For :math:`n > 2` random variables, ``dit`` generalizes :math:`U` analogously to how :doc:`mss_common_information` extends:

.. math::

   U(X_{0:n}) = \max_i \H{\Phi^{X_{\setminus i}}_{X_i}}.


Worked example
==============

The paper's reference example (Section III-A) is the joint distribution

.. math::

   p(x, y) =
   \frac{1}{37}
   \begin{pmatrix}
     4 &  0 & 0 & 0 \\
     0 &  9 & 2 & 3 \\
     0 & 12 & 3 & 4 \\
   \end{pmatrix}

with rows indexed by :math:`X \in \{a, b, c\}` and columns by :math:`Y \in \{\alpha, \beta, \gamma, \delta\}`. Conditioning on :math:`Y`, the values :math:`\beta` and :math:`\delta` induce the same conditional distribution :math:`p(X \mid Y) = (0, 3/7, 4/7)` and so collapse under :math:`\Phi^X_Y`; the other values are distinct. So :math:`\Phi^X_Y` takes three values with probabilities :math:`(4/37, 28/37, 5/37)`.

.. ipython::

   In [1]: from dit import Distribution as D

   In [2]: from dit.multivariate import kamath_common_information as U

   In [3]: from dit.multivariate import directed_kamath_common_information as G_dir

   In [4]: outcomes = ['aα', 'bβ', 'bγ', 'bδ', 'cβ', 'cγ', 'cδ']

   In [5]: pmf = [4/37, 9/37, 2/37, 3/37, 12/37, 3/37, 4/37]

   In [6]: d = D(outcomes, pmf)

   @doctest float
   In [7]: G_dir(d, rvs=[1], about=[0])
   Out[7]: 1.0414647631411194

   @doctest float
   In [8]: G_dir(d, rvs=[0], about=[1])
   Out[8]: 1.3712481855145016

   @doctest float
   In [9]: U(d)
   Out[9]: 1.3712481855145016

The conditional law :math:`\Phi^Y_X` is injective on the support of :math:`X` (all three rows of the joint matrix have distinct shape), so :math:`G(X \to Y) = \H{X}` and :math:`U(X; Y) = G(X \to Y)`.


Properties
==========

For two variables, :math:`U` satisfies

.. math::

   \K{X : Y} \leq \I{X : Y} \leq \C{X : Y} \leq G(Y \to X) \leq \H{Y}

and likewise for :math:`G(X \to Y) \leq \H{X}`. The right-hand side bounds are tight on generic joint distributions, where no two columns of the joint matrix coincide as conditional laws and :math:`U(X; Y) = \max\{\H{X}, \H{Y}\}`.


API
===

.. autofunction:: kamath_common_information
.. autofunction:: directed_kamath_common_information
