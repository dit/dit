.. logarithmic_decomposition.rst
.. py:currentmodule:: dit.multivariate

.. _logarithmic_decomposition:

**************************
Logarithmic Decomposition
**************************

Down & Mediano :cite:`down2023logarithmic,down2024logarithmic` refine Yeung's
I-measure into *logarithmic atoms* — one for every subset of the joint outcome
space :math:`\Omega` with two or more elements. Each atom has an intrinsic sign
given by its degree (even positive, odd negative) and an interior-loss measure
:math:`\mu` that sums to entropy, mutual information, co-information, and so on.

A random variable :math:`X` corresponds to its *content* :math:`\Delta X`, the
atoms holding two outcomes that :math:`X` tells apart. Every region of the
I-diagram is the matching combination of contents, and its I-measure is the
measure of those atoms :cite:`down2023logarithmic`:

.. math::

   \H{X} = \mu(\Delta X), \qquad
   \H{X | Y} = \mu(\Delta X \setminus \Delta Y), \qquad
   \I{X : Y | Z} = \mu\left((\Delta X \cap \Delta Y) \setminus \Delta Z\right).

:meth:`~LogarithmicDecomposition.region` returns the atoms of the region
:math:`\I{X_1 : \ldots : X_r | Z}`, and ``crvs`` conditions the information
quantities:

.. ipython::

   In [1]: from dit.multivariate import LogarithmicDecomposition

   In [2]: from dit.example_dists import Xor

   In [3]: ld = LogarithmicDecomposition(Xor())

   @doctest
   In [4]: ld
   Out[4]: LogarithmicDecomposition(|Omega|=4, atoms=11)

   @doctest float
   In [5]: ld.coinformation()
   Out[5]: -1.0

   @doctest float
   In [6]: ld.coinformation([[0], [1]], crvs=[2])
   Out[6]: 1.0

The number of atoms is :math:`2^{|\Omega|} - |\Omega| - 1`, and measuring all of
them takes about three times as long with each added outcome, so the
decomposition is practical only for supports of up to about 15 outcomes.


Discernible Subsets and Common Information
==========================================

A set of atoms is *discernible* when it is the content of some random variable
on :math:`\Omega` (the expanded paper :cite:`down2024logarithmic` calls it
*representable*). Contents are closed under union, so every set :math:`S` of
atoms has a largest discernible subset :math:`\operatorname{Dis}(S)`: the
content of the finest partition of :math:`\Omega` that keeps together the
outcomes of each atom outside :math:`S`. The mutual information content is
rarely discernible, and its largest discernible subset is the content of the
Gács-Körner common random variable :math:`X \meet Y`
:cite:`down2023logarithmic`:

.. math::

   \K{X : Y} = \mu\left(\operatorname{Dis}(\Delta X \cap \Delta Y)\right).

Here the common variable keeps outcomes ``22`` and ``33`` apart and lumps the
other four together; the mutual information is larger than the common
information because the lumped outcomes are correlated:

.. ipython::

   In [7]: from dit import Distribution

   In [8]: d = Distribution(['00', '01', '10', '11', '22', '33'], [3/16, 1/16, 1/16, 3/16, 1/4, 1/4])

   In [9]: ld = LogarithmicDecomposition(d)

   In [10]: mi = ld.region([[0], [1]])

   @doctest
   In [11]: ld.is_discernible(mi)
   Out[11]: False

   @doctest
   In [12]: sorted(len(block) for block in ld.discernible_partition(mi))
   Out[12]: [1, 1, 4]

   @doctest float
   In [13]: ld.measure_set(ld.discernible(mi))
   Out[13]: 1.5

   @doctest float
   In [14]: ld.measure_set(mi)
   Out[14]: 1.5943609377704289

:meth:`~LogarithmicDecomposition.discernible` accepts any set of atoms, not
only intersections of contents, and
:meth:`~LogarithmicDecomposition.gk_common_information` computes the
Gács-Körner common information this way.


Formal Sums
===========

Entropy expressions that count atoms more than once, such as the total
correlation, are *formal sums*: integer combinations of atoms
:cite:`down2024logarithmic`. If an atom lies in the contents of :math:`k` of the
variables, the total correlation counts it :math:`k - 1` times, the dual total
correlation counts it once when :math:`k \geq 2` (so the dual total correlation
is the measure of a set of atoms), and the O-information counts it
:math:`\max(k - 2, 0)` times. :meth:`~LogarithmicDecomposition.formal_sum`
returns the multiplicity of every atom in an expression
:math:`\sum_k c_k \H{X_k | Z}`:

.. ipython::

   In [15]: ld = LogarithmicDecomposition(Xor())

   In [16]: tc = ld.formal_sum([(1, [0]), (1, [1]), (1, [2]), (-1, [0, 1, 2])])

   @doctest
   In [17]: sorted(set(tc.values()))
   Out[17]: [1, 2]

   @doctest float
   In [18]: ld.measure_set(tc)
   Out[18]: 1.0

   @doctest float
   In [19]: ld.o_information()
   Out[19]: -1.0


API
===

.. autoclass:: LogarithmicDecomposition
   :members:
   :no-index:

.. autofunction:: logarithmic_decomposition
   :no-index:
