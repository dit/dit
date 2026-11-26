.. functional_information.rst
.. py:currentmodule:: dit.other

**********************
Functional Information
**********************

The functional information :cite:`flamich2026functional` of the channel :math:`X \to Y` is the number of bits needed to simulate the channel once. Suppose an encoder sees :math:`X` and the decoder must output a :math:`Y` with the right joint distribution, and that both share randomness :math:`Z` independent of :math:`X`. Then the least achievable expected entropy :math:`\H{Y | Z}` is between :math:`I_F(X \to Y)` and :math:`I_F(X \to Y) + \log_2 e`.

It is built from the :doc:`channel simulation divergence </measures/divergences/channel_simulation_divergence>` :math:`D_{CS}` of the posterior :math:`p(x|y)` from the prior :math:`p(x)`, averaged over :math:`Y`:

.. math::

   I_F(X \to Y) = \sum_y p(y)\, D_{CS}\left(p(X | y) \,\|\, p(X)\right)
                = \sum_y p(y)\, h(w_y),

where :math:`w_y(\ell) = \Pr\left(p(y | X) / p(y) \geq \ell\right)` is a probability density on :math:`\ell \in [0, \infty)` and :math:`h` is differential entropy.

Properties
==========

The functional information behaves much like the mutual information, but exceeds it except for singular channels :cite:`flamich2026functional`:

* :math:`\I{X : Y} \leq I_F(X \to Y) \leq \I{X : Y} + \log_2(\I{X : Y} + 1) + 1`. Equality on the left holds exactly when the channel is *singular*: :math:`p(y|x) / p(y)` depends only on :math:`y` wherever it is positive. Erasure channels and deterministic functions are singular, so in particular :math:`I_F(X \to X) = \H{X}`.
* It is not symmetric.
* It satisfies a data processing inequality: :math:`I_F(X \to Y) \geq I_F(X \to Z)` for a Markov chain :math:`X - Y - Z`.
* It is concave in :math:`p(x)` and convex in :math:`p(y|x)`.

With :math:`\eta(s) = -s \log_2 s`, it has a closed form. Sort each column of :math:`p(y|x)` as :math:`k_{(0)} \geq \cdots \geq k_{(n-1)} \geq k_{(n)} = 0`, and let :math:`S_i` be the probability of the first :math:`i + 1` values of :math:`X`. Then

.. math::

   I_F(X \to Y) = \sum_y \sum_i \left(k_{(i)} - k_{(i+1)}\right) \eta(S_i),

which is piecewise linear and convex in the channel.

For a doubly symmetric binary source, :math:`I_F` exceeds the mutual information:

.. ipython::

   In [1]: from dit.other import functional_information

   In [2]: d = dit.Distribution(['00', '01', '10', '11'], [0.45, 0.05, 0.05, 0.45])

   @doctest float
   In [3]: functional_information(d, [0], [1])
   Out[3]: 0.8

   @doctest float
   In [4]: dit.shannon.mutual_information(d, [0], [1])
   Out[4]: 0.5310044064107187

The two directions of a channel generally differ:

.. ipython::

   In [5]: d = dit.Distribution(['00', '01', '11', '12'], [0.3, 0.2, 0.1, 0.4])

   @doctest float
   In [6]: functional_information(d, [0], [1])
   Out[6]: 0.8

   @doctest float
   In [7]: functional_information(d, [1], [0])
   Out[7]: 0.94106402938001

Channel Simulation
==================

The least expected entropy :math:`\H{Y | Z}` over functional representations :math:`Y = f(X, Z)` is :math:`\I{X : Y}` plus the excess functional information :cite:`li2018strong`. dit computes it exactly with :func:`dit.rate_distortion.excess_functional_information`, using a linear program over random functions. Write :math:`R^*` for this least entropy. Theorem 1 of :cite:`flamich2026functional` brackets it:

.. math::

   I_F(X \to Y) \leq R^* \leq I_F(X \to Y) + \log_2 e.

The lower bound holds with equality whenever :math:`Y` is binary (Proposition 7 of :cite:`flamich2026functional`). It also holds whenever :math:`X` is binary, because a maximal coupling of the two rows of :math:`p(y|x)` meets the condition of their Proposition 6. It is strict, for example, for the cyclic channel

.. math::

   p(y|x) = \begin{pmatrix} 1 - \epsilon & 0 & \epsilon \\ \epsilon & 1 - \epsilon & 0 \\ 0 & \epsilon & 1 - \epsilon \end{pmatrix}

with :math:`1/3 < \epsilon < 1/2`. For channels too large for the linear program, :func:`dit.rate_distortion.excess_functional_information_bounds` returns this bracket. The :ref:`functional rate-distortion function <functional_rate_distortion>` is the analogous lossy compression quantity.

Rényi Functional Information
============================

For :math:`\alpha \in (0, 1)`, the Rényi functional information :cite:`flamich2026functional` replaces each differential entropy with a differential Rényi entropy:

.. math::

   I_{F,\alpha}(X \to Y) = \frac{1}{1 - \alpha} \log_2 \sum_y p(y) \int_0^\infty w_y(\ell)^\alpha \, d\ell.

It has the following properties:

* It tends to :math:`I_F` as :math:`\alpha \to 1`.
* :math:`I_{F,\alpha}(X \to X) = \RE{\alpha}{X}`.
* It bounds the :doc:`Sibson mutual information <sibson_mutual_information>` of order :math:`1 / \alpha` from above, with equality exactly for singular channels.
* It characterizes channel simulation under an exponential (Campbell) cost on the message length.

Pass the order as ``order``:

.. ipython::

   In [8]: from dit.other import sibson_mutual_information

   In [9]: d = dit.Distribution(['00', '01', '10', '11'], [0.45, 0.05, 0.05, 0.45])

   @doctest float
   In [10]: functional_information(d, [0], [1], order=0.5)
   Out[10]: 0.8258249723836791

   @doctest float
   In [11]: sibson_mutual_information(d, [0], [1], 2)
   Out[11]: 0.7136958148433591

API
===

.. autofunction:: functional_information
   :no-index:

.. autofunction:: functional_information_pmf
   :no-index:
