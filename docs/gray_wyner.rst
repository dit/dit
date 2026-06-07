.. gray_wyner.rst
.. py:module:: dit.rate_distortion.gray_wyner

The Gray-Wyner Network
======================

.. note::

   We use :math:`p` to denote fixed probability distributions, and :math:`q` to denote probability distributions that are optimized.

The Gray-Wyner network :cite:`gray1974source` is a simple source-coding network with one encoder and :math:`n` decoders. The encoder observes a correlated source vector :math:`(X_1, \dots, X_n) \sim p` and emits one *common* message at rate :math:`R_0`, delivered to every decoder, together with :math:`n` *private* messages, where message :math:`i` (at rate :math:`R_i`) is delivered only to decoder :math:`i`. Decoder :math:`i` reconstructs :math:`X_i` to within an average distortion :math:`D_i`.

The achievable rate region (lossless: :cite:`gray1974source`; lossy: :cite:`viswanatha2014lossy`) is the set of rate tuples :math:`(R_0, R_1, \dots, R_n)` for which there exists an auxiliary random variable :math:`W` satisfying

.. math::

   R_0 &\ge \I{X_1, \dots, X_n : W} \\
   R_i &\ge R_{X_i \mid W}(D_i) \qquad i = 1, \dots, n

where :math:`R_{X_i \mid W}(D_i)` is the conditional rate-distortion function of :math:`X_i` given :math:`W`. In the lossless case (:math:`D_i = 0` under a Hamming distortion) this reduces to :math:`R_i \ge \H{X_i \mid W}`.

The region is convex, so its lower boundary is traced by minimizing a weighted sum of rates over :math:`W` (and, in the lossy case, the reconstruction test channels subject to the distortion budgets):

.. math::

   \min_{W} \quad \lambda_0 \I{X_{1:n} : W} + \sum_{i} \lambda_i R_{X_i \mid W}(D_i)
   ~.

Sweeping the weights :math:`(\lambda_0, \dots, \lambda_n)` sweeps the Pareto surface.

Common informations as operating points
----------------------------------------

Several of the :doc:`common informations <measures/multivariate/multivariate>` are particular operating points of the lossless Gray-Wyner region:

- The :doc:`measures/multivariate/wyner_common_information` :math:`\C{\cdot}` is the smallest common rate :math:`R_0` on the minimum sum-rate face.
- The :doc:`measures/multivariate/gk_common_information` :math:`\K{\cdot}` is the largest common rate incurring no sum-rate penalty.
- The :doc:`measures/multivariate/exact_common_information` :math:`\G{\cdot}` and :doc:`measures/multivariate/kamath_common_information` :math:`\operatorname{U}(\cdot)` are further extreme points.

``GrayWynerNetwork.corner_points`` returns these values by delegating to their canonical implementations.

Lossy common information
------------------------

Generalizing the minimum-sum-rate operating point to positive distortion gives the lossy common information :math:`C(D_1, \dots, D_n)` :cite:`viswanatha2014lossy`, available as :func:`lossy_wyner_common_information`. For :math:`D_i = 0` it coincides with Wyner's common information.

Example
-------

The trade-off between the common rate and the total private rate is traced by ``GrayWynerCurve``:

.. ipython::
   :verbatim:

   In [1]: from dit.rate_distortion import GrayWynerCurve

   In [2]: d = dit.Distribution(['00', '01', '10', '11'], [0.4, 0.1, 0.1, 0.4])

   In [3]: GrayWynerCurve(d, s_num=21).plot();

The named corner points and the lossy common information:

.. ipython::
   :verbatim:

   In [4]: from dit.rate_distortion import GrayWynerNetwork, lossy_wyner_common_information

   In [5]: from dit.rate_distortion.gray_wyner import hamming_matrix

   In [6]: GrayWynerNetwork(d).corner_points()

   In [7]: dm = [hamming_matrix(2), hamming_matrix(2)]

   In [8]: lossy_wyner_common_information(d, bounds=[0.1, 0.1], distortions=dm)

APIs
====

.. autoclass:: GrayWynerNetwork
   :members:

.. autoclass:: GrayWynerCurve

.. autofunction:: lossy_wyner_common_information

.. autoclass:: GrayWynerOptimizer
