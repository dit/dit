.. optimization.rst
.. py:module:: dit.algorithms.distribution_optimizers

************
Optimization
************

It is often useful to construct a distribution :math:`d^\prime` which is consistent with some marginal aspects of :math:`d`, but otherwise optimizes some information measure. For example, perhaps we are interested in constructing a distribution which matches pairwise marginals with another, but otherwise has maximum entropy:

.. ipython::

    In [1]: from dit.algorithms.distribution_optimizers import MaxEntOptimizer

    In [2]: xor = dit.example_dists.Xor()

    In [3]: meo = MaxEntOptimizer(xor, [[0,1], [0,2], [1,2]])

    In [4]: meo.optimize()

    In [5]: dp = meo.construct_dist()

    In [6]: print(dp)
    Class:          Distribution
    Alphabet:       ('0', '1') for all rvs
    Base:           linear
    Outcome Class:  str
    Outcome Length: 3
    RV Names:       None
    x     p(x)
    000   0.125
    001   0.125
    010   0.125
    011   0.125
    100   0.125
    101   0.125
    110   0.125
    111   0.125

================
Helper Functions
================

There are three special functions to handle common optimization problems:

.. ipython::

    In [7]: from dit.algorithms import maxent_dist, marginal_maxent_dists

The first is maximum entropy distributions with specific fixed marginals. It encapsulates the steps run above:

.. ipython::

    In [8]: print(maxent_dist(xor, [[0,1], [0,2], [1,2]]))
    Class:          Distribution
    Alphabet:       ('0', '1') for all rvs
    Base:           linear
    Outcome Class:  str
    Outcome Length: 3
    RV Names:       None
    x     p(x)
    000   0.125
    001   0.125
    010   0.125
    011   0.125
    100   0.125
    101   0.125
    110   0.125
    111   0.125

The second constructs several maximum entropy distributions, each with all subsets of variables of a particular size fixed:

.. ipython::

    In [9]: k0, k1, k2, k3 = marginal_maxent_dists(xor)

where ``k0`` is the maxent dist corresponding the same alphabets as ``xor``; ``k1`` fixes :math:`p(x_0)`, :math:`p(x_1)`, and :math:`p(x_2)`; ``k2`` fixes :math:`p(x_0, x_1)`, :math:`p(x_0, x_2)`, and :math:`p(x_1, x_2)` (as in the ``maxent_dist`` example above), and finally ``k3`` fixes :math:`p(x_0, x_1, x_2)` (e.g. is the distribution we started with).

=================================
M-Flat m-Projections
=================================

The dual of the MaxEnt / e-flat ladder is Amari's m-flat hierarchy
:cite:`Amari2001`. Instead of matching :math:`k`-way marginals, one truncates
the additive (ANOVA) expansion of the pmf and takes the reverse-KL m-projection
onto that mixture family. Sparse supports use the symmetric smooth
:math:`P_\varepsilon=(1-\varepsilon)P+\varepsilon U` (or the
:math:`\varepsilon\downarrow 0` schedule via ``m_projection_eps_limit``):

.. ipython::

    In [9a]: from dit.algorithms import m_projection, m_projection_eps_limit, mflat_mprojection_dists

    In [9b]: q2 = m_projection(xor, 2, eps=1e-6)

    In [9c]: q2_lim = m_projection_eps_limit(xor, order=2)

    In [9d]: ladder = mflat_mprojection_dists(xor, eps_schedule=(1e-4, 1e-6, 1e-8))

See :doc:`profiles` (:class:`~dit.profiles.MFlatConnectedInformations` and
:class:`~dit.profiles.DualDependencyDecomposition`) for reverse-KL gaps along
the order chain and the full dependency lattice.

=================================
Maximum Entropy Solver (IPF)
=================================

By default, ``maxent_dist`` computes the maximum entropy distribution using `Iterative Proportional Fitting <https://en.wikipedia.org/wiki/Iterative_proportional_fitting>`_ (IPF), the classic algorithm from reconstructability analysis and log-linear modeling. Starting from the uniform distribution, IPF cyclically rescales the working distribution so each constrained marginal matches the data, iterating until convergence. It is typically far faster than the general scipy convex optimizer. IPF converges only linearly on cyclic structures with induced structural zeros, however, so when it fails to converge within its iteration budget ``maxent_dist`` automatically falls back to the scipy optimizer to preserve accuracy. The scipy backend can also be requested explicitly via ``method='scipy'``:

.. ipython::

    In [10]: print(maxent_dist(xor, [[0,1], [0,2], [1,2]], method='scipy'))

=============================
Reconstructability Analysis
=============================

The maximum entropy reconstruction underlies reconstructability analysis, which decomposes a distribution into a *structure* of marginals and assesses each structure by two quantities: its error (:doc:`transmission </measures/multivariate/transmission>`, the information lost relative to the data) and its complexity (degrees of freedom, the number of free parameters). The dependency decomposition (see :doc:`profiles`) evaluates these over the whole lattice of structures, yielding the "decomposition spectrum":

.. ipython::

    In [11]: from dit.algorithms import degrees_of_freedom

    In [12]: from dit.multivariate import transmission

    In [13]: from dit.profiles import DependencyDecomposition

    In [14]: from dit.multivariate import entropy

    In [15]: print(DependencyDecomposition(xor, measures={'H': entropy, 'T': transmission, 'df': degrees_of_freedom}))

=================================
Mixtures of product distributions
=================================

Shared-randomness profiles fit *mixtures of fully factorized* distributions
(naive-Bayes / latent-class models) by EM. Maximum likelihood equals the
forward-KL projection onto

.. math::

   \mathcal{F}_k = \Bigl\{
       Q : Q(x) = \sum_{\alpha=1}^{k} \pi_\alpha \prod_i Q_i(x_i \mid \alpha)
   \Bigr\}.

Helpers :func:`~dit.algorithms.fit_mixture_of_products` and
:func:`~dit.algorithms.mixture_of_products_dists` feed
:class:`~dit.profiles.BindingMixtureProfile` (see :doc:`profiles`).

For *additive* building blocks (Amari ANOVA / fixed marginal lifts), see
:func:`~dit.algorithms.m_projection` (criteria ``jsd``, ``forward_kl``,
``reverse_kl``) and :func:`~dit.algorithms.fit_marginal_lift_mixture`.

.. ipython::

    In [15a]: from dit.algorithms import fit_mixture_of_products

    In [15b]: gb = dit.Distribution(['000', '111'], [0.5, 0.5])

    In [15c]: q2 = fit_mixture_of_products(gb, k=2, n_init=8, seed=0)['dist']

    In [15d]: print(round(dit.divergences.kullback_leibler_divergence(gb, q2), 8))
    0.0

=====================
Optimization Backends
=====================

By default, ``dit`` uses NumPy and SciPy for numerical optimization. Three additional backends are available that leverage automatic differentiation for computing exact gradients, which can improve convergence for large or complex problems:

JAX Backend
~~~~~~~~~~~

Install with ``pip install "dit[jax]"``. The JAX backend (``dit.algorithms.optimization_jax``) provides:

* Automatic differentiation via ``jax.grad`` for exact gradient computation
* JIT compilation for improved performance
* GPU/TPU acceleration when available

PyTorch Backend
~~~~~~~~~~~~~~~

Install with ``pip install "dit[torch]"``. The PyTorch backend (``dit.algorithms.optimization_torch``) provides:

* Automatic differentiation via ``torch.autograd`` for exact gradient computation
* GPU acceleration via CUDA or MPS when available
* ``torch.compile`` support for PyTorch 2.0+

PyTensor Backend
~~~~~~~~~~~~~~~~

Install with ``pip install "dit[pytensor]"``. The PyTensor backend (``dit.algorithms.optimization_pytensor``) uses `PyTensor <https://pytensor.readthedocs.io>`_ (the maintained successor to the archived Aesara) and provides:

* Symbolic graph compilation of the objective, its exact gradient via ``pytensor.grad``, and each constraint Jacobian, compiled once and reused across the optimization
* A native augmented-Lagrangian solver (compiled value/gradient with ``L-BFGS-B`` inner solves) for moderate problem sizes, with a SciPy SLSQP fallback
* Optional Numba compilation of the compiled functions (set ``DIT_PYTENSOR_MODE=NUMBA``)

Set ``DIT_PYTENSOR_COMPILEDIR`` to persist PyTensor's compilation cache across process launches.

For measures that use the Markov variable optimizer (e.g. common informations), the backend can be selected via the ``backend`` parameter:

.. code-block:: python

    from dit.multivariate import wyner_common_information

    # Default NumPy backend
    wyner_common_information(d)

    # JAX backend (requires jax)
    wyner_common_information(d, backend='jax')

    # PyTorch backend (requires torch)
    wyner_common_information(d, backend='torch')

    # PyTensor backend (requires pytensor)
    wyner_common_information(d, backend='pytensor')

=======
Logging
=======

The optimization modules emit structured log messages via `loguru <https://loguru.readthedocs.io>`_. Logging is disabled by default. To enable it:

.. code-block:: python

    from loguru import logger
    logger.enable("dit")

This will show optimization progress including problem dimensions, convergence status, and objective values.
