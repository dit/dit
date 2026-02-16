.. optimization.rst
.. py:module:: dit.algorithms.scipy_optimizers

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

=====================
Optimization Backends
=====================

By default, ``dit`` uses NumPy and SciPy for numerical optimization. Two additional backends are available that leverage automatic differentiation for computing exact gradients, which can improve convergence for large or complex problems:

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

For measures that use the Markov variable optimizer (e.g. common informations), the backend can be selected via the ``backend`` parameter:

.. code-block:: python

    from dit.multivariate import wyner_common_information

    # Default NumPy backend
    wyner_common_information(d)

    # JAX backend (requires jax)
    wyner_common_information(d, backend='jax')

    # PyTorch backend (requires torch)
    wyner_common_information(d, backend='torch')

=======
Logging
=======

The optimization modules emit structured log messages via `loguru <https://loguru.readthedocs.io>`_. Logging is disabled by default. To enable it:

.. code-block:: python

    from loguru import logger
    logger.enable("dit")

This will show optimization progress including problem dimensions, convergence status, and objective values.
