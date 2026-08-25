.. constructors.rst

.. py:currentmodule:: dit

Constructors and examples
=========================

Besides the :class:`~dit.Distribution` constructors on :doc:`npdist`,
:mod:`dit` ships helpers for building joints and a catalog of named examples.

``dit.distconst``
-----------------

These are imported at the package root:

- :func:`~dit.uniform` / :func:`~dit.uniform_distribution` / :func:`~dit.uniform_like`
- :func:`~dit.mixture_distribution` — convex combination of distributions
- :func:`~dit.product_distribution` — product of specified marginals of a joint
- :func:`~dit.insert_rvf` / :class:`~dit.RVFunctions` — append a deterministic
  function of existing variables
- :func:`~dit.noisy` / :func:`~dit.erasure` — pass each variable through a
  noisy or erasure channel
- :func:`~dit.random_distribution` / :func:`~dit.simplex_grid` —
  random or gridded pmfs
- :func:`~dit.distribution_from_bayesnet` — joint from a NetworkX Bayesian
  network whose nodes carry local distributions

.. ipython::

   In [1]: import dit

   In [2]: d = dit.uniform_distribution(2, 2)

   @doctest
   In [3]: print(d)
   Class:    Distribution
   Alphabet: ((0, 1), (0, 1))
   Base:     linear
   <BLANKLINE>
   x        p(X0,X1)
   (0, 0)   0.25
   (0, 1)   0.25
   (1, 0)   0.25
   (1, 1)   0.25

   In [4]: xor = dit.insert_rvf(d, lambda x: (int(x[0]) ^ int(x[1]),))

   @doctest
   In [5]: print(xor)
   Class:    Distribution
   Alphabet: ((0, 1), (0, 1), (0, 1))
   Base:     linear
   <BLANKLINE>
   x           p(X0,X1,X2)
   (0, 0, 0)   0.25
   (0, 1, 1)   0.25
   (1, 0, 1)   0.25
   (1, 1, 0)   0.25

``dit.example_dists``
---------------------

Parametric and hand-built examples:

- Circuits: :func:`~dit.example_dists.Xor`, :func:`~dit.example_dists.And`,
  :func:`~dit.example_dists.Or`, :func:`~dit.example_dists.Rdn`,
  :func:`~dit.example_dists.Unq`, and related PID illustrations
- Counts: :func:`~dit.example_dists.uniform`, :func:`~dit.example_dists.bernoulli`,
  :func:`~dit.example_dists.binomial`, :func:`~dit.example_dists.multinomial`
- Structured joints: :func:`~dit.example_dists.giant_bit`,
  :func:`~dit.example_dists.n_mod_m`, :func:`~dit.example_dists.dyadic`,
  :func:`~dit.example_dists.triadic`, :func:`~dit.example_dists.pr_box`

Empirical constructors under :mod:`dit.example_dists.empirical`
(:func:`~dit.example_dists.titanic`, :func:`~dit.example_dists.penguins`,
:func:`~dit.example_dists.blood_types`, :func:`~dit.example_dists.congress`,
:func:`~dit.example_dists.student`, :func:`~dit.example_dists.car`,
:func:`~dit.example_dists.bach`, :func:`~dit.example_dists.corelli`)
**fetch their source data at call time** and return the estimated joint.
The music examples additionally require ``dit[music]``.

API
===

.. autofunction:: dit.uniform

.. autofunction:: dit.uniform_distribution

.. autofunction:: dit.mixture_distribution

.. autofunction:: dit.product_distribution

.. autofunction:: dit.insert_rvf

.. autofunction:: dit.distribution_from_bayesnet
