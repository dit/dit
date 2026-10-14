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
- :func:`~dit.kirkwood_approximation` / :func:`~dit.lancaster_approximation` /
  :func:`~dit.fejer_riesz_approximation` — order-:math:`k` multiplicative,
  additive, and squared-amplitude approximations built from marginals (see
  :ref:`below <marginal_approximations>`)
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

.. _marginal_approximations:

Kirkwood and Lancaster approximations
-------------------------------------

Both approximations truncate an interaction expansion of :math:`p` at subsets of at most :math:`k` variables, and share the coefficients

.. math::

   c_S = (-1)^{k-|S|} \binom{n-|S|-1}{k-|S|}, \qquad |S| \leq k.

The Kirkwood approximation :cite:`watanabe1960information,killian2007extraction` combines the marginals geometrically, and the Lancaster approximation :cite:`lancaster1969chi` arithmetically:

.. math::

   \tilde{p}_k(x) = \prod_{|S| \leq k} p(x_S)^{c_S}, \qquad
   p_k(x) = \sum_{|S| \leq k} c_S \, p(x_S) \prod_{i \notin S} p(x_i).

The default order is :math:`k = n-1`. For three variables these are :math:`p(x, y) p(x, z) p(y, z) / \left(p(x) p(y) p(z)\right)` and :math:`p(x, y) p(z) + p(x, z) p(y) + p(y, z) p(x) - 2 p(x) p(y) p(z)`. Both return dense, unvalidated distributions.

The Kirkwood approximation is nonnegative but not normalized. Its total mass is the normalizer of the :ref:`kirkwood_mutual_information`. For the giant bit, every outcome off the diagonal has a vanishing pairwise marginal:

.. ipython::

   In [6]: from dit.example_dists import giant_bit

   In [7]: dit.kirkwood_approximation(giant_bit(3, 2)).pmf
   Out[7]: array([1., 0., 0., 0., 0., 0., 0., 1.])

The order-:math:`k` Lancaster approximation keeps the Lancaster interactions of at most :math:`k` variables in the expansion :math:`p = \prod_i p(x_i) \sum_T \delta_T(x_T)`. At the default order it is :math:`p - \Delta_L p`, where :math:`\Delta_L p = \prod_i \left(p^*_{X_i} - p_{X_i}\right)` is the Lancaster interaction measure :cite:`sejdinovic2013kernel`. It sums to one and matches every marginal of :math:`p` on at most :math:`k` variables, but it may be negative:

.. ipython::

   In [8]: dit.lancaster_approximation(dit.Distribution(['000', '010', '100', '111'], [1/4] * 4)).pmf
   Out[8]: array([ 0.3125, -0.0625,  0.1875,  0.0625,  0.1875,  0.0625,  0.0625,  0.1875])

For :math:`n \leq 3` the default-order Lancaster approximation reproduces :math:`p` whenever :math:`p` factors nontrivially, though not only then. For :math:`n > 3` it can miss factorizations such as :math:`(X_0, X_1) \perp (X_2, X_3)`, which Streitberg's partition-lattice correction repairs :cite:`streitberg1990lancaster`. No canonical literature source is known for Lancaster orders below :math:`n-1`.

The Fejér-Riesz approximation trades the marginal constraints for nonnegativity. Writing :math:`\pi(x) = \prod_i p(x_i)`, the Lancaster approximation truncates the density ratio :math:`p / \pi`. Following the Fejér-Riesz theorem, which writes a nonnegative trigonometric polynomial as :math:`|h|^2` :cite:`fejer1916trigonometrische`, this approximation instead truncates the amplitude :math:`s = \sqrt{p / \pi}` and squares it:

.. math::

   s_k(x) = \sum_{|S| \leq k} c_S \, \mathbb{E}_\pi[s \mid x_S], \qquad
   q_k(x) = \frac{\pi(x) \, s_k(x)^2}{Z_k}.

Here :math:`s_k` is the :math:`L^2(\pi)` projection of :math:`s` onto functions of at most :math:`k` variables, so :math:`Z_k = \|s_k\|^2_\pi \leq 1`. The result is a probability distribution, and it equals :math:`p` whenever :math:`p` is a product of marginals. However, it need not match the marginals of :math:`p`. For the distribution above:

.. ipython::

   In [9]: dit.fejer_riesz_approximation(dit.Distribution(['000', '010', '100', '111'], [1/4] * 4)).pmf.round(4)
   Out[9]: array([0.3879, 0.0248, 0.1946, 0.0248, 0.1946, 0.0248, 0.0083, 0.1404])

No canonical literature source is known for this construction.

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

.. autofunction:: dit.kirkwood_approximation

.. autofunction:: dit.lancaster_approximation

.. autofunction:: dit.fejer_riesz_approximation

.. autofunction:: dit.insert_rvf

.. autofunction:: dit.distribution_from_bayesnet
