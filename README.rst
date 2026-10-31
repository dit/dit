``dit`` is a Python package for discrete information theory.

|build| |codecov| |docs| |conda|

|joss| |zenodo| |slack|

Introduction
------------

.. start-intro

Information theory is a powerful extension to probability and statistics, quantifying dependencies
among arbitrary random variables in a way that is consistent and comparable across systems and
scales. Information theory was originally developed to quantify how quickly and reliably information
could be transmitted across an arbitrary channel. The demands of modern, data-driven science have
been coopting and extending these quantities and methods into unknown, multivariate settings where
the interpretation and best practices are not known. For example, there are at least four reasonable
multivariate generalizations of the mutual information, none of which inherit all the
interpretations of the standard bivariate case. Which is best to use is context-dependent. ``dit``
implements a vast range of multivariate information measures in an effort to allow information
practitioners to study how these various measures behave and interact in a variety of contexts. We
hope that having all these measures and techniques implemented in one place will allow the
development of robust techniques for the automated quantification of dependencies within a system
and concrete interpretation of what those dependencies mean.

.. end-intro

Citing
------

.. start-citing

If you use ``dit`` in your research, please cite it as::

   @article{dit,
     Author = {James, R. G. and Ellison, C. J. and Crutchfield, J. P.},
     Title = {{dit}: a {P}ython package for discrete information theory},
     Journal = {The Journal of Open Source Software},
     Volume = {3},
     Number = {25},
     Pages = {738},
     Year = {2018},
     Doi = {https://doi.org/10.21105/joss.00738}
   }

.. end-citing

Basic Information
-----------------

.. start-basic-info

Documentation
   https://docs.dit.io

Downloads
   https://pypi.org/project/dit/ and https://anaconda.org/conda-forge/dit

Code and bug tracker
   https://github.com/dit/dit

License
   BSD 3-Clause, see ``LICENSE.txt`` for details.

Dependencies
   * Python 3.11+
   * `boltons <https://boltons.readthedocs.io>`_
   * `debtcollector <https://docs.openstack.org/debtcollector/>`_
   * `lattices <https://github.com/dit/lattices>`_
   * `loguru <https://loguru.readthedocs.io>`_
   * `networkx <https://networkx.org/>`_
   * `numpy <https://numpy.org/>`_
   * `PLTable <https://github.com/platomav/PLTable>`_
   * `scipy <https://scipy.org/>`_
   * `xarray <https://docs.xarray.dev/>`_

.. end-basic-info

Install
-------

.. start-install

The easiest way to install is:

.. code-block:: bash

   pip install dit

If you want to install ``dit`` within a conda environment, you can simply do:

.. code-block:: bash

   conda install -c conda-forge dit

Optional features are grouped into extras:

=============== ===========================================================================
Extra           Provides
=============== ===========================================================================
``optional``    matplotlib (plots), plotly (interactive plots), colorama (colored PID
                tables), numdifftools (numerical gradients/Hessians), pint (units),
                pycddlib-standalone + pypoman (polytope vertex enumeration), scikit-learn
                (fast nearest-neighbor estimators)
``jax``         JAX optimization backend with autodiff (jax, jaxlib, optimistix, slsqp-jax)
``torch``       PyTorch optimization backend with autodiff and GPU support
``pytensor``    PyTensor optimization backend with autodiff
``symbolic``    sympy-valued (symbolic) distributions
``broja``       ECOS cone solver for the BROJA partial information decomposition
``music``       music21, for the empirical Bach / Corelli example distributions
``test``        pytest, hypothesis, and plugins for running the test suite
``docs``        Sphinx and friends for building the documentation
``dev``         everything above, plus ruff and ty
=============== ===========================================================================

For example:

.. code-block:: bash

   pip install "dit[optional,jax]"

For development, we recommend `uv <https://docs.astral.sh/uv/>`_:

.. code-block:: bash

   git clone https://github.com/dit/dit.git
   cd dit
   uv sync --extra dev

This installs ``dit`` in editable mode with all development dependencies
(tests, docs, linting, type checking, and optional backends).

Testing
*******

.. code-block:: bash

   # Using uv (recommended)
   uv run pytest

   # Or with pip
   pip install -e ".[test]"
   pytest

By default the test suite skips tests marked ``slow`` and ``very_slow``. To run
them as well, override the marker selection:

.. code-block:: bash

   uv run pytest -m ""

Building the documentation
**************************

.. code-block:: bash

   uv run sphinx-build -W --keep-going -b html docs docs/_build/html

.. end-install

Implemented Measures
--------------------

.. start-measures

``dit`` implements the following information measures and tools. Most measures are implemented in
multivariate and conditional generality, where such generalizations either exist in the literature
or are relatively obvious --- for example, though it is not in the literature, the multivariate
conditional exact common information is implemented here. Each heading links to the relevant
section of the `documentation <https://docs.dit.io>`__.

`Entropies <https://docs.dit.io/en/latest/measures/measures.html>`__
   Shannon entropy (joint and conditional) · Rényi entropy · Tsallis entropy · residual entropy /
   independent information / variation of information · necessary conditional entropy ·
   cumulative residual entropy (generalized and conditional) · extropy · perplexity · negentropy ·
   disequilibrium and LMPR complexity · α-synergy entropy spectra (synergistic, unique, union,
   intersection, and self-synergy entropies)

`Multivariate mutual informations <https://docs.dit.io/en/latest/measures/multivariate/multivariate.html>`__
   mutual information · co-information · interaction information · total correlation /
   multi-information · dual total correlation / binding information (and its generalization) ·
   CAEKL mutual information · O-information · S-information · cohesion · Kirkwood mutual
   information · frustration (order-k and pairwise) · transmission · DeWeese mutual informations ·
   cross mutual informations · union information · Quax synergy · synergistic disclosure (and its
   modified and backbone variants) · logarithmic decomposition · :math:`\Delta^k` / :math:`\Gamma^k` ·
   TSE complexity · α-synergy information spectra

`Common informations <https://docs.dit.io/en/latest/measures/common_informations/common_informations.html>`__
   Gács–Körner · stochastic Gács–Körner · Wyner · exact · functional · MSS · Kamath–Anantharam
   (and its directed variant) · β common information · tension common information · channel
   synthesis rate · Salamatian–Cohen–Médard maximum entropy function

`Other measures <https://docs.dit.io/en/latest/measures/other/other.html>`__
   lautum information · Sibson mutual information (and its conditional variants) · maximal leakage

`Divergences <https://docs.dit.io/en/latest/measures/divergences/divergences.html>`__
   Kullback–Leibler divergence / relative entropy · cross entropy · Jensen–Shannon divergence ·
   variational distance · Hellinger distance and divergence · Bhattacharyya coefficient · Chernoff
   information · Rényi, Tsallis, α-, and f-divergences · earth mover's distance · coupling metrics ·
   copy mutual information · maximum correlation · hypercontractivity coefficient

`Secret key agreement <https://docs.dit.io/en/latest/measures/secret_keys.html>`__
   secret key capacity (multiterminal, with omniscience rate) · no-communication, one-way, and
   two-way secret key agreement rates · secrecy capacity · intrinsic, reduced intrinsic, minimal
   intrinsic, two-part intrinsic, relaxed two-part, upper, lower, necessary, interactive, and
   less-noisy intrinsic mutual informations (with total correlation, dual total correlation, and
   CAEKL variants) · iterated discarding · secret key cost / information of formation

`Partial information decomposition <https://docs.dit.io/en/latest/decompositions/pid.html>`__
   :math:`I_{min}` · :math:`I_{MMI}` · :math:`I_{\wedge}` · :math:`I_{proj}` · :math:`I_{BROJA}` ·
   :math:`I_{ccs}` · :math:`I_{\pm}` · :math:`I_{sx}` · :math:`I_{dep}` · :math:`I_{RA}` ·
   :math:`I_{RAV}` · :math:`I_{RR}` · :math:`I_{RDR}` · :math:`I_{CT}` · :math:`I_{deg}` ·
   :math:`I_{MC}` · :math:`I_{LN}` · :math:`I_{GH}` · :math:`I_{IG}` · :math:`I_{IPID}` ·
   :math:`I_{\delta}` · :math:`I_{\delta\Lambda}` · :math:`I_{ME}` · :math:`I_{\prec}` ·
   :math:`I_{do}` · :math:`I_{SKAR}` (four variants) · :math:`I_{\cup}^{CI}` (union-information
   synergy) · partial entropy decompositions (:math:`H_{cs}`, :math:`H_{MOS}`, :math:`H_{sx}`) ·
   synergistic disclosure decomposition · integrated information decomposition (ΦID) ·
   generalized information decomposition (GID) · representational complexity

`Profiles and visualization <https://docs.dit.io/en/latest/decompositions/profiles.html>`__
   Shannon and extropy partitions (information diagrams) · complexity profile · marginal utility
   of information · connected informations (Schneidman, dual, and m-flat / Amari) · marginal lift
   profile · binding mixture / shared randomness decomposition · dependency decompositions (and
   dual) · Shapley decompositions · entropy triangles · information UpSet plots

`Channels and coding <https://docs.dit.io/en/latest/channels/channels.html>`__
   channel capacity (certified, accelerated Blahut–Arimoto) · capacity–cost function · Sibson
   capacity · a catalog of example channels · channel orderings (degraded, less noisy, more
   capable, Blackwell, Shannon inclusion) and Le Cam / KL deficiencies · source codes (Shannon,
   Fano, Shannon–Fano–Elias, Huffman, length-limited Huffman, Golomb/Rice, Tunstall, Elias,
   Fibonacci, unary, polar) · channel codes (repetition, parity-check, Hamming, Reed–Muller,
   Golay, LDPC, polar, convolutional)

`Rate-distortion and Gray–Wyner <https://docs.dit.io/en/latest/channels/rate_distortion.html>`__
   rate-distortion curves · information bottleneck (generalized, deterministic, and divergence
   variants) · privacy funnel (and perfect privacy) · Gray–Wyner network, region, and extension
   profile · lossy Wyner common information · Körner graph entropy · maximal and private
   interaction informations · excess functional information

`Estimation <https://docs.dit.io/en/latest/estimation/inference.html>`__
   distributions from data and time series · 17 plug-in and bias-corrected entropy estimators
   (Miller–Madow, Grassberger, Chao–Shen, NSB, Zhang, shrinkage, Bayesian, jackknife, …) ·
   alphabet-aware conditional mutual information · k-NN differential entropy and KSG total
   correlation · permutation entropy · entropy rates (block, Lempel–Ziv, Bayesian posterior) ·
   Markov order tests · surrogate data

`Optimization and constructions <https://docs.dit.io/en/latest/estimation/optimization.html>`__
   maximum entropy distributions · iterative proportional fitting · m-projections · mixtures of
   products · marginal lifts · convex maximization over polytopes · minimal sufficient statistics
   and information trimming · lattice joins and meets · Kirkwood, Lancaster, and Fejér–Riesz
   approximations · symbolic (sympy) distributions · JAX, PyTorch, and PyTensor backends

.. end-measures

Quickstart
----------

.. start-quickstart

The basic usage of ``dit`` corresponds to creating distributions, modifying them
if need be, and then computing properties of those distributions. First, we
import:

.. code:: python

   >>> import dit

Suppose we have a really thick coin, one so thick that there is a reasonable
chance of it landing on its edge. Here is how we might represent the coin in
``dit``. Each outcome is a tuple with one entry per random variable; unnamed
variables are called ``X0``, ``X1``, ….

.. code:: python

   >>> d = dit.Distribution(['H', 'T', 'E'], [.4, .4, .2])
   >>> print(d)
   Class:    Distribution
   Alphabet: (('E', 'H', 'T'),)
   Base:     linear
   <BLANKLINE>
   x        p(X0)
   ('E',)   0.2
   ('H',)   0.4
   ('T',)   0.4

Calculate the probability of ``H`` and also of the combination ``H or T``.

.. code:: python

   >>> d['H']
   0.4
   >>> d.event_probability(['H', 'T'])
   0.8

Calculate the Shannon entropy and extropy of the joint distribution.

.. code:: python

   >>> float(dit.shannon.entropy(d))
   1.5219280948873621
   >>> float(dit.other.extropy(d))
   1.1419011889093373

Create a distribution where ``Z = xor(X, Y)``.

.. code:: python

   >>> import dit.example_dists
   >>> d = dit.example_dists.Xor()
   >>> d.set_rv_names(['X', 'Y', 'Z'])
   >>> print(d)
   Class:    Distribution
   Alphabet: (('0', '1'), ('0', '1'), ('0', '1'))
   Base:     linear
   <BLANKLINE>
   x                 p(X,Y,Z)
   ('0', '0', '0')   0.25
   ('0', '1', '1')   0.25
   ('1', '0', '1')   0.25
   ('1', '1', '0')   0.25

Calculate the Shannon mutual informations ``I[X:Z]``, ``I[Y:Z]``, and
``I[X,Y:Z]``.

.. code:: python

   >>> float(dit.shannon.mutual_information(d, ['X'], ['Z']))
   0.0
   >>> float(dit.shannon.mutual_information(d, ['Y'], ['Z']))
   0.0
   >>> float(dit.shannon.mutual_information(d, ['X', 'Y'], ['Z']))
   1.0

Calculate the marginal distribution ``P(X,Z)`` and print its probabilities as
fractions.

.. code:: python

   >>> d2 = d.marginal(['X', 'Z'])
   >>> print(d2.to_string(exact=True))
   Class:    Distribution
   Alphabet: (('0', '1'), ('0', '1'))
   Base:     linear
   <BLANKLINE>
   x            p(X,Z)
   ('0', '0')   1/4
   ('0', '1')   1/4
   ('1', '0')   1/4
   ('1', '1')   1/4

Convert the distribution probabilities to log (base 3.5) probabilities, and
access its probability mass function.

.. code:: python

   >>> d2.set_base(3.5)
   >>> d2.pmf
   array([-1.10658951, -1.10658951, -1.10658951, -1.10658951])

Draw 5 random samples from this distribution.

.. code:: python

   >>> dit.math.prng.seed(1)
   >>> d2.rand(5)
   [('0', '1'), ('1', '0'), ('0', '0'), ('0', '1'), ('0', '0')]

Source and Channel Coding
*************************

Beyond measures, ``dit`` builds explicit codes and ships a catalog of channels.

The ``dit.coding`` module constructs lossless *source codes* (Shannon, Fano,
Shannon-Fano-Elias, Huffman, length-limited Huffman, Golomb/Rice, Tunstall, and
the universal integer codes) and reports their code-theoretic properties (rate,
redundancy, efficiency, the Kraft sum, and whether the code is prefix-free /
uniquely decodable / optimal).

.. code:: python

   >>> from dit.coding import huffman
   >>> d = dit.Distribution(['a', 'b', 'c', 'd', 'e'], [0.4, 0.2, 0.2, 0.1, 0.1])
   >>> code = huffman(d)
   >>> code.average_length()
   2.2
   >>> float(code.source_entropy())
   2.1219280948873624
   >>> code.is_optimal(), code.is_prefix_free()
   (True, True)

It also builds binary (GF(2)) *channel codes* --- linear block codes
(repetition, parity-check, Hamming, Reed-Muller, Golay) as well as LDPC, polar,
and convolutional codes --- and evaluates them against a noisy channel supplied
as a conditional ``Distribution`` ``p(Y|X)``.

.. code:: python

   >>> from dit.coding import hamming
   >>> code = hamming(3)
   >>> code.length, code.dimension, code.minimum_distance()
   (7, 4, 3)
   >>> bsc = dit.example_channels.binary_symmetric_channel(0.05)
   >>> float(code.probability_of_error(bsc, method='exact'))
   0.04438054218749993

The ``dit.example_channels`` module is a catalog of canonical discrete
memoryless channels (binary symmetric, binary erasure, Z-channel, q-ary
symmetric/erasure, noisy typewriter, ...). Each constructor returns a
conditional ``Distribution`` ``p(Y|X)`` ready for
``dit.algorithms.channel_capacity`` or the coding layer above.

.. code:: python

   >>> from dit.algorithms import channel_capacity
   >>> bec = dit.example_channels.binary_erasure_channel(0.25)
   >>> round(float(channel_capacity(bec)[0]), 10)
   0.75

.. end-quickstart

Contributions & Help
--------------------

If you'd like a feature added to ``dit``, please file an issue. Or, better yet, open a pull request.
Ideally, all code should be tested and documented, but please don't let this be a barrier to
contributing. We'll work with you to ensure that all pull requests are in a mergeable state.

If you'd like to get in contact about anything, you can reach us through our
`slack channel <https://dit-python.slack.com/>`_.

Thanks
------

Thank you to everyone who has contributed bug reports or pull requests to ``dit``:
Aaron Griffith, Abhishek Bhatia, @alimasn, André F. C. Gomes, Angus Forbes, Artemy Kolchinsky,
Benjamin Lee, Bochen Zhao, Casper van Elteren, Charles Kemp, @d-jeon, Daniel Feldspar,
David A. Ehrlich, Diego Volpatto, Elias Jaffe, Fabio Anza, Freya Behrens, Haochen Wu,
Harald Schilly, Himanshu Mishra, @ivan-marroquin, Jaewon Chung, James K Johnson, Jan Olszak,
Jeffrey Emenheiser, Juan Carlos Farah, Keenan Down, @kokokostation, Kunal Marwaha, Lee Junho,
Longshan Du, @LuisHeinzlmeier, Marc Harper, Nathaniel Chen, @OrdinarySK, Pattarawat Chormai, QZP,
Rion Brattig Correia, Robin Ince, Rodrigo Silva, Scott Sievert, Tai C. Huang, Talha Anwar,
Thomas Kluyver, Tobias Mages, Vigneswaran Chandrasekaran, 赵丰 (Zhao Feng), and @zuiyanzuiyan.


.. badges:

.. |build| image:: https://github.com/dit/dit/actions/workflows/build.yml/badge.svg
   :target: https://github.com/dit/dit/actions/workflows/build.yml
   :alt: Continuous Integration Status

.. |codecov| image:: https://codecov.io/gh/dit/dit/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/dit/dit
   :alt: Test Coverage Status

.. |docs| image:: https://readthedocs.org/projects/dit/badge/?version=latest
   :target: https://docs.dit.io/en/latest/
   :alt: Documentation Status

.. |conda| image:: https://anaconda.org/conda-forge/dit/badges/installer/conda.svg
   :target: https://anaconda.org/conda-forge/dit
   :alt: Conda installation

.. |zenodo| image:: https://zenodo.org/badge/13201610.svg
   :target: https://zenodo.org/badge/latestdoi/13201610
   :alt: DOI

.. |slack| image:: https://img.shields.io/badge/Slack-dit--python-lightgrey.svg
   :target: https://dit-python.slack.com/
   :alt: dit chat

.. |joss| image:: http://joss.theoj.org/papers/10.21105/joss.00738/status.svg
   :target: https://doi.org/10.21105/joss.00738
   :alt: JOSS Status
