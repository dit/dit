``dit`` is a Python package for information theory.

|build| |codecov| |docs| |conda|

|joss| |zenodo| |slack|

Introduction
------------

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

Citing
------

If you use `dit` in your research, please cite it as::

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

Basic Information
-----------------

Documentation
*************

http://docs.dit.io

Downloads
*********

https://pypi.org/project/dit/

https://anaconda.org/conda-forge/dit

+-------------------------------------------------------------------+
| Dependencies                                                      |
+===================================================================+
| * Python 3.11+                                                    |
| * `boltons <https://boltons.readthedocs.io>`_                     |
| * `debtcollector <https://docs.openstack.org/debtcollector/>`_    |
| * `lattices <https://github.com/dit/lattices>`_                   |
| * `loguru <https://loguru.readthedocs.io>`_                       |
| * `networkx <https://networkx.github.io/>`_                       |
| * `numpy <http://www.numpy.org/>`_                                |
| * `PLTable <https://github.com/platomav/PLTable>`_                |
| * `scipy <https://www.scipy.org/>`_                               |
| * `xarray <https://docs.xarray.dev/>`_                            |
+-------------------------------------------------------------------+

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~
* colorama: colored column heads in PID indicating failure modes
* cython: faster sampling from distributions
* hypothesis: random sampling of distributions
* jax, jaxlib: JAX-based optimization backend with autodiff support
* matplotlib, python-ternary: plotting of various information-theoretic expansions
* numdifftools: numerical evaluation of gradients and hessians during optimization
* pint: add units to informational values
* pycddlib-standalone, pypoman: polytope vertex enumeration for convex-maximization-based measures
* pytensor: PyTensor-based optimization backend with autodiff support
* scikit-learn: faster nearest-neighbor lookups during entropy/mutual information estimation from samples
* torch: PyTorch-based optimization backend with autodiff and GPU support

Install
*******

The easiest way to install is:

.. code-block:: bash

  pip install dit

If you want to install ``dit`` within a conda environment, you can simply do:

.. code-block:: bash

  conda install -c conda-forge dit

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

Code and bug tracker
********************

https://github.com/dit/dit

License
*******

BSD 3-Clause, see LICENSE.txt for details.

Implemented Measures
--------------------

``dit`` implements the following information measures. Most of these are implemented in multivariate & conditional
generality, where such generalizations either exist in the literature or are relatively obvious --- for example,
though it is not in the literature, the multivariate conditional exact common information is implemented here.

+------------------------------------------+-----------------------------------------+-----------------------------------+
| Entropies                                | Mutual Informations                     | Divergences                       |
|                                          |                                         |                                   |
| * Shannon Entropy                        | * Co-Information                        | * Variational Distance            |
| * Renyi Entropy                          | * Interaction Information               | * Kullback-Leibler Divergence /   |
| * Tsallis Entropy                        | * Total Correlation / Multi-Information |   Relative Entropy                |
| * Necessary Conditional Entropy          | * Dual Total Correlation / Binding      | * Cross Entropy                   |
| * Residual Entropy / Independent         |   Information                           | * Jensen-Shannon Divergence       |
|   Information / Variation of Information | * CAEKL Multivariate Mutual Information | * Earth Mover's Distance          |
+------------------------------------------+ * O-Information                         +-----------------------------------+
| Common Informations                      | * Cohesion                              | Other Measures                    |
|                                          | * Transmission                          |                                   |
| * Gacs-Korner Common Information         | * Union Information                     | * Channel Capacity                |
| * Wyner Common Information               | * Logarithmic Decomposition             | * Complexity Profile              |
| * Exact Common Information               | * Delta^k / Gamma^k                     | * Connected Informations          |
| * Functional Common Information          | * DeWeese Mutual Information            | * Copy Mutual Information         |
| * MSS Common Information                 +-----------------------------------------+ * Cumulative Residual Entropy     |
| * Kamath-Anantharam Dual Common          | Partial Information Decomposition       | * Extropy                         |
|   Information                            |                                         | * Hypercontractivity Coefficient  |
| * Beta Common Information                | * :math:`I_{min}`                       | * Information Bottleneck          |
| * Salamatian-Cohen-Medard Maximum Entropy| * :math:`I_{\wedge}`                    | * Information Diagrams            |
|   Function                               | * :math:`I_{RR}`                        | * Information Trimming            |
+------------------------------------------+ * :math:`I_{\downarrow}`                | * Lautum Information              |
| Secret Key Agreement Bounds              | * :math:`I_{proj}`                      | * LMPR Complexity                 |
|                                          | * :math:`I_{BROJA}`                     | * Marginal Utility of Information |
| * Secrecy Capacity                       | * :math:`I_{ccs}`                       | * Maximum Correlation             |
| * Intrinsic Mutual Information           | * :math:`I_{\pm}`                       | * Maximum Entropy Distributions   |
| * Reduced Intrinsic Mutual Information   | * :math:`I_{sx}`                        | * Perplexity                      |
| * Minimal Intrinsic Mutual Information   | * :math:`I_{dep}`                       | * Rate-Distortion Theory          |
| * Necessary Intrinsic Mutual Information | * :math:`I_{RAV}`                       | * TSE Complexity                  |
| * Two-Part Intrinsic Mutual Information  | * :math:`I_{mmi}`                       | * Gray-Wyner Network              |
|                                          | * :math:`I_{\prec}`                     |                                   |
|                                          | * :math:`I_{RA}`                        |                                   |
|                                          | * :math:`I_{SKAR}`                      |                                   |
|                                          | * :math:`I_{IG}`                        |                                   |
|                                          | * :math:`I_{RDR}`                       |                                   |
|                                          | * :math:`I_{do}`                        |                                   |
+------------------------------------------+-----------------------------------------+-----------------------------------+

Quickstart
----------

The basic usage of ``dit`` corresponds to creating distributions, modifying them
if need be, and then computing properties of those distributions. First, we
import:

.. code:: python

   >>> import dit

Suppose we have a really thick coin, one so thick that there is a reasonable
chance of it landing on its edge. Here is how we might represent the coin in
``dit``.

.. code:: python

   >>> d = dit.Distribution(['H', 'T', 'E'], [.4, .4, .2])
   >>> print(d)
   Class:          Distribution
   Alphabet:       ('E', 'H', 'T') for all rvs
   Base:           linear
   Outcome Class:  str
   Outcome Length: 1
   RV Names:       None

   x   p(x)
   E   0.2
   H   0.4
   T   0.4

Calculate the probability of ``H`` and also of the combination ``H or T``.

.. code:: python

   >>> d['H']
   0.4
   >>> d.event_probability(['H','T'])
   0.8

Calculate the Shannon entropy and extropy of the joint distribution.

.. code:: python

   >>> dit.shannon.entropy(d)
   1.5219280948873621
   >>> dit.other.extropy(d)
   1.1419011889093373

Create a distribution where ``Z = xor(X, Y)``.

.. code:: python

   >>> import dit.example_dists
   >>> d = dit.example_dists.Xor()
   >>> d.set_rv_names(['X', 'Y', 'Z'])
   >>> print(d)
   Class:          Distribution
   Alphabet:       ('0', '1') for all rvs
   Base:           linear
   Outcome Class:  str
   Outcome Length: 3
   RV Names:       ('X', 'Y', 'Z')

   x     p(x)
   000   0.25
   011   0.25
   101   0.25
   110   0.25

Calculate the Shannon mutual informations ``I[X:Z]``, ``I[Y:Z]``, and
``I[X,Y:Z]``.

.. code:: python

   >>> dit.shannon.mutual_information(d, ['X'], ['Z'])
   0.0
   >>> dit.shannon.mutual_information(d, ['Y'], ['Z'])
   0.0
   >>> dit.shannon.mutual_information(d, ['X', 'Y'], ['Z'])
   1.0

Calculate the marginal distribution ``P(X,Z)``.
Then print its probabilities as fractions, showing the mask.

.. code:: python

   >>> d2 = d.marginal(['X', 'Z'])
   >>> print(d2.to_string(show_mask=True, exact=True))
   Class:          Distribution
   Alphabet:       ('0', '1') for all rvs
   Base:           linear
   Outcome Class:  str
   Outcome Length: 2 (mask: 3)
   RV Names:       ('X', 'Z')

   x     p(x)
   0*0   1/4
   0*1   1/4
   1*0   1/4
   1*1   1/4

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
   ['01', '10', '00', '01', '00']

Source and Channel Coding
-------------------------

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
   >>> float(channel_capacity(bec)[0])
   0.75

Contributions & Help
--------------------

If you'd like to feature added to ``dit``, please file an issue. Or, better yet, open a pull request. Ideally, all code should be tested and documented, but please don't let this be a barrier to contributing. We'll work with you to ensure that all pull requests are in a mergable state.

If you'd like to get in contact about anything, you can reach us through our `slack channel <https://dit-python.slack.com/>`_.


.. badges:

.. |build| image:: https://github.com/dit/dit/actions/workflows/build.yml/badge.svg
   :target: https://github.com/dit/dit/actions/workflows/build.yml
   :alt: Continuous Integration Status

.. |codecov| image:: https://codecov.io/gh/dit/dit/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/dit/dit
   :alt: Test Coverage Status

.. |docs| image:: https://readthedocs.org/projects/dit/badge/?version=latest
   :target: http://dit.readthedocs.org/en/latest/?badge=latest
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
