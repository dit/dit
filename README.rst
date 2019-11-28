``dit`` is a Python package for information theory.

|build| |build_windows| |codecov| |codacy| |deps|

|docs| |slack| |saythanks| |conda|

|joss| |zenodo|

Try ``dit`` live: |binder|

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
| * Python 3.3+                                                     |
| * `boltons <https://boltons.readthedocs.io>`_                     |
| * `contextlib2 <https://contextlib2.readthedocs.io>`_             |
| * `debtcollector <https://docs.openstack.org/debtcollector/>`_    |
| * `lattices <https://github.com/dit/lattices>`_                   |
| * `networkx <https://networkx.github.io/>`_                       |
| * `numpy <http://www.numpy.org/>`_                                |
| * `PLTable <https://github.com/platomav/PLTable>`_                |
| * `scipy <https://www.scipy.org/>`_                               |
| * `six <http://pythonhosted.org/six/>`_                           |
+-------------------------------------------------------------------+

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~
* colorama: colored column heads in PID indicating failure modes
* cython: faster sampling from distributions
* hypothesis: random sampling of distributions
* matplotlib, python-ternary: plotting of various information-theoretic expansions
* numdifftools: numerical evaluation of gradients and hessians during optimization
* pint: add units to informational values
* scikit-learn: faster nearest-neighbor lookups during entropy/mutual information estimation from samples

Install
*******

The easiest way to install is:

.. code-block:: bash

  pip install dit

If you want to install `dit` within a conda environment, you can simply do:

.. code-block:: bash

  conda install -c conda-forge dit

Alternatively, you can clone this repository, move into the newly created
``dit`` directory, and then install the package:

.. code-block:: bash

  git clone https://github.com/dit/dit.git
  cd dit
  pip install .

.. note::

  The cython extensions are currently not supported on windows. Please install
  using the ``--nocython`` option.


Testing
*******
.. code-block:: shell

  $ git clone https://github.com/dit/dit.git
  $ cd dit
  $ pip install -r requirements_testing.txt
  $ py.test

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
| * Renyi Entropy                          | * Interaction Information               | * Kullback-Leibler Divergence \   |
| * Tsallis Entropy                        | * Total Correlation /                   |   Relative Entropy                |
| * Necessary Conditional Entropy          |   Multi-Information                     | * Cross Entropy                   |
| * Residual Entropy /                     | * Dual Total Correlation /              | * Jensen-Shannon Divergence       |
|   Independent Information /              |   Binding Information                   | * Earth Mover's Distance          |
|   Variation of Information               | * CAEKL Multivariate Mutual Information +-----------------------------------+
+------------------------------------------+-----------------------------------------+ Other Measures                    |
| Common Informations                      | Partial Information Decomposition       |                                   |
|                                          |                                         | * Channel Capacity                |
| * Gacs-Korner Common Information         | * :math:`I_{min}`                       | * Complexity Profile              |
| * Wyner Common Information               | * :math:`I_{\wedge}`                    | * Connected Informations          |
| * Exact Common Information               | * :math:`I_{\downarrow}`                | * Cumulative Residual Entropy     |
| * Functional Common Information          | * :math:`I_{proj}`                      | * Extropy                         |
| * MSS Common Information                 | * :math:`I_{BROJA}`                     | * Hypercontractivity Coefficient  |
+------------------------------------------+ * :math:`I_{ccs}`                       | * Information Bottleneck          |
| Secret Key Agreement Bounds              | * :math:`I_{\pm}`                       | * Information Diagrams            |
|                                          | * :math:`I_{dep}`                       | * Information Trimming            |
| * Intrinsic Mutual Information           | * :math:`I_{RAV}`                       | * Lautum Information              |
| * Reduced Intrinsic Mutual Information   |                                         | * LMPR Complexity                 |
| * Minimal Intrinsic Mutual Information   |                                         | * Marginal Utility of Information |
| * Necessary Intrinsic Mutual Information |                                         | * Maximum Correlation             |
| * Secrecy Capacity                       |                                         | * Maximum Entropy Distributions   |
|                                          |                                         | * Perplexity                      |
|                                          |                                         | * Rate-Distortion Theory          |
|                                          |                                         | * TSE Complexity                  |
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

Contributions & Help
--------------------

If you'd like to feature added to ``dit``, please file an issue. Or, better yet, open a pull request. Ideally, all code should be tested and documented, but please don't let this be a barrier to contributing. We'll work with you to ensure that all pull requests are in a mergable state.

If you'd like to get in contact about anything, you can reach us through our `slack channel <https://dit-python.slack.com/>`_.


.. badges:

.. |build| image:: https://travis-ci.org/dit/dit.png?branch=master
   :target: https://travis-ci.org/dit/dit
   :alt: Continuous Integration Status

.. |build_windows| image:: https://ci.appveyor.com/api/projects/status/idb5hc5gm59whf8m?svg=true
   :target: https://ci.appveyor.com/project/Autoplectic/dit
   :alt: Continuous Integration Status (windows)

.. |codecov| image:: https://codecov.io/gh/dit/dit/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/dit/dit
  :alt: Test Coverage Status

.. |coveralls| image:: https://coveralls.io/repos/dit/dit/badge.svg?branch=master
   :target: https://coveralls.io/r/dit/dit?branch=master
   :alt: Test Coverage Status

.. |docs| image:: https://readthedocs.org/projects/dit/badge/?version=latest
   :target: http://dit.readthedocs.org/en/latest/?badge=latest
   :alt: Documentation Status

.. |health| image:: https://landscape.io/github/dit/dit/master/landscape.svg?style=flat
   :target: https://landscape.io/github/dit/dit/master
   :alt: Code Health

.. |codacy| image:: https://api.codacy.com/project/badge/Grade/b1beeea8ada647d49f97648216fd9687
   :target: https://www.codacy.com/app/Autoplectic/dit?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=dit/dit&amp;utm_campaign=Badge_Grade
   :alt: Code Quality

.. |deps| image:: https://requires.io/github/dit/dit/requirements.svg?branch=master
   :target: https://requires.io/github/dit/dit/requirements/?branch=master
   :alt: Requirements Status

.. |conda| image:: https://anaconda.org/conda-forge/dit/badges/installer/conda.svg
   :target: https://anaconda.org/conda-forge/dit
   :alt: Conda installation

.. |zenodo| image:: https://zenodo.org/badge/13201610.svg
   :target: https://zenodo.org/badge/latestdoi/13201610
   :alt: DOI

.. |gitter| image:: https://badges.gitter.im/Join%20Chat.svg
   :target: https://gitter.im/dit/dit?utm_source=badge&utm_medium=badge
   :alt: Join the Chat

.. |saythanks| image:: https://img.shields.io/badge/SayThanks.io-%E2%98%BC-1EAEDB.svg
   :target: https://saythanks.io/to/Autoplectic
   :alt: Say Thanks!

.. |depsy| image:: http://depsy.org/api/package/pypi/dit/badge.svg
   :target: http://depsy.org/package/python/dit
   :alt: Research software impact

.. |waffle| image:: https://badge.waffle.io/dit/dit.png?label=ready&title=Ready
   :target: https://waffle.io/dit/dit?utm_source=badge
   :alt: Stories in Ready

.. |slack| image:: https://img.shields.io/badge/Slack-dit--python-lightgrey.svg
   :target: https://dit-python.slack.com/
   :alt: dit chat

.. |joss| image:: http://joss.theoj.org/papers/10.21105/joss.00738/status.svg
   :target: https://doi.org/10.21105/joss.00738
   :alt: JOSS Status

.. |binder| image:: https://mybinder.org/badge.svg
   :target: https://mybinder.org/v2/gh/dit/dit/master?filepath=examples
   :alt: Run `dit` live!
