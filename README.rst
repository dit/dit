``dit`` is a Python package for information theory.

.. image:: https://travis-ci.org/dit/dit.png?branch=master
   :target: https://travis-ci.org/dit/dit
.. image:: https://coveralls.io/repos/dit/dit/badge.png?branch=master
   :target: https://coveralls.io/r/dit/dit?branch=master
.. image:: https://readthedocs.org/projects/dit/badge/?version=latest
   :target: https://readthedocs.org/projects/dit/?badge=latest

Documentation:
  http://docs.dit.io

Downloads:
  Coming soon.

Dependencies:
  * Python 2.6, 2.7, 3.2, or 3.3
  * numpy
  * iterutils
  * six

Optional Dependencies:
  * cython

Install:
  Until ``dit`` is available on PyPI, the easiest way to install is:

  .. code-block:: bash

      pip install git+https://github.com/dit/dit/#egg=dit

  Alternatively, you can clone this repository, move into the newly created ``dit`` directory, and then install the package:

  .. code-block:: bash

      git clone https://github.com/dit/dit.git
      cd dit
      pip install .

Mailing list:
  None

Code and bug tracker:
  https://github.com/dit/dit

License:
  BSD 2-Clause, see LICENSE.txt for details.

Quickstart
----------

The basic usage of ``dit`` corresponds to creating distributions, modifying
them if need be, and then computing properties of those distributions.
First, we import:

.. code:: python

   >>> import dit

Suppose we have a really thick coin, one so thick that there is a reasonable
chance of it landing on its edge. Here is how we might represent the coin in
``dit``.

.. code:: python

   >>> d = dit.Distribution(['H', 'T', 'E'], [.4, .4, .2])
   >>> print d
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
   >>> print d
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

Calculate the Shannon mutual informations ``I[X:Z]``, ``I[Y:Z]``, and ``I[X,Y:Z]``.

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
   >>> print d2.to_string(show_mask=True, exact=True)
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

Convert the distribution probabilities to log (base 3.5)
probabilities, and access its probability mass function.

.. code:: python

   >>> d2.set_base(3.5)
   >>> d2.pmf
   array([-1.10658951, -1.10658951, -1.10658951, -1.10658951])

Draw 5 random samples from this distribution.

.. code:: python

   >>> dit.math.prng.seed(1)
   >>> d2.rand(5)
   ['01', '10', '00', '01', '00']

Enjoy!
