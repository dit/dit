
`dit` is a Python package for information theory.

.. image:: https://travis-ci.org/dit/dit.png?branch=master
   :target: https://travis-ci.org/dit/dit
.. image:: https://coveralls.io/repos/dit/dit/badge.png?branch=master
   :target: https://coveralls.io/r/dit/dit?branch=master

Documentation:
  Coming soon.

Downloads:
  Coming soon.
  
Dependencies:
  * Python 2.6, 2.7, 3.2, 3.3
  * numpy >= 1.6
  * iterutils >= 0.1.6
  * six >= 1.4.0

Optional Dependencies:
  * cython

Install:
  Until `dit` is on PyPI, the easiest way to install is::
  
      pip install git+https://github.com/dit/dit/#egg=dit
      
  Alternatively, you can clone this repository, move into the newly created `dit` directory, and then install the package.
  
      git clone https://github.com/dit/dit.git
      cd dit
      pip install .

Mailing list:
  None

Code and bug tracker:
  https://github.com/dit/dit

License:
  BSD 2-Clause License, see LICENSE.txt for details.

Quickstart
----------

Basic usage is as follows:

    >>> import dit

Create a biased coin and print it.

    >>> d = dit.Distribution(['H', 'T'], [.4, .6])
    >>> print d
    Class:          Distribution
    Alphabet:       ('H', 'T') for all rvs
    Base:           linear
    Outcome Class:  str
    Outcome Length: 1
    RV Names:       None

    x   p(x)
    H   0.4
    T   0.6
    
Calculate the probability of 'H' and also of 'H' or 'T'.

    >>> d['H']
    0.4
    >>> d.event_probability(['H','T'])
    1.0

Create a distribution representing the XOR logic function.  Here, we have two inputs, X and Y, and then an output 
Z = XOR(X,Y).

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

Calculate the Shannon entropy and extropy of the joint distribution.

    >>> dit.algorithms.entropy(d)
    0.97095059445466858
    >>> dit.algorithms.extropy(d)
    1.2451124978365313

Calculate the Shannon mutual informations I[X:Z], I[Y:Z], I[X,Y:Z].

    >>> dit.algorithms.mutual_information(d, ['X'], ['Z'])
    0.0
    >>> dit.algorithms.mutual_information(d, ['Y'], ['Z'])
    0.0
    >>> dit.algorithms.mutual_information(d, ['X', 'Y'], ['Z'])
    1.0

Calculate the marginal distribution P(X,Z). Then print its probabilities as fractions, showing the mask.

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

Convert the distribution probabilities to log (base 3.5) probabilities, and access its pmf.

    >>> d2.set_base(3.5)
    >>> d2.pmf
    array([-1.10658951, -1.10658951, -1.10658951, -1.10658951])
    
Draw 5 random samples from this distribution.

    >>> d2.rand(5)
    ['10', '11', '00', '01', '10']
    
    
