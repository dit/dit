dit
===

`dit` is a Python package for information theory.  It's undergoing substantial
development, so the API will change as well.


Installation
============

Installation is easy, provided you have `git` and `pip` set up:

    pip install git+https://github.com/dit/dit/#egg=dit
    
And that's it! Alternatively, you can clone this repository:

    git clone https://github.com/dit/dit

Then, move into the newly created `dit` directory, and install the package 
using `pip`:

    pip install .

Usage
=====

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

Calculate the marginal distribution P(X,Z) and print its probabilities as fractions.

    >>> d2 = d.marginal(['X', 'Z'])
    >>> print d2.to_string(exact=True)
    Class:          Distribution
    Alphabet:       ('0', '1') for all rvs
    Base:           linear
    Outcome Class:  str
    Outcome Length: 2
    RV Names:       ('X', 'Z')

    x    p(x)
    00   1/4
    01   1/4
    10   1/4
    11   1/4

Convert the distribution probabilities to log (base 3.5) probabilities, and access its pmf.

    >>> d2.set_base(3.5)
    >>> d2.pmf
    array([-1.10658951, -1.10658951, -1.10658951, -1.10658951])
    
Draw 5 random samples from this distribution.

    >>> d2.rand(5)
    ['10', '11', '00', '01', '10']
    
    
