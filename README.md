dit
===

`dit` is a Python package for information theory.  It's undergoing substantial
development, so the API will change as well.


Installation
============

Installation is easy, provided you have `git` and `pip` set up:

    pip install git+https://github.com/dit/dit/#egg=dit

If you want to do this manually, clone this repository:

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

Create a distribution representing the XOR logic function.  We have two
inputs, X and Y, and then Z = XOR(X,Y).

    >>> import dit.example_dists
    >>> d = dit.example_dists.Xor()
    >>> d.set_rv_names(['X', 'Y', 'Z'])

Calculate the Shannon entropy of the joint distribution.

    >>> dit.algorithms.entropy(d)
    0.97095059445466858

Calculate the Shannon mutual informations I[X:Z], I[Y:Z], I[X,Y:Z].

    >>> dit.algorithms.mutual_information(d, ['X'], ['Z'])
    0.0
    >>> dit.algorithms.mutual_information(d, ['Y'], ['Z'])
    0.0
    >>> dit.algorithms.mutual_information(d, ['X', 'Y'], ['Z'])
    1.0

Calculate the marginal distribution P(X,Z).

    >>> d2 = d.marginal(['X', 'Z'])
    >>> print d2
    Class:          Distribution
    Alphabet:       ('0', '1') for all rvs
    Base:           linear
    Outcome Class:  str
    Outcome Length: 2
    RV Names:       ('X', 'Z')

    x    p(x)
    00   0.25
    01   0.25
    10   0.25
    11   0.25

