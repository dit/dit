.. tutorial.rst

***********
Basic Usage
***********

The basic usage of ``dit`` corresponds to creating distributions, modifying
them if need be, and then computing properties of those distributions. For
example::

   >>> from dit.example_dists import Xor
   >>> from dit.algorithms import entropy
   >>> d = Xor()
   >>> print(d)
   Class:          Distribution
   Alphabet:       ('0', '1') for all rvs
   Base:           linear
   Outcome Class:  str
   Outcome Length: 3
   RV Names:       None

   x     p(x)
   000   0.25
   011   0.25
   101   0.25
   110   0.25
   >>> print(entropy(d))
   2.0

Here, we imported an example distribution constructor (that of the logical
exclusive or) and the entropy function. Then we instantiated the XOR
distribution, printed it, and computed its entropy.
