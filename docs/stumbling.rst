.. stumbling.rst

(Stumbling Blocks) On the Road to Understanding Multivariate Information Theory
===============================================================================

A theory is more than just a set of measures. It also attributes meaning to
those measures, and ties that meaning to some sort of objective reality. At this
point, most understanding of multivariate information theory is flawed. In part,
this is due to several counter-intuitive situations which can arise in the study
of joint distributions.

In this document, we will discuss many of these examples which generally impact
the ability to construct universal understanding of multivariate mutual
information theory. It remains to be seen whether this is because multivariate
information theory is simply richer and more nuanced than our intuitions would
like, or if it is simply inadequate for the jobs it has been tasked with.


Necessity of Common Informations
--------------------------------

Our first pitfall is that the mutual information between two random variables
can not be embodied. That is, generically there does not exist a variable
:math:`Z` such that:

.. math::

   \I{X : Y | Z} = 0 \\
   \textrm{and} \\
   \H{Z} - \I{X : Y} = 0

That is, there is no :math:`Z` which captures the entirety of :math:`\I{X : Y}`
and nothing more.

When embodying the information shared by :math:`X` and :math:`Y` is desired,
one must make a choice. Choosing the variable capturing as much of
:math:`I{X : Y}` and nothing more results in the
:ref:`Gács-Körner Common Information`. Choosing the variable capturing all of
:math:`I{X : Y}` with as little else as possible results in the
:ref:`Exact Common Information`. If one chooses to incorporate only information
from :math:`X` we arrive as :math:`Z = X \mss Y`. Other choices are described
in :ref:`Common Informations`.

.. ipython::

   In [1]: d = Distribution(['00', '01', '10'], [1/3]*3)

   In [2]: I(d, [[0], [1]])
   Out[2]: 0.25162916738782304

   In [3]: K(d, [[0], [1]])
   Out[3]: 0.0

   In [4]: G(d, [[0], [1]])
   Out[4]: 0.9182909718428677

In this case, we see a wide gap. The largest random variable capturing nothing
outside of :math:`\I{X : Y}` is null, indicated by the Gács-Körner common
information being zero, while the smallest variable capturing all of
:math:`\I{X : Y}` is much larger, capturing two-thirds of a bit more than the
actual shared information.

Conditional Dependence
----------------------

Consider the duality between set theory and information theory. One simple
inequality in set theory is:

.. math::

   | X - Y | \leq | X |

and indeed the corresponding information theoretic inequality holds:

.. math::

   \H{X | Y} \leq \H{X}

Since the intersection of two sets is itself a set, the following inequality
also holds:

.. math::

   | (X \cap Y) - Z | \leq | X \cap Y |

We might then assume that its corresponding information-theoretic inequality
would hold:

.. math::

   \I{X : Y | Z} \leq \I{X : Y}

This, however, has a couple major difficulties. Firstly, the mutual information
between two variables does not itself correspond to a random variable, as we saw
in :ref:`Necessity of Common Informations` and so the analogy does not hold.
Secondly, the inequality does not hold. The most simplest counterexample is the
``xor`` distribution:

.. ipython::

   In [5]: d = Distribution(['000', '011', '101', '110'], [1/4]*4)

   In [6]: I(d, [[0], [1]])
   Out[6]: 0.0

   In [7]: I(d, [[0], [1]], [2])
   Out[7]: 1.0


Zero Probabilities
------------------

The following implication holds, so long as :math:`p(w, x, y, z) > 0`:

.. math::

   \left. \begin{array}{l} W \perp Z | (X, Y) \\ W \perp Y | (X, Z) \end{array} \right\} \implies W \perp (Y, Z) | X

This demonstrates that structural properties, such as conditional independence,
is sensitive to the distinction between "small" probability and zero
probability.

This becomes an issue when, for example, Bayesian methods are used to infer the
probability distribution. These methods will generally never set a probability
to zero and so will always exhibit this conditional independence even if the
underlying reality does not due to null probabilities. In this way, Bayesian
methods can systematically mislead a practitioner regarding the structural
independencies in a system.


Shannon-like Information Measures Are Insensitive to Structural Differences
---------------------------------------------------------------------------

Consider two distributions of three variables, each taking on four values. One
built by flipping three coins and assigning each to a different pair of
variables, the variable's state is then the concatenation of the two coins it
has access to. The second built by again flipping three coins, but this time
all variables share one of the coin flips, and then the other two coins and
their ``xor`` are each assigned to a variable. The first is constructed using
solely pairwise (dyadic) interactions, while the second using three-way
(triadic) interactions.

In spite of the fact that these two distributions are qualitatively quite
distinct, their informational signatures are all identical:

.. ipython::

   In [8]: from dit.example_dists import dyadic, triadic

   In [9]: from dit.profiles import ShannonPartition

   In [10]: ShannonPartition(dyadic)
   Out[10]:
   +----------+--------+
   | measure  |  bits  |
   +----------+--------+
   | H[0|1,2] |  0.000 |
   | H[1|0,2] |  0.000 |
   | H[2|0,1] |  0.000 |
   | I[0:1|2] |  1.000 |
   | I[0:2|1] |  1.000 |
   | I[1:2|0] |  1.000 |
   | I[0:1:2] |  0.000 |
   +----------+--------+

   In [11]: ShannonPartition(triadic)
   Out[11]:
   +----------+--------+
   | measure  |  bits  |
   +----------+--------+
   | H[0|1,2] |  0.000 |
   | H[1|0,2] |  0.000 |
   | H[2|0,1] |  0.000 |
   | I[0:1|2] |  1.000 |
   | I[0:2|1] |  1.000 |
   | I[1:2|0] |  1.000 |
   | I[0:1:2] |  0.000 |
   +----------+--------+

This result implies that any measure built form Shannon-like information
measures necessarily can not distinguish between distributions with different
scales of interaction.


Local Modifications Can Create Redundancy
-----------------------------------------

It is commonly believed that a non-zero coinformation value is a signature of
some sort of triadic interactions. Positive values indicate "redundancy", for
example a giant bit:

.. ipython::

   In [12]: d = Distribution(['000', '111'], [1/2]*2)

   In [13]: I(d)
   Out[13]: 1.0

Negative values indicate "synergy", for example the ``xor``:

.. ipython::

   In [14]: d = Distribution(['000', '011', '101', '110'], [1/4]*4)

   In [15]: I(d)
   Out[15]: -1.0


As seen in  :ref:`Shannon-like Information Measures Are Insensitive to
Structural Differences`, zero coinformation does not indicate a lack of triadic
interactions.

If we begin with a distribution lacking triadic interactions by construction,
the dyadic distribution from :ref:`Shannon-like Information Measures Are
Insensitive to Structural Differences`. If we then allow each variable to be
modified independent of the others while maximizing the coinformation, we
arrive at the :ref:`DeWeese-like Measures <DeWeese coinformation>`:

.. ipython::

   In [16]: from dit.multivariate import deweese_coinformation

   In [17]: deweese_coinformation(dyadic)
   Out[17]: 0.06127812445775139

This implies that cyclic pairwise interactions can be utilized to construct
triadic interactions.

Negative Coinformation Does Not Imply Threeway Interactions
-----------------------------------------------------------

Finally, does a negative coinformation imply triadic interactions? Consider
a distribution consisting of two random bits and their logical ``and``. This
distribution has a negative coinformation, implying conditional dependence and
some sort of triadic interaction. However, if we consider the family of
distributions which match ``and`` on its pairwise marginals, this family
consists of exactly one distribution: the ``and`` distribution!

.. ipython::

   In [18]: d = Distribution(['000', '010', '100', '111'], [1/4]*4)

   In [19]: I(d)
   Out[19]: -0.18872187554086706

   In [20]: maxent_dist(d, [[0, 1], [0, 2], [1, 2]])
   Out[20]:
   Class:          Distribution
   Alphabet:       ('0', '1') for all rvs
   Base:           linear
   Outcome Class:  str
   Outcome Length: 3
   RV Names:       None

   x     p(x)
   000   1/4
   010   1/4
   100   1/4
   111   1/4

   In [21]: from dit.algorithms.distribution_optimizers import MinEntOptimizer

   In [22]: meo = MinEntOptimizer(d, [[0, 1], [0, 2], [1, 2]])

   In [23]: meo.optimize()

   In [24]: meo.construct_dist()
   Out[24]:
   Class:          Distribution
   Alphabet:       ('0', '1') for all rvs
   Base:           linear
   Outcome Class:  str
   Outcome Length: 3
   RV Names:       None

   x     p(x)
   000   1/4
   010   1/4
   100   1/4
   111   1/4

And so this negative coinformation arises from cyclic, but strictly pairwise
interactions. We do note that a negative coinformation is not possible without
at least the cyclic pairwise constraints. But this raises an important
observation: negative coinformations can be constructed solely with pairwise
interactions, and so conditional dependence is not a phenomena which requires
triadic interactions.

Closing
-------

At this point one might suspect that information theory is in shambles, and not
up for the task of accurately detecting and quantifying dependencies. However,
I believe the limitation lies not with information theory but rather with our
impression of what it should be.
