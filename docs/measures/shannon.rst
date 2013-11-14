.. shannon.rst
.. py:module :: dit.algorithms.shannon

**********************
Basic Shannon measures
**********************

The information on this page is drawn from the fantastic text book **Elements of
Information Theory** by Cover and Thomas :cite:`Cover2006`. Other good choices
are **Information Theory, Inference and Learning Algorithms** by MacKay
:cite:`MacKay2003` and **Information Theory and Network Coding** by Yeung
:cite:`Yeung2008`.

Entropy
=======

The entropy measures how much information is in a random variable :math:`X`.

.. math::

   \H[X] = - \sum_{x \in X} p(x) \log_2 p(x)

What do we mean by "how much information"? Basically, we mean the average number
of yes-no questions one would have to ask to determine an outcome from the
distribution. In the simplest case, consider a sure thing:

.. code-block:: python

   >>> d = dit.Distribution(['H'], [1])
   >>> dit.algorithms.entropy(d)
   0.0

So is we know that the outcome from our distribution will always be `H`, we have
to ask zero questions to figure that out. If however we have a fair coin:

.. code-block:: python

   >>> d = dit.Distribution(['H', 'T'], [1/2, 1/2])
   >>> dit.algorithms.entropy(d)
   1.0

The entropy tells us that we must ask one question to determine whether an `H`
or `T` was the outcome of the coin flip. Now what if there are three outcomes?
Let's consider the following situation:

.. code-block:: python

   >>> d = dit.Distribution(['A', 'B', 'C'], [1/2, 1/4, 1/4])
   >>> dit.algorithms.entropy(d)
   1.5

Here we find that the entropy is 1.5 bits. How do we ask one and a half
questions on average? Well, if our first question is "was it `A`?" and it is
true, then we are done, and that occurs half the time. The other half of the
time we need to ask a follow up question: "was it `B`?". So half the time we
need to ask one question, and the other half of the time we need to ask two
questions. In other words, we need to ask 1.5 questions on average.

Joint Entropy
-------------

The entropy of multiple variables is computed in a similar manner:

.. math::

   \H[X_{0:n}] = \sum_{x_{0:n} \in X_{0:n}} p(x_{0:n}) \log_2 p(x_{0:n})

Its intuition is also the same: the average number of binary questions required
to identify a joint event from the distribution.

.. autofunction:: dit.algorithms.shannon.entropy

Conditional Entropy
===================

The conditional entropy is the amount of information in variable :math:`X`
beyond that which is in variable :math:`Y`.

.. math::

   \H[X|Y] = \sum_{x \in X, y \in Y} p(x, y) \log_2 p(x|y)

.. autofunction:: dit.algorithms.shannon.conditional_entropy

Mutual Information
==================

The mutual information is the amount of information shared by :math:`X` and
:math:`Y`.

.. math::

   \I[X:Y] &= \H[X,Y] - \H[X|Y] - \H[Y|X] \\
           &= \H[X] + \H[Y] - \H[X,Y] \\
           &= \sum_{x \in X, y \in Y} p(x, y) \log_2 \frac{p(x, y)}{p(x)p(y)}

.. todo::

   Add i-diagrams.

.. todo::

   Add discussion.

.. todo::

   Add examples.

.. autofunction:: dit.algorithms.shannon.mutual_information
