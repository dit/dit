.. notation.rst

********
Notation
********

``dit`` is a scientific tool, and so, much of this documentation will contain mathematical expressions. Here we will describe this notation.

Basic Notation
==============

A random variable :math:`X` consists of *outcomes* :math:`x` from an *alphabet* :math:`\mathcal{X}`. As such, we write the entropy of a distribution as :math:`\H{X} = \sum_{x \in \mathcal{X}} p(x) \log_2 p(x)`, where :math:`p(x)` denote the probability of the outcome :math:`x` occuring.

Many distributions are *joint* distribution. In the absence of variable names, we index each random variable with a subscript. For example, a distribution over three variables is written :math:`X_0X_1X_2`. As a shorthand, we also denote those random variables as :math:`X_{0:3}`, meaning start with :math:`X_0` and go through, but not including :math:`X_3` — just like python slice notation.

If a set of variables :math:`X_{0:n}` are independent, we will write :math:`\ind X_{0:n}`. If a set of variables :math:`X_{0:n}` are independent conditioned on :math:`V`, we write :math:`\ind X_{0:n} \mid V`.

If we ever need to describe an infinitely long chain of variables we drop the index from the side that is infinite. So :math:`X_{:0} = \ldots X_{-3}X_{-2}X_{-1}` and :math:`X_{0:} = X_0X_1X_2\ldots`. For an arbitrary set of indices :math:`A`, the corresponding collection of random variables is denoted :math:`X_A`. For example, if :math:`A = \{0,2,4\}`, then :math:`X_A = X_0 X_2 X_4`. The complement of :math:`A` (with respect to some universal set) is denoted :math:`\overline{A}`.

Furthermore, we define :math:`0 \log_2 0 = 0`.

Advanced Notation
=================

When there exists a function :math:`Y = f(X)` we write :math:`X \imore Y` meaning that :math:`X` is *informationally richer* than :math:`Y`. Similarly, if :math:`f(Y) = X` then we write :math:`X \iless Y` and say that :math:`X` is *informationally poorer* than :math:`Y`. If :math:`X \iless Y` and :math:`X \imore Y` then we write :math:`X \ieq Y` and say that :math:`X` is *informationally equivalent* to :math:`Y`. Of all the variables that are poorer than both :math:`X` and :math:`Y`, there is a richest one. This variable is known as the *meet* of :math:`X` and :math:`Y` and is denoted :math:`X \meet Y`. By definition, :math:`\forall Z s.t. Z \iless X` and :math:`Z \iless Y, Z \iless X \meet Y`. Similarly of all variables richer than both :math:`X` and :math:`Y`, there is a poorest. This variable is known as the *join* of :math:`X` and :math:`Y` and is denoted :math:`X \join Y`. The joint random variable :math:`(X,Y)` and the join are informationally equivalent: :math:`(X,Y) \ieq X \join Y`.

Lastly, we use :math:`X \mss Y` to denote the minimal sufficient statistic of :math:`X` about the random variable :math:`Y`.
