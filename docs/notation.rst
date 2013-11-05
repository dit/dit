.. notation.rst

********
Notation
********

``dit`` is a scientific tool, and so, much of this documentation will contain
mathematical expressions. Here we will describe this notation.

Many distributions are *joint* distribution. In the absence of variable names,
we index each random variable with a subscript. For example, a distribution
over three variables is written :math:`X_0X_1X_2`. As a shorthand, we also
denote those random variables as :math:`X_{0:3}`, meaning start with
:math:`X_0` and go through, but not including :math:`X_3` — just like python
slice notation.

If we ever need to describe an infinitely long chain of
variables we drop the index from the side that is infinite. So
:math:`X_{:0} = \ldots X_{-3}X_{-2}X_{-1}` and :math:`X_{0:} = X_0X_1X_2\ldots`.
For an arbitrary set of indices :math:`A`, the corresponding collection of
random variables is denoted :math:`X_A`. For example, if :math:`A = \{0,2,4\}`,
then :math:`X_A = X_0 X_2 X_4`. The complement of :math:`A`
(with respect to some universal set) is denoted :math:`\bar{A}`.

When there exists a function :math:`Y = f(X)` we write :math:`X \imore Y`
meaning that :math:`X` is *informationally richer* than :math:`Y`. Similarly, if
:math:`f(Y) = X` then we write :math:`X \iless Y` and say that :math:`X` is
*informationally poorer* than :math:`Y`.

.. note::
   need to describe the meet :math:`\meet` and join :math:`\join`.
